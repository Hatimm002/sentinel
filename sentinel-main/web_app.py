from __future__ import annotations

from pathlib import Path
import json
import queue
import re
import threading
import time
import os

import cv2
from flask import Flask, Response, jsonify, redirect, render_template_string, request, session, url_for
from functools import wraps
import joblib
import pandas as pd
import serial
from serial import SerialException
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(name, value)


load_local_env()
configured_arduino_port = os.getenv("ARDUINO_PORT", "COM11")
if os.name == "nt" and configured_arduino_port.startswith("/dev/"):
    configured_arduino_port = "COM11"
ARDUINO_PORT = configured_arduino_port
ARDUINO_BAUDRATE = int(os.getenv("ARDUINO_BAUDRATE", "115200"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_EVERY_N_FRAMES = 3
YOLO_IMAGE_SIZE = 320
JPEG_QUALITY = 75
DEFAULT_CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
MODEL_PATH = BASE_DIR / "Model" / "isolation_forest_model.joblib"
MODEL_FEATURES = ["temperature_C", "humidite_pct"]
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY", "sentinel-x-local-admin-key-change-me"
)
state_lock = threading.Lock()
state = {
    "temperature": None,
    "humidity": None,
    "movement": False,
    "person_detected": False,
    "serial_connected": False,
    "camera_connected": False,
    "message": "Démarrage...",
    "updated_at": None,
    "last_serial_line": "",
    "anomaly": None,
    "anomaly_score": None,
    "critical": False,
}
latest_jpeg: bytes | None = None
subscribers: list[queue.Queue[str]] = []
requested_camera_index = DEFAULT_CAMERA_INDEX
arduino_connection: serial.Serial | None = None
arduino_write_lock = threading.Lock()
BUZZER_COMMAND_COOLDOWN = 8000
last_buzzer_command = 0.0


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("authenticated") is not True:
            if request.path.startswith("/api/") or request.path in {"/events", "/camera"}:
                return jsonify({"error": "Authentification requise"}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


@app.get("/login")
def login() -> str:
    return render_template_string(
        """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SENTINEL-X | Connexion</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#070b14;color:#f3f7ff;font-family:system-ui}
.box{width:min(390px,calc(100% - 40px));padding:34px;border:1px solid #25334d;border-radius:22px;background:#101827;box-shadow:0 20px 60px #0009}
h1{letter-spacing:3px;margin:0 0 8px}.sub{color:#8fa1bd;margin:0 0 25px}label{display:block;color:#8fa1bd;font-size:13px;margin:15px 0 7px}
input{width:100%;box-sizing:border-box;padding:13px;border-radius:10px;border:1px solid #31415d;background:#0b1322;color:#fff;font-size:15px}
button{width:100%;margin-top:22px;padding:13px;border:0;border-radius:10px;background:linear-gradient(135deg,#54d8ff,#7a8cff);font-weight:700;cursor:pointer}
.error{color:#ff6d86;margin-top:15px}
</style></head><body><form class="box" method="post"><h1>SENTINEL-X</h1>
<p class="sub">Accès administrateur sécurisé</p><label>Utilisateur</label>
<input name="username" autocomplete="username" required><label>Mot de passe</label>
<input name="password" type="password" autocomplete="current-password" required>
<button type="submit">Se connecter</button>{% if error %}<div class="error">{{ error }}</div>{% endif %}
</form></body></html>""",
        error=request.args.get("error"),
    )


@app.post("/login")
def login_submit():
    if (
        request.form.get("username") == ADMIN_USERNAME
        and request.form.get("password") == ADMIN_PASSWORD
    ):
        session["authenticated"] = True
        return redirect(url_for("index"))
    return redirect(url_for("login", error="Identifiants invalides"))


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


def load_isolation_forest() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle joblib introuvable: {MODEL_PATH}")
    package = joblib.load(MODEL_PATH)
    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("Le fichier joblib ne contient pas un package de modèle valide.")
    features = package.get("features")
    if features != MODEL_FEATURES:
        raise ValueError(
            f"Features joblib invalides: {features!r}; attendu {MODEL_FEATURES!r}"
        )
    model = package["model"]
    if not hasattr(model, "predict"):
        raise TypeError("Le modèle joblib ne fournit pas de méthode predict().")
    print(f"Modèle Isolation Forest chargé depuis {MODEL_PATH}.")
    return model


def parse_sensor_values(line: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in line.replace(";", ",").split(",")]
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    match = re.search(
        r"Temp(?:erature)?\s*:\s*(-?\d+(?:[.,]\d+)?)"
        r".*?Humidit(?:e|é)\s*:\s*(-?\d+(?:[.,]\d+)?)",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1).replace(",", ".")), float(
            match.group(2).replace(",", ".")
        )
    return None


def update_state(**values: object) -> None:
    with state_lock:
        state.update(values)
        state["updated_at"] = time.strftime("%H:%M:%S")
        message = json.dumps(state, ensure_ascii=False)
        for subscriber in subscribers[:]:
            try:
                subscriber.put_nowait(message)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(message)
                except queue.Empty:
                    pass


def refresh_alert_state() -> None:
    with state_lock:
        critical = bool(
            state["anomaly"] or state["person_detected"]
        )
    update_state(critical=critical)


def request_buzzer_alert(source: str) -> None:
    global last_buzzer_command
    now = time.monotonic()
    if now - last_buzzer_command < BUZZER_COMMAND_COOLDOWN / 1000:
        return
    with arduino_write_lock:
        if arduino_connection is None or not arduino_connection.is_open:
            return
        try:
            arduino_connection.write(b"BUZZER_ALERT\n")
            arduino_connection.flush()
            last_buzzer_command = now
            print(f"[BUZZER] Alerte envoyée ({source}).")
        except SerialException as error:
            print(f"[BUZZER] Commande impossible ({source}): {error}")


def sensor_worker(anomaly_model: object) -> None:
    global arduino_connection
    try:
        arduino = serial.Serial(
            ARDUINO_PORT,
            ARDUINO_BAUDRATE,
            timeout=1,
            dsrdtr=False,
            rtscts=False,
        )
    except SerialException as error:
        update_state(
            serial_connected=False,
            message=(
                f"Arduino indisponible sur {ARDUINO_PORT}: {error}. "
                "Ferme le moniteur série et vérifie le port."
            ),
        )
        return
    arduino.dtr = False
    arduino.rts = False
    with arduino_write_lock:
        arduino_connection = arduino
    update_state(serial_connected=True, message="Arduino connecté")
    print(f"Lecture série active sur {ARDUINO_PORT} à {ARDUINO_BAUDRATE} bauds.")
    print("Attente des mesures ESP32...")
    try:
        while True:
            line = arduino.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            print(f"[ESP32] {line}")
            update_state(last_serial_line=line)
            line_lower = line.lower()
            if (
                "mouvement termine" in line_lower
                or "mouvement_termine" in line_lower
            ):
                update_state(movement=False, message="Surveillance normale")
                refresh_alert_state()
                continue
            if "mouvement" in line_lower:
                update_state(movement=True, message="Mouvement détecté")
                refresh_alert_state()
                continue
            values = parse_sensor_values(line)
            if values is None:
                continue
            temperature, humidity = values
            features = pd.DataFrame(
                [[temperature, humidity]], columns=MODEL_FEATURES
            )
            prediction = int(anomaly_model.predict(features)[0])
            anomaly = prediction == -1
            anomaly_score = float(anomaly_model.decision_function(features)[0])
            update_state(
                temperature=temperature,
                humidity=humidity,
                anomaly=anomaly,
                anomaly_score=round(anomaly_score, 4),
                critical=anomaly,
                message="ANOMALIE DÉTECTÉE PAR LE MODÈLE" if anomaly else "Mesures normales selon le modèle",
            )
            refresh_alert_state()
            if anomaly:
                request_buzzer_alert("Isolation Forest")
    finally:
        with arduino_write_lock:
            if arduino_connection is arduino:
                arduino_connection = None
        arduino.close()


def camera_worker() -> None:
    global latest_jpeg

    camera = None
    active_index = None
    frame_number = 0
    last_person_detected = False
    vision_model = YOLO("yolov8n.pt")

    try:
        while True:
            with state_lock:
                selected_index = requested_camera_index
            if camera is None or selected_index != active_index:
                if camera is not None:
                    camera.release()
                camera = cv2.VideoCapture(selected_index, cv2.CAP_DSHOW)
                if not camera.isOpened():
                    camera.release()
                    camera = None
                    update_state(camera_connected=False, message=f"Caméra {selected_index} indisponible")
                    time.sleep(1)
                    continue
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                active_index = selected_index
                update_state(camera_connected=True, camera_index=active_index, message=f"Caméra {active_index} active")
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame_number += 1
            if frame_number % YOLO_EVERY_N_FRAMES == 0:
                result = vision_model(
                    frame,
                    classes=[0],
                    imgsz=YOLO_IMAGE_SIZE,
                    verbose=False,
                )[0]
                last_person_detected = len(result.boxes) > 0
                update_state(person_detected=last_person_detected)
                refresh_alert_state()
                if last_person_detected:
                    request_buzzer_alert("personne détectée")
                frame = result.plot()
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if ok:
                with state_lock:
                    latest_jpeg = encoded.tobytes()
            time.sleep(0.01)
    finally:
        if camera is not None:
            camera.release()


@app.get("/")
@login_required
def index() -> str:
    return """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SENTINEL-X | Centre de surveillance</title>
<style>
:root{--bg:#070b14;--panel:#101827;--panel2:#151f32;--line:#25334d;--text:#f3f7ff;--muted:#8fa1bd;--cyan:#54d8ff;--green:#49e19a;--red:#ff6d86;--shadow:0 20px 60px #0008}
*{box-sizing:border-box}body{margin:0;min-height:100vh;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text);background:radial-gradient(circle at 85% 0,#123455 0,transparent 30%),radial-gradient(circle at 10% 20%,#102441 0,transparent 32%),var(--bg)}
.shell{max-width:1440px;margin:auto;padding:28px 34px 42px}.topbar{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:30px}
.brand{display:flex;align-items:center;gap:14px}.brand-mark{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;color:#06101e;background:linear-gradient(135deg,var(--cyan),#7a8cff);font-weight:900;font-size:20px;box-shadow:0 0 35px #54d8ff55}.eyebrow{font-size:11px;letter-spacing:2px;color:var(--cyan);text-transform:uppercase}.brand h1{font-size:25px;letter-spacing:3px;margin:2px 0 0}.connection{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px}.dot{width:9px;height:9px;background:var(--green);border-radius:50%;box-shadow:0 0 14px var(--green)}.dot.off{background:var(--red);box-shadow:0 0 14px var(--red)}
.hero{display:flex;justify-content:space-between;align-items:end;margin-bottom:22px}.hero h2{font-size:clamp(25px,4vw,42px);margin:0 0 8px;letter-spacing:-1px}.hero p{margin:0;color:var(--muted)}.live{border:1px solid #49e19a55;color:var(--green);padding:8px 12px;border-radius:99px;font-size:12px;letter-spacing:1px;background:#49e19a0d}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:18px}.card,.video-panel,.side-panel{background:linear-gradient(145deg,#162239e8,#0d1524e8);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}.metric{padding:22px 24px;position:relative;overflow:hidden}.metric:after{content:"";position:absolute;width:100px;height:100px;right:-40px;bottom:-55px;border-radius:50%;background:#54d8ff18}.metric-top{display:flex;justify-content:space-between;color:var(--muted);font-size:13px}.metric-icon{color:var(--cyan);font-size:18px}.value{font-size:34px;font-weight:750;letter-spacing:-1px;margin-top:18px}.unit{font-size:15px;color:var(--muted);font-weight:500}.ok{color:var(--green)}.alert{color:var(--red);text-shadow:0 0 18px #ff6d8666}.muted{color:var(--muted)}
.critical{animation:dangerPulse 1s ease-in-out infinite;background:radial-gradient(circle at 50% 0,#8d173b 0,transparent 45%),var(--bg)}.urgent{display:none;position:fixed;z-index:5;inset:0;background:#ff174455;pointer-events:none;animation:flash .9s infinite}.critical .urgent{display:block}@keyframes dangerPulse{50%{box-shadow:inset 0 0 100px #e21b4f33}}@keyframes flash{50%{opacity:.15}}
.workspace{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.55fr);gap:18px}.video-panel{padding:14px}.panel-head{display:flex;justify-content:space-between;align-items:center;padding:4px 8px 14px}.panel-title{font-weight:700}.panel-subtitle{font-size:12px;color:var(--muted);margin-top:3px}.camera-wrap{position:relative;overflow:hidden;border-radius:14px;background:#05080e;aspect-ratio:16/9}.camera-wrap img{width:100%;height:100%;object-fit:cover;display:block}.camera-badge{position:absolute;left:14px;top:14px;background:#05080ecc;border:1px solid #ffffff1f;padding:7px 10px;border-radius:8px;font-size:11px;color:var(--green)}
.side-panel{padding:22px}.side-panel h3{font-size:14px;margin:0 0 18px}.status-row{display:flex;justify-content:space-between;gap:15px;padding:14px 0;border-bottom:1px solid var(--line);font-size:13px}.status-row span:first-child{color:var(--muted)}select{width:100%;margin-top:10px;padding:12px 13px;color:var(--text);background:#0b1322;border:1px solid #31415d;border-radius:11px;font-size:14px;outline:none}select:focus{border-color:var(--cyan);box-shadow:0 0 0 3px #54d8ff22}.serial{margin-top:20px;font-size:11px;color:var(--muted);line-height:1.6;word-break:break-word}.footer{display:flex;justify-content:space-between;gap:15px;margin-top:20px;color:#60718e;font-size:11px}
@media(max-width:900px){.workspace{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:620px){.shell{padding:20px 15px}.topbar,.hero{align-items:flex-start;flex-direction:column}.metrics{grid-template-columns:1fr}.connection{align-self:flex-end}}
</style></head><body><div class="urgent"></div><main class="shell">
<header class="topbar"><div class="brand"><div class="brand-mark">S</div><div><div class="eyebrow">Smart monitoring system</div><h1>SENTINEL-X</h1></div></div><div class="connection"><span id="connectionDot" class="dot"></span><span id="connectionText">Initialisation...</span></div></header>
<section class="hero"><div><h2>Centre de surveillance</h2><p id="message">Connexion aux capteurs en cours...</p></div><div class="live">● LIVE</div></section>
<section class="metrics">
<article class="card metric"><div class="metric-top"><span>Température</span><span class="metric-icon">°C</span></div><div id="temperature" class="value">-- <span class="unit">°C</span></div></article>
<article class="card metric"><div class="metric-top"><span>Humidité relative</span><span class="metric-icon">%</span></div><div id="humidity" class="value">-- <span class="unit">%</span></div></article>
<article class="card metric"><div class="metric-top"><span>Isolation Forest</span><span class="metric-icon">AI</span></div><div id="anomaly" class="value muted">--</div><div id="anomalyScore" class="panel-subtitle">Score modèle : --</div></article>
<article class="card metric"><div class="metric-top"><span>Détection mouvement</span><span class="metric-icon">◉</span></div><div id="movement" class="value ok">Aucun</div></article>
</section>
<section class="workspace"><article class="video-panel"><div class="panel-head"><div><div class="panel-title">Flux caméra</div><div class="panel-subtitle">Analyse vidéo intelligente en direct</div></div><div id="cameraStatus" class="panel-subtitle">Caméra --</div></div><div class="camera-wrap"><img src="/camera" alt="Flux caméra"><div class="camera-badge">● LIVE / YOLO</div></div></article>
<aside class="side-panel"><h3>État du système</h3><div class="status-row"><span>Arduino</span><strong id="arduinoStatus">--</strong></div><div class="status-row"><span>Personne</span><strong id="personStatus">--</strong></div><div class="status-row"><span>Dernière mise à jour</span><strong id="updatedAt">--</strong></div><label class="panel-subtitle">Source caméra<select id="cameraSelect"></select></label><div id="serial" class="serial">Dernière trame série : --</div><form method="post" action="/logout"><button type="submit">Se déconnecter</button></form></aside></section>
<footer class="footer"><span>SENTINEL-X / Monitoring local</span><span>Temps réel via EventSource</span></footer></main>
<script>
let movementDisplayTimer=null;
function render(s){
document.body.classList.toggle('critical',Boolean(s.critical));
document.getElementById('temperature').innerHTML=s.temperature===null?'-- <span class="unit">°C</span>':s.temperature.toFixed(1)+' <span class="unit">°C</span>';
document.getElementById('humidity').innerHTML=s.humidity===null?'-- <span class="unit">%</span>':s.humidity.toFixed(1)+' <span class="unit">%</span>';
const m=document.getElementById('movement');m.textContent=s.movement?'DÉTECTÉ':'Aucun';m.className='value muted';
if(movementDisplayTimer!==null)clearTimeout(movementDisplayTimer);
if(s.movement){
 movementDisplayTimer=setTimeout(()=>{
  m.textContent='Aucun';m.className='value ok';
  if(!s.anomaly&&!s.person_detected)document.body.classList.remove('critical');
 },2500);
}
const a=document.getElementById('anomaly');a.textContent=s.anomaly===null?'--':(s.anomaly?'ANOMALIE':'NORMAL');a.className='value '+(s.anomaly?'alert':'ok');
document.getElementById('anomalyScore').textContent='Score modèle : '+(s.anomaly_score===null?'--':s.anomaly_score.toFixed(4));
document.getElementById('message').textContent=s.message;
document.getElementById('connectionText').textContent=s.serial_connected?'Système opérationnel':'Arduino hors ligne';
document.getElementById('connectionDot').className='dot '+(s.serial_connected?'':'off');
document.getElementById('arduinoStatus').textContent=s.serial_connected?'Connecté':'Hors ligne';
document.getElementById('arduinoStatus').className=s.serial_connected?'ok':'alert';
document.getElementById('personStatus').textContent=s.person_detected?'Détectée':'Aucune';
document.getElementById('personStatus').className=s.person_detected?'alert':'ok';
document.getElementById('updatedAt').textContent=s.updated_at||'--';
document.getElementById('cameraStatus').textContent=s.camera_connected?'Caméra '+(s.camera_index??'')+' active':'Caméra indisponible';
document.getElementById('serial').textContent='Dernière trame série : '+(s.last_serial_line||'aucune');
if(s.camera_index!==undefined)document.getElementById('cameraSelect').value=s.camera_index;
}
const events=new EventSource('/events');
events.onmessage=(event)=>render(JSON.parse(event.data));
events.onerror=()=>document.getElementById('message').textContent='Connexion temps réel interrompue';
fetch('/api/status').then(response=>response.json()).then(render);
fetch('/api/cameras').then(response=>response.json()).then(cameras=>{
const select=document.getElementById('cameraSelect');
cameras.forEach(camera=>{const option=document.createElement('option');option.value=camera.index;option.textContent=camera.label;option.disabled=!camera.available;if(camera.index===1)option.selected=true;select.appendChild(option)});
select.onchange=()=>fetch('/api/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:Number(select.value)})});
});
</script></body></html>"""


@app.get("/api/status")
@login_required
def api_status() -> Response:
    with state_lock:
        return jsonify(state)


@app.get("/api/cameras")
@login_required
def api_cameras() -> Response:
    cameras = []
    for index in range(5):
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        available = camera.isOpened()
        camera.release()
        cameras.append(
            {"index": index, "label": f"Caméra {index}", "available": available}
        )
    return jsonify(cameras)


@app.post("/api/camera")
@login_required
def select_camera() -> Response:
    global requested_camera_index
    payload = request.get_json(silent=True) or {}
    index = payload.get("index")
    if not isinstance(index, int) or not 0 <= index <= 4:
        return jsonify({"error": "Index caméra invalide"}), 400
    requested_camera_index = index
    return jsonify({"ok": True, "index": index})


@app.get("/events")
@login_required
def events() -> Response:
    subscriber: queue.Queue[str] = queue.Queue(maxsize=10)
    with state_lock:
        subscribers.append(subscriber)
        initial_state = json.dumps(state, ensure_ascii=False)

    def stream():
        try:
            yield f"data: {initial_state}\n\n"
            while True:
                try:
                    message = subscriber.get(timeout=15)
                    yield f"data: {message}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with state_lock:
                if subscriber in subscribers:
                    subscribers.remove(subscriber)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/movement/reset")
@login_required
def reset_movement() -> Response:
    update_state(movement=False)
    return jsonify({"ok": True})


@app.get("/camera")
@login_required
def camera_feed() -> Response:
    def frames():
        while True:
            with state_lock:
                frame = latest_jpeg
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    isolation_forest = load_isolation_forest()
    threading.Thread(
        target=sensor_worker, args=(isolation_forest,), daemon=True
    ).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=False,
        threaded=True,
    )
