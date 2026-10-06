from __future__ import annotations

from pathlib import Path
import re

import cv2
import pandas as pd
from sklearn.ensemble import IsolationForest
from ultralytics import YOLO

try:
    import serial
    from serial import SerialException
except ImportError as error:
    raise SystemExit(
        "pyserial est requis pour lire l'Arduino. "
        "Installe-le avec: pip install pyserial"
    ) from error


BASE_DIR = Path(__file__).resolve().parent
NORMAL_DATA_PATH = BASE_DIR / "Model" / "temperature_humidite_normal.csv"
FALLBACK_DATA_PATH = BASE_DIR / "Model" / "temperature_humidite_test_aberrant.csv"
ARDUINO_PORT = "COM11"
ARDUINO_BAUDRATE = 115200
FEATURE_COLUMNS = ["temperature_C", "humidite_pct"]


def train_isolation_forest() -> IsolationForest:
    """Apprend le comportement normal au démarrage du programme."""
    data_path = NORMAL_DATA_PATH
    if not data_path.exists():
        data_path = FALLBACK_DATA_PATH
        print(
            "AVERTISSEMENT: temperature_humidite_normal.csv est absent. "
            "Le CSV de test est utilisé; crée un CSV normal pour une détection fiable."
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Fichier d'apprentissage introuvable: {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans {data_path}: {missing_columns}")

    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(features) < 10:
        raise ValueError("Le fichier d'apprentissage doit contenir au moins 10 mesures valides.")

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)
    print(f"IsolationForest entraîné instantanément sur {len(features)} mesures.")
    return model


def read_sensor_values(line: str) -> tuple[float, float] | None:
    """Lit les formats CSV et texte envoyés par l'ESP32."""
    csv_parts = [part.strip() for part in line.replace(";", ",").split(",")]
    if len(csv_parts) == 2:
        try:
            return float(csv_parts[0]), float(csv_parts[1])
        except ValueError:
            pass

    match = re.search(
        r"Temp(?:erature)?\s*:\s*(-?\d+(?:[.,]\d+)?)"
        r".*?Humidite\s*:\s*(-?\d+(?:[.,]\d+)?)",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1).replace(",", ".")), float(
            match.group(2).replace(",", ".")
        )
    return None


def main() -> None:
    anomaly_model = train_isolation_forest()
    vision_model = YOLO("yolov8n.pt")

    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=1)
    except SerialException as error:
        raise SystemExit(
            f"Impossible d'ouvrir {ARDUINO_PORT}. "
            "Modifie ARDUINO_PORT en haut du fichier."
        ) from error

    camera_indexes = [1, 0, 2, 3, 4]
    cap = None
    for index in camera_indexes:
        candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if candidate.isOpened():
            cap = candidate
            print(f"USB camera connected on index {index}")
            break
        candidate.release()

    if cap is None:
        arduino.close()
        raise SystemExit("Impossible d'ouvrir la webcam")

    latest_sensor_text = "Capteur: en attente"
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Impossible de lire la caméra")
                break

            while arduino.in_waiting:
                line = arduino.readline().decode("utf-8", errors="ignore").strip()
                sensor_values = read_sensor_values(line)
                if sensor_values is None:
                    continue

                temperature, humidity = sensor_values
                prediction = anomaly_model.predict([[temperature, humidity]])[0]
                is_anomaly = prediction == -1
                state = "ANOMALIE" if is_anomaly else "NORMAL"
                latest_sensor_text = (
                    f"T: {temperature:.1f} C | H: {humidity:.1f} % | {state}"
                )
                if is_anomaly:
                    print(f"ALERTE CAPTEUR: {latest_sensor_text}")

            results = vision_model(frame, classes=[0], verbose=False)
            person_detected = len(results[0].boxes) > 0
            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                latest_sensor_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if "ANOMALIE" in latest_sensor_text else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if person_detected:
                cv2.putText(
                    annotated_frame,
                    "PERSON DETECTED",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            cv2.imshow("SENTINEL-X - Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        arduino.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
