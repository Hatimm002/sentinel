# SENTINEL-X

SENTINEL-X est une plateforme locale de surveillance temps réel construite
autour d'un ESP32, d'un capteur de température/humidité DHT22, d'un capteur de
mouvement PIR, d'un buzzer, d'une caméra et d'une interface web Flask.

Le système combine :

- la lecture des capteurs ESP32 par liaison série ;
- une détection d'anomalies température/humidité avec Isolation Forest ;
- une détection de personne par YOLO ;
- une interface web temps réel ;
- une authentification administrateur ;
- un flux vidéo MJPEG ;
- des alertes visuelles et un buzzer commandé localement ou depuis le PC ;
- un déploiement Docker avec Mosquitto TLS.

> Le projet est actuellement principalement prévu pour un lancement local sous
> Windows, car l'ESP32 est connecté sur `COM11` et la caméra est une ressource
> Windows. Docker est fourni pour la partie web/broker et pour un déploiement
> Linux où les périphériques peuvent être exposés au conteneur.

---

## 1. Architecture générale

```text
                         USB série 115200 bauds
ESP32 ----------------------------------------------------+
  |                                                       |
  |-- DHT22 : température / humidité                      |
  |-- PIR  : événement mouvement                          |
  |-- Buzzer : alerte sonore                              |
                                                          v
                                                web_app.py (Flask)
                                                          |
                  +---------------------------------------+----------------+
                  |                                       |                |
                  v                                       v                v
       Isolation Forest Joblib                    YOLO + caméra       SSE / API
       température + humidité                    personne             navigateur
                  |                                       |
                  +-------------------+-------------------+
                                      v
                              Dashboard SENTINEL-X
                                      |
                            BUZZER_ALERT vers ESP32

Docker Compose :
  sentinel_web       application Flask
  mosquitto_sentinel broker MQTT TLS
```

### Répartition des responsabilités

#### ESP32

L'ESP32 :

- lit le DHT22 ;
- lit le PIR ;
- filtre le signal PIR ;
- écrit les mesures sur le port série ;
- écrit les événements de mouvement ;
- commande directement le buzzer pour les commandes reçues du PC.

#### Ordinateur / application Flask

Le PC :

- lit `COM11` ;
- parse les lignes série ;
- exécute la prédiction Isolation Forest ;
- capture la caméra ;
- exécute YOLO sur les images ;
- envoie `BUZZER_ALERT` à l'ESP32 en cas d'anomalie ou de personne ;
- sert l'interface web ;
- diffuse l'état par Server-Sent Events (SSE).

#### Navigateur

Le navigateur :

- affiche les valeurs DHT ;
- affiche l'état Isolation Forest ;
- affiche le score du modèle ;
- affiche le mouvement comme information ;
- affiche la détection de personne ;
- reçoit les mises à jour sans rechargement grâce à `EventSource` ;
- affiche l'animation rouge pour une anomalie modèle ou une personne.

---

## 2. Matériel et branchements

Le sketch actuel utilise les broches suivantes :

| Composant | Broche ESP32 |
|---|---:|
| Buzzer | GPIO 14 |
| Sortie PIR | GPIO 13 |
| Signal DATA DHT22 | GPIO 32 |
| DHT22 | type `DHT22` |

Branchement PIR recommandé :

```text
PIR VCC  -> 5V ou VIN selon le module
PIR GND  -> GND
PIR OUT  -> GPIO 13
```

Branchement DHT22 :

```text
DHT22 VCC  -> alimentation adaptée
DHT22 GND  -> GND
DHT22 DATA -> GPIO 32
```

Le Moniteur série Arduino et Python ne doivent pas ouvrir `COM11` en même
temps. Ferme le Moniteur série avant de lancer Flask ou avant un téléversement.

---

## 3. Code ESP32

Le fichier [sentinel_esp32.ino](./sentinel_esp32.ino) contient le programme à
téléverser sur la carte.

### Initialisation

Au démarrage, l'ESP32 écrit :

```text
SENTINEL_X_READY
Capteurs initialises
Attente stabilisation PIR...
PIR_READY
```

Le PIR attend 30 secondes afin de se stabiliser.

### Format des mesures

Toutes les deux secondes environ, l'ESP32 écrit :

```text
Temp: 26.20 C | Humidite: 50.50 %
```

Python accepte aussi les formats numériques séparés par une virgule ou un point
virgule.

### Filtrage PIR

Le code actuel utilise :

```cpp
const unsigned long PIR_SAMPLE_INTERVAL = 20;
const unsigned int PIR_HIGH_SAMPLES_REQUIRED = 15;
const unsigned long PIR_RESET_DELAY = 1000;
const unsigned long PIR_ALERT_COOLDOWN = 3000;
```

Cela signifie :

- lecture du PIR toutes les 20 ms ;
- confirmation après environ 15 lectures HIGH, soit environ 300 ms ;
- retour à l'état terminé après environ 1 seconde LOW ;
- séparation minimale de 3 secondes entre deux alertes PIR.

Le mouvement ne déclenche plus le buzzer et ne déclenche plus l'animation rouge
du site. Il est conservé comme information dans l'interface.

L'ESP32 écrit :

```text
ALERTE : Mouvement detecte !
MOUVEMENT_TERMINE
```

Le backend reconnaît les deux variantes `MOUVEMENT_TERMINE` et
`Mouvement termine`, ce qui évite un blocage d'état lié à l'underscore.

### Commande distante du buzzer

Le PC peut envoyer :

```text
BUZZER_ALERT
```

L'ESP32 lit cette commande et joue un bip de 180 ms à 1800 Hz. Un cooldown
interne de 1 seconde évite les bips distants répétés en boucle.

---

## 4. Réglage du capteur PIR

Sur un module PIR de type HC-SR501, les deux potentiomètres sont généralement :

- `Sx` / `SENS` : sensibilité et portée ;
- `Tx` / `TIME` : durée du signal HIGH.

Réglage conseillé :

```text
SENS : position centrale, puis ajuster
TIME : presque au minimum
Jumper : L / Single si disponible
```

L'ordre physique gauche/droite dépend de l'orientation de la carte. Il faut
suivre les marquages `Sx` et `Tx`, et non supposer que le potentiomètre de
gauche est toujours la sensibilité.

Procédure :

1. redémarrer l'ESP32 ;
2. ne pas bouger pendant les 30 secondes de stabilisation ;
3. tester à courte distance ;
4. ajuster uniquement `SENS` ;
5. laisser `TIME` bas pour éviter un signal HIGH trop long.

Si le PIR reste HIGH sans mouvement, vérifier l'alimentation, le GND, les
sources de chaleur, les ventilateurs, les fenêtres et les câbles.

---

## 5. Isolation Forest

Le modèle persistant se trouve dans :

[Model/isolation_forest_model.joblib](./Model/isolation_forest_model.joblib)

Le package Joblib contient :

- le modèle `IsolationForest` ;
- la liste des features ;
- les paramètres du modèle.

Les features attendues sont exactement :

```text
temperature_C
humidite_pct
```

À chaque ligne DHT valide, Flask construit un DataFrame nommé :

```python
pd.DataFrame(
    [[temperature, humidity]],
    columns=["temperature_C", "humidite_pct"],
)
```

Le modèle retourne :

| Valeur | Signification |
|---:|---|
| `1` | mesure normale |
| `-1` | anomalie |

Le backend expose également `decision_function()` sous la forme
`anomaly_score`.

### Important : aucun seuil fixe de température

Le projet n'utilise plus de règle du type :

```text
température >= 35 °C
```

La décision dépend uniquement de la combinaison température/humidité apprise
par Isolation Forest.

Conséquences :

- une température élevée n'est pas automatiquement une anomalie si elle
  ressemble aux données d'apprentissage ;
- une température modérée peut être anormale si la combinaison avec l'humidité
  est inhabituelle ;
- le score affiché sert à comprendre la décision du modèle, mais le statut
  officiel reste `NORMAL` ou `ANOMALIE`.

Quand le modèle retourne `-1` :

- le statut web passe en anomalie ;
- l'interface active le mode rouge urgent ;
- le PC envoie `BUZZER_ALERT` à l'ESP32.

Le modèle ne s'entraîne pas à chaque mesure live. Il est chargé au démarrage
depuis le fichier Joblib.

---

## 6. Caméra et détection YOLO

La caméra par défaut est l'index `1`, configurable avec :

```env
CAMERA_INDEX=1
```

Le modèle YOLO utilisé est :

[yolov8n.pt](./yolov8n.pt)

Optimisations utilisées :

- résolution cible `640x480` ;
- buffer caméra réduit à 1 image ;
- YOLO exécuté une image sur trois ;
- taille d'image YOLO `320` ;
- JPEG qualité `75` ;
- flux MJPEG avec attente de 50 ms entre les images.

La détection est limitée à la classe COCO `0`, correspondant à `person`.

Quand YOLO détecte une personne :

- `person_detected` passe à `true` ;
- le dashboard affiche `Détectée` ;
- le mode rouge urgent est activé ;
- `BUZZER_ALERT` est envoyé à l'ESP32 ;
- le cooldown Python empêche une répétition immédiate.

Le dashboard permet de sélectionner les caméras disponibles via
`/api/cameras`. La caméra 1 est sélectionnée par défaut, mais une caméra 0 à 4
peut être choisie.

Les avertissements OpenCV `VideoCapture` indiquent généralement qu'un index
n'est pas disponible. Ils ne concernent pas le modèle Isolation Forest ni le
port série.

---

## 7. Interface web

L'interface est servie par Flask depuis [web_app.py](./web_app.py).

Elle contient :

- une page de connexion ;
- le dashboard ;
- les cartes température et humidité ;
- la carte Isolation Forest ;
- le score du modèle ;
- la carte mouvement ;
- l'état Arduino ;
- l'état personne ;
- le choix de caméra ;
- le flux vidéo ;
- la dernière trame série ;
- la déconnexion.

### Authentification

Routes :

| Route | Fonction |
|---|---|
| `GET /login` | formulaire de connexion |
| `POST /login` | validation des identifiants |
| `POST /logout` | fermeture de session |

Les routes dashboard, caméra, SSE et API nécessitent une session authentifiée.

Les identifiants sont configurables dans `.env` :

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin
```

`admin/admin` est acceptable pour un prototype local, mais doit être remplacé
dans un vrai déploiement.

La clé Flask est également configurable :

```env
FLASK_SECRET_KEY=une-cle-longue-et-aleatoire
```

### Mise à jour temps réel

Le navigateur ouvre :

```text
GET /events
```

Le serveur diffuse les changements JSON avec Server-Sent Events. Le navigateur
utilise `EventSource` et met à jour les cartes sans polling permanent.

---

## 8. État transmis par l'application

L'état central contient notamment :

```json
{
  "temperature": 26.2,
  "humidity": 50.5,
  "movement": false,
  "person_detected": false,
  "serial_connected": true,
  "camera_connected": true,
  "message": "Mesures normales selon le modèle",
  "updated_at": "13:52:10",
  "last_serial_line": "Temp: 26.20 C | Humidite: 50.50 %",
  "anomaly": false,
  "anomaly_score": 0.2241,
  "critical": false
}
```

`critical` est vrai uniquement lorsqu'une anomalie Isolation Forest ou une
personne est détectée. Le mouvement seul n'est pas critique.

---

## 9. API et routes techniques

| Route | Méthode | Authentification | Fonction |
|---|---|---|---|
| `/` | GET | oui | dashboard |
| `/login` | GET/POST | non pour l'affichage | connexion |
| `/logout` | POST | oui | déconnexion |
| `/health` | GET | non | healthcheck |
| `/api/status` | GET | oui | état JSON |
| `/api/cameras` | GET | oui | caméras détectées |
| `/api/camera` | POST | oui | sélection caméra |
| `/events` | GET | oui | flux SSE |
| `/camera` | GET | oui | flux MJPEG |
| `/api/movement/reset` | POST | oui | remet le mouvement à false |

Exemple :

```powershell
curl http://127.0.0.1:5000/health
```

Réponse :

```json
{"status":"ok"}
```

---

## 10. Configuration `.env`

Le fichier `.env` local actuel doit ressembler à ceci sous Windows :

```env
FLASK_SECRET_KEY=une-cle-secrete-longue-et-aleatoire
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin
ARDUINO_PORT=COM11
CAMERA_INDEX=1
```

Le fichier [`.env.example`](./.env.example) est fourni comme modèle.

Variables utilisées :

| Variable | Windows | Docker/Linux |
|---|---|---|
| `FLASK_SECRET_KEY` | clé Flask | clé Flask |
| `ADMIN_USERNAME` | utilisateur web | utilisateur web |
| `ADMIN_PASSWORD` | mot de passe web | mot de passe web |
| `ARDUINO_PORT` | `COM11` | généralement non utilisé |
| `DOCKER_ARDUINO_PORT` | généralement non utilisé | `/dev/ttyUSB0` |
| `CAMERA_INDEX` | `1` | `1` |
| `ARDUINO_BAUDRATE` | `115200` | `115200` |
| `FLASK_HOST` | `127.0.0.1` par défaut | `0.0.0.0` |
| `FLASK_PORT` | `5000` | `5000` |

`web_app.py` charge automatiquement `.env` au démarrage sans dépendance
supplémentaire à `python-dotenv`. La valeur Windows `/dev/ttyUSB0` est
également remplacée automatiquement par `COM11` si elle est utilisée par
erreur sous Windows.

---

## 11. Lancement local sous Windows

### Prérequis

- Windows ;
- Python et l'environnement `venv` fourni ;
- ESP32 connecté ;
- port série correct ;
- caméra disponible ;
- dépendances installées.

### Démarrage

Fermer le Moniteur série Arduino, puis :

```powershell
cd C:\Users\elala\OneDrive\Desktop\projet\WORKSHOP\WORKSHOP\sentinel-main
..\venv\Scripts\python.exe .\web_app.py
```

Ouvrir ensuite :

<http://127.0.0.1:5000>

Connexion prototype :

```text
Utilisateur : admin
Mot de passe : admin
```

Logs attendus :

```text
Modèle Isolation Forest chargé depuis ...\Model\isolation_forest_model.joblib
Lecture série active sur COM11 à 115200 bauds.
Attente des mesures ESP32...
```

Si le serveur a été arrêté, le relancer après toute modification de
`web_app.py` ou de `.env`. Un simple rafraîchissement du navigateur ne
redémarre pas Python.

---

## 12. Déploiement Docker

Les fichiers Docker sont :

- [Dockerfile](./Dockerfile) ;
- [docker-compose.yml](./docker-compose.yml) ;
- [`.dockerignore`](./.dockerignore).

### Préparer les variables

```powershell
cd C:\Users\elala\OneDrive\Desktop\projet\WORKSHOP\WORKSHOP\sentinel-main
Copy-Item .env.example .env
```

Modifier au minimum :

```env
FLASK_SECRET_KEY=une-vraie-cle-secrete-longue
ADMIN_PASSWORD=un-vrai-mot-de-passe
```

### Construire et démarrer

```powershell
docker compose up -d --build
```

Services :

```text
sentinel_web        http://127.0.0.1:5000
mosquitto_sentinel  TLS MQTT sur le port 8883
```

Vérifier l'état :

```powershell
docker compose ps
docker compose logs -f sentinel_web
docker compose logs -f mosquitto_sentinel
```

Healthcheck :

<http://127.0.0.1:5000/health>

Arrêter :

```powershell
docker compose down
```

Reconstruire après une modification du code ou du Dockerfile :

```powershell
docker compose up -d --build --force-recreate
```

Modifier seulement `.env` :

```powershell
docker compose up -d --force-recreate
```

### Limitation Docker Desktop Windows

Docker Desktop ne transmet pas automatiquement `COM11` ni les caméras
Windows au conteneur. Par conséquent :

- le lancement local avec le venv est recommandé pour le matériel réel ;
- Docker peut servir l'interface et Mosquitto ;
- sur Linux, il faut exposer explicitement `/dev/ttyUSB0` et la caméra ;
- `DOCKER_ARDUINO_PORT` est séparé de `ARDUINO_PORT` pour éviter de mélanger
  les chemins Windows et Linux.

---

## 13. Mosquitto et TLS

Le service Mosquitto existant utilise :

- authentification obligatoire ;
- fichier de mots de passe ;
- ACL ;
- certificats TLS ;
- port sécurisé `8883` ;
- réseau Docker `192.168.10.0/24`.

Configuration :

- [config/mosquitto.conf](./config/mosquitto.conf) ;
- [config/aclfile](./config/aclfile) ;
- [config/passwords](./config/passwords) ;
- [config/certs/](./config/certs/).

Les volumes Docker persistants sont :

```text
sentinel_mosquitto_data
sentinel_mosquitto_logs
```

Le dashboard Flask actuel communique directement avec l'ESP32 par série. Le
broker Mosquitto est préparé pour une extension MQTT et n'est pas nécessaire
pour la lecture actuelle de `COM11`.

---

## 14. Structure du projet

```text
sentinel-main/
├── web_app.py
├── sentinel_esp32.ino
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .env.example
├── .dockerignore
├── yolov8n.pt
├── Model/
│   ├── isolation_forest_model.joblib
│   ├── IsolateForest.ipynb
│   ├── data.csv
│   └── isolation_forest_results.csv
└── config/
    ├── mosquitto.conf
    ├── aclfile
    ├── passwords
    └── certs/
```

Fichiers importants :

- `web_app.py` : point d'entrée Flask, série, modèle, caméra et API ;
- `sentinel_esp32.ino` : firmware ESP32 ;
- `Model/isolation_forest_model.joblib` : modèle utilisé en live ;
- `yolov8n.pt` : modèle de détection personne ;
- `docker-compose.yml` : web + Mosquitto ;
- `.env` : configuration locale, à ne pas publier ;
- `README.md` : documentation du projet.

---

## 15. Dépannage

### Arduino hors ligne

Vérifier :

1. l'ESP32 est branché ;
2. le bon port est sélectionné ;
3. `.env` contient `ARDUINO_PORT=COM11` sous Windows ;
4. le Moniteur série est fermé ;
5. aucun autre programme n'utilise `COM11` ;
6. le serveur a été redémarré après modification de `.env`.

Log attendu :

```text
Lecture série active sur COM11 à 115200 bauds.
```

### Température et humidité à `--`

Le backend n'a pas reçu de ligne DHT valide. Vérifier :

- le sketch téléversé ;
- le câblage DHT22 ;
- la ligne `Temp: ... | Humidite: ...` dans les logs ;
- le port série ;
- la vitesse `115200`.

### Le mouvement reste affiché

Le backend reconnaît `MOUVEMENT_TERMINE` et remet l'état à `false`.
Redémarrer Flask après une mise à jour de `web_app.py`, puis actualiser avec
`Ctrl+F5`.

### Caméra indisponible

Essayer l'index 0 puis 1 dans la liste. Fermer les applications qui utilisent la
caméra : Teams, Zoom, navigateur, application Caméra Windows, etc.

Les messages OpenCV concernant `DSHOW` signifient qu'un index de caméra n'est
pas accessible ; ils ne signifient pas que l'Arduino est déconnecté.

### Login refusé

Vérifier :

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin
```

Puis redémarrer Flask. Avec Docker, recréer le conteneur :

```powershell
docker compose up -d --force-recreate
```

### Le modèle ne se charge pas

Vérifier que le fichier existe :

```text
Model/isolation_forest_model.joblib
```

Le package doit contenir un modèle avec les features :

```text
temperature_C
humidite_pct
```

---

## 16. Vérifications réalisées

Les contrôles suivants ont été effectués pendant l'intégration :

- syntaxe Python de `web_app.py` valide ;
- modèle Joblib chargé correctement ;
- prédiction Isolation Forest testée ;
- score `decision_function` testé ;
- accès non authentifié redirigé vers `/login` ;
- accès API non authentifié refusé ;
- login `admin/admin` testé ;
- endpoint `/health` retournant HTTP 200 ;
- configuration Docker Compose validée ;
- port Windows chargé comme `COM11` ;
- index caméra par défaut confirmé à `1` ;
- commande série `BUZZER_ALERT` générée et gérée ;
- reconnaissance de `MOUVEMENT_TERMINE` avec underscore validée.

---
