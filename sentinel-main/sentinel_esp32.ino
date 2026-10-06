#include "DHT.h"

#define BUZZER_PIN 14
#define PIR_PIN 13
#define DHT_PIN 32
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

const unsigned long DHT_INTERVAL = 2000;

// Filtrage anti-parasites : un HIGH bref ou aleatoire n'est pas un mouvement.
const unsigned long PIR_SAMPLE_INTERVAL = 20;
const unsigned int PIR_HIGH_SAMPLES_REQUIRED = 15;  // environ 300 ms HIGH
const unsigned long PIR_RESET_DELAY = 1000;         // environ 1 s LOW
// Delai minimal entre deux mouvements distincts.
const unsigned long PIR_ALERT_COOLDOWN = 3000;
// Evite seulement les bips distants repetes en boucle.
const unsigned long REMOTE_ALERT_COOLDOWN = 1000;

unsigned long lastDhtReading = 0;
unsigned long lastPirSample = 0;
unsigned long lastMotionAlert = 0;
unsigned long motionStartedAt = 0;
unsigned long lowSince = 0;
unsigned long lastRemoteAlert = 0;
unsigned int consecutiveHighSamples = 0;
bool movementReported = false;

void soundBuzzer(unsigned long now) {
  if (now - lastRemoteAlert < REMOTE_ALERT_COOLDOWN) {
    return;
  }
  tone(BUZZER_PIN, 1800);
  delay(180);
  noTone(BUZZER_PIN);
  lastRemoteAlert = now;
}

void readRemoteCommands() {
  if (!Serial.available()) {
    return;
  }
  String command = Serial.readStringUntil('\n');
  command.trim();
  if (command == "BUZZER_ALERT") {
    soundBuzzer(millis());
  }
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(PIR_PIN, INPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.begin(115200);
  delay(1000);
  dht.begin();

  Serial.println("SENTINEL_X_READY");
  Serial.println("Capteurs initialises");
  Serial.println("Attente stabilisation PIR...");
  delay(30000);
  Serial.println("PIR_READY");
  lastDhtReading = millis() - DHT_INTERVAL;
}

void detectMovement(unsigned long now) {
  if (now - lastPirSample < PIR_SAMPLE_INTERVAL) {
    return;
  }
  lastPirSample = now;

  bool pirHigh = digitalRead(PIR_PIN) == HIGH;

  if (pirHigh) {
    lowSince = 0;
    consecutiveHighSamples++;
    if (motionStartedAt == 0) {
      motionStartedAt = now;
    }

    bool enoughSamples =
        consecutiveHighSamples >= PIR_HIGH_SAMPLES_REQUIRED;
    bool cooldownOver =
        now - lastMotionAlert >= PIR_ALERT_COOLDOWN;

    if (enoughSamples && cooldownOver && !movementReported) {
      Serial.println("ALERTE : Mouvement detecte !");
      lastMotionAlert = now;
      movementReported = true;
    }
  } else {
    consecutiveHighSamples = 0;
    motionStartedAt = 0;
    if (lowSince == 0) {
      lowSince = now;
    }
    if (now - lowSince >= PIR_RESET_DELAY) {
      if (movementReported) {
        Serial.println("MOUVEMENT_TERMINE");
      }
      movementReported = false;
    }
  }
}

void readDht(unsigned long now) {
  if (now - lastDhtReading < DHT_INTERVAL) {
    return;
  }
  lastDhtReading = now;

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("ERREUR_CAPTEUR_DHT22");
    return;
  }

  Serial.print("Temp: ");
  Serial.print(temperature, 2);
  Serial.print(" C | Humidite: ");
  Serial.print(humidity, 2);
  Serial.println(" %");
}

void loop() {
  unsigned long now = millis();
  readRemoteCommands();
  detectMovement(now);
  readDht(now);
  delay(5);
}
