from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import time

import pandas as pd
import serial


SERIAL_PORT = "COM11"
BAUDRATE = 115200
COLLECTION_SECONDS = 300
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "Model"
    / "temperature_humidite_normal.csv"
)


def parse_sensor_line(line: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in line.replace(";", ",").split(",")]
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
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
    print(f"Collecte pendant {COLLECTION_SECONDS} secondes sur {SERIAL_PORT}...")
    print("Laisse le capteur dans des conditions normales.")
    rows: list[dict[str, object]] = []

    try:
        connection = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except serial.SerialException as error:
        raise SystemExit(
            f"Impossible d'ouvrir {SERIAL_PORT}: {error}\n"
            "Ferme le moniteur série Arduino, puis vérifie le port sélectionné."
        ) from error

    deadline = time.monotonic() + COLLECTION_SECONDS
    try:
        while time.monotonic() < deadline:
            line = connection.readline().decode("utf-8", errors="ignore").strip()
            values = parse_sensor_line(line)
            if values is None:
                continue
            temperature, humidity = values
            rows.append(
                {
                    "horaire": datetime.now().strftime("%H:%M:%S"),
                    "temperature_C": temperature,
                    "humidite_pct": humidity,
                }
            )
            print(f"{temperature:.2f} C | {humidity:.2f} %")
    finally:
        connection.close()

    if len(rows) < 10:
        raise SystemExit("Moins de 10 mesures valides ont été reçues.")

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"{len(output)} mesures normales enregistrées dans: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
