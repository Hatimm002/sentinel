import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

camera_indexes = [1, 0, 2, 3, 4]
cap = None

for index in camera_indexes:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"USB camera connected on index {index}")
        break
    cap.release()
    cap = None

if cap is None or not cap.isOpened():
    print("Impossible d'ouvrir la webcam")
    raise SystemExit

person_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Impossible de lire la caméra")
        break

    results = model(frame, classes=[0], verbose=False)
    person_detected = len(results[0].boxes) > 0

    annotated_frame = results[0].plot()

    if person_detected:
        cv2.putText(
            annotated_frame,
            "PERSON DETECTED",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.imshow("SENTINEL-X - Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()