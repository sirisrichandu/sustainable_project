import cv2
from ultralytics import YOLO
import os
import time
import uuid
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'helmet_model.pt')
model = YOLO(MODEL_PATH)

violation_tracker = {} 
def process_image(image_path, output_path):
    image = cv2.imread(image_path)

    results = model(image)[0]
    helmet_status = "Helmet Not Detected"

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:
                helmet_status = "Helmet Detected"
                color = (0, 255, 0)
                label = "Helmet"
            else:
                color = (0, 0, 255)
                label = "No Helmet"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

    cv2.imwrite(output_path, image)
    return helmet_status, None


def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    helmet_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:
                    helmet_count += 1
                    label = "Helmet"
                    color = (0, 255, 0)
                else:
                    label = "No Helmet"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return {
        "frames_processed": frame_count,
        "helmet_detections": helmet_count
    }
def webcam_stream():
    cap = cv2.VideoCapture(0)
    violation_dir = os.path.join(BASE_DIR, "media", "violations")

    os.makedirs(violation_dir, exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            person_id = f"{cx//50}_{cy//50}"

  # simple identity

            # Assume:
            # cls == 1 → person without helmet
            # cls == 0 → helmet
            if cls == 1:
                if person_id not in violation_tracker:
                    violation_tracker[person_id] = current_time

                elapsed = current_time - violation_tracker[person_id]

                cv2.putText(
                    frame,
                    f"No Helmet {int(elapsed)}s",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if elapsed >= 10:
                    filename = f"violation_{uuid.uuid4().hex}.jpg"
                    filepath = os.path.join(violation_dir, filename)

                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(filepath, crop)
                    print("Violation captured:", filepath)

                    violation_tracker.pop(person_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            else:
                if person_id in violation_tracker:
                    violation_tracker.pop(person_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Helmet",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
