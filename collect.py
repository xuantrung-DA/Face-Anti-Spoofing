import cv2
from mtcnn import MTCNN
import os
from PIL import Image

save_dir = "data_collect/real"
os.makedirs(save_dir, exist_ok=True)

detector = MTCNN()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
print("'s' to save | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    if len(faces) > 0:
        faces.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)
        face = faces[0] 

        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        x2, y2 = x + w, y + h

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        face_crop = frame_rgb[y:y2, x:x2]
        if face_crop.size == 0:
            continue

        face_pil = Image.fromarray(face_crop).resize((256, 256))

        if cv2.waitKey(1) & 0xFF == ord('s'):
            save_path = os.path.join(save_dir, f"real_{count:04d}.jpg")
            face_pil.save(save_path)
            print(f"[INFO] Saved {save_path}")
            count += 1

    cv2.imshow("Capture Faces", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
