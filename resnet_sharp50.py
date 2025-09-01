import cv2
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

# ----------------------------
# 1. Load ResNet18 model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 emotions
model.load_state_dict(torch.load("Resnet50_Stack_Ori_Sharpen.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# 2. Emotion labels & face detector
# ----------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----------------------------
# 3. Real-time video capture
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Face preprocessing
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))  # ResNet expects 224x224
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
        face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        face_tensor = face_tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, 1).item()
            emotion = emotion_labels[pred]

        # Display
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
