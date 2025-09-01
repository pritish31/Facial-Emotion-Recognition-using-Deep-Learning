import cv2
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----Load ResNet50 model ---
sharp_model = models.resnet50(pretrained=False)
sharp_model.fc = nn.Linear(sharp_model.fc.in_features, 7)
sharp_model.load_state_dict(torch.load("Resnet50_Stack_Ori_Sharpen.pth", map_location=device))
sharp_model = sharp_model.to(device)
sharp_model.eval()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------Preprocessing-----
def gaussian_kernel(size=3, sigma=1.0):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

def unsharp_mask_numpy(img_np, strength=1.5, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = convolve(img_np, kernel)
    mask = img_np - blurred
    sharpened = img_np + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# Transform for ResNet (224x224, normalized)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (224, 224))

        original_img = face_resized.copy()
        sharpened_img = unsharp_mask_numpy(original_img)

        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        sharpened_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_GRAY2RGB)

        original_tensor = transform(original_rgb).unsqueeze(0).to(device)
        sharpened_tensor = transform(sharpened_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output_orig = sharp_model(original_tensor)
            pred_orig = torch.argmax(output_orig, dim=1).item()
            emotion_orig = emotion_labels[pred_orig]

            output_sharp = sharp_model(sharpened_tensor)
            pred_sharp = torch.argmax(output_sharp, dim=1).item()
            emotion_sharp = emotion_labels[pred_sharp]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"Sharpened: {emotion_sharp}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Base: {emotion_orig}", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Compare Base vs Sharpened Model", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
