import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Classes
classes = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, len(classes))
)

model.load_state_dict(
    torch.load("best_asl_model.pth", map_location=device)
)

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ROI box
    x1, y1 = 100, 100
    x2, y2 = 400, 400

    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(tensor)

        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    label = classes[pred.item()]
    confidence = confidence.item() * 100

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show prediction
    cv2.putText(
        frame,
        f"{label} ({confidence:.1f}%)",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()