import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp

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

# Initialize MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)

label = ""
confidence = 0
roi = None

while True:
    ret, frame = cap.read()

    roi = None

    if not ret:
        break

    # Detect Hand Inside Loop
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # Find Hand Bounding Box
    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            xs = [lm.x * w for lm in hand.landmark]
            ys = [lm.y * h for lm in hand.landmark]

            x_min = max(int(min(xs)) - 30, 0)
            y_min = max(int(min(ys)) - 30, 0)

            x_max = min(int(max(xs)) + 30, w)
            y_max = min(int(max(ys)) + 30, h)

            roi = frame[y_min:y_max, x_min:x_max]

    # if not ret:
    #     break

    # ROI box
    # x1, y1 = 100, 100
    # x2, y2 = 400, 400

    # calculate the box from the frame size
    # h, w, _ = frame.shape

    # box_size = 400

    # x1 = (w - box_size) // 2
    # y1 = (h - box_size) // 2

    # x2 = x1 + box_size
    # y2 = y1 + box_size

    # roi = frame[y1:y2, x1:x2]

    # Preprocess
    # img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(img)

    # tensor = transform(img).unsqueeze(0).to(device)

    # # Prediction
    # with torch.no_grad():
    #     output = model(tensor)

    #     probs = torch.softmax(output, dim=1)

    #     confidence, pred = torch.max(probs, dim=1)

    # label = classes[pred.item()]

    if roi is not None:

        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, dim=1)

        label = classes[pred.item()]
        confidence = confidence.item() * 100
    # confidence = confidence.item() * 100

    # Draw box
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

    # Show prediction
    # cv2.putText(
    #     frame,
    #     f"{label} ({confidence:.1f}%)",
    #     (50, 50),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2
    # )
        cv2.putText(
            frame,
            f"{label} ({confidence:.1f}%)",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()