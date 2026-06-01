import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

classes = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

device = torch.device("cpu")

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, len(classes))
)

model.load_state_dict(
    torch.load("best_asl_model.pth", map_location=device)
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def predict(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()

    return classes[pred]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ASL Sign Language Translator"
)

demo.launch()