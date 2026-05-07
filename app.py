import os
import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import io
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH   = 'model/cattle_classifier_deploy.pth'
CLASSES_PATH = 'model/class_names.json'
IMG_SIZE     = 300
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]

# ── Load class names ────────────────────────────────────────────
with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

# ── Build model architecture (must match training) ──────────────
def build_model(num_classes):
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1024),
        nn.SiLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

# ── Load weights ────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = build_model(NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded — {NUM_CLASSES} breeds | Device: {DEVICE}")

# ── Transforms ──────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ── Grad-CAM setup ──────────────────────────────────────────────
cam = GradCAM(model=model, target_layers=[model.features[-1]])

# ── Breed info dictionary ────────────────────────────────────────
BREED_INFO = {
    "Gir": {"region": "Gujarat", "use": "Milk", "fact": "Famous for high milk production, used to improve Brazilian cattle."},
    "Sahiwal": {"region": "Punjab", "use": "Milk", "fact": "One of the best dairy breeds in Asia, highly heat-tolerant."},
    "Kankrej": {"region": "Gujarat/Rajasthan", "use": "Draft & Milk", "fact": "Known for immense strength and used as draught animals."},
    "Ongole": {"region": "Andhra Pradesh", "use": "Draft & Milk", "fact": "Exported worldwide, especially to Brazil and USA."},
    "Hariana": {"region": "Haryana", "use": "Draft & Milk", "fact": "Popular dual-purpose breed across North India."},
    "Tharparkar": {"region": "Rajasthan", "use": "Milk", "fact": "Thrives in desert conditions with low water needs."},
    "Red_Sindhi": {"region": "Sindh (Pakistan/Rajasthan)", "use": "Milk", "fact": "High fat content in milk, very popular in South India."},
    "Hallikar": {"region": "Karnataka", "use": "Draft", "fact": "Considered the parent breed of many South Indian breeds."},
    "Khillari": {"region": "Maharashtra", "use": "Draft", "fact": "One of the fastest-moving draught cattle in India."},
    "Deoni": {"region": "Maharashtra/Karnataka", "use": "Milk & Draft", "fact": "Distinctively spotted pattern, dual-purpose breed."},
}

def get_breed_info(breed_name):
    info = BREED_INFO.get(breed_name, {
        "region": "India",
        "use": "Mixed",
        "fact": "A recognized Indian cattle breed with unique characteristics."
    })
    return info

def predict_image(pil_img):
    """Run inference + Grad-CAM on a PIL image."""
    img_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idxs = probs.topk(5)
    top5 = [
        {"breed": CLASS_NAMES[i], "confidence": round(p.item() * 100, 2)}
        for i, p in zip(top5_idxs.cpu(), top5_probs.cpu())
    ]

    # Grad-CAM
    targets = [ClassifierOutputTarget(top5_idxs[0].item())]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    cam_image   = show_cam_on_image(img_resized.astype(np.float32), grayscale_cam, use_rgb=True)

    # Encode Grad-CAM as base64 for response
    cam_pil    = Image.fromarray(cam_image)
    buf        = io.BytesIO()
    cam_pil.save(buf, format='PNG')
    cam_b64    = base64.b64encode(buf.getvalue()).decode()

    return top5, cam_b64


@app.route('/')
def index():
    return render_template('index.html', num_breeds=NUM_CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        top5, cam_b64 = predict_image(pil_img)

        breed_name = top5[0]['breed']
        info = get_breed_info(breed_name)

        return jsonify({
            'success': True,
            'predictions': top5,
            'gradcam': cam_b64,
            'breed_info': info,
            'top_breed': breed_name,
            'confidence': top5[0]['confidence']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/breeds')
def breeds():
    return jsonify({'breeds': CLASS_NAMES, 'total': NUM_CLASSES})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
