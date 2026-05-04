# 🐄 GauDrishti — Indian Cattle Breed Classification
### Complete Setup & Running Guide

---

## 📁 Files 

| File | What it is |
|---|---|
| `cattle_classifier_deploy.pth` | Trained model weights |
| `class_names.json` | List of 50 breed names |
| `confusion_matrix.png` | Evaluation chart |
| `gradcam_results.png` | Grad-CAM sample output |
| `training_curves.png` | Training history chart |
| `Indian_Cattle_Breed_Classification.ipynb` | Colab training notebook |

---

## 🗂️ Project Folder Structure

```
cattle_app/
├── app.py                  ← Flask backend (main server)
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Frontend web page
├── static/
│   └── uploads/            ← Temp image storage (auto-created)
└── model/                  ← PUT YOUR MODEL FILES HERE
    ├── cattle_classifier_deploy.pth
    └── class_names.json
```

---

## 🚀 Setup & Run (Step by Step)

### Step 1 — Copy model files into the model/ folder
```
cattle_app/model/cattle_classifier_deploy.pth   ← your .pth file
cattle_app/model/class_names.json               ← your class_names.json
```

### Step 2 — Create a virtual environment (recommended)
```bash
cd cattle_app
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ If you don't have a GPU, PyTorch CPU version is fine — install from https://pytorch.org

### Step 4 — Run the server
```bash
python app.py
```

You should see:
```
Loading model from model/cattle_classifier_deploy.pth...
✅ Model loaded — 50 breeds | Device: cpu
 * Running on http://0.0.0.0:5000
```

### Step 5 — Open in browser
```
http://localhost:5000
```

---

## 🌐 How the App Works

```
User uploads image
      ↓
Flask backend (app.py) receives it
      ↓
Image → val_transform (resize 320→300, center crop, normalize)
      ↓
EfficientNetB3 → logits → softmax → Top-5 predictions
      ↓
Grad-CAM → heatmap PNG → base64 encoded
      ↓
JSON response → Frontend renders results
```

---

## 📂 What to Do With Each File

| File | Action |
|---|---|
| `.pth` model | → Put in `cattle_app/model/` folder |
| `class_names.json` | → Put in `cattle_app/model/` folder |
| `.ipynb` notebook | → Keep for reference / re-training. Upload to Google Drive |



---

## 🧪 Testing the API directly

```bash
# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -F "image=@your_cow_photo.jpg"

# List all breeds
curl http://localhost:5000/breeds
```

---

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: model/cattle_classifier_deploy.pth` | Make sure .pth file is in the `model/` folder |
| CUDA out of memory | App auto-falls back to CPU — no action needed |
| Port 5000 already in use | Change `port=5000` to `port=5001` in app.py |
| Slow first prediction | Normal — model loads on first request. Second is fast. |

---

## 📊 Model Performance Summary

| Metric | Value |
|---|---|
| Architecture | EfficientNetB3 (pretrained ImageNet) |
| Training Strategy | Two-phase fine-tuning |
| Image Size | 300×300 |
| Number of Classes | 50 Indian cattle breeds |
| Best Val Accuracy | ~72% |
| Training Images | ~12,000–15,000 |
| Interpretability | Grad-CAM on final conv block |

---



** methodology:**
> "The system uses EfficientNetB3 pretrained on ImageNet, fine-tuned in two phases on the Indian Cattle Image Dataset (Kaggle, ~15,000 images, 50 breeds). Transfer learning was applied with selective layer unfreezing and differential learning rates. Grad-CAM provides visual interpretability by highlighting discriminative regions such as the animal's head, horns, and body markings."
