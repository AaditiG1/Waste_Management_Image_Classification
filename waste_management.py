import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODEL ARCHITECTURES ---
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(64*28*28, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, x): return self.fc_layers(self.conv_layers(x))

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(64*28*28, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
    def forward(self, x): return self.fc_layers(self.conv_layers(x))

# --- LOADING LOGIC ---
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

models_dict = {}
try:
    m1 = BaselineCNN(); m1.load_state_dict(torch.load('baseline_waste_model.pth', map_location=device))
    models_dict['baseline'] = m1.eval()

    m2 = ImprovedCNN(); m2.load_state_dict(torch.load('improved_waste_model.pth', map_location=device))
    models_dict['improved'] = m2.eval()

    m3 = models.resnet18(); m3.fc = nn.Linear(m3.fc.in_features, 2)
    m3.load_state_dict(torch.load('resnet_waste_model.pth', map_location=device))
    models_dict['resnet'] = m3.eval()
    print("Dashboard Ready: All 3 models loaded.")
except Exception as e:
    print(f"Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    model_type = request.form.get('model_type') # Get the chosen model from UI
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.jpg')
    file.save(img_path)

    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    model = models_dict[model_type]
    with torch.no_grad():
        out = model(img_t)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
        label = ['Organic', 'Recyclable'][pred.item()]

    return jsonify({"label": label, "confidence": f"{conf.item()*100:.2f}%"})

if __name__ == '__main__':
    # use_reloader=False prevents Flask from running the code twice
    app.run(debug=True, use_reloader=False)