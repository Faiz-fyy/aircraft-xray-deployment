from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

IMG_HEIGHT = 350
IMG_WIDTH = 512

class ResNetWaterDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetWaterDetector, self).__init__()
        
        self.backbone = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output.squeeze()

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    if model is None:
        model = ResNetWaterDetector()
        model.load_state_dict(torch.load('Aircraft_Flap_Water_Detection_PyTorch.pth', map_location=device))
        model.to(device)
        model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = image.convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def interpret_prediction(confidence_score):
    if confidence_score > 0.8:
        return f"HIGH CONFIDENCE: {confidence_score:.1%}"
    elif confidence_score > 0.6:
        return f"MEDIUM CONFIDENCE: {confidence_score:.1%}"
    else:
        return f"LOW CONFIDENCE: {confidence_score:.1%} - Manual review recommended"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Load model and predict
        model = load_model()
        processed_image = preprocess_image(pil_image)
        
        with torch.no_grad():
            prediction = model(processed_image).item()
        
        predicted_class = 'Water Detected' if prediction > 0.5 else 'No Water Ingression'
        confidence_interp = interpret_prediction(prediction)
        
        return JSONResponse({
            "prediction": predicted_class,
            "confidence": prediction,
            "interpretation": confidence_interp
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)