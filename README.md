# Aircraft X-Ray Water Detection - Production Deployment

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![Computer Vision](https://img.shields.io/badge/CV-CNN-orange) ![Industry](https://img.shields.io/badge/Industry-Aviation-darkblue) ![Deployment](https://img.shields.io/badge/Deploy-HuggingFace-yellow)

Production deployment of ResNet50-based CNN for automated water ingression detection in aircraft composite structures using X-ray radiographic analysis.

**Deployment: Hugging Face Spaces | Framework: PyTorch + FastAPI | Performance: 92.9% accuracy**

## Overview

This deployment implements a deep learning model trained on 138 X-ray images of aircraft flap honeycomb structures. The system provides automated water ingression detection with conservative confidence scoring designed for aviation safety applications.

### Key Features
- **Real-time Analysis**: Direct image upload and instant prediction
- **Mobile Support**: Camera capture and gallery selection
- **Interactive Cropping**: Drag-and-drop ROI selection
- **Safety Framework**: Three-tier confidence scoring (High/Medium/Low)
- **Production Ready**: FastAPI backend with error handling

### Model Performance
- Accuracy: 92.9%
- Precision: 100% (Water), 83% (Nil)  
- Recall: 89% (Water), 100% (Nil)
- ROC AUC: 1.00

## Technical Stack

- **Backend**: FastAPI with PyTorch model serving
- **Frontend**: Vanilla JavaScript with HTML5 Canvas
- **ML Framework**: PyTorch with ResNet50 transfer learning
- **Deployment**: Hugging Face Spaces with Docker
- **Image Processing**: PIL with torchvision transforms

## Model Architecture

```python
class ResNetWaterDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetWaterDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
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
```

## API Usage

### POST /predict
```bash
curl -X POST "https://huggingface.co/spaces/Faiz-fyy/xray_water_detection/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@xray_image.jpg"
```

**Response:**
```json
{
    "prediction": "Water Detected",
    "confidence": 0.85,
    "interpretation": "HIGH CONFIDENCE: 85%"
}
```

## Confidence Framework

| Level | Threshold | Action |
|-------|-----------|--------|
| High | ≥80% | Automated decision |
| Medium | 60-80% | Confirmation needed |
| Low | <60% | Manual review recommended |

## Technology Migration

This deployment represents a complete migration from the original TensorFlow implementation to PyTorch:

**Original (TensorFlow):**
```python
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

**Deployment (PyTorch):**
```python
model = ResNetWaterDetector()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Migration Benefits
- Simplified deployment pipeline
- Better production serving integration
- Consistent framework across portfolio projects
- More granular control over model architecture

## Installation

### Local Development
```bash
# Clone repository
git clone https://github.com/your-username/aircraft-xray-deployment
cd aircraft-xray-deployment

# Install dependencies
pip install fastapi uvicorn torch torchvision pillow

# Run server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Requirements
```
torch
torchvision
fastapi
uvicorn
python-multipart
pillow
numpy
```

## Project Structure

```
├── notebooks/              # PyTorch reimplementation and deployment preparation
├── docs/                   # Methodology documentation
├── app.py                 # FastAPI application
├── index.html            # Web interface
├── Aircraft_Flap_Water_Detection_PyTorch.pth
├── requirements.txt
└── README.md
```

## Model Training Background

**Dataset**: 138 X-ray images from aircraft maintenance operations  
**Training Method**: ResNet50 transfer learning with fine-tuning  
**Preprocessing**: Manual ROI extraction, standardized resizing  
**Validation**: Stratified split with reproducible results  

**Training Pipeline:**
1. Progressive training: frozen → fine-tuned layers
2. Deterministic operations for reproducibility  
3. Conservative confidence calibration
4. Expert validation of uncertain cases

## License

This deployment is provided for demonstration purposes. Model training data remains proprietary to aviation maintenance operations.
