#!/bin/bash

# Quick Start Script for Document Scanner ML Model

echo "======================================"
echo "Document Scanner ML Model - Quick Start"
echo "======================================"
echo ""

# 1. 가상환경 확인
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 2. 의존성 설치
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# 3. 디렉토리 생성
echo ""
echo "Creating directories..."
mkdir -p data/dataset
mkdir -p checkpoints
mkdir -p converted_models

# 4. 데이터셋 다운로드 안내
echo ""
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo ""
echo "1. Set your Roboflow API key:"
echo "   export ROBOFLOW_API_KEY='your_api_key_here'"
echo ""
echo "2. Download dataset:"
echo "   python data/download_dataset.py --output ./data/dataset"
echo ""
echo "3. Train U-Net model:"
echo "   python train.py --model unet --epochs 50 --batch-size 8"
echo ""
echo "4. Train DeepLabV3+ model:"
echo "   python train.py --model deeplabv3plus --epochs 50 --batch-size 8"
echo ""
echo "5. Start web app:"
echo "   cd web_app && python app.py"
echo ""
echo "6. Convert to ONNX/TFLite:"
echo "   python convert_models.py --checkpoint ./checkpoints/unet_best.pth --model-type unet --onnx --tflite"
echo ""
echo "======================================"
echo "Setup completed!"
echo "======================================"
