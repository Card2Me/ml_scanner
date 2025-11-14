# Document Scanner ML Model

모바일 앱의 Document Scanner를 위한 경량 ML 모델 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 모바일 환경에 최적화된 문서 세그멘테이션 모델을 개발하고 테스트하는 것을 목표로 합니다.

### 주요 기능

- **2가지 경량 모델 구현**
  - Lightweight U-Net
  - DeepLabV3+ MobileNet

- **웹 기반 테스트 환경**
  - 이미지 업로드 테스트
  - 실시간 카메라 테스트
  - 두 모델 비교 가능

- **모바일 배포 준비**
  - ONNX 변환 지원
  - TFLite 변환 지원
  - 양자화 옵션 제공

## 프로젝트 구조

```
ml_scanner_model/
├── data/
│   └── download_dataset.py      # Roboflow 데이터셋 다운로드
├── models/
│   ├── unet.py                  # 경량 U-Net 모델
│   └── deeplabv3_mobilenet.py   # DeepLabV3+ MobileNet 모델
├── utils/
│   └── dataset.py               # 데이터셋 로더 및 전처리
├── web_app/
│   ├── app.py                   # Flask 웹 서버
│   ├── templates/
│   │   └── index.html           # 웹 UI
│   └── static/
│       ├── css/style.css        # 스타일시트
│       └── js/script.js         # 클라이언트 JavaScript
├── train.py                     # 모델 학습 스크립트
├── convert_models.py            # ONNX/TFLite 변환 스크립트
├── requirements.txt             # Python 의존성
└── README.md
```

## 설치 방법

### 1. 환경 설정

```bash
# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터셋 다운로드

Roboflow에서 Document Segmentation 데이터셋을 다운로드합니다.

```bash
# Roboflow API 키 설정 (https://app.roboflow.com/에서 발급)
export ROBOFLOW_API_KEY="your_api_key_here"

# 데이터셋 다운로드
python data/download_dataset.py --output ./data/dataset
```

## 사용 방법

### 1. 모델 학습

#### U-Net 모델 학습

```bash
python train.py \
    --model unet \
    --data-dir ./data/dataset \
    --epochs 50 \
    --batch-size 8 \
    --base-channels 32 \
    --checkpoint-dir ./checkpoints
```

#### DeepLabV3+ MobileNet 모델 학습

```bash
python train.py \
    --model deeplabv3plus \
    --data-dir ./data/dataset \
    --epochs 50 \
    --batch-size 8 \
    --encoder-weights imagenet \
    --checkpoint-dir ./checkpoints
```

#### 학습 파라미터 설명

- `--model`: 모델 종류 (`unet` 또는 `deeplabv3plus`)
- `--data-dir`: 데이터셋 디렉토리
- `--epochs`: 학습 에폭 수
- `--batch-size`: 배치 크기 (GPU 메모리에 맞게 조정)
- `--img-size`: 입력 이미지 크기 (기본값: 512)
- `--lr`: 학습률 (기본값: 1e-4)
- `--base-channels`: U-Net의 기본 채널 수 (16, 32, 64 등)
- `--encoder-weights`: DeepLabV3+의 사전학습 가중치 (`imagenet` 또는 `None`)

### 2. 모델 변환 (ONNX/TFLite)

#### ONNX 변환

```bash
python convert_models.py \
    --checkpoint ./checkpoints/unet_best.pth \
    --model-type unet \
    --onnx \
    --output-dir ./converted_models \
    --test
```

#### TFLite 변환 (양자화 포함)

```bash
python convert_models.py \
    --checkpoint ./checkpoints/deeplabv3plus_best.pth \
    --model-type deeplabv3plus \
    --onnx \
    --tflite \
    --quantize \
    --output-dir ./converted_models
```

### 3. 웹 테스트 환경 실행

```bash
cd web_app
python app.py
```

브라우저에서 http://localhost:5000 접속

#### 웹 앱 기능

1. **모델 선택**: U-Net 또는 DeepLabV3+ 선택
2. **입력 방식**:
   - 📁 Upload Image: 로컬 이미지 업로드
   - 📷 Use Camera: 실시간 카메라 사용
3. **결과 확인**:
   - Overlay: 원본 이미지에 세그멘테이션 오버레이
   - Mask Only: 세그멘테이션 마스크만 표시

## 모델 정보

### 1. Lightweight U-Net

- **특징**: 경량화된 U-Net 구조
- **파라미터**: ~1-5M (base_channels에 따라 조정)
- **장점**: 빠른 추론 속도, 적은 메모리 사용
- **적합한 환경**: 저사양 모바일 기기

### 2. DeepLabV3+ MobileNet

- **특징**: MobileNetV2 백본 + ASPP + Decoder
- **파라미터**: ~5-10M
- **장점**: 높은 정확도, 세밀한 경계 검출
- **적합한 환경**: 중급 이상 모바일 기기

## 성능 최적화 팁

### 학습 최적화

1. **배치 크기 조정**: GPU 메모리에 맞게 조정
2. **이미지 크기**: 512x512 (모바일 환경 고려)
3. **데이터 증강**: Albumentations로 다양한 augmentation 적용
4. **학습률 스케줄러**: ReduceLROnPlateau 사용

### 모바일 배포 최적화

1. **양자화**: TFLite 변환 시 float16 양자화 적용
2. **모델 크기 감소**: U-Net의 base_channels를 16 또는 32로 설정
3. **추론 속도**: ONNX Runtime 또는 TFLite 인터프리터 사용

## 모바일 통합 가이드

### Android (TFLite)

```kotlin
// TFLite 모델 로드 및 추론 예시
val interpreter = Interpreter(loadModelFile())
val inputArray = preprocessImage(bitmap)
val outputArray = Array(1) { Array(512) { FloatArray(512) } }
interpreter.run(inputArray, outputArray)
```

### iOS (Core ML / ONNX)

```swift
// Core ML 모델 로드 (ONNX -> Core ML 변환 필요)
let model = try VNCoreMLModel(for: DocumentScanner().model)
let request = VNCoreMLRequest(model: model)
```

## 문제 해결

### 데이터셋 다운로드 실패

- Roboflow API 키가 올바른지 확인
- 네트워크 연결 확인

### GPU 메모리 부족

- 배치 크기 감소 (`--batch-size 4`)
- 이미지 크기 감소 (`--img-size 384`)

### 웹 앱 모델 로드 실패

- 체크포인트 파일 경로 확인 (`./checkpoints/`)
- 모델 학습 완료 여부 확인

## 라이선스

MIT License

## 참고 자료

- [Roboflow Dataset](https://universe.roboflow.com/maulvi-zm/document-segmentation-j6olp/dataset/2)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [DeepLabV3+ Paper](https://arxiv.org/abs/1802.02611)
- [Qualcomm AI Hub - DeepLabV3+ MobileNet](https://aihub.qualcomm.com/mobile/models/deeplabv3_plus_mobilenet)

## 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.
