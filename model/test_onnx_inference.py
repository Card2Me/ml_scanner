#!/usr/bin/env python3
"""
ONNX 모델을 실제 이미지로 테스트
"""
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from models.deeplabv3_mobilenet import DeepLabV3PlusMobileNet
import torch.nn as nn

def preprocess_image(image_path, size=512):
    """이미지 전처리"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size))

    # ImageNet normalization
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_array = (img_array - mean) / std

    # NCHW format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()

    return img_tensor

def test_onnx_model(image_path, checkpoint_path, onnx_path):
    """PyTorch vs ONNX 비교"""

    print(f"📸 테스트 이미지: {image_path}")

    # 1. PyTorch 모델 로드
    print("\n✓ PyTorch 모델 로드...")
    model = DeepLabV3PlusMobileNet(n_classes=2, encoder_name='mobilenet_v2', encoder_weights=None)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Binary output으로 변경
    old_conv = model.model.segmentation_head[0]
    model.model.segmentation_head[0] = nn.Conv2d(
        old_conv.in_channels, 1, kernel_size=old_conv.kernel_size, padding=old_conv.padding
    )
    with torch.no_grad():
        model.model.segmentation_head[0].weight.data = old_conv.weight.data[1:2]
        model.model.segmentation_head[0].bias.data = old_conv.bias.data[1:2]

    model.eval()

    # 2. ONNX 모델 로드
    print("✓ ONNX 모델 로드...")
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = ort_session.get_inputs()[0]
    onnx_shape = onnx_input.shape
    target_size = None
    if len(onnx_shape) == 4:
        h, w = onnx_shape[2], onnx_shape[3]
        if isinstance(h, int) and isinstance(w, int):
            target_size = h
    if target_size is None:
        target_size = 512
    print(f"  - ONNX 입력 해상도: {target_size}x{target_size}")

    # 3. 이미지 전처리
    print("✓ 이미지 전처리...")
    input_tensor = preprocess_image(image_path, size=target_size)

    # 4. PyTorch 추론
    print("\n✓ PyTorch 추론...")
    with torch.no_grad():
        pytorch_logits = model(input_tensor)
        pytorch_probs = torch.sigmoid(pytorch_logits)
        pytorch_mask = (pytorch_probs > 0.5).float()

    print(f"  - 문서 픽셀 비율: {pytorch_mask.mean()*100:.2f}%")

    # 5. ONNX 추론
    print("\n✓ ONNX 추론...")
    ort_inputs = {onnx_input.name: input_tensor.numpy()}
    onnx_logits = ort_session.run(None, ort_inputs)[0]
    if onnx_logits.shape[1] > 1:
        onnx_logits = onnx_logits[:, 1:2]
    onnx_probs = 1.0 / (1.0 + np.exp(-onnx_logits))
    onnx_mask = (onnx_probs > 0.5).astype(np.float32)

    print(f"  - 문서 픽셀 비율: {onnx_mask.mean()*100:.2f}%")

    # 6. 비교
    print("\n📊 비교 결과:")

    # 로짓 차이
    logit_diff = np.abs(pytorch_logits.numpy() - onnx_logits).max()
    print(f"  - 최대 로짓 차이: {logit_diff:.6f}")

    # 확률 차이
    prob_diff = np.abs(pytorch_probs.numpy() - onnx_probs).max()
    print(f"  - 최대 확률 차이: {prob_diff:.6f} ({prob_diff*100:.2f}%)")

    # IoU
    pytorch_mask_np = pytorch_mask.numpy()
    intersection = (pytorch_mask_np * onnx_mask).sum()
    union = pytorch_mask_np.sum() + onnx_mask.sum() - intersection
    iou = intersection / (union + 1e-8)

    print(f"  - Binary Mask IoU: {iou*100:.2f}%")

    # 픽셀 일치율
    pixel_acc = (pytorch_mask_np == onnx_mask).mean()
    print(f"  - 픽셀 일치율: {pixel_acc*100:.2f}%")

    if iou > 0.95:
        print("\n✅ 변환 성공! ONNX 모델이 PyTorch와 거의 동일합니다.")
    elif iou > 0.90:
        print("\n✅ 변환 양호! ONNX 모델이 사용 가능합니다.")
    else:
        print(f"\n⚠️  경고: IoU가 낮습니다 ({iou*100:.2f}%)")

if __name__ == '__main__':
    import sys
    import os

    # 테스트 이미지 찾기
    test_images = []

    # data/valid에서 이미지 찾기
    if os.path.exists('data/valid/images'):
        for fname in os.listdir('data/valid/images'):
            if fname.endswith(('.jpg', '.png', '.jpeg')):
                test_images.append(os.path.join('data/valid/images', fname))

    if not test_images:
        print("❌ 테스트 이미지를 찾을 수 없습니다.")
        print("   data/valid/images/ 디렉토리에 이미지를 넣어주세요.")
        sys.exit(1)

    # 첫 번째 이미지로 테스트
    test_image = test_images[0]

    test_onnx_model(
        image_path=test_image,
        checkpoint_path='checkpoints/deeplabv3plus_best.pth',
        onnx_path='checkpoints/deeplabv3plus_mobilenet_512.onnx'
    )
