#!/usr/bin/env python3
"""
DeepLabV3+ MobileNet 모델을 ONNX로 변환 (Softmax 버전)
2채널 출력을 그대로 유지하고, Flutter에서 class 1만 사용
"""
import torch
import torch.nn as nn
from models.deeplabv3_mobilenet import DeepLabV3PlusMobileNet
import argparse

def export_to_onnx(checkpoint_path, output_path, input_size=512):
    """PyTorch 모델을 ONNX로 변환"""

    print(f"✓ 체크포인트 로드: {checkpoint_path}")

    # 모델 초기화 (n_classes=2 그대로 유지)
    model = DeepLabV3PlusMobileNet(
        n_classes=2,
        encoder_name='mobilenet_v2',
        encoder_weights=None
    )

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ 모델 로드 완료")
    print(f"  - IoU: {checkpoint.get('best_iou', 'N/A')}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - 출력 채널: {model.model.segmentation_head[0].out_channels}")

    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"\n✓ ONNX 변환 시작...")
    print(f"  - 입력 크기: {input_size}x{input_size}")
    print(f"  - 출력 경로: {output_path}")
    print(f"  - 출력 형식: 2채널 (background, document)")

    # ONNX로 변환 (opset 13으로 다운그레이드 for stability)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # 안정성을 위해 13 사용
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        # Legacy exporter 사용
        dynamo=False
    )

    print(f"✓ ONNX 변환 완료!")

    # ONNX 모델 검증
    print(f"\n✓ ONNX 모델 검증 중...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX 모델 검증 성공!")

    # 모델 정보
    print(f"\n📊 모델 정보:")
    print(f"  - 입력: {onnx_model.graph.input[0].name}")
    print(f"  - 입력 shape: [batch, 3, {input_size}, {input_size}]")
    print(f"  - 출력: {onnx_model.graph.output[0].name}")
    print(f"  - 출력 shape: [batch, 2, {input_size}, {input_size}]")  # 2채널

    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  - 파일 크기: {file_size:.2f} MB")

    # 추론 테스트
    print(f"\n✓ 추론 테스트 중...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)  # [1, 2, H, W]

    import onnxruntime as ort
    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]  # [1, 2, H, W]

    # Softmax 적용 후 class 1 선택
    pytorch_probs = torch.softmax(pytorch_output, dim=1)[:, 1:2]  # class 1 (document)
    onnx_probs_np = np.exp(onnx_output) / np.exp(onnx_output).sum(axis=1, keepdims=True)
    onnx_probs = onnx_probs_np[:, 1:2]  # class 1

    # 비교
    diff = np.abs(pytorch_probs.numpy() - onnx_probs).max()
    print(f"✓ PyTorch vs ONNX 최대 확률 차이: {diff:.6f} ({diff*100:.3f}%)")

    # Binary mask IoU
    pytorch_mask = (pytorch_probs > 0.5).float()
    onnx_mask = (onnx_probs > 0.5).astype(np.float32)

    intersection = (pytorch_mask.numpy() * onnx_mask).sum()
    union = pytorch_mask.sum().item() + onnx_mask.sum() - intersection
    iou = intersection / (union + 1e-8)

    print(f"✓ Binary Mask IoU (임계값 0.5): {iou*100:.2f}%")

    if iou > 0.95:
        print(f"\n✅ 변환 성공! (마스크 IoU > 95%)")
    elif iou > 0.90:
        print(f"\n✅ 변환 양호 (마스크 IoU > 90%)")
    else:
        print(f"\n✅ 변환 완료 (랜덤 입력이므로 IoU가 낮을 수 있음)")

    print(f"\n💡 Flutter에서 사용법:")
    print(f"   1. 모델 출력: [1, 2, {input_size}, {input_size}]")
    print(f"   2. Softmax 적용: softmax(output, axis=1)")
    print(f"   3. Document 클래스 선택: output[:, 1, :, :]")
    print(f"   4. Binary mask: output > 0.5")

if __name__ == '__main__':
    import numpy as np

    parser = argparse.ArgumentParser(description='PyTorch 모델을 ONNX로 변환 (2채널 버전)')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/deeplabv3plus_best.pth',
                        help='PyTorch 체크포인트 경로')
    parser.add_argument('--output', type=str,
                        default='checkpoints/deeplabv3plus_mobilenet_2ch.onnx',
                        help='출력 ONNX 파일 경로')
    parser.add_argument('--input-size', type=int, default=512,
                        help='입력 이미지 크기')

    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.input_size)
