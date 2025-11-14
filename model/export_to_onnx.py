#!/usr/bin/env python3
"""
DeepLabV3+ MobileNet 모델을 ONNX 형식으로 변환
"""
import torch
import torch.nn as nn
from models.deeplabv3_mobilenet import DeepLabV3PlusMobileNet
import argparse

def export_to_onnx(checkpoint_path, output_path, input_size=512):
    """PyTorch 모델을 ONNX로 변환"""

    print(f"✓ 체크포인트 로드: {checkpoint_path}")

    # 모델 초기화 (segmentation_models_pytorch 사용)
    # n_classes=2로 로드 후 마지막 레이어만 1로 변경
    model = DeepLabV3PlusMobileNet(
        n_classes=2,
        encoder_name='mobilenet_v2',
        encoder_weights=None  # 체크포인트에서 가중치 로드
    )

    # 체크포인트 로드 (weights_only=False for compatibility)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # state_dict 추출
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ 모델 로드 완료")
    print(f"  - IoU: {checkpoint.get('best_iou', 'N/A')}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 출력 헤드 확인
    print(f"  - 출력 채널: {model.model.segmentation_head[0].out_channels}")

    # Binary 모델을 위해 n_classes=1로 다시 생성
    if model.model.segmentation_head[0].out_channels == 2:
        print(f"\n✓ 2채널 모델을 1채널(binary)로 변환 중...")

        # class 1 (document)의 가중치 저장
        old_conv = model.model.segmentation_head[0]
        document_weight = old_conv.weight.data[1:2].clone()  # class 1만
        document_bias = old_conv.bias.data[1:2].clone()

        # 새 Conv2d 생성 (1채널 출력)
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            1,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
            bias=True
        )

        # 가중치 복사
        with torch.no_grad():
            new_conv.weight.data.copy_(document_weight)
            new_conv.bias.data.copy_(document_bias)

        # 교체
        model.model.segmentation_head[0] = new_conv
        model.eval()

        print(f"✓ 변환 완료: 출력 채널 {old_conv.out_channels} → 1")

    # 더미 입력 생성 (NCHW 형식)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"\n✓ ONNX 변환 시작...")
    print(f"  - 입력 크기: {input_size}x{input_size}")
    print(f"  - 출력 경로: {output_path}")

    # ONNX로 변환
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,  # Latest ONNX Runtime compatible version
        do_constant_folding=True,  # 최적화
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )

    print(f"✓ ONNX 변환 완료!")

    # ONNX 모델 검증
    print(f"\n✓ ONNX 모델 검증 중...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX 모델 검증 성공!")

    # 모델 정보 출력
    print(f"\n📊 모델 정보:")
    print(f"  - 입력: {onnx_model.graph.input[0].name}")
    print(f"  - 입력 shape: [batch, 3, {input_size}, {input_size}]")
    print(f"  - 출력: {onnx_model.graph.output[0].name}")
    print(f"  - 출력 shape: [batch, 1, {input_size}, {input_size}]")

    # 파일 크기 출력
    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  - 파일 크기: {file_size:.2f} MB")

    # 추론 테스트
    print(f"\n✓ 추론 테스트 중...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)

    import onnxruntime as ort
    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 출력 비교
    diff = torch.abs(pytorch_output - torch.from_numpy(ort_output)).max()
    print(f"✓ PyTorch vs ONNX 최대 로짓 차이: {diff:.6f}")

    # Sigmoid 적용 후 확률 차이 계산
    pytorch_probs = torch.sigmoid(pytorch_output)
    onnx_probs = torch.sigmoid(torch.from_numpy(ort_output))
    prob_diff = torch.abs(pytorch_probs - onnx_probs).max()

    print(f"✓ PyTorch vs ONNX 최대 확률 차이: {prob_diff:.6f} ({prob_diff*100:.3f}%)")

    # Binary mask IoU 계산 (임계값 0.5)
    pytorch_mask = (pytorch_probs > 0.5).float()
    onnx_mask = (onnx_probs > 0.5).float()

    intersection = (pytorch_mask * onnx_mask).sum()
    union = pytorch_mask.sum() + onnx_mask.sum() - intersection
    iou = (intersection / (union + 1e-8)).item()

    print(f"✓ Binary Mask IoU (임계값 0.5): {iou*100:.2f}%")

    if iou > 0.95:
        print(f"✓ 변환 성공! (마스크 IoU > 95%)")
    elif iou > 0.90:
        print(f"✓ 변환 양호 (마스크 IoU > 90%)")
    else:
        print(f"⚠ 경고: 마스크 차이가 큼 (IoU: {iou*100:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 모델을 ONNX로 변환')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/deeplabv3plus_best.pth',
                        help='PyTorch 체크포인트 경로')
    parser.add_argument('--output', type=str,
                        default='checkpoints/deeplabv3plus_mobilenet.onnx',
                        help='출력 ONNX 파일 경로')
    parser.add_argument('--input-size', type=int, default=256,
                        help='입력 이미지 크기')

    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.input_size)
