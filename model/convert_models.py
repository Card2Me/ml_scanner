"""
학습된 모델을 ONNX 및 TFLite 형식으로 변환
모바일 배포를 위한 모델 변환 스크립트
"""
import os
import argparse
import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

from models.unet import get_lightweight_unet
from models.deeplabv3_mobilenet import get_deeplabv3plus_mobilenet

def load_model(model_type, checkpoint_path, n_classes=2, base_channels=32):
    """학습된 모델 로드"""
    if model_type == 'unet':
        model = get_lightweight_unet(n_classes=n_classes, base_channels=base_channels)
    elif model_type == 'deeplabv3plus':
        model = get_deeplabv3plus_mobilenet(n_classes=n_classes, encoder_weights=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    if 'val_iou' in checkpoint:
        print(f"   Validation IoU: {checkpoint['val_iou']:.4f}")

    return model

def convert_to_onnx(model, output_path, img_size=512, opset_version=11):
    """PyTorch 모델을 ONNX로 변환"""
    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # ONNX 모델 검증
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"✅ ONNX model saved to {output_path}")

    # 모델 크기 출력
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Model size: {file_size:.2f} MB")

    return output_path

def convert_to_tflite(onnx_path, output_path, quantize=False):
    """ONNX 모델을 TFLite로 변환"""
    try:
        # ONNX -> TensorFlow
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        # TensorFlow SavedModel로 저장
        tf_model_dir = output_path.replace('.tflite', '_saved_model')
        tf_rep.export_graph(tf_model_dir)

        # TFLite 변환
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

        if quantize:
            # 양자화 적용 (모바일 최적화)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("   Applying float16 quantization...")

        tflite_model = converter.convert()

        # TFLite 모델 저장
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✅ TFLite model saved to {output_path}")

        # 모델 크기 출력
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Model size: {file_size:.2f} MB")

        # SavedModel 디렉토리 정리 (옵션)
        import shutil
        if os.path.exists(tf_model_dir):
            shutil.rmtree(tf_model_dir)

        return output_path

    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        print("   Note: TFLite conversion may require additional dependencies.")
        return None

def test_onnx_inference(onnx_path, img_size=512):
    """ONNX 모델 추론 테스트"""
    import onnxruntime as ort
    import numpy as np

    # ONNX Runtime 세션 생성
    session = ort.InferenceSession(onnx_path)

    # 더미 입력
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # 추론
    outputs = session.run(None, {'input': dummy_input})

    print(f"✅ ONNX inference test passed")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {outputs[0].shape}")

def convert_model(args):
    """모델 변환 메인 함수"""

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    model = load_model(
        args.model_type,
        args.checkpoint,
        n_classes=args.n_classes,
        base_channels=args.base_channels
    )

    # 출력 파일명 생성
    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # ONNX 변환
    if args.onnx:
        onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
        convert_to_onnx(model, onnx_path, img_size=args.img_size, opset_version=args.opset_version)

        # ONNX 추론 테스트
        if args.test:
            test_onnx_inference(onnx_path, img_size=args.img_size)

        # TFLite 변환
        if args.tflite:
            tflite_path = os.path.join(args.output_dir, f"{base_name}.tflite")
            convert_to_tflite(onnx_path, tflite_path, quantize=args.quantize)

    print("\n✅ Model conversion completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained models to ONNX/TFLite")

    # 모델 관련
    parser.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 체크포인트 경로')
    parser.add_argument('--model-type', type=str, required=True, choices=['unet', 'deeplabv3plus'],
                        help='모델 타입')
    parser.add_argument('--n-classes', type=int, default=2, help='클래스 수')
    parser.add_argument('--base-channels', type=int, default=32, help='U-Net 기본 채널 수')
    parser.add_argument('--img-size', type=int, default=512, help='이미지 크기')

    # 변환 옵션
    parser.add_argument('--onnx', action='store_true', default=True, help='ONNX로 변환')
    parser.add_argument('--tflite', action='store_true', help='TFLite로 변환')
    parser.add_argument('--quantize', action='store_true', help='TFLite 양자화 적용')
    parser.add_argument('--opset-version', type=int, default=11, help='ONNX opset 버전')

    # 출력 관련
    parser.add_argument('--output-dir', type=str, default='./converted_models', help='출력 디렉토리')
    parser.add_argument('--test', action='store_true', help='변환된 모델 추론 테스트')

    args = parser.parse_args()

    convert_model(args)
