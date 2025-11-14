"""
단일 이미지 추론 스크립트
학습된 모델로 이미지를 테스트합니다.
"""
import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from models.unet import get_lightweight_unet
from models.deeplabv3_mobilenet import get_deeplabv3plus_mobilenet

def load_model(model_type, checkpoint_path, n_classes=2, base_channels=32):
    """학습된 모델 로드"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'unet':
        model = get_lightweight_unet(n_classes=n_classes, base_channels=base_channels)
    elif model_type == 'deeplabv3plus':
        model = get_deeplabv3plus_mobilenet(n_classes=n_classes, encoder_weights=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    print(f"Model loaded from {checkpoint_path}")
    if 'val_iou' in checkpoint:
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")

    return model, device

def preprocess_image(image_path, img_size=512):
    """이미지 전처리"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    # 리사이즈
    image_resized = cv2.resize(image, (img_size, img_size))

    # 정규화
    image_normalized = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std

    # Tensor로 변환
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)

    return image, image_resized, image_tensor, original_size

def postprocess_mask(mask, original_size):
    """마스크 후처리"""
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def visualize_results(original_image, pred_mask, save_path=None):
    """결과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 예측 마스크
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    # 오버레이
    mask_colored = np.zeros_like(original_image)
    mask_colored[pred_mask == 1] = [0, 255, 0]
    overlay = cv2.addWeighted(original_image, 0.6, mask_colored, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to {save_path}")

    plt.show()

def find_document_corners(mask):
    """문서 모서리 찾기"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    return approx

def inference(args):
    """추론 실행"""
    # 모델 로드
    model, device = load_model(
        args.model_type,
        args.checkpoint,
        n_classes=args.n_classes,
        base_channels=args.base_channels
    )

    # 이미지 전처리
    original_image, resized_image, image_tensor, original_size = preprocess_image(
        args.image,
        img_size=args.img_size
    )
    image_tensor = image_tensor.to(device)

    # 추론
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # 후처리
    pred_mask = postprocess_mask(pred_mask, original_size)

    # 문서 모서리 찾기
    corners = find_document_corners(pred_mask)
    if corners is not None:
        print(f"Found {len(corners)} corners")

    # 결과 시각화
    save_path = args.output if args.output else None
    visualize_results(original_image, pred_mask, save_path)

    # 마스크만 저장
    if args.save_mask:
        mask_path = args.save_mask
        cv2.imwrite(mask_path, pred_mask * 255)
        print(f"Mask saved to {mask_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Segmentation Inference")

    parser.add_argument('--image', type=str, required=True, help='입력 이미지 경로')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--model-type', type=str, required=True, choices=['unet', 'deeplabv3plus'],
                        help='모델 타입')
    parser.add_argument('--n-classes', type=int, default=2, help='클래스 수')
    parser.add_argument('--base-channels', type=int, default=32, help='U-Net 기본 채널 수')
    parser.add_argument('--img-size', type=int, default=512, help='이미지 크기')
    parser.add_argument('--output', type=str, help='결과 저장 경로')
    parser.add_argument('--save-mask', type=str, help='마스크 저장 경로')

    args = parser.parse_args()
    inference(args)
