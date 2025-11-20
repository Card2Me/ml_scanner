"""
Document Scanner Web Application
웹 브라우저에서 모델을 테스트할 수 있는 Flask 애플리케이션
"""
import os
import sys
import io
import base64
import numpy as np
import cv2
from PIL import Image
import torch
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import get_lightweight_unet
from models.deeplabv3_mobilenet import get_deeplabv3plus_mobilenet

app = Flask(__name__)
CORS(app)

# 전역 변수로 모델 저장
models = {}
onnx_models = {}
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def load_models():
    """학습된 모델들을 로드"""
    checkpoint_dir = '../checkpoints'

    # U-Net 모델 로드
    unet_path = os.path.join(checkpoint_dir, 'unet_best.pth')
    if os.path.exists(unet_path):
        try:
            model = get_lightweight_unet(n_classes=2, base_channels=32)
            checkpoint = torch.load(unet_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(device)
            models['unet'] = model
            print(f"✅ U-Net loaded (IoU: {checkpoint.get('val_iou', 'N/A')})")
        except Exception as e:
            print(f"❌ Failed to load U-Net: {e}")

    # DeepLabV3+ 모델 로드
    deeplabv3_path = os.path.join(checkpoint_dir, 'deeplabv3plus_best.pth')
    if os.path.exists(deeplabv3_path):
        try:
            model = get_deeplabv3plus_mobilenet(n_classes=2, encoder_weights=None)
            checkpoint = torch.load(deeplabv3_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(device)
            models['deeplabv3plus'] = model
            print(f"✅ DeepLabV3+ loaded (IoU: {checkpoint.get('val_iou', 'N/A')})")
        except Exception as e:
            print(f"❌ Failed to load DeepLabV3+: {e}")

    if not models:
        print("⚠️  No models loaded. Please train models first.")

    # ONNX 모델 로드 (DeepLabV3+)
    onnx_path = os.path.join(checkpoint_dir, 'deeplabv3plus_mobilenet_512.onnx')
    if os.path.exists(onnx_path):
        try:
            session = ort.InferenceSession(onnx_path)
            input_meta = session.get_inputs()[0]
            input_shape = input_meta.shape
            input_size = 512
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
                h, w = input_shape[2], input_shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    input_size = int(h)
            onnx_models['deeplabv3plus_onnx'] = {
                'session': session,
                'input_name': input_meta.name,
                'input_size': input_size
            }
            print(f"✅ DeepLabV3+ ONNX loaded (input: {input_size}x{input_size})")
        except Exception as e:
            print(f"❌ Failed to load DeepLabV3+ ONNX: {e}")

    if not onnx_models:
        print("ℹ️  No ONNX models loaded. Place ONNX files in checkpoints/ if needed.")

def preprocess_image(image, img_size=512):
    """이미지 전처리"""
    # RGB로 변환
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 리사이즈
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (img_size, img_size))

    # 정규화 (float32로 명시)
    image_normalized = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_normalized - mean) / std

    # Tensor로 변환 (C, H, W) - float32 명시
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

    return image_tensor, original_size

def postprocess_mask(mask, original_size):
    """마스크 후처리"""
    # numpy array로 변환 (MPS tensor인 경우 CPU로 이동)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # uint8로 변환
    mask = mask.astype(np.uint8)

    # 원본 크기로 복원
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def create_overlay(image, mask, alpha=0.5):
    """원본 이미지에 마스크 오버레이"""
    # 마스크를 컬러로 변환
    mask_colored = np.zeros_like(image)
    mask_colored[mask == 1] = [0, 255, 0]  # 녹색

    # 오버레이
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay

def find_document_contours(mask):
    """문서 경계선 및 근사 다각형 계산"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    return largest_contour, approx


def get_document_quadrilateral(contour, approx):
    """컨투어를 기반으로 문서 4각형 근사"""
    if contour is None:
        return None

    if approx is not None and len(approx) >= 4:
        approx_points = approx.reshape(-1, 2).astype(np.float32)
        if len(approx_points) == 4:
            return approx_points
        elif len(approx_points) > 4:
            hull = cv2.convexHull(approx_points)
            if hull.ndim == 2:
                hull_contour = hull.reshape(-1, 1, 2)
            else:
                hull_contour = hull
            perimeter = cv2.arcLength(hull_contour, True)
            hull_poly = cv2.approxPolyDP(hull_contour, 0.05 * perimeter, True)
            if len(hull_poly) == 4:
                return hull_poly.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def order_points(points):
    """관점 변환을 위한 4개의 점 정렬"""
    pts = np.array(points, dtype=np.float32)
    if pts.shape[0] != 4:
        return None

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def flatten_document(image, quad_points):
    """4각형 근사 결과를 사용해 문서를 평탄화"""
    if quad_points is None or len(quad_points) != 4:
        return None

    rect = order_points(quad_points)
    if rect is None:
        return None

    width_a = np.linalg.norm(rect[2] - rect[3])
    width_b = np.linalg.norm(rect[1] - rect[0])
    height_a = np.linalg.norm(rect[1] - rect[2])
    height_b = np.linalg.norm(rect[0] - rect[3])

    max_width = max(int(round(width_a)), int(round(width_b)))
    max_height = max(int(round(height_a)), int(round(height_b)))

    if max_width < 1 or max_height < 1:
        return None

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

@app.route('/')
def index():
    """메인 페이지"""
    available = list(models.keys()) + list(onnx_models.keys())
    return render_template('index.html', models=available)

@app.route('/predict', methods=['POST'])
def predict():
    """이미지 예측 엔드포인트"""
    try:
        # 모델 선택
        model_name = request.form.get('model', 'unet')
        use_torch = model_name in models
        use_onnx = model_name in onnx_models
        if not (use_torch or use_onnx):
            return jsonify({'error': f'Model {model_name} not available'}), 400

        # 이미지 로드
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        # BGR to RGB (if needed)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # 모델별 입력 크기 설정
        target_size = 512
        if use_onnx:
            target_size = onnx_models[model_name]['input_size']

        # 전처리
        image_tensor, original_size = preprocess_image(image_rgb, img_size=target_size)

        if use_torch:
            image_tensor = image_tensor.to(device)
            model = models[model_name]
            with torch.no_grad():
                outputs = model(image_tensor)
                pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        else:
            session_info = onnx_models[model_name]
            ort_inputs = {session_info['input_name']: image_tensor.numpy()}
            onnx_logits = session_info['session'].run(None, ort_inputs)[0]

            if onnx_logits.ndim != 4:
                raise ValueError('Unexpected ONNX output shape')

            if onnx_logits.shape[1] > 1:
                exp_logits = np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                pred_mask = (np.argmax(probs, axis=1) == 1).astype(np.uint8)
            else:
                probs = 1.0 / (1.0 + np.exp(-onnx_logits))
                pred_mask = (probs > 0.5).astype(np.uint8)

            pred_mask = np.squeeze(pred_mask)
            if pred_mask.ndim == 0:
                pred_mask = np.zeros((target_size, target_size), dtype=np.uint8)

        # 후처리
        pred_mask = postprocess_mask(pred_mask, original_size)

        # 오버레이 생성
        overlay = create_overlay(image_rgb, pred_mask)

        # 경계선 찾기
        contour, approx = find_document_contours(pred_mask)
        if approx is not None:
            overlay_with_contours = overlay.copy()
            cv2.drawContours(overlay_with_contours, [approx], -1, (255, 0, 0), 3)
        elif contour is not None:
            overlay_with_contours = overlay.copy()
            cv2.drawContours(overlay_with_contours, [contour], -1, (255, 0, 0), 2)
        else:
            overlay_with_contours = overlay

        # 문서 평탄화
        quad = get_document_quadrilateral(contour, approx)
        flattened_scan = flatten_document(image_rgb, quad)
        scan_data = None
        if flattened_scan is not None:
            if len(flattened_scan.shape) == 3 and flattened_scan.shape[2] == 3:
                flattened_bgr = cv2.cvtColor(flattened_scan, cv2.COLOR_RGB2BGR)
            else:
                flattened_bgr = flattened_scan
            _, scan_buffer = cv2.imencode('.png', flattened_bgr)
            scan_base64 = base64.b64encode(scan_buffer).decode('utf-8')
            scan_data = f'data:image/png;base64,{scan_base64}'

        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay_with_contours, cv2.COLOR_RGB2BGR))
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')

        _, mask_buffer = cv2.imencode('.png', pred_mask * 255)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'overlay': f'data:image/png;base64,{overlay_base64}',
            'mask': f'data:image/png;base64,{mask_base64}',
            'scan': scan_data,
            'model': model_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록"""
    return jsonify({
        'models': list(models.keys()) + list(onnx_models.keys())
    })

if __name__ == '__main__':
    print("Loading models...")
    load_models()

    print(f"Starting web server on http://localhost:5001")
    print(f"Available models: {list(models.keys())}")
    app.run(host='0.0.0.0', port=5001, debug=True)
