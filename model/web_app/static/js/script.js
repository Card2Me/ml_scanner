// Document Scanner Web App - JavaScript

let currentImage = null;
let currentMode = 'upload'; // 'upload' or 'camera'
let stream = null;

// DOM 요소
const uploadBtn = document.getElementById('upload-btn');
const cameraBtn = document.getElementById('camera-btn');
const uploadArea = document.getElementById('upload-area');
const cameraArea = document.getElementById('camera-area');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewImage = document.getElementById('preview-image');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const predictBtn = document.getElementById('predict-btn');
const modelSelect = document.getElementById('model-select');
const loadingOverlay = document.getElementById('loading-overlay');

// 탭 전환
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        // 탭 버튼 활성화
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // 탭 콘텐츠 활성화
        document.querySelectorAll('.output-tab').forEach(tab => tab.classList.remove('active'));
        document.getElementById(`${tabName}-output`).classList.add('active');
    });
});

// 입력 모드 전환
uploadBtn.addEventListener('click', () => {
    switchMode('upload');
});

cameraBtn.addEventListener('click', () => {
    switchMode('camera');
});

function switchMode(mode) {
    currentMode = mode;

    if (mode === 'upload') {
        uploadBtn.classList.add('active');
        cameraBtn.classList.remove('active');
        uploadArea.style.display = 'block';
        cameraArea.style.display = 'none';

        // 카메라 스트림 중지
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    } else {
        cameraBtn.classList.add('active');
        uploadBtn.classList.remove('active');
        uploadArea.style.display = 'none';
        cameraArea.style.display = 'block';

        // 카메라 시작
        startCamera();
    }
}

// 카메라 시작
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' } // 후면 카메라
        });
        video.srcObject = stream;
    } catch (err) {
        alert('카메라 접근 실패: ' + err.message);
        switchMode('upload');
    }
}

// 파일 업로드
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.background = '#f9f9ff';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#ddd';
    uploadArea.style.background = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#ddd';
    uploadArea.style.background = '';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = file;
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// 카메라 캡처
captureBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
        currentImage = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
        predictBtn.disabled = false;
    }, 'image/jpeg');
});

// 예측 실행
predictBtn.addEventListener('click', async () => {
    if (!currentImage) {
        alert('이미지를 선택하거나 캡처하세요.');
        return;
    }

    loadingOverlay.style.display = 'flex';

    const formData = new FormData();
    formData.append('image', currentImage);
    formData.append('model', modelSelect.value);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // 결과 표시
            displayResult(data);
        } else {
            alert('예측 실패: ' + data.error);
        }
    } catch (err) {
        alert('서버 오류: ' + err.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
});

function displayResult(data) {
    // Overlay 결과
    const overlayOutput = document.getElementById('overlay-output');
    overlayOutput.innerHTML = `<img src="${data.overlay}" alt="Overlay Result">`;

    // Mask 결과
    const maskOutput = document.getElementById('mask-output');
    maskOutput.innerHTML = `<img src="${data.mask}" alt="Mask Result">`;

    // 정보 표시
    const infoSection = document.getElementById('info-section');
    infoSection.style.display = 'block';
    document.getElementById('info-model').textContent = data.model;
    document.getElementById('info-status').textContent = 'Success ✅';
}

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 사용 가능한 모델 확인
    fetch('/models')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('model-select');
            select.innerHTML = '';

            if (data.models.length === 0) {
                select.innerHTML = '<option value="">No models available</option>';
                predictBtn.disabled = true;
                alert('⚠️ 학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.');
            } else {
                const labels = {
                    'unet': 'Lightweight U-Net',
                    'deeplabv3plus': 'DeepLabV3+ MobileNet (PyTorch)',
                    'deeplabv3plus_onnx': 'DeepLabV3+ MobileNet (ONNX)'
                };
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = labels[model] || model;
                    select.appendChild(option);
                });
            }
        })
        .catch(err => {
            console.error('모델 목록 로드 실패:', err);
        });
});
