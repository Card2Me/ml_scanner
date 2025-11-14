"""
데이터셋 로딩 및 전처리 유틸리티
"""
import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DocumentSegmentationDataset(Dataset):
    """Document Segmentation 데이터셋"""

    def __init__(self, root_dir, split='train', transform=None, img_size=512):
        """
        Args:
            root_dir: 데이터셋 루트 디렉토리
            split: 'train', 'valid', 'test'
            transform: augmentation 변환
            img_size: 이미지 크기
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # 이미지 및 마스크 경로
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')

        # 이미지 파일 목록 가져오기
        self.image_files = sorted([f for f in os.listdir(self.img_dir)
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])

        # Transform 설정
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 파일명
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        # 마스크 파일명 (확장자를 png로 변경)
        mask_filename = os.path.splitext(img_filename)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 마스크 로드 (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        # 마스크를 0과 1로 정규화 (0: 배경, 1: 문서)
        mask = (mask > 127).astype(np.uint8)

        # Transform 적용
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()

        return image, mask

def get_transforms(img_size=512, split='train'):
    """데이터 augmentation transform 반환"""
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
