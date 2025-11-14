"""
Document Segmentation 모델 학습 스크립트
U-Net과 DeepLabV3+ MobileNet 모델을 학습합니다.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from models.unet import get_lightweight_unet
from models.deeplabv3_mobilenet import get_deeplabv3plus_mobilenet
from utils.dataset import DocumentSegmentationDataset

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        dice = 0.0
        for class_idx in range(num_classes):
            pred = probs[:, class_idx]
            target = (targets == class_idx).float()

            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()

            dice += (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice / num_classes

class CombinedLoss(nn.Module):
    """Cross Entropy + Dice Loss"""

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice

def calculate_iou(pred, target, n_classes=2):
    """Calculate IoU (Intersection over Union)"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return np.nanmean(ious)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        preds = torch.argmax(outputs, dim=1)
        iou = calculate_iou(preds, masks)

        total_loss += loss.item()
        total_iou += iou

        pbar.set_postfix({'loss': loss.item(), 'iou': iou})

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou

def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds, masks)

            total_loss += loss.item()
            total_iou += iou

            pbar.set_postfix({'loss': loss.item(), 'iou': iou})

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou

def train(args):
    """메인 학습 함수"""

    # Device 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 데이터셋 로드
    train_dataset = DocumentSegmentationDataset(
        root_dir=args.data_dir,
        split='train',
        img_size=args.img_size
    )

    val_dataset = DocumentSegmentationDataset(
        root_dir=args.data_dir,
        split='valid',
        img_size=args.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid batch norm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 모델 생성
    if args.model == 'unet':
        model = get_lightweight_unet(n_classes=args.n_classes, base_channels=args.base_channels)
    elif args.model == 'deeplabv3plus':
        model = get_deeplabv3plus_mobilenet(n_classes=args.n_classes, encoder_weights=args.encoder_weights)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 학습
    best_iou = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")

        # Validation
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Scheduler step
        scheduler.step(val_loss)

        # 모델 저장
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss
            }, save_path)
            print(f"Model saved to {save_path} (IoU: {val_iou:.4f})")

        # 주기적으로 저장
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.checkpoint_dir, f"{args.model}_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss
            }, save_path)

    print(f"\nTraining completed! Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Document Segmentation Model")

    # 데이터 관련
    parser.add_argument('--data-dir', type=str, default='./data/dataset', help='데이터셋 디렉토리')
    parser.add_argument('--img-size', type=int, default=512, help='이미지 크기')
    parser.add_argument('--n-classes', type=int, default=2, help='클래스 수')

    # 모델 관련
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3plus'],
                        help='모델 선택')
    parser.add_argument('--base-channels', type=int, default=32, help='U-Net 기본 채널 수')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='DeepLabV3+ 인코더 사전학습 가중치')

    # 학습 관련
    parser.add_argument('--epochs', type=int, default=50, help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='데이터 로더 워커 수')

    # 저장 관련
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='체크포인트 저장 디렉토리')
    parser.add_argument('--save-interval', type=int, default=10, help='모델 저장 간격')

    args = parser.parse_args()

    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)
