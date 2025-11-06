# train_cdcn_improved.py
# CDCN++ cải tiến (CBAM + MSF + Focal+BCE + Augment) - train & eval & export
# Usage: python train_cdcn_improved.py

import os
import math
import time
import random
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

SEED = 2025
DATA_DIR = "......."  
IMG_SIZE = 256
EPOCHS = 5
BATCH = 32
LR = 1e-4
WD = 1e-4
PATIENCE_ES = 6
USE_SEED = True
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
SAVE_DIR = "./checkpoints_cdcn_improved"
os.makedirs(SAVE_DIR, exist_ok=True)

if USE_SEED:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if USE_CUDA: torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

def apcer_bpcer_acer(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    fake_m = (y_true==1); real_m = (y_true==0)
    apcer = ((y_pred[fake_m]==0).sum() / max(1, fake_m.sum()))
    bpcer = ((y_pred[real_m]==1).sum() / max(1, real_m.sum()))
    acer  = 0.5*(apcer+bpcer)
    return apcer, bpcer, acer

def best_thr_by_acer(y_true, y_score):
    cands = np.quantile(y_score, np.linspace(0.01, 0.99, 99))
    best = (1.0, 0.5)
    for t in cands:
        _, _, acer = apcer_bpcer_acer(y_true, y_score, t)
        if acer < best[0]:
            best = (acer, float(t))
    return best[1], best[0]

def compute_binary_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    apcer = fp / (tn + fp + 1e-8)
    bpcer = fn / (tp + fn + 1e-8)
    acer = 0.5*(apcer+bpcer)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return dict(acc=acc, apcer=apcer, bpcer=bpcer, acer=acer, prec=prec, rec=rec, f1=f1)

class Conv2d_cd(nn.Module):
    """Central-difference 2D conv (original idea)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        kernel_diff = self.conv.weight.sum(2).sum(2)[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff,
                            bias=self.conv.bias, stride=self.conv.stride,
                            padding=0, groups=self.conv.groups)
        return out_normal - self.theta * out_diff

class CBAM(nn.Module):
    """CBAM: Channel + Spatial attention"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channels = channels
        # channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # spatial
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size//2), bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # channel
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca
        # spatial
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg, mx], dim=1)))
        x = x * sa
        return x

class CDCNpp_CBAM(nn.Module):
    def __init__(self, theta=0.7):
        super().__init__()
        self.theta = theta
        self.conv1 = nn.Sequential(
            Conv2d_cd(3, 64, 3, 1, 1, theta=theta), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        # Block1
        self.block1 = nn.Sequential(
            Conv2d_cd(64, 128, 3, 1, 1, theta=theta), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            Conv2d_cd(128, int(128*1.6), 3, 1, 1, theta=theta), nn.BatchNorm2d(int(128*1.6)), nn.ReLU(inplace=True),
            Conv2d_cd(int(128*1.6), 128, 3, 1, 1, theta=theta), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        # Block2
        self.block2 = nn.Sequential(
            Conv2d_cd(128, int(128*1.2), 3,1,1, theta=theta), nn.BatchNorm2d(int(128*1.2)), nn.ReLU(inplace=True),
            Conv2d_cd(int(128*1.2), 128, 3,1,1, theta=theta), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        # Block3
        self.block3 = nn.Sequential(
            Conv2d_cd(128, 128, 3,1,1, theta=theta), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(128)

        self.down = nn.Upsample(size=(32,32), mode='bilinear', align_corners=False)
        self.fusion = nn.Sequential(
            Conv2d_cd(128*3, 128, 3,1,1, theta=theta), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            Conv2d_cd(128, 64, 3,1,1, theta=theta), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)            
        f1 = self.block1(x)        
        f2 = self.block2(f1)         
        f3 = self.block3(f2)         

        f1 = self.cbam1(f1)
        f2 = self.cbam2(f2)
        f3 = self.cbam3(f3)

        f1s = self.down(f1)
        f2s = self.down(f2)
        f3s = self.down(f3)
        fused = torch.cat([f1s, f2s, f3s], dim=1)   
        fused = self.fusion(fused)                 
        out = self.head(fused)                      
        return out.squeeze(1)                       

class FASDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for lab_name in ["real","fake"]:
            folder = os.path.join(root_dir, lab_name)
            if not os.path.isdir(folder): continue
            label = 0 if lab_name=="real" else 1 
            for p in glob(os.path.join(folder, "*")):
                if p.lower().endswith((".jpg",".jpeg",".png")):
                    self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop((IMG_SIZE,IMG_SIZE), scale=(0.8,1.15)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def evaluate(model, dl, device, use_best_thr=False):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labs in dl:
            imgs = imgs.to(device)
            labs = labs.to(device)
            logits = model(imgs)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labs.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    probs = 1/(1+np.exp(-all_logits))
    all_labels = np.concatenate(all_labels, axis=0).astype(int)
    try:
        auc = roc_auc_score(all_labels, probs)
    except:
        auc = float("nan")
    thr, acer = best_thr_by_acer(all_labels, probs)
    metrics = compute_binary_metrics(all_labels, probs, thr)
    metrics['auc'] = auc
    metrics['thr'] = thr
    metrics['acer_thr'] = acer
    return metrics, probs, all_labels

def train_loop(data_dir, epochs=EPOCHS, batch_size=BATCH, lr=LR):
    device = DEVICE
    print("[INFO] Device:", device)
    ds = FASDataset(data_dir, transform=train_tfms)
    n = len(ds)
    n_train = int(0.7 * n); n_val = int(0.15 * n); n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    val_ds.dataset.transform = eval_tfms
    test_ds.dataset.transform = eval_tfms

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CDCNpp_CBAM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_acer = 1e9
    best_ckpt = None
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for imgs, labs in pbar:
            imgs = imgs.to(device)
            labs = labs.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(imgs)
                loss_bce = bce(logits, labs)
                loss_focal = focal(logits, labs)
                loss = 0.6*loss_bce + 0.4*loss_focal
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f"{total_loss/((pbar.n+1)*imgs.size(0)):.4f}"})

        val_metrics, val_probs, val_labels = evaluate(model, val_loader, device)
        print(f"[EP {ep}] TrainLoss={total_loss/len(train_loader.dataset):.4f} | Val Acc={val_metrics['acc']:.4f} APCER={val_metrics['apcer']:.4f} BPCER={val_metrics['bpcer']:.4f} ACER={val_metrics['acer']:.4f} AUC={val_metrics['auc']:.4f}")

        if val_metrics['acer'] + 1e-6 < best_acer:
            best_acer = val_metrics['acer']
            best_ckpt = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': ep,
                'val_metrics': val_metrics
            }
            torch.save(best_ckpt, os.path.join(SAVE_DIR, "best_cdcnpp_cbam.pth"))
            print(f"[INFO] New best ACER {best_acer:.5f} saved.")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE_ES:
                print(f"[INFO] Early stopping at epoch {ep}. Best ACER {best_acer:.4f}")
                break

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt['model_state'])
    test_metrics, test_probs, test_labels = evaluate(model, test_loader, device)
    print("=== FINAL TEST ===")
    print(f"Test: Acc={test_metrics['acc']:.4f}, APCER={test_metrics['apcer']:.4f}, BPCER={test_metrics['bpcer']:.4f}, ACER={test_metrics['acer']:.4f}, AUC={test_metrics['auc']:.4f}, F1={test_metrics['f1']:.4f}")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cdcnpp_cbam_final.pth"))
    try:
        model_cpu = model.to('cpu').eval()
        dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
        torch.onnx.export(model_cpu, dummy, os.path.join(SAVE_DIR, "cdcnpp_cbam.onnx"),
                          input_names=['input'], output_names=['output'], opset_version=17,
                          dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}})
        print("Done")
    except Exception as e:
        print("Export failed:", e)

    return model, test_metrics

# Test: Acc=0.9596, APCER=0.0471, BPCER=0.0339, ACER=0.0405, AUC=0.9920, F1=0.9601
