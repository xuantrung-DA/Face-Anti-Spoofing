import argparse
import random
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from data_splits import manifest_datasets

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2).unsqueeze(2).unsqueeze(3)
            out_diff = F.conv2d(x, kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff

class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CDCNpp(nn.Module):
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCNpp, self).__init__()
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(128), nn.ReLU(),
            basic_conv(128, int(128*1.6), 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128*1.6)), nn.ReLU(),
            basic_conv(int(128*1.6), 128, 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(int(128*1.2)), nn.ReLU(),
            basic_conv(int(128*1.2), 128, 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.sa1, self.sa2, self.sa3 = SpatialAttention(7), SpatialAttention(5), SpatialAttention(3)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear')

        self.lastconv = nn.Sequential(
            basic_conv(128*3, 64, 3, 1, 1, bias=False, theta=theta),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.Block1(x)
        x2 = self.Block2(x1)
        x3 = self.Block3(x2)
        x1, x2, x3 = self.sa1(x1)*x1, self.sa2(x2)*x2, self.sa3(x3)*x3
        x_cat = torch.cat([self.downsample(x1), self.downsample(x2), self.downsample(x3)], dim=1)
        return self.lastconv(x_cat)

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples, self.transform = [], transform
        for label_name in ["real", "fake"]:
            folder = os.path.join(root_dir, label_name)
            label = 1 if label_name == "real" else 0
            if not os.path.exists(folder): continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# === Metrics ===
def compute_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    apcer, bpcer = fp / (tn + fp + 1e-8), fn / (tp + fn + 1e-8)
    acer = (apcer + bpcer) / 2
    prec = precision_score(y_true, y_pred_bin)
    rec = recall_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)
    return acc, apcer, bpcer, acer, prec, rec, f1

def train_model(data_dir, epochs=50, batch_size=16, lr=1e-4, *, split_manifest, seed=2025,
                output_dir="checkpoints/sa"):
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 256)),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    full_dataset = FaceDataset(data_dir)
    train_ds, val_ds, test_ds = manifest_datasets(
        full_dataset, data_dir, split_manifest, transform, eval_transform, real_label=1
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CDCNpp().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = torch.sigmoid(model(imgs)).cpu().numpy().flatten()
                preds.extend(outputs)
                gts.extend(labels.numpy())
        acc, apcer, bpcer, acer, prec, rec, f1 = compute_metrics(np.array(gts), np.array(preds))
        print(f"\nEpoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Acc={acc:.4f}, APCER={apcer:.4f}, BPCER={bpcer:.4f}, ACER={acer:.4f}, "
              f"P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "sa_final.pt"))

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = torch.sigmoid(model(imgs)).cpu().numpy().flatten()
            preds.extend(outputs)
            gts.extend(labels.numpy())
    acc, apcer, bpcer, acer, prec, rec, f1 = compute_metrics(np.array(gts), np.array(preds))
    print(f"Test results: Acc={acc:.4f}, APCER={apcer:.4f}, BPCER={bpcer:.4f}, "
          f"ACER={acer:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SA on explicit group-disjoint splits.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-dir", default="checkpoints/sa")
    args = parser.parse_args()
    train_model(args.data_dir, epochs=args.epochs, split_manifest=args.split_manifest,
                seed=args.seed, output_dir=args.output_dir)
