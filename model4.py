import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os

def dice_loss(pred, target, smooth=1e-6):
    # Liczymy dice loss na surowych wyjściach (bez binaryzacji)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    inter = (pred * target).sum()
    return (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)

def loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    d_loss = 1 - dice_loss(pred, target)
    return bce + d_loss

def iou_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)

def binary_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum()
    all_pred = pred_bin.sum()
    all_true = target.sum()
    sensitivity = tp / (all_true + 1e-6)
    precision = tp / (all_pred + 1e-6)
    f1 = 2 * tp / (all_pred + all_true + 1e-6)
    return sensitivity, precision, f1

class QRSModel(nn.Module):
    def __init__(self, input_size=1, conv_channels=64, lstm_hidden1=64, lstm_hidden2=32, dense_size=32, num_dense=64, num_classes=1, dropout=0.3):
        super(QRSModel, self).__init__()
        # Pięć warstw konwolucyjnych z BatchNorm, ReLU i Dropout
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=7, padding=7//2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=5, padding=5//2)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        
        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=3//2)
        self.bn3 = nn.BatchNorm1d(conv_channels)
        
        self.conv4 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=3//2)
        self.bn4 = nn.BatchNorm1d(conv_channels)
        
        self.conv5 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=3//2)
        self.bn5 = nn.BatchNorm1d(conv_channels)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Dwie warstwy LSTM
        self.lstm1 = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden1*2, hidden_size=lstm_hidden2, batch_first=True, bidirectional=True)
        
        # Dwie warstwy gęste
        self.fc1 = nn.Linear(lstm_hidden2*2, dense_size)
        self.fc2 = nn.Linear(dense_size, num_dense)
        self.out_fc = nn.Linear(num_dense, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch_size, num_samples, channels=1]
        x = x.transpose(1, 2)  # [batch_size, channels, num_samples]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [batch_size, num_samples, conv_channels]
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_fc(x)
        out = self.sigmoid(x)
        return out

class QRSLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, threshold=0.5, fs=500):
        super(QRSLightning, self).__init__()
        self.save_hyperparameters()
        self.model = QRSModel(input_size=1, conv_channels=32)
        self.lr = lr
        self.threshold = threshold
        self.fs = fs
        self.test_outs = []  # Będziemy zbierać wyniki dla obu leadów razem

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, masks):
        return loss(logits, masks)

    def training_step(self, batch, batch_idx):
        # Batch zawiera: signal1, mask1, signal2, mask2
        signal1, mask1, signal2, mask2 = batch
        # Łączymy dane z obu leadów wzdłuż wymiaru batcha
        signals = torch.cat([signal1, signal2], dim=0)
        masks = torch.cat([mask1, mask2], dim=0)
        preds = self(signals)
        loss_value = self.compute_loss(preds, masks)
        dice_val = dice_loss(preds, masks)
        self.log('train_loss', loss_value, prog_bar=True)
        self.log('train_dice', dice_val, prog_bar=True)
        return loss_value

    def validation_step(self, batch, batch_idx):
        signal1, mask1, signal2, mask2 = batch
        signals = torch.cat([signal1, signal2], dim=0)
        masks = torch.cat([mask1, mask2], dim=0)
        preds = self(signals)
        loss_value = self.compute_loss(preds, masks)
        dice_val = dice_loss(preds, masks)
        self.log('val_loss', loss_value, prog_bar=True)
        self.log('val_dice', dice_val, prog_bar=True)
        return {'loss': loss_value}

    def test_step(self, batch, batch_idx):
        signal1, mask1, signal2, mask2 = batch
        signals = torch.cat([signal1, signal2], dim=0)
        masks = torch.cat([mask1, mask2], dim=0)
        preds = self(signals)
        loss_value = self.compute_loss(preds, masks)
        self.test_outs.append({'loss': loss_value, 'preds': preds, 'masks': masks, 'signals': signals})
        return {'loss': loss_value}

    def test_pred(self):
        # Przetwarzamy zgromadzone wyniki wspólnie dla obu leadów
        outs = self.test_outs
        self.test_outs = []
        all_preds = []
        all_masks = []
        all_signals = []
        losses = []
        for o in outs:
            losses.append(o['loss'].item())
            all_preds.append(o['preds'].detach().cpu().numpy())
            all_masks.append(o['masks'].detach().cpu().numpy())
            all_signals.append(o['signals'].detach().cpu().numpy())
        avg_loss = np.mean(losses)
        preds_all = np.concatenate(all_preds, axis=0)
        masks_all = np.concatenate(all_masks, axis=0)
        preds_bin = (preds_all > self.threshold).astype(np.uint8)
        masks_bin = (masks_all > 0.5).astype(np.uint8)
        flat_preds = preds_bin.flatten()
        flat_masks = masks_bin.flatten()
        acc = accuracy_score(flat_masks, flat_preds)
        prec = precision_score(flat_masks, flat_preds, zero_division=0)
        rec = recall_score(flat_masks, flat_preds, zero_division=0)
        f1_val = f1_score(flat_masks, flat_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(flat_masks, flat_preds)
        except Exception:
            roc_auc = float('nan')
        loss_dice_val = dice_loss(torch.tensor(flat_preds), torch.tensor(flat_masks)).item()
        metrics = {
            "Loss": [avg_loss],
            "Accuracy": [acc],
            "Precision": [prec],
            "Recall": [rec],
            "F1 Score": [f1_val],
            "ROC AUC": [roc_auc],
            "Dice": [loss_dice_val]
        }
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df)

        # Wizualizacja – przycinamy krawędzie, by uniknąć artefaktów paddingu
        n_trim = 50
        num_plot = min(10, np.concatenate(all_signals, axis=0).shape[0])
        for i in range(num_plot):
            sig = all_signals[i].squeeze(-1)
            if len(sig) > 2 * n_trim:
                sig = sig[n_trim:-n_trim]
                mask_plot = masks_bin[i][n_trim:-n_trim, 0]
            else:
                mask_plot = masks_bin[i][:, 0]
            time = np.arange(len(sig)) / self.fs
            plt.figure(figsize=(12, 4))
            plt.plot(time, sig, label='ECG Signal', color='blue')
            if np.any(mask_plot == 1):
                gt_indices = np.where(mask_plot == 1)[0]
                groups = np.split(gt_indices, np.where(np.diff(gt_indices) != 1)[0] + 1)
                first = True
                for group in groups:
                    if len(group) > 0:
                        onset = group[0]
                        offset = group[-1]
                        plt.axvline(x=time[onset], color='green', linestyle='--',
                                    label='GT Onset' if first else "")
                        plt.axvline(x=time[offset], color='green', linestyle=':',
                                    label='GT Offset' if first else "")
                        first = False
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title(f"Example {i+1}")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

class ECGDataset(Dataset):
    def __init__(self, signals_file_lead1, signals_file_lead2, masks_file_lead1, masks_file_lead2):
        # Ładujemy sygnały dla obu leadów
        signals_dict1 = np.load(signals_file_lead1, allow_pickle=True).item()
        signals_dict2 = np.load(signals_file_lead2, allow_pickle=True).item()
        # Ładujemy maski osobno
        masks_dict1 = np.load(masks_file_lead1, allow_pickle=True).item()
        masks_dict2 = np.load(masks_file_lead2, allow_pickle=True).item()
        
        # Zakładamy, że klucze (record_ids) są takie same we wszystkich plikach
        self.record_ids = list(signals_dict1.keys())
        self.signals1 = []
        self.signals2 = []
        self.masks1 = []
        self.masks2 = []
        for rec_id in self.record_ids:
            s1 = np.squeeze(signals_dict1[rec_id])
            s2 = np.squeeze(signals_dict2[rec_id])
            m1 = np.squeeze(masks_dict1[rec_id])
            m2 = np.squeeze(masks_dict2[rec_id])
            # Upewnij się, że sygnały mają kształt (num_samples, 1)
            s1 = np.expand_dims(s1, axis=-1) if s1.ndim == 1 else s1
            s2 = np.expand_dims(s2, axis=-1) if s2.ndim == 1 else s2
            m1 = np.expand_dims(m1, axis=-1) if m1.ndim == 1 else m1
            m2 = np.expand_dims(m2, axis=-1) if m2.ndim == 1 else m2
            self.signals1.append(s1)
            self.signals2.append(s2)
            self.masks1.append(m1)
            self.masks2.append(m2)

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        signal1 = self.signals1[idx].astype(np.float32)
        signal2 = self.signals2[idx].astype(np.float32)
        mask1 = self.masks1[idx].astype(np.float32)
        mask2 = self.masks2[idx].astype(np.float32)
        # Upewnij się, że maski mają wartość 0 lub 1
        mask1 = (mask1 > 0).astype(np.float32)
        mask2 = (mask2 > 0).astype(np.float32)
        return (torch.tensor(signal1), torch.tensor(mask1),
                torch.tensor(signal2), torch.tensor(mask2))

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    train_dataset = ECGDataset(
        os.path.join("signals", "train", "train_lead1.npy"),
        os.path.join("signals", "train", "train_lead2.npy"),
        os.path.join("masks", "train", "train_mask.npy")
    )
    val_dataset = ECGDataset(
        os.path.join("signals", "val", "val_lead1.npy"),
        os.path.join("signals", "val", "val_lead2.npy"),
        os.path.join("masks", "val", "val_mask.npy")
    )
    test_dataset = ECGDataset(
        os.path.join("signals", "test", "test_lead1.npy"),
        os.path.join("signals", "test", "test_lead2.npy"),
        os.path.join("masks", "test", "test_mask.npy")
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)

    model = QRSLightning(lr=1e-4, threshold=0.5, fs=500)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop, checkpoint]
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    model.test_pred()
