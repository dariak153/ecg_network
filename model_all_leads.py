
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def dice_loss(pred, target, smooth=1e-6):
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
   
    # def __init__(self, input_size=1, conv_channels=64, gru_hidden1=64,gru_hidden2=32, dense_size=32, num_classes=1, dropout=0.3):
    def __init__(self, input_size=1, conv_channels=64, lstm_hidden1=64, lstm_hidden2=32, dense_size=32, num_classes=1, dropout=0.3):
        super(QRSModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=7, padding=7//2)
        self.bn1   = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=5, padding=5//2)
        self.bn2   = nn.BatchNorm1d(conv_channels)
        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=3//2)
        self.bn3   = nn.BatchNorm1d(conv_channels)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden1*2, hidden_size=lstm_hidden2, batch_first=True, bidirectional=True)
        # self.gru1 = nn.GRU(input_size=conv_channels, hidden_size=gru_hidden1, batch_first=True, bidirectional=True)
        # self.gru2 = nn.GRU(input_size=lstm_hidden1*2, hidden_size=gru_hidden2, batch_first=True, bidirectional=True)
        self.fc     = nn.Linear(lstm_hidden2*2, dense_size)
        self.out_fc = nn.Linear(dense_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
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
        x = x.transpose(1, 2)
           

        #x, _ = self.gru1(x)
        #x = self.dropout(x)
        #x, _ = self.gru2(x)
        #x = self.dropout(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_fc(x)
        out = self.sigmoid(x)
        return out

class QRSLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, threshold=0.5, fs=500):
        super(QRSLightning, self).__init__()
        self.save_hyperparameters()
        self.model = QRSModel()
        self.lr = lr
        self.threshold = threshold
        self.fs = fs
        self.test_outs = []

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, masks):
        return loss(logits, masks)

    def training_step(self, batch, batch_idx):
        signals, masks = batch
        preds = self(signals)
        loss_value = self.compute_loss(preds, masks)
        d = dice_loss((preds > self.threshold).float(), masks)
        self.log('train_loss', loss_value, prog_bar=True)
        self.log('train_dice', d, prog_bar=True)
        return loss_value

    def validation_step(self, batch, batch_idx):
        signals, masks = batch
        preds = self(signals)
        loss_value = self.compute_loss(preds, masks)
        d = dice_loss((preds > self.threshold).float(), masks)
        self.log('val_loss', loss_value, prog_bar=True)
        self.log('val_dice', d, prog_bar=True)
        return {'loss': loss_value}

    def test_step(self, batch, batch_idx):
        signals, masks = batch
        preds = self(signals)
        loss_value = self.compute_loss(preds, masks)
        out = {'loss': loss_value, 'preds': preds, 'masks': masks, 'signals': signals}
        self.test_outs.append(out)
        return out

    def on_test_epoch_end(self):
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

        n = 1000
        preds_trim = preds_all[:, n:-n, :]
        masks_trim = masks_all[:, n:-n, :]

        preds_bin = (preds_trim > self.threshold).astype(np.uint8)
        masks_bin = (masks_trim > 0.5).astype(np.uint8)

        flat_preds = preds_bin.flatten()
        flat_masks = masks_bin.flatten()

        acc = accuracy_score(flat_masks, flat_preds)
        prec = precision_score(flat_masks, flat_preds, zero_division=0)
        rec = recall_score(flat_masks, flat_preds, zero_division=0)
        f1_val = f1_score(flat_masks, flat_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(flat_masks, flat_preds)
        except:
            roc_auc = float('nan')
        d = dice_loss(torch.tensor(flat_preds), torch.tensor(flat_masks)).item()

        import pandas as pd
        metrics = {
            "Loss": [avg_loss],
            "Accuracy": [acc],
            "Precision": [prec],
            "Recall": [rec],
            "F1 Score": [f1_val],
            "ROC AUC": [roc_auc],
            "Dice": [d]
        }
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df)

        duration_info = []
        num_ex = preds_trim.shape[0]
        for i in range(num_ex):
            pred_qrs = preds_trim[i, :, 0]
            gt_qrs   = masks_trim[i, :, 0]
            pred_dur = np.sum(pred_qrs) / self.fs
            gt_dur   = np.sum(gt_qrs) / self.fs
            diff     = pred_dur - gt_dur
            duration_info.append({"example": i+1, "GT QRS": gt_dur, "Predicted QRS": pred_dur, "Diff (s)": diff})
        duration_df = pd.DataFrame(duration_info)
        print(duration_df)

        with open("duration_info.txt", "w") as f:
            f.write("QRS Duration:\n")
            f.write(duration_df.to_string(index=False))

        all_signals_full = np.concatenate(all_signals, axis=0)
        signals_trim = all_signals_full[:, n:-n, :]

        num_plot = min(10, signals_trim.shape[0])
        for i in range(num_plot):
            sig = signals_trim[i].squeeze(-1)
            gt_mask = masks_trim[i][:, 0]
            pred_mask = preds_trim[i][:, 0]
            time = np.arange(len(sig)) / self.fs
            plt.figure(figsize=(12, 4))
            plt.plot(time, sig, label='ECG Signal', color='blue')
            if np.any(gt_mask == 1):
                gt_indices = np.where(gt_mask == 1)[0]
                groups = np.split(gt_indices, np.where(np.diff(gt_indices) != 1)[0] + 1)
                first_gt = True
                for group in groups:
                    if len(group) > 0:
                        onset = group[0]
                        offset = group[-1]
                        plt.axvline(x=time[onset], color='green', linestyle='--', label='GT Onset' if first_gt else "")
                        plt.axvline(x=time[offset], color='green', linestyle=':', label='GT Offset' if first_gt else "")
                        first_gt = False
            if np.any(pred_mask > self.threshold):
                pred_indices = np.where(pred_mask > self.threshold)[0]
                groups = np.split(pred_indices, np.where(np.diff(pred_indices) != 1)[0] + 1)
                first_pred = True
                for group in groups:
                    if len(group) > 0:
                        onset = group[0]
                        offset = group[-1]
                        plt.axvline(x=time[onset], color='red', linestyle='--', label='Pred Onset' if first_pred else "")
                        plt.axvline(x=time[offset], color='red', linestyle=':', label='Pred Offset' if first_pred else "")
                        first_pred = False
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title(f"Example {i+1}: GT vs Pred")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

class ECGDataset(Dataset):
    def __init__(self, signals_file, masks_file):
        self.signals = np.load(signals_file, allow_pickle=True)
        self.masks   = np.load(masks_file, allow_pickle=True)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)
        qrs = (mask > 0).astype(np.float32)
        mask = np.expand_dims(qrs, axis=-1)
        if signal.ndim == 1:
            signal = np.expand_dims(signal, axis=-1)
        return torch.tensor(signal), torch.tensor(mask)

# Ścieżka do katalogu wyjściowego
output_dir = "."

train_dataset = ECGDataset(os.path.join(output_dir, "train_signals.npy"),
                           os.path.join(output_dir, "train_masks.npy"))
val_dataset   = ECGDataset(os.path.join(output_dir, "val_signals.npy"),
                           os.path.join(output_dir, "val_masks.npy"))
test_dataset  = ECGDataset(os.path.join(output_dir, "test_signals.npy"),
                           os.path.join(output_dir, "test_masks.npy"))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

model = QRSLightning(lr=1e-4, threshold=0.5, fs=500)

early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_qrs_model")

trainer = pl.Trainer(
    max_epochs=50, 
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[early_stop, checkpoint]
)

print("\nTraining\n")
trainer.fit(model, train_loader, val_loader)

print("\nTesting model\n")
trainer.test(model, test_loader)

