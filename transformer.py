import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class PositionEmbedding(nn.Module):
    def __init__(self, emb_type: str, input_dim: int, max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.emb_type = emb_type
        if emb_type == "static":
            pe = torch.zeros(max_length, input_dim)
            pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, input_dim, 2).float() * (-np.log(10000.0) / input_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
            self.dropout = nn.Dropout(dropout)
        elif emb_type == "learn":
            self.pe = nn.Embedding(max_length, input_dim)
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown emb_type {emb_type}")

    def forward(self, x):
        B, T, D = x.shape
        if self.emb_type == "static":
            x = x + self.pe[:, :T, :]
        else:
            idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            x = x + self.pe(idx)
        return self.dropout(x)


def generate_mask_square(sz, prev=0):
    mask = (torch.triu(torch.ones(sz, sz), -prev) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, 0.0)
    return mask

class BatchFirstTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, batch_first=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, src, *args, **kwargs):
        if self.batch_first:
            src = src.transpose(0,1)
        out = super().forward(src, *args, **kwargs)
        return out.transpose(0,1)

class TSTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        emb_dim: int,
        emb_type: str,
        hidden_size: int,
        dropout: float,
        num_layers: int = 2,
        num_heads: int = 4,
        is_bidir: bool = False,
        use_last: bool = False,
        max_length: int = 5000,
    ):
        super().__init__()
        if input_size is not None:
            self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        else:
            self.encoder = None

        self.emb_dim = emb_dim
        if emb_dim > 0 and input_size is not None:
            self.pe = PositionEmbedding(emb_type, input_size, max_length, dropout)

        layer = BatchFirstTransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers)
        self.is_bidir = is_bidir
        self.use_last = use_last

    def forward(self, x):
        if self.emb_dim > 0:
            x = self.pe(x)
        z = self.encoder(x) if self.encoder is not None else x

        mask = generate_mask_square(z.size(1), 0).to(z.device)
        if self.is_bidir:
            mask = torch.zeros_like(mask)


        h = self.temporal_encoder(z, mask)  
        out = h[:, -1, :] if self.use_last else h.mean(dim=1)
        return h, out


def dice_coeff(pred, target, smooth=1e-6):
    p = pred.view(-1); t = target.view(-1)
    inter = (p * t).sum()
    return (2 * inter + smooth) / (p.sum() + t.sum() + smooth)

def composite_loss(pred, target, λ_bce=1.0, λ_dice=1.0, λ_mse=1.0):
    bce = nn.BCELoss()(pred, target)
    dice = 1 - dice_coeff(pred, target)
    mse  = nn.MSELoss()(pred, target)
    return λ_bce * bce + λ_dice * dice + λ_mse * mse

class QRSModel(nn.Module):
    def __init__(self,
        input_size=1,
        conv_channels=64,
        emb_dim=64,
        emb_type="learn",
        hidden_size=64,
        transformer_layers=2,
        transformer_heads=4,
        dropout=0.2,
        dense_size=32,
        num_classes=1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_channels, 7, padding=3)
        self.bn1   = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(conv_channels)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(conv_channels)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)

        self.ts = TSTransformer(
            input_size=conv_channels,
            emb_dim=emb_dim,
            emb_type=emb_type,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            is_bidir=False,
            use_last=False,
            max_length=5000
        )

        self.fc     = nn.Linear(hidden_size, dense_size)
        self.out_fc = nn.Linear(dense_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1,2)
        for conv,bn in ((self.conv1,self.bn1),(self.conv2,self.bn2),(self.conv3,self.bn3)):
            x = conv(x); x = bn(x)
            x = self.relu(x); x = self.drop(x)
        x = x.transpose(1,2)
        h, _ = self.ts(x)
        out = self.fc(h)
        out = self.relu(out); out = self.drop(out)
        out = self.out_fc(out)
        return self.sigmoid(out)

class QRSLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, threshold=0.5, fs=500):
        super().__init__()
        self.save_hyperparameters()
        self.model = QRSModel()
        self.lr = lr
        self.threshold = threshold
        self.fs = fs
        self.test_outs = []

    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        sig, m = batch
        p = self(sig)
        loss = composite_loss(p, m, λ_bce=1.0, λ_dice=1.0, λ_mse=0.5)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sig, m = batch
        p = self(sig)
        loss = composite_loss(p, m, λ_bce=1.0, λ_dice=1.0, λ_mse=0.5)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        sig, m = batch
        p = self(sig)
        self.test_outs.append((p.detach().cpu().numpy(), m.detach().cpu().numpy()))

    def on_test_epoch_end(self):
        preds, masks = zip(*self.test_outs)
        preds = np.concatenate(preds,0)[...,0]
        masks = np.concatenate(masks,0)[...,0]
        n = 1000
        preds = preds[:, n:-n]
        masks = masks[:, n:-n]
        pred_b = (preds > self.threshold).astype(int)
        dur_p = pred_b.sum(axis=1)/self.fs
        dur_t = masks.sum(axis=1)/self.fs
        err   = dur_p - dur_t

        stats = {
            "Mean (s)"   : err.mean(),
            "Median (s)" : np.median(err),
            "Std (s)"    : err.std(ddof=1),
            "MSE (s²)"   : np.mean(err**2),
            "RMSE (s)"   : np.sqrt(np.mean(err**2)),
            "5th pct (s)": np.percentile(err,5),
            "95th pct (s)": np.percentile(err,95),
        }
        df_stats = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        print("\n=== Duration error statistics ===")
        print(df_stats)

        plt.figure(figsize=(8,4))
        sns.histplot(err, bins=40, kde=True, edgecolor='black')
        plt.title("Detection error distribution")
        plt.xlabel("Error (s)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

class ECGDataset(Dataset):
    def __init__(self, signals_file, masks_file):
        self.signals = np.load(signals_file, allow_pickle=True)
        self.masks   = np.load(masks_file, allow_pickle=True)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        s = self.signals[idx].astype(np.float32)
        m = (self.masks[idx].astype(np.float32) > 0).astype(np.float32)
        if s.ndim == 1:
            s = s[:, None]
        return torch.tensor(s), torch.tensor(m[..., None])

if __name__ == "__main__":
    output_dir = "."
    train_ds = ECGDataset(os.path.join(output_dir,"train_signals.npy"),
                          os.path.join(output_dir,"train_masks.npy"))
    val_ds   = ECGDataset(os.path.join(output_dir,"val_signals.npy"),
                          os.path.join(output_dir,"val_masks.npy"))
    test_ds  = ECGDataset(os.path.join(output_dir,"test_signals.npy"),
                          os.path.join(output_dir,"test_masks.npy"))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4)
    test_loader  = DataLoader(test_ds,  batch_size=4)

    model = QRSLightning(lr=1e-4, threshold=0.5, fs=500)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    ckpt       = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_qrs_transformer")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop, ckpt],
    )

    print("\n=== Training ===")
    trainer.fit(model, train_loader, val_loader)

    print("\n=== Testing ===")
    trainer.test(model, test_loader)
