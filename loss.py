#!/usr/bin/env python3
import os
import sys
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from pytorch_tcn import TCN



def dice_coef(pred, target, smooth=1e-6):
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    inter = (p * t).sum()
    return (2. * inter + smooth) / (p.sum() + t.sum() + smooth)



def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, reduction="mean"):
    loss = -pos_weight * targets * sigmoid_x.log() - (1 - targets) * (1 - sigmoid_x).log()
    if weight is not None:
        loss = loss * weight
    return loss.mean() if reduction == "mean" else loss.sum()

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, weight: torch.Tensor = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("weight", weight if weight is not None else torch.ones_like(pos_weight))
    def forward(self, input, target):
        return weighted_binary_cross_entropy(torch.sigmoid(input), target, self.pos_weight, self.weight)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight
        self.d_w = dice_weight
    def forward(self, input, target):
        probs = torch.sigmoid(input)
        b = self.bce(input, target)
        d = 1 - dice_coef(probs, target)
        return self.bce_w * b + self.d_w * d

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = torch.exp(-bce)
        fl = (1 - p_t) ** self.gamma * bce
        return fl.mean() if self.reduction == "mean" else fl.sum()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, prob_margin=0.05, reduction="mean"):
        super().__init__()
        self.gn, self.gp, self.pm, self.red = gamma_neg, gamma_pos, prob_margin, reduction
        self.eps = 1e-8
    def forward(self, input, target):
        pos = torch.sigmoid(input)
        neg = (1 - pos + self.pm).clamp(max=1)
        lp = target * torch.log(pos.clamp(min=self.eps)) * (1 - pos) ** self.gp
        ln = (1 - target) * torch.log(neg.clamp(min=self.eps)) * pos ** self.gn
        return -(lp + ln).mean() if self.red=="mean" else -(lp+ln).sum()

class MSEProbLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    def forward(self, input, target):
        return self.mse(torch.sigmoid(input), target)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, input, target):
        probs = torch.sigmoid(input).view(-1)
        t = target.view(-1)
        tp = (probs * t).sum()
        fp = ((1 - t) * probs).sum()
        fn = (t * (1 - probs)).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

def setup_criterion(name: str, **kwargs):
    mapping = {
        "WeightedBCELoss": WeightedBCELoss,
        "BCEDiceLoss":  BCEDiceLoss,
        "FocalLoss":    FocalLoss,
        "AsymmetricLoss": AsymmetricLoss,
        "MSEProbLoss":  MSEProbLoss,
        "TverskyLoss":  TverskyLoss,
    }
    if name in mapping:
        return mapping[name](**kwargs)
    if name.startswith("nn."):
        return eval(name)(**kwargs)
    return getattr(nn, name)(**kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
      
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class HybridTCNTransformer(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 crit_name: str,
                 crit_kwargs: dict,
              
                 tcn_channels=(256,128,64),
                 kernel_size=4,
                 dropout=0.4,
                 causal=False,
             
                 d_model=64,
                 nhead=4,
                 dim_feedforward=128,
                 num_layers=1,
                 max_pos_len=10000,
             
                 lr=1e-4,
                 thr=0.5):
        super().__init__()
        self.save_hyperparameters()

    
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=list(tcn_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            input_shape='NCL'
        )
        tcn_out_ch = tcn_channels[-1]

     
        self.linear_proj = nn.Linear(tcn_out_ch, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_pos_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- head ---
        self.classifier = nn.Linear(d_model, output_size)
        self.act = nn.Sigmoid() if output_size==1 else None

        # --- loss ---
        self.crit = setup_criterion(crit_name, **crit_kwargs)

    def forward(self, x):
        # x: (B, L, C)
        # TCN expects (B, C, L)
        x = x.transpose(1,2)
        x = self.tcn(x)                          # (B, tcn_out_ch, L)
        x = x.transpose(1,2)                     # (B, L, tcn_out_ch)
        x = self.linear_proj(x)                  # (B, L, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                  # (B, L, d_model)
        x = self.classifier(x)                   # (B, L, output_size)
        if self.act: x = self.act(x)
        return x

    def compute_loss(self, preds, target):
        return self.crit(preds, target)

    def training_step(self, batch, batch_idx):
        s, m = batch
        preds = self(s)
        loss  = self.compute_loss(preds, m)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, m = batch
        preds = self(s)
        loss  = self.compute_loss(preds, m)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}



class dataset_apnea(Dataset):
    def __init__(self, sf, mf):
        self.sig  = np.load(sf, allow_pickle=True)
        self.mask = np.load(mf, allow_pickle=True)
    def __len__(self): return len(self.sig)
    def __getitem__(self, idx):
        s = self.sig[idx].astype(np.float32)
        m = self.mask[idx].astype(np.float32)
        m = np.expand_dims((m>0).astype(np.float32), -1)
        if s.ndim==1:
            s = np.expand_dims(s, -1)
        return torch.tensor(s), torch.tensor(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
   
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    if len(sys.argv)==1:
        parser.print_help(); sys.exit(0)


    parser.add_argument('--base_dir',   type=str,   default=os.getcwd())
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--max_epochs', type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--thr',        type=float, default=0.5)

    # loss
    parser.add_argument('--crit_name',  type=str, required=True,
                        choices=['nn.MSELoss','WeightedBCELoss','BCEDiceLoss',
                                 'FocalLoss','AsymmetricLoss','MSEProbLoss','TverskyLoss'])
    parser.add_argument('--pos_weight', type=float, nargs='+')
    parser.add_argument('--class_weight', type=float, nargs='+')
    parser.add_argument('--gamma',      type=float, default=2.0)
    parser.add_argument('--gamma_neg',  type=float, default=4.0)
    parser.add_argument('--gamma_pos',  type=float, default=1.0)
    parser.add_argument('--prob_margin',type=float, default=0.05)
    parser.add_argument('--alpha',      type=float, default=0.5)
    parser.add_argument('--beta',       type=float, default=0.5)

    # TCN params
    parser.add_argument('--tcn_channels',     type=int, nargs='+', default=[256,128,64])
    parser.add_argument('--kernel_size',      type=int, default=4)
    parser.add_argument('--dropout',          type=float, default=0.4)
    parser.add_argument('--causal',           action='store_true')

    # transformer params
    parser.add_argument('--d_model',         type=int,   default=64)
    parser.add_argument('--nhead',           type=int,   default=4)
    parser.add_argument('--dim_feedforward', type=int,   default=128)
    parser.add_argument('--num_layers',      type=int,   default=1)
    parser.add_argument('--max_pos_len',     type=int,   default=10000)

    args = parser.parse_args()

  
    crit_kwargs = {}
    if args.crit_name == 'WeightedBCELoss' and args.pos_weight:
        crit_kwargs['pos_weight'] = torch.tensor(args.pos_weight)
    if args.crit_name == 'BCEWithLogitsWithClassWeightLoss' and args.class_weight:
        crit_kwargs['class_weight'] = torch.tensor(args.class_weight)
    if args.crit_name == 'FocalLoss':
        crit_kwargs['gamma'] = args.gamma
    if args.crit_name == 'AsymmetricLoss':
        crit_kwargs.update({'gamma_neg':args.gamma_neg,'gamma_pos':args.gamma_pos,'prob_margin':args.prob_margin})
    if args.crit_name == 'TverskyLoss':
        crit_kwargs.update({'alpha':args.alpha,'beta':args.beta})


    train_ds = dataset_apnea(os.path.join(args.base_dir,"train_signals.npy"),
                             os.path.join(args.base_dir,"train_masks.npy"))
    val_ds   = dataset_apnea(os.path.join(args.base_dir,"val_signals.npy"),
                             os.path.join(args.base_dir,"val_masks.npy"))
    test_ds  = dataset_apnea(os.path.join(args.base_dir,"test_signals.npy"),
                             os.path.join(args.base_dir,"test_masks.npy"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = HybridTCNTransformer(
        input_size=1,
        output_size=1,
        crit_name=args.crit_name,
        crit_kwargs=crit_kwargs,
        tcn_channels=tuple(args.tcn_channels),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        causal=args.causal,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        max_pos_len=args.max_pos_len,
        lr=args.lr,
        thr=args.thr
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(model, train_loader, val_loader)

    ys_pred, ys_true = [], []
    model.eval()
    with torch.no_grad():
        for s, m in test_loader:
            preds = model(s.to(model.device))
            probs = preds.cpu().numpy().flatten()
            bin_pred = (probs > args.thr).astype(np.float32)
            ys_pred.append(bin_pred)
            ys_true.append(m.numpy().flatten())

    y_pred = np.concatenate(ys_pred)
    y_true = np.concatenate(ys_true)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1s  = f1_score(y_true, y_pred, zero_division=0)
    dice = dice_coef(torch.tensor(y_pred), torch.tensor(y_true)).item()
    iou  = (y_pred * y_true).sum() + 1e-6
    iou /= (y_pred + y_true - y_pred*y_true).sum() + 1e-6

    print("Test metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1s:.4f}")
    print(f"  Dice:      {dice:.4f}")
    print(f"  IoU:       {iou:.4f}")

    df = pd.DataFrame([{
        "Loss Function": args.crit_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1s,
        "Dice": dice,
        "IoU": iou
    }])
    out_csv = os.path.join(args.base_dir, "loss_metrics.csv")
    df.to_csv(out_csv, index=False)
