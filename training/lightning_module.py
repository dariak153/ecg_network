import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from models.model import Model  
from utils.loss import dice_loss, loss_fn

class LightningModel(pl.LightningModule):
    def __init__(self, lr=1e-4, threshold=0.5, fs=500, output_length=5000):
        super().__init__()
        self.save_hyperparameters(ignore=["lr", "threshold", "fs", "output_length"])
        self.lr = lr
        self.threshold = threshold
        self.fs = fs
        self.model = Model(
            input_channels=1, 
            dropout=0.2, 
            lstm_hidden1=64, 
            lstm_hidden2=32,
            fc_hidden=32, 
            num_classes=1, 
            output_length=output_length
        )
        self.test_results = []  

    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, preds, masks):
        return loss_fn(preds, masks)

    def training_step(self, batch, batch_idx):
        signals, masks = batch
        preds = self(signals)
        loss_val = self.compute_loss(preds, masks)
        dice_val = dice_loss((preds > self.threshold).float(), masks)
        self.log('train_loss', loss_val, prog_bar=True)
        self.log('train_dice', dice_val, prog_bar=True)
        return loss_val
    
    def validation_step(self, batch, batch_idx):
        signals, masks = batch
        preds = self(signals)
        loss_val = self.compute_loss(preds, masks)
        dice_val = dice_loss((preds > self.threshold).float(), masks)
        self.log('val_loss', loss_val, prog_bar=True)
        self.log('val_dice', dice_val, prog_bar=True)
        return {"val_loss": loss_val}
    
    def test_step(self, batch, batch_idx):
        signals, masks = batch
        preds = self(signals)
        loss_val = self.compute_loss(preds, masks)
        self.test_results.append({
            'loss': loss_val.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy(),
            'masks': masks.detach().cpu().numpy(),
            'signals': signals.detach().cpu().numpy()
        })
        return {"test_loss": loss_val}
    
    def on_test_epoch_end(self):
        preds_all = np.concatenate([res['preds'] for res in self.test_results], axis=0)
        masks_all = np.concatenate([res['masks'] for res in self.test_results], axis=0)
        signals_all = np.concatenate([res['signals'] for res in self.test_results], axis=0)
        avg_loss = np.mean([res['loss'] for res in self.test_results])
        
        preds_trim = preds_all[:, 1000:-1000, :]
        masks_trim = masks_all[:, 1000:-1000, :]
        
      
        flat_preds = (preds_trim > self.threshold).astype(np.uint8).flatten()
        flat_masks = (masks_trim > 0.5).astype(np.uint8).flatten()
  
        metrics = {
            "Loss": avg_loss,
            "Accuracy": accuracy_score(flat_masks, flat_preds),
            "Precision": precision_score(flat_masks, flat_preds, zero_division=0),
            "Recall": recall_score(flat_masks, flat_preds, zero_division=0),
            "F1 Score": f1_score(flat_masks, flat_preds, zero_division=0),
            "ROC AUC": roc_auc_score(flat_masks, flat_preds) if np.any(flat_masks) else float('nan'),
            "Dice": dice_loss(torch.tensor(flat_preds), torch.tensor(flat_masks)).item()
        }
        metrics_df = pd.DataFrame(metrics, index=[0])
    
        print(metrics_df)
        
    
        duration_data = []
        for i in range(preds_trim.shape[0]):
            pred_qrs = preds_trim[i, :, 0]
            gt_qrs = masks_trim[i, :, 0]
            pred_duration = np.sum(pred_qrs) / self.fs
            gt_duration = np.sum(gt_qrs) / self.fs
            duration_data.append({
                "example": i + 1,
                "GT QRS": gt_duration,
                "Predicted QRS": pred_duration,
                "Diff (s)": pred_duration - gt_duration
            })
        duration_df = pd.DataFrame(duration_data)
       
        print(duration_df)
        with open("duration_info.txt", "w") as f:
            f.write(duration_df.to_string(index=False))
        
     
        signals_trim = signals_all[:, 1000:-1000, :]

        num = min(10, signals_trim.shape[0])
        for i in range(num):
            s = signals_trim[i].squeeze(-1)
            m = masks_trim[i][:, 0]
            p = preds_trim[i][:, 0]
            t = np.arange(len(s)) / self.fs
            plt.figure(figsize=(12, 4))
            plt.plot(t, s, label='ECG', color='blue')
           
            for idx in np.split(np.where(m == 1)[0], np.where(np.diff(np.where(m == 1)[0]) != 1)[0] + 1):
                if idx.size:
                    plt.axvline(x=t[idx[0]], color='green', linestyle='--', label='GT')
                    plt.axvline(x=t[idx[-1]], color='green', linestyle=':', label='GT')
       
            for idx in np.split(np.where(p > self.threshold)[0], np.where(np.diff(np.where(p > self.threshold)[0]) != 1)[0] + 1):
                if idx.size:
                    plt.axvline(x=t[idx[0]], color='red', linestyle='--', label='Pred')
                    plt.axvline(x=t[idx[-1]], color='red', linestyle=':', label='Pred')
            plt.xlabel("Time[s]")
            plt.ylabel("Amplitude")
            plt.title(f"Example {i+1}")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

        self.test_results = []  

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
