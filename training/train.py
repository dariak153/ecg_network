import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import ECGDataset
from training.lightning_module import LightningModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import yaml

def main(config):
  
    train_set = ECGDataset(config["data"]["train_signals"], config["data"]["train_masks"])
    val_set = ECGDataset(config["data"]["val_signals"], config["data"]["val_masks"])
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    

    model = LightningModel(lr=config["learning_rate"], threshold=config["threshold"],
                           fs=config["fs"], output_length=config["output_length"])
    
    
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_model")
    
 
    trainer = pl.Trainer(max_epochs=config["max_epochs"],
                         accelerator="cuda" if torch.cuda.is_available() else "cpu",
                         devices=1,
                         callbacks=[early_stop, checkpoint])
    

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)
