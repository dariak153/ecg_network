import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import ECGDataset
from training.lightning_module import LightningModel
import yaml

def main(config):
  
    test_set = ECGDataset(config["data"]["test_signals"], config["data"]["test_masks"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
 
    model = LightningModel(lr=config["learning_rate"], threshold=config["threshold"],
                           fs=config["fs"], output_length=config["output_length"])
    
   
    # checkpoint_path = "best_model.ckpt"
    # model = LightningModel.load_from_checkpoint(checkpoint_path)
    
    trainer = pl.Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)
