import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from data import WikiartDataset
from augmentation import augment_images

from efficientnet_pytorch import EfficientNet

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.metrics import TensorMetric
from pytorch_lightning.loggers.neptune import NeptuneLogger

from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Accuracy

import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "efficientnet"

train_dir = os.path.join("..", "..", "Data", "train2.csv")
test_dir = os.path.join("..", "..", "Data", "test2.csv")
img_path = os.path.join("..", "..", "Imgs")
root_dir = os.path.join("..", "..", "Data", "wikiart", "images")
labels_dir = os.path.join("..", "..", "Data", "labels.csv")
accuracy_path = os.path.join(img_path, str(MODEL_NAME) + "_accuracy.png")
loss_path = os.path.join(img_path, str(MODEL_NAME) + "_loss.png")


# Parsing hyperparameters
parser = argparse.ArgumentParser(description='WikiArt Inception Neural Network to identify styles of images.')
parser.add_argument('--batch_size', default=128, help='provide an integer batch size (8-256)')
parser.add_argument('--epochs', default=20, help='provide an integer epoch number (1-100)')
hyperparams = parser.parse_args()

labels = pd.read_csv(labels_dir)

EPOCHS = int(hyperparams.epochs)
NUM_CLASSES = len(labels)
BATCH_SIZE = int(hyperparams.batch_size)
LR = 0.001


class LitTransferLearning(pl.LightningModule):
    def __init__(self):
        super(LitTransferLearning, self).__init__()   
        self.model = EfficientNet.from_pretrained("efficientnet-b7")
        self.model._fc = nn.Linear(self.model._fc.in_features, NUM_CLASSES)
        
        self.metric = Accuracy(num_classes=NUM_CLASSES)
        #self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)
    

    def training_step(self, batch, batch_idx):
        images, labels = batch
        print(f"im size: {images.size()}")
        augmented_images = augment_images(images)
        print(f"aug im size: {augmented_images.size()}")
        labels_predicted = self(augmented_images)
        loss = F.cross_entropy(labels_predicted, labels)
        
        avg_acc = self.metric(labels_predicted, labels)
        
        tensorboard_logs = {'train_loss': loss, 'train_acc': avg_acc}  
        return {'loss': loss, 'log': tensorboard_logs}
    
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels_predicted = self(images)
        loss = F.cross_entropy(labels_predicted, labels)   
        
        avg_val_acc = self.metric(labels_predicted, labels)
        tensorboard_logs = {'val_loss': loss, 'val_acc': avg_val_acc}  
        return {'val_loss': loss, 'log': tensorboard_logs}
    
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_val_loss}    
        return {'val_loss': avg_val_loss, 'log': tensorboard_logs}
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(WikiartDataset(csv_file=train_dir, root_dir=root_dir, mode='train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(WikiartDataset(csv_file=test_dir, root_dir=root_dir, mode='val'), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
       
        
if __name__ == "__main__":
    CHECKPOINTS_DIR = '../../Models/checkpoints'

    
    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        experiment_name="default",  
        params={"max_epochs": EPOCHS, "batch_size": BATCH_SIZE},  
        tags=["pytorch-lightning", "mlp"],
        close_after_fit=False)
    
    
    seed_everything(42)
    device = 'gpu'
    
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=2, verbose=True, mode='auto')
    
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR, save_top_k=1,
                                                    verbose=True,
                                                    monitor='val_loss',
                                                    mode='min',
                                                    prefix='')

    trainer = Trainer(auto_scale_batch_size=False,
                      auto_lr_find=False, 
                      max_epochs=EPOCHS,
                      min_epochs=1,
                      fast_dev_run=False,
                      early_stop_callback=early_stop_callback,
                      checkpoint_callback=model_checkpoint,
                      logger=neptune_logger,
                      profiler=True)
    
    
    model = LitTransferLearning()
    
    print('Model loaded on {} - Start training...'.format(device))
    trainer.fit(model)
    trainer.save_checkpoint("model.ckpt")
    print("Model saved and training finished.")
    model.to_onnx(CHECKPOINTS_DIR, export_params=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    