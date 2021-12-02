from torch.utils.data import Dataset
import torch as t
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import skimage
from skimage import io, transform, color
from torchvision import transforms, utils
import os
from PIL import Image
import PIL
import pandas as pd
        
        
class WikiartDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode="train"):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode
        
        
        self._transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            }
            
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations['path'].iloc[index])
        
        image = Image.open(img_path).convert('RGB')            
        image = np.asarray(image).astype(np.uint8)
        #image = skimage.color.gray2rgb(image)

        y_label = int(self.annotations['style'].iloc[index])
        
        
        if self.mode == 'train':
            image = self._transform['train'](image)
        else:
            image = self._transform['val'](image)
        
            
        return (image, y_label)
    
    def __len__(self):
        return len(self.annotations)

   