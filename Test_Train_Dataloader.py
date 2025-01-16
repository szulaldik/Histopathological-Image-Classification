import torch
import numpy as np
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class My_dataloader(Dataset):
    def __init__(self, train=True):
        if train:
            with open('ColHis-IDS\\train_paths.txt', 'r') as f:
                paths = [line.strip() for line in f]
        else:
            with open('ColHis-IDS\\test_paths.txt', 'r') as f:
                paths = [line.strip() for line in f]
        self.data=[] 
        for path in paths:
            if train is True:
                selected_patches_paths = glob.glob(path.replace('.jpg','_selectedpatches')+ '\\*.jpg')
            else:
                selected_patches_paths = glob.glob(path.replace('.jpg','_patches')+ '\\*.jpg')
            for selected_patch in selected_patches_paths:
                    img_class = selected_patch.split('\\')[1]           
                    embedding_path = selected_patch.replace('.jpg', '_embedding.npy')
                    self.data.append([img_class, selected_patch, embedding_path])     
        
        self.class_map= {
        "Adenocarcinoma": 4,
        "High-grade IN": 3,
        "Low-grade IN": 2,
        "Polyp": 1,
        "Normal": 0
        }
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    
    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx): 
        class_name, img_path, embedding_path = self.data[idx]
        image_name = os.path.basename(img_path).split('_patch')[0]
        class_id = self.class_map[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        embedding = np.load(embedding_path)
        return img, img_path, image_name, embedding, class_id
    


    

if __name__ == '__main__':

    train_dataset = My_dataloader(True)
    print('Len:',train_dataset.__len__())

    img, image_name, img_path,  embedding, class_id = train_dataset.__getitem__(0)

    print('image_name:', image_name )
    print('img_path:', img_path )
    print('class_id:', class_id )
    print('embedding:', embedding )
    print('img:', img )


    test_dataset = My_dataloader(False)
    print('Len:',test_dataset.__len__())

    img, image_name, img_path,  embedding, class_id = test_dataset.__getitem__(0)

    print('image_name:', image_name )
    print('img_path:', img_path )
    print('class_id:', class_id )
    print('embedding:', embedding )
    print('img:', img )