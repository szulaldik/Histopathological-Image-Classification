from _future_ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch.nn.functional as F
from PIL import Image
import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class My_dataloader(Dataset):
    def __init__(self, path):
        dir_list = glob.glob(path + "*")
        self.data=[]
        for class_path in dir_list:
            img_class = class_path.split("\\")[-1]
            img_paths = glob.glob(class_path + "\\\\\*.jpg", recursive=True)
            for img_path in img_paths:
                if "patch" in img_path:
                    self.data.append([img_class, img_path])
        
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
        class_name, img_path = self.data[idx]
        image_name = os.path.basename(img_path)
        class_id = self.class_map[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path


def test_model(model, criterion, num_epochs=25):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    for i in range(1):
        slide_patches_dict_1024 = {}
        path = "ColHis-IDS\\"
        test_imagedataset = My_dataloader(path)
        dataloader_test = torch.utils.data.DataLoader(test_imagedataset, batch_size=16,shuffle=False)
        counter = 0
        # Iterate over data.
        for ii, (inputs, img_paths) in enumerate(dataloader_test):
            inputs = inputs.to(device)
            output1, outputs = model(inputs)
            output_1024 = output1.cpu().detach().numpy()
            # output_128 = output2.cpu().detach().numpy()
            for j in range(len(outputs)):
                slide_patches_dict_1024[img_paths[j]] = output_1024[j]
        
        time_elapsed = time.time() - since
        print('Evaluation completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return slide_patches_dict_1024


class fully_connected(nn.Module):
    """docstring for BottleNeck"""
    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self)._init_()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs,num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return  out_1, out_3


if __name__ == '__main__':
    model = torchvision.models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
    num_ftrs = model.classifier.in_features
    model_final = fully_connected(model.features, num_ftrs, 30)
    model = model.to(device)

    model_final = model_final.to(device)
    model_final = nn.DataParallel(model_final)
    params_to_update = []
    criterion = nn.CrossEntropyLoss()

    model_final.load_state_dict(torch.load('./weights/KimiaNetPyTorchWeights.pth'))

    embedding_dict = test_model(model_final, criterion, num_epochs=1)

    for image_path, embedding in embedding_dict.items():
        embedding_path = image_path.replace('.jpg', '_embedding.npy')
        np.save(embedding_path, embedding)


