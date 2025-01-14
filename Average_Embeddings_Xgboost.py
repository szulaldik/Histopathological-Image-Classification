import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import time
import pickle
from torchvision import datasets, models, transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from Test_Train_Dataloader import My_dataloader
from collections import Counter
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


random.seed(42)

class My_dataloader(Dataset):
    def __init__(self, train=True):
        if train:
            with open('ColHis-IDS\\train_paths.txt', 'r') as f:
                paths = ['new_data\\' + line.strip() for line in f]
        else:
            with open('ColHis-IDS\\test_paths.txt', 'r') as f:
                paths = ['new_data\\' +line.strip() for line in f]
        self.data=[] 
        for path in paths:
            patches_paths = glob(path.replace('.jpg','_patches')+ '\\*.jpg')
            embeddings=[]
            for patch in patches_paths:
                embedding_path = patch.replace('.jpg', '_embedding.npy')
                embeddings.append(np.load(embedding_path))

            img_class = patch.split('\\')[2]         
            average_embeddings = np.mean(embeddings, axis=0)
            self.data.append([img_class, path, average_embeddings])     
        
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
        class_name, img_path, average_embedding = self.data[idx]
        # image_name = os.path.basename(img_path).split('_patch')[0]
        class_id = self.class_map[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path, average_embedding, class_id
    
    
    
def predict(flattened_img_path_list,ypred):
    # find indexes of all elements
    indexes = {img: [idx for idx, img_path in enumerate(flattened_img_path_list) if img_path == img] for img in set(flattened_img_path_list)}

    #print(indexes)
    

    img_prediction={}

    for img, idx_list in indexes.items():
        patch_predictions = [ypred[i] for i in idx_list]
        img_prediction[img]=Counter(patch_predictions).most_common(1)[0][0]
        
    class_map={
                "Adenocarcinoma": 4, #158
                "High-grade IN": 3, #26
                "Low-grade IN": 2, #121
                "Polyp": 1, #51
                "Normal": 0 #13
            }  
    keys = list(img_prediction.keys())
    keys[0].split('_')
    true_values= 0
    false_values= 0
    conf_matrix = np.zeros((5, 5))
    for img, pred in img_prediction.items():
        class_name = img.split('\\')[2]
        class_id = class_map[class_name]
        conf_matrix[class_id, pred] += 1
        if class_id == pred:
            true_values+=1
        else:
            false_values+=1

    accuracy = true_values/(true_values+false_values)

    print("Accuracy: ", accuracy)
    print("true_values",true_values)
    print("false_values",false_values)

    conf_matrix = conf_matrix.astype(int)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                    xticklabels=['Normal', 'Polyp', 'Low-grade IN','High-grade IN','Adenocarcinoma'], 
                    yticklabels=['Normal', 'Polyp', 'Low-grade IN','High-grade IN','Adenocarcinoma'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'XGboost\nTraining with average embeddings on preprocessed data\nConfusion Matrix\nAccuracy: {accuracy:.2f}')  # Başlıkta doğruluk oranını göster
    plt.show()

    conf_matris_path= f"ConfMatrix\\xgb_avrg_embddngs_discarding.png"
    fig = ax.get_figure()
    fig.savefig(conf_matris_path)  





if __name__ == '__main__':
    start_time = time.time() 
    print("Program started")

    train_dataset = My_dataloader(True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("train_dataloader is created")

    train_embeds_flat = []
    train_labels_flat  = []
    train_img_paths = []

    for imgs, img_paths, embeddings, class_ids in train_dataloader:
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        train_embeds_flat.append(embeddings_flat.numpy())
        train_img_paths.append(img_paths)

        # flat labels
        labels_flat = class_ids.numpy().flatten()
        train_labels_flat.append(labels_flat)
        

    # convert embeddings and labels to numpy arrays 
    train_embeds_flat = np.concatenate(train_embeds_flat, axis=0)
    train_labels_flat = np.concatenate(train_labels_flat, axis=0)


    end_time = time.time()  
    print("Execution time of creating train embeds and labels:", end_time - start_time, "seconds")  # Çalışma süresi

    print("Train embeddings shape:", train_embeds_flat.shape)
    print("Train labels shape:", train_labels_flat.shape)

    train_paths = [list(i) for i in train_img_paths]
    flattened_img_path_list = sum(train_paths, [])
    

    start_time = time.time()  

    test_dataset = My_dataloader(False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    test_embeds_flat = []
    test_labels_flat  = []
    test_img_paths = []

    for imgs, img_paths, embeddings, class_ids in test_dataloader:
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        test_embeds_flat.append(embeddings_flat.numpy())
        test_img_paths.append(img_paths)

        # flat labels
        labels_flat = class_ids.numpy().flatten()
        test_labels_flat.append(labels_flat)
        

    # convert embeddings and labels to numpy arrays 
    test_embeds_flat = np.concatenate(test_embeds_flat, axis=0)
    test_labels_flat = np.concatenate(test_labels_flat, axis=0)


    end_time = time.time()  
    print("Execution time of creating test embeds and labels:", end_time - start_time, "seconds")  # Çalışma süresi

    print("Test embeddings shape:", test_embeds_flat.shape)
    print("Test labels shape:", test_labels_flat.shape)

    test_paths = [list(i) for i in test_img_paths]
    flattened_img_path_list = sum(test_paths, [])
    

    start_time = time.time() 

    # create the model
    xgb_model = xgb.XGBClassifier()
    
    # train the model
    xgb_model.fit(train_embeds_flat, train_labels_flat)

    # test the model
    ypred =xgb_model.predict(test_embeds_flat)

    end_time = time.time() 
    print("Execution time of prediction:", end_time - start_time, "seconds")  # Çalışma süresi

    # save the test result
    predict(flattened_img_path_list, ypred)