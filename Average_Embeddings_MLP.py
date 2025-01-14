import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import numpy as np
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


random.seed(42)

class My_dataloader(Dataset):
    def __init__(self, split_mode='train'):
        if split_mode=='train':
            with open('ColHis-IDS\\mlp_train_paths.txt', 'r') as f:
                paths = ['new_data\\'+line.strip() for line in f]
        elif split_mode == 'val':
            with open('ColHis-IDS\\mlp_val_paths.txt', 'r') as f:
                paths = ['new_data\\'+line.strip() for line in f]
        else:
            with open('ColHis-IDS\\test_paths.txt', 'r') as f:
                paths = ['new_data\\'+line.strip() for line in f]
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
    
    
    
def compare(img_labels,patch_predictions):
    global selected_patch_num
    img_prediction={}

    for img_path, prediction_list in patch_predictions.items():
        img_prediction[img_path]=Counter(prediction_list).most_common(1)[0][0]
        
    true_values= 0
    false_values= 0
    conf_matrix = np.zeros((5, 5))
    for img, pred in img_prediction.items():
        class_id = img_labels[img]
        conf_matrix[class_id, pred] += 1
        if class_id == pred:
            true_values+=1
        else:
            false_values+=1

    print(true_values/(true_values+false_values))
    print("true_values",true_values)
    print("false_values",false_values)

    conf_matrix = conf_matrix.astype(int)

    accuracy = true_values / (true_values + false_values)
    print("Accuracy: ", accuracy)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                    xticklabels=['Normal', 'Polyp', 'Low-grade IN','High-grade IN','Adenocarcinoma'], 
                    yticklabels=['Normal', 'Polyp', 'Low-grade IN','High-grade IN','Adenocarcinoma'])
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.ylabel('Gerçek Etiketler')
    plt.title(f'MLP\nTraining with average embeddings on preprocessed data\nConfusion Matrix\nAccuracy: {accuracy:.2f}')  
    plt.show()

    conf_matris_path= f"ConfMatrix\\mlp_avrg_embddngs_discarding.png"
    fig = ax.get_figure()
    fig.savefig(conf_matris_path)  

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1024, 512)  # Input layer (1024 input, 512 hidden)
        self.fc2 = nn.Linear(512, 1024)  # Hidden layer 1 (512 hidden, 1024 hidden)
        self.fc3 = nn.Linear(1024, 512)  # Hidden layer 2 (1024 hidden, 512 hidden)
        self.fc4 = nn.Linear(512, 5)     # Output layer (512 hidden, 5 output)
        self.relu = nn.ReLU()             # ReLU activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # First hidden layer ve ReLU activation
        x = self.relu(self.fc2(x))  # Second hidden layer ve ReLU activation
        x = self.relu(self.fc3(x))  # Third hidden layer ve ReLU activation
        x = self.fc4(x)             # Output layer
        x = self.softmax(x)         # Softmax activation function 
        return x


if __name__ == '__main__':
    start_time = time.time() 
    print("Program started")

    train_dataset = My_dataloader(split_mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("train_dataloader is created")
    
    test_dataset = My_dataloader(split_mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    val_dataset = My_dataloader(split_mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = MLP().to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    best_loss = float('inf')  # best loss initialization with infinity 

    # training
    epochs = 200
    validation_interval=20
    for epoch in range(epochs):
        model.train()  # train mode
        running_loss = 0.0
        for x,y, inputs, labels in train_dataloader:  # Eğitim veri kümesi üzerinde döngü
            inputs, labels = inputs.to(device), labels.to(device)  # Girdi tensörlerini cihaza taşı

            optimizer.zero_grad()  
            
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # compute the loss
            
            loss.backward()  # backward pass
            optimizer.step()  # update the prameters with optimizer
            
            running_loss += loss.item()

        # validate every 20 epoch
        if (epoch + 1) % validation_interval == 0:
            model.eval()  # evaluation mode
            validation_loss = 0.0
            with torch.no_grad():  
                for x, y, inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

            validation_loss /= len(val_dataloader)
            print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader):.4f}, Validation Loss: {validation_loss:.4f}")

            # save the best model
            if validation_loss < best_loss:
                best_loss = validation_loss
                model_file_name=f"best_model_trained_with_average_embdngs_with_preprocessed_data.pth"
                torch.save(model.state_dict(), model_file_name)
                print("Best model saved with validation loss: {:.4f}".format(validation_loss))
        else:
            print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader):.4f}")
    
   
    # load the best model
    best_model = MLP().to(device)
    best_model.load_state_dict(torch.load(model_file_name))
    best_model.to(device)
    best_model.eval()  # evaluation mode



    # empty dicts for predictions and labels
    all_predictions = {}
    all_labels = {}

    with torch.no_grad():
        # img, img_path, average_embedding, class_id
        for imgs, img_paths, embeddings, class_ids in test_dataloader:
            embeddings, class_ids = embeddings.to(device), class_ids.to(device)
            outputs = best_model(embeddings)  # Modelden tahminler al
            
            # choose the max prob of class
            _, predicted = torch.max(outputs.data, 1)

            for idx in range(len(predicted)):
                key = os.path.dirname(img_paths[idx])

                if key not in all_predictions.keys():
                    all_predictions[key]=[]

                all_predictions[key].append(predicted[idx])

                if key not in all_labels.keys():
                    all_labels[key] = class_ids[idx] 
            
    # save the test result
    compare(all_labels, all_predictions)


  

