import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torchvision import transforms
import os
from PIL import Image
import glob
import numpy as np
import random
import joblib
import cv2 as cv
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
from sklearn.ensemble import RandomForestClassifier
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
from glob import glob
from collections import Counter

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)
random.seed(42)


model_name = "MLP"
def orb_features(img_file):
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
 
    # Initiate ORB detector
    orb = cv.ORB_create()
    
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return des

class My_dataloader(Dataset):
    def __init__(self, split_mode='train'):
        if split_mode=='train':
            with open('ColHis-IDS\\mlp_train_paths.txt', 'r') as f:
                paths = [line.strip() for line in f]
        elif split_mode == 'val':
            with open('ColHis-IDS\\mlp_val_paths.txt', 'r') as f:
                paths = [line.strip() for line in f]
        else:
            with open('ColHis-IDS\\test_paths.txt', 'r') as f:
                paths = [line.strip() for line in f]
        self.data=[]
        for path in paths:
            img_path = path.replace('\\', '/')
            img_class = img_path.split('/')[1]  # Example: extract label from filename
            des = orb_features(img_path)
            if des is not None:
                feature = des.mean(axis=0)
                self.data.append([img_class, path, feature])
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
        class_name, img_path, feature = self.data[idx]
        class_id = self.class_map[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Özellik vektörünü float32'ye dönüştür
        feature = torch.tensor(feature, dtype=torch.float32)
        
        return img, img_path, feature, class_id
    
    

def plot_and_save_conf_matrix(y_true, y_pred, model_name, output_dir='ConfMatrix'):
    """
    Plots a confusion matrix, computes accuracy and F1 score, and saves it as a PNG file.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of the data.

    y_pred : array-like of shape (n_samples,)
        Predicted labels by the model.

    labels : list of str
        List of class labels to be displayed on the x and y axes.

    model_name : str
        Name of the model to be included in the file name and plot title.

    output_dir : str, optional, default='ConfMatrix'
        Directory where the confusion matrix image will be saved.
    """

    labels=['Normal', 'Polyp', 'Low-grade IN','High-grade IN','Adenocarcinoma']
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Compute accuracy and F1 score
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                     xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{model_name}\nConfusion Matrix\nAccuracy: {accuracy:.2f}, F1 Score: {f1:.2f}')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    conf_matrix_path = os.path.join(output_dir, f"{model_name}_orb_rawdata_conf_matrix.png")
    fig = ax.get_figure()
    fig.savefig(conf_matrix_path, bbox_inches='tight')
    plt.show()

    print(f"Confusion matrix saved to {conf_matrix_path}")

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
    plt.title(f'MLP\nORB\nConfusion Matrix\nAccuracy: {accuracy:.2f}')  # Başlıkta doğruluk oranını göster
    plt.show()

    conf_matris_path= f"ConfMatrix\\mlp_orb.png"
    fig = ax.get_figure()
    fig.savefig(conf_matris_path)  # PNG formatında kaydet


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32, 64)  
        self.fc2 = nn.Linear(64, 128)  # Gizli katman 1 (512 gizli, 1024 gizli)
        self.fc3 = nn.Linear(128, 64)  # Gizli katman 2 (1024 gizli, 512 gizli)
        self.fc4 = nn.Linear(64, 5)     # Çıkış katmanı (512 gizli, 5 çıkış)
        self.relu = nn.ReLU()             # ReLU aktivasyon fonksiyonu
        self.softmax = nn.Softmax(dim=1)  # Softmax aktivasyon fonksiyonu

    def forward(self, x):
        x = self.relu(self.fc1(x))  # İlk gizli katman ve ReLU aktivasyonu
        x = self.relu(self.fc2(x))  # İkinci gizli katman ve ReLU aktivasyonu
        x = self.relu(self.fc3(x))  # Üçüncü gizli katman ve ReLU aktivasyonu
        x = self.fc4(x)             # Çıkış katmanı
        x = self.softmax(x)         # Softmax aktivasyon fonksiyonu ile çıkış
        return x
    



if __name__ == '__main__':
    start_time = time.time()  # Başlangıç zamanı
    print("Program started")
    
    # Örnek etiketler ve embeddingler
    train_dataset = My_dataloader(split_mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("train_dataloader is created")
    
    test_dataset = My_dataloader(split_mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("test_dataloader is created")
    
    val_dataset = My_dataloader(split_mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    
    # model = MLP().to(device)

    # # Kayıp fonksiyonu (loss function)
    # criterion = nn.CrossEntropyLoss()

    # # Optimizer 
    # optimizer = optim.SGD(model.parameters(), lr=0.01)


    # best_loss = float('inf')  # En iyi kayıp başlangıçta sonsuz

    # # Eğitim döngüsü
    # epochs = 200
    # validation_interval=20
    # for epoch in range(epochs):
    #     model.train()  # Modeli eğitim moduna al
    #     running_loss = 0.0
    #     for x,y, inputs, labels in train_dataloader:  # Eğitim veri kümesi üzerinde döngü
    #         inputs, labels = inputs.to(device), labels.to(device)  # Girdi tensörlerini cihaza taşı

    #         optimizer.zero_grad()  # Gradyanları sıfırla
            
    #         outputs = model(inputs)  # İleri geçiş (forward pass)
    #         loss = criterion(outputs, labels)  # Kaybı hesapla
            
    #         loss.backward()  # Geriye doğru geçiş (backward pass)
    #         optimizer.step()  # Optimizer ile parametreleri güncelle
            
    #         running_loss += loss.item()

    #     # Her 20 epoch'ta bir validasyon yap
    #     if (epoch + 1) % validation_interval == 0:
    #         model.eval()  # Modeli değerlendirme moduna al
    #         validation_loss = 0.0
    #         with torch.no_grad():  # Gradyan hesaplamasını devre dışı bırak
    #             for x, y, inputs, labels in val_dataloader:
    #                 inputs, labels = inputs.to(device), labels.to(device)

    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 validation_loss += loss.item()

    #         validation_loss /= len(val_dataloader)
    #         print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader):.4f}, Validation Loss: {validation_loss:.4f}")

    #         # En iyi modeli kaydet
    #         if validation_loss < best_loss:
    #             best_loss = validation_loss
    #             model_file_name=f"best_model_orb.pth"
    #             torch.save(model.state_dict(), model_file_name)
    #             print("Best model saved with validation loss: {:.4f}".format(validation_loss))
    #     else:
    #         print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader):.4f}")
    
    # #Epoch 180, Training Loss: 1.0566, Validation Loss: 1.2066
    # #Best model saved with validation loss: 1.2066
    model_file_name = "C:\\Users\\FSM\\Desktop\\SUMEYYE-HATICE\\EBH-HE-IDS\\best_model_orb.pth"
    # En iyi modeli yükle
    best_model = MLP().to(device)
    best_model.load_state_dict(torch.load(model_file_name))
    best_model.to(device)
    best_model.eval()  # Modeli değerlendirme moduna al



    # Tahminler ve gerçek etiketler için boş listeler
    all_predictions = {}
    all_labels = {}

    y_pred=[]
    y_true =[]

    with torch.no_grad():
        # img, img_path, average_embedding, class_id
        for imgs, img_paths, embeddings, class_ids in test_dataloader:
            embeddings, class_ids = embeddings.to(device), class_ids.to(device)
            outputs = best_model(embeddings)  # Modelden tahminler al
            
            # En yüksek olasılığa sahip sınıfı seç
            _, predicted = torch.max(outputs.data, 1)

            for idx in range(len(predicted)):

                key = img_paths[idx]

                if key not in all_predictions.keys():
                    all_predictions[key]=[]

                all_predictions[key].append(predicted[idx])

                if key not in all_labels.keys():
                    all_labels[key] = class_ids[idx] 
                    
                y_pred.append(predicted[idx])
                y_true.append(class_ids[idx])

            
    

    compare(all_labels, all_predictions)

    plot_and_save_conf_matrix(y_true, y_pred, model_name, output_dir='ConfMatrix')
    
    
    
