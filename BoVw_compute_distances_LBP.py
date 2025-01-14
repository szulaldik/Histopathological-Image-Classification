
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from imageio import imread
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import feature
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_dic = 200

class_map= { 
        "Normal": 0,
        "Polyp": 1,
        "Low-grade IN": 2,
        "High-grade IN": 3,
        "Adenocarcinoma": 4
}


# Load train histograms and labels
train_histograms = []
train_labels = []


# Load histograms and labels for each train image
with open('EBH-HE-IDS/ColHis-IDS/train_paths.txt', 'r') as f:
    paths = [line.strip() for line in f]

for path in paths:
    image_name = os.path.basename(path).split('.')[0]
    save_dir = path.replace('.jpg', '_patches')
    hist_path = "EBH-HE-IDS/"+os.path.join(save_dir, f"{image_name}_histogram_lbp_{n_dic}clusters.npy").replace('\\', '/') 
    histogram = np.load(hist_path)
    train_histograms.append(histogram)

    # Get the class label for the image
    class_name = path.split('\\')[1] 
    class_id =class_map[class_name]  # map class name to class id
    train_labels.append(class_id)

train_histograms = np.array(train_histograms)
train_labels = np.array(train_labels)

# Load test histograms
test_histograms = []
test_labels = []

with open('EBH-HE-IDS/ColHis-IDS/test_paths.txt', 'r') as f:
    test_paths = [line.strip() for line in f]

for path in test_paths:
    image_name = os.path.basename(path).split('.')[0]
    save_dir = path.replace('.jpg', '_patches')
    hist_path = "EBH-HE-IDS/"+os.path.join(save_dir, f"{image_name}_histogram_lbp_{n_dic}clusters.npy").replace('\\', '/') 
    histogram = np.load(hist_path)
    test_histograms.append(histogram)

    # Get the class label for the image
    class_name = path.split('\\')[1]
    class_id = class_map[class_name]
    test_labels.append(class_id)

test_histograms = np.array(test_histograms)
test_labels = np.array(test_labels)

# Parameter for the nearest k_neighbors images 
k_neighbors =  10

predicted_labels = []

for test_hist in test_histograms:
    # Euclidean distance
    distances = cdist([test_hist], train_histograms, metric='euclidean')[0]
    nearest_indices = np.argsort(distances)[:k_neighbors]
    
    # Get the labels of the nearest k_neighbors
    nearest_labels = train_labels[nearest_indices]

    # Majority Vote
    majority_label = np.bincount(nearest_labels).argmax()
    predicted_labels.append(majority_label)

# Confusion Matrix 
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)


# Accuracy 
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))

class_names = list(class_map.keys())


plt.figure(figsize=(10, 8))
ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=True,
                 xticklabels=class_names, yticklabels=class_names, square=True, annot_kws={"size": 12})

plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.title(f'Nearest {k_neighbors} images\n{n_dic} Cluster\nBag of Visual Words Confusion Matrix\nAccuracy:{accuracy:.2f}', fontsize=12)

ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(class_names, rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(f"EBH-HE-IDS/conf_matrix_bovw_lbp_{k_neighbors}neighbors_{n_dic}clusters.png", dpi=400, bbox_inches='tight', pad_inches=0.5)
plt.show()
