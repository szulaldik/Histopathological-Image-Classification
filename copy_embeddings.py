import glob
import os
import shutil


path = "new_data\\ColHis-IDS\\"
dir_list = glob.glob(path + "*")
data=[]

for class_path in dir_list:
    img_paths = glob.glob(class_path + "\\*\\*\\*.jpg", recursive=True)
    for img_path in img_paths:
        # Dosya adını al
        img_name = os.path.basename(img_path)
        # Dosya adından uzantıyı kaldır
        patch_path = os.path.splitext(img_path)[0]
        magnitude_dir = patch_path.split('\\')[-2]
        if magnitude_dir == '200':        
            #print("img_name",img_name)          
            # Yama klasörü
            patch_folder = os.path.splitext(img_path)[0]+"_patches"
            not_discarded_patches = glob.glob(patch_folder+ '\\*')
            for patch in not_discarded_patches:
                source_path = patch.replace("new_data\\","")
                source_embedding_path = source_path.replace(".jpg","_embedding.npy")
                destination_embedding_path = patch.replace(".jpg","_embedding.npy")
                shutil.copyfile(source_embedding_path, destination_embedding_path)       
            
