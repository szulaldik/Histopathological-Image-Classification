from PIL import Image
import os
import glob
import numpy as np
import cv2


def calculate_patch_statistics(patch):
    mean_intensity = np.mean(patch)
    std_intensity = np.std(patch)
    return mean_intensity, std_intensity   
				

def image_to_tiles(image_path, tile_size, patch_folder):
    
    #Resmin ismi
    patches_name = os.path.basename(image_path).split('.')[0] + "_patch_"
    
    # Resmi yükle
    image = Image.open(image_path)
    
    # Resmin genişliği ve yüksekliği
    width, height = image.size
    
    # Parçalama için gerekli tam boyutları hesapla
    new_width = width + (tile_size - (width % tile_size)) % tile_size
    new_height = height + (tile_size - (height % tile_size)) % tile_size
    
    # Resmi yeniden boyutlandır ve padding ekle
    padded_image = Image.new("RGB", (new_width, new_height))
    padded_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    
    # Parça sayısı hesapla
    num_tiles_x = new_width // tile_size
    num_tiles_y = new_height // tile_size
    
    # Resmi parçalara ayır
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = x * tile_size
            upper = y * tile_size
            right = left + tile_size
            lower = upper + tile_size
            
            # Parçayı al
            tile = padded_image.crop((left, upper, right, lower))
            
            # Parça adını oluştur
            tile_name = patches_name + str(x) + "_" + str(y) + ".jpg"
            #print(tile_name)
            
            # Parçayı kaydet
            patch_path = os.path.join(patch_folder, tile_name)
            tile.save(patch_path)

            im = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
            mean_intensity, std_intensity = calculate_patch_statistics(im)

            patch_var = cv2.Laplacian(im, cv2.CV_64F).var()
            # eğer mean_int > 165 veya path_var > 700 ve std<42 ise görüntüyü discard et...
            if ((mean_intensity>160 or patch_var>650) & (std_intensity < 41)):
                discarded_folder = patch_folder.replace("patches", "discardedPatches")
                os.makedirs(discarded_folder, exist_ok=True)
                os.rename(patch_path, os.path.join(discarded_folder, os.path.basename(patch_path)))
            

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
            # Klasör adı oluştur
            patch_folder = os.path.splitext(img_path)[0]+"_patches"
            #print("folder_name",patch_folder)
        
            # Klasörü oluştur
            os.makedirs(patch_folder, exist_ok=True)
            image_to_tiles(img_path, 128, patch_folder)
        
        