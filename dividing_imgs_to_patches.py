from PIL import Image
import os
import glob


def image_to_tiles(image_path, tile_size, patch_folder):
    
    patches_name = os.path.basename(image_path).split('.')[0] + "patch"
    
    image = Image.open(image_path)
    
    width, height = image.size
    
    new_width = width + (tile_size - (width % tile_size)) % tile_size
    new_height = height + (tile_size - (height % tile_size)) % tile_size
    
    padded_image = Image.new("RGB", (new_width, new_height))
    padded_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    
    num_tiles_x = new_width // tile_size
    num_tiles_y = new_height // tile_size
    
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = x * tile_size
            upper = y * tile_size
            right = left + tile_size
            lower = upper + tile_size
            
            tile = padded_image.crop((left, upper, right, lower))
            
            tile_name = patches_name + str(x) + "_" + str(y) + ".jpg"
            print(tile_name)
            
            tile.save(os.path.join(patch_folder, tile_name))



if __name__ == '__main__':
    path = "ColHis-IDS\\"
    dir_list = glob.glob(path + "*")
    data=[]
    #print(dir_list)
    for class_path in dir_list:
        img_paths = glob.glob(class_path + "\\\\\\*.jpg", recursive=True)
        for img_path in img_paths:
            # folder name
            img_name = os.path.basename(img_path)
            # get patch path from folder name
            patch_path = os.path.splitext(img_path)[0]
            magnitude_dir = patch_path.split('\\')[-2]
            if magnitude_dir == '200':        
                #print("img_name",img_name)          
                # name of directory
                patch_folder = os.path.splitext(img_path)[0]+"_patches"
                print("folder_name",patch_folder)
            
                # create the directory save all patches into that
                os.makedirs(patch_folder, exist_ok=True)
                image_to_tiles(img_path, 128, patch_folder)