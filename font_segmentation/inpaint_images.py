import os
import shutil

def inpaintImages(images_path, masks_path, output_path):
    image_names = [x.name for x in os.scandir(images_path)]
    # replace masked areas by inpainting

    for image_name in image_names:
        assert os.path.exists(masks_path + image_name), f"file not found, image name: {image_name} at location {masks_path}"
        
        if not os.path.exists(output_path + "masks/" + image_name[:-4]):
            os.makedirs(output_path + "masks/" + image_name[:-4])
        
        if not os.path.exists(output_path + "resulting_images/"):
            os.makedirs(output_path  + "resulting_images/")
        
        assert os.path.exists("InpaintAnything/remove_anything.py"), "InpaintAnything seems to be missing or to not be at the expected place. Please set it up according to the Readme file"
        
        os.system(f"python3 InpaintAnything/remove_anything.py --input_img {images_path + image_name} --coords_type ignore --mask {masks_path + image_name} --point_labels 1 --dilate_kernel_size 4 --output_dir {output_path}masks/ --sam_model_type \"vit_h\" --sam_ckpt sam_vit_h_4b8939.pth --lama_config ./InpaintAnything/lama/configs/prediction/default.yaml --lama_ckpt ./InpaintAnything/pretrained_models/big-lama")
    
    for image_name in image_names:
        shutil.move(output_path +"masks/" + image_name[:-4] + "/inpainted_with_mask_0.png", output_path + "resulting_images/" + image_name)