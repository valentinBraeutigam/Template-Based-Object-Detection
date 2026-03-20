import numpy as np
import PIL.Image as Image
import os
import concurrent.futures


def image_processing(input):
    image_name, output_folder, bboxes_crops_path,  image_folder_path = input
    
    img = np.asarray(Image.open(os.path.join(image_folder_path, image_name)))[:,:,0:3]
        
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    image_size = [img.width, img.height]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder + image_name[:-4]):
        os.makedirs(output_folder + image_name[:-4])

    print("processing: ", image_name)
    
    # bbox input and output file
    bbox_crop_file = open(bboxes_crops_path + image_name[:-4] + "/bbox_file.txt", "r") 
    bbox_output = open(output_folder  + image_name[:-4] + "/bbox_file.txt","w")

    index = 0
    for line in bbox_crop_file.readlines():
        id = line[:line.find(" ")]

        bbox = line[line.find(" ")+1:].replace("[", "").replace("]","")
        
        bbox = np.asarray([float(x) for x in bbox.split(",")])
        
        # clip bboxes to image size
        bbox[0] = min(image_size[0]-1, max(0,bbox[0]))
        bbox[1] = min(image_size[1]-1, max(0,bbox[1]))
        bbox[2] = min(image_size[0]-1, max(0,bbox[2]))
        bbox[3] = min(image_size[1]-1, max(0,bbox[3]))
        
        # skip images that have a width or height of 0
        if bbox[3] - bbox[1] < 1 or bbox[2] - bbox[0] < 1:
            continue
        
        print(bbox)
        
        crop = img.crop(bbox)
        if not os.path.exists(output_folder + image_name[:-4]):
            os.makedirs(os.path.join(output_folder + image_name[:-4]) + "/")
        
        print("save to " + output_folder + image_name[:-4] + "/" + image_name[:-4] + "_" + str(index) + ".png")
        crop.save(output_folder + image_name[:-4] + "/" + image_name[:-4] + "_" + str(index) + ".png")
        bbox_output.write(str(index) + " " + np.array2string(bbox,separator=", ") + "\n")
        index += 1

def generate_crops(image_folder_path, output_folder, bbox_folder, number_threads=1):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    image_names = [x.name for x in os.scandir(image_folder_path)]
    
    # replace masked areas by uniform color
    image_processing_input = []
    for image_name in image_names:
        image_processing_input.append([image_name, output_folder, bbox_folder, image_folder_path])
        
    
    if number_threads > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_threads) as ex:
            ex.map(image_processing, image_processing_input)
    else:
        for img_data in image_processing_input:
            image_processing(img_data)
    

