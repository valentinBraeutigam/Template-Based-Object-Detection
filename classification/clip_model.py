import torch
import clip
import torch
from PIL import Image
import os
import numpy as np
from sklearn import preprocessing

def normalize_vectors_min_max(input_vectors, min, max):
    """
        Normalizes the input vectors by min and max values given for each position (which should be computed from the template data).   
    
        Args:
            input_vectors (np.array(float)) | list(np.array(float)) | dict(str : np.array(float))): input data
            min (np.array(float)): minimal values found in the template data
            max (np.array(float)): maximum values found in the template data
    """
    range_vector = max - min
    range_vector[range_vector == 0.0] = 1.0
    if type(input_vectors) == dict:
        for key in input_vectors.keys():
            input_vectors[key] = (input_vectors[key] - min) / range_vector
    elif type(input_vectors) == np.array:
        input_vectors = (input_vectors - min) / range_vector
    elif type(input_vectors) == list:
        for i in len(input_vectors):
            input_vectors[i] = (input_vectors[i] - min) / range_vector
    
    return input_vectors

def normalize_vectors_l2(input_vectors):
    """
        Applies L2 norm to input vectors.
    """
    return preprocessing.normalize(input_vectors, "l2")

def parse_feature_vector_file(file, feature_length):
    """
        Open file with computed features and computes the min and max values for each value position. 

        Args: 
            file(file): text file containing feature vectors. 
            feature_length: the length of a feature vector.
    """
    min = np.zeros((feature_length), dtype=float)
    max = np.zeros((feature_length), dtype=float)
    feature_vectors = []
    labels = []
    for line in file.readlines():
        line = line.replace("[","").replace("]","").split(",")
        idx = 0
        labels.append(line[0])
        for val in line[1:]:
            val = float(val)
            if min[idx] > val:
                min[idx] = val
            if max[idx] < val:
                max[idx] = val
            idx += 1
        feature_vectors.append(np.array(line[1:], dtype=float))
    return np.array(feature_vectors), min, max, labels

def load_input_vectors(file):
    """
        Converts the input vectors from the given file. 
        
        Args:
            file (file): file containing the input vectors. 

        Returns:
            np.array : 
            list : the labels/file paths to each vector
    """
    input_vectors = []
    labels = []
    for line in file.readlines():
        print(line)
        line = line.replace("[","").replace("]","")
        line = line.split(",")
        labels.append(line[0])
        input_vectors.append(np.array(line[1:], dtype=float))
    return np.array(input_vectors), labels

def compute_clip_features_to_file(crop, output_file, preprocess, model, device):
    """
        Computes the CLIP features and writes them to the output_file.

        Args:
            crop (str): path to crop file
            output_file (file): file where to write the feature vectors
    """
    image_crop = Image.open(crop)
    image = preprocess(image_crop).unsqueeze(0).to(device)
    image_features = model.encode_image(image).cpu().reshape((-1))
    image_features = np.array2string(image_features.numpy(), separator=", ").replace("\n", "")
    output_file.write(f"{crop}, {image_features} \n")
    return image_features
    
def compute_clip_features(image_crop, preprocess, model, device):
    """
        Computes the CLIP features and returns them.

        Args:
            crop (str): path to crop file
        
        Return value:
            list : the CLIP features computed from the image
    """
    image = preprocess(image_crop).unsqueeze(0).to(device)
    image_features = model.encode_image(image).cpu().reshape((-1))
    return image_features

def convert_bbox(bbox_string, image_size, unconverted_gt_bboxes):
    """_summary_

    Args:
        bbox_string (string): bbox in format [x_0, y_0, x_1, y_1]
        image_size (array): image size [width, height]
        unconverted_gt_bboxes (bool): whether to scale bbox from normalized xywh format to xyxy format or not

    Returns:
        list(float): bbox in xyxy format
    """
    
    bbox = bbox_string[bbox_string.find(" ")+1:]
        
    bbox = np.asarray([float(x) for x in bbox.split(" ")])
    if unconverted_gt_bboxes:
        bbox[2] *= image_size[0]
        bbox[3] *= image_size[1]
        bbox[0] = bbox[0] * image_size[0] - bbox[2]/2
        bbox[1] = bbox[1] * image_size[1] - bbox[3]/2
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        
        bbox[0] = min(1799, max(0,bbox[0]))
        bbox[1] = min(696, max(0,bbox[1]))
        bbox[2] = min(1799, max(0,bbox[2]))
        bbox[3] = min(696, max(0,bbox[3]))
    if bbox[3] - bbox[1] < 1 or bbox[2] - bbox[0] < 1:
        return None
    
    return bbox


def load_CLIP_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    return [device, model, preprocess]

def precompute_templates_CLIP(template_crop_folder_path, model_information, output_file_template_path=None):
    preprocess, model, device = model_information
    crop_paths_template = []
            
    for crop in list(os.scandir(template_crop_folder_path)):
        crop_paths_template.append(template_crop_folder_path + crop.name)
    if not output_file_template_path is None:
        output_file_template = open(output_file_template_path,"w")
    
    templates = {}
    
    with torch.no_grad():
        for i in range(len(crop_paths_template)):
            template_name = crop_paths_template[i][crop_paths_template[i].rfind("/"):]
            if not output_file_template_path is None:
                features = compute_clip_features_to_file(crop_paths_template[i], output_file_template, preprocess, model, device)
                templates[template_name] = features
            else:
                features = compute_clip_features(crop_paths_template[i], preprocess, model, device)
                templates[template_name] = features
                
    # compute min and max value for each feature position
    min, max = compute_min_max_normalization_values(templates, 512)
    
    # scale templates
    templates = normalize_vectors_min_max(templates, min, max)
           
    return templates, [min, max]

def compute_min_max_normalization_values(templates, feature_length):
    min = np.zeros((feature_length), dtype=float)
    max = np.zeros((feature_length), dtype=float)
    
    for template_name in templates.keys():
        idx = 0
        for val in templates[template_name]:
            val = float(val)
            if min[idx] > val:
                min[idx] = val
            if max[idx] < val:
                max[idx] = val
            idx += 1
    return min, max

    

def compute_all_clip_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    if not feature_vectors_only_test:  
        # get fitting data for normalization:
        # if template only one template images is used, else from a set of crops an amount of templates per category is selected
        if template:
            crop_paths_template = []
            
            for crop in list(os.scandir(template_crop_folder_path)):
                crop_paths_template.append(template_crop_folder_path + crop.name)
            output_file_template = open(output_file_template_path,"w")
        else:
            crop_paths_template = []
            for crop_id_folder in list(os.scandir(template_crop_folder_path)):
                counter = 0
                for crop in list(os.scandir(template_crop_folder_path + crop_id_folder.name)):
                    if count_of_crops > 0 and counter >= count_of_crops:
                        continue
                    crop_paths_template.append(template_crop_folder_path + crop_id_folder.name + "/" + crop.name)
                    counter += 1
            output_file_template = open(output_file_template_path,"w")
    
    if not feature_vectors_only_template:
        # get the paths to the crop images
        crop_paths_test = []
        folders = os.scandir(image_folder_path)

        for folder in folders:
            image_files = os.scandir(image_folder_path + folder.name)
            for crop in image_files:
                if crop.name.__contains__("annotated_image") or crop.name.__contains__("bbox_file"):
                    continue
                crop_paths_test.append(image_folder_path + folder.name + "/" + crop.name)

        output_file_test = open(output_file_test_path,"w")

        crop_features_template = []
    
    # compute the clip features to the templates and crops
    with torch.no_grad():
        if not feature_vectors_only_test:
            for i in range(len(crop_paths_template)):
                compute_clip_features(crop_paths_template[i], output_file_template, preprocess, model, device,)
            
        if not feature_vectors_only_template:  
            for i in range(len(crop_paths_test)):
                compute_clip_features(crop_paths_test[i], output_file_test, preprocess, model, device,)

def normalize_all_feature_vectors():
    # normalize the computed CLIP features by MinMax-Scaling (according to the templates) or by L2-Scaling
    techniques = []
    if min_max_scaling:
        techniques.append([min_max_scaling, l2_normalization])
    elif both_normalizations:
        techniques.append([True, False])
        techniques.append([False, True])
    for norm in techniques:
        min_max_scaling, l2_normalization = norm 
        
        normalization_str = ""
        if min_max_scaling:
            normalization_str += "minmax"
        if l2_normalization:
            normalization_str += "l2"
        
        file_template_data = open(output_file_template_path,"r") #"VGG_Data/template_crops_reduced_new.txt","r")
        file_test_data = open(output_file_test_path,"r")
        
        fitting_data, min, max, labels_template = parse_feature_vector_file(file_template_data, feature_vector_length)
        test_data, labels_test = load_input_vectors(file_test_data)
        print(f"found {len(fitting_data)} feature vectors for template and {len(test_data)} for testing")
            
            
        if min_max_scaling:
            fitting_data = normalize_vectors_min_max(fitting_data, min, max)
            test_data = normalize_vectors_min_max(test_data, min, max)
        if l2_normalization:
            fitting_data = normalize_vectors_l2(fitting_data)
            test_data = normalize_vectors_l2(test_data)
        
        output_file_template_normalized = open(output_file_template_path[:-4].replace("_no_norm","") + "_" + normalization_str + ".txt", "w")
        output_file_test_normalized = open(output_file_test_path[:-4].replace("_no_norm","") + "_" + normalization_str + f"{count_of_crops_str}.txt", "w")
        
        for idx in range(len(fitting_data)):  
            label = labels_template[idx]
            if template:
                label = label[label.rfind("/")+1:][:-4]
                if label.__contains__("_"):
                    label = label[:label.find("_")]
            else:
                label = label[:label.rfind("/")]
                label = label[label.rfind("/")+1:]
            output_file_template_normalized.write(f"{label}")
            print("label: ", labels_template[idx])
            for i in range(len(fitting_data[idx])):
                output_file_template_normalized.write(f", {fitting_data[idx][i]}")
            output_file_template_normalized.write("\n")
        
        for idx in range(len(test_data)):   
            output_file_test_normalized.write(f"{labels_test[idx]}")
            print("label: ", labels_test[idx])
            for i in range(len(test_data[idx])):
                output_file_test_normalized.write(f", {test_data[idx][i]}")
            output_file_test_normalized.write("\n")

# if __name__ == "__main__":
#     # configuration
#     compute_image_crops = False # whether to first compute the image crops from the bounding boxes 
#     
#     compute_clip_feature_vectors = True # whether to generate the clip features
#     feature_vectors_only_template = False # whether to only generate the template features
#     feature_vectors_only_test = False # whether to only generate the test features
# 
#     assert not feature_vectors_only_template or not feature_vectors_only_test, "Impossible configuration selected. Only one of feature_vectors_only_template and feature_vectors_only_test can be True."
#     
#     crops_no_overwrite = False # whether to check if crops already exist and skipping them if they do
# 
#     image_size = [1800, 697] # image width, image height
# 
#     unconverted_gt_bboxes = False # whether the input bounding box is in xywh format
#     filter_ids = [] # ignore other class ids in gt_annotations
#     
#     # Paths to configure
#     SAM_run_name = "run_2025-5-2_15-42" # SAM segmentation run
#     image_folder_path_start = "input/dataset/audi-Essen-Köln/images/" # path to original images
#     inpainted_images = "input/inpainted_images/" # path to inpainted images
#     template_crop_folder_path = "input/templates/audi_png/" # path to template images
#     
#     bboxes_crops_path = f"output/SAM/{SAM_run_name}/" # path to bboxes
# 
# 
#     template = True
#     crops_computed = True # in case of SAM results
# 
#     for masking_str in ["no_mask"]: # from ["no_mask", "fill_black", "fill_white", "inpaint"]:
#         if masking_str != "no_mask" or not crops_computed:
#             computed_crops_path = f"output/SAM/{SAM_run_name}_masked/{masking_str}/"
#         elif masking_str == "no_mask" and crops_computed:
#             computed_crops_path = f"output/SAM/{SAM_run_name}/"
# 
#         image_folder_path = image_folder_path_start
#         normalize_feature_vectors = True
#         
#         output_folder = f"output/feature_matching/features/results_{SAM_run_name}/"
#         feature_vector_length = 512 # length of the feature vectors
#         
#         # which scaling techniques to apply
#         min_max_scaling = False
#         l2_normalization = False
#         both_normalizations = True
#         assert not((min_max_scaling or l2_normalization) and both_normalizations), "use only one variant defining the normalization technique"
#         
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#         
#         # if multiple example images are used instead of one template
#         count_of_crops = -1 # 30
#         count_of_crops_str = "_" + str(count_of_crops)
#         if count_of_crops < 0:
#             count_of_crops_str = ""
#         
#         # output file names
#         run_name = f"mask_{masking_str}_no_norm"
#         output_file_template_path = f"{output_folder}template_crops{count_of_crops_str}_{run_name}.txt" # f"{output_folder}template_crops_{count_of_crops}_clip_template.txt"
#         output_file_test_path = f"{output_folder}crops_gt_{run_name}.txt" # f"{output_folder}crops_finetuned_threshold_0.7_res1_post_clip_template.txt"
#         
#         # compute and save crops to each image
#         if compute_image_crops:
#             if masking_str == "no_mask":
#                 generate_crops.run([computed_crops_path, bboxes_crops_path, filter_ids, False, None, False, image_folder_path])
#             elif masking_str == "fill_black":
#                 generate_crops.run([computed_crops_path, bboxes_crops_path, True, [0,0,0], False, image_folder_path])
#             elif masking_str == "fill_white":
#                 generate_crops.run([computed_crops_path, bboxes_crops_path, filter_ids, True, [255,255,255], False, image_folder_path])
#             elif masking_str == "inpaint":
#                 generate_crops.run([computed_crops_path, bboxes_crops_path, filter_ids, False, None, True, inpainted_images])
# 
#         image_folder_path = computed_crops_path
#             
#         if compute_clip_feature_vectors:
#             compute_all_clip_features()
# 
#         
#         if normalize_feature_vectors:
#             normalize_all_feature_vectors()
