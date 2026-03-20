import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import classification.clip_model as clip_model

def compute_lpips_precomputed(lpips_network, feats_crop, feats_template):        
    d = lpips_network(None, None, precomputed_feats_in0=feats_crop, precomputed_feats_in1=feats_template).item()
    return d


def compute_metric(crop_image, template_features, correlations, model_information, template_image_sizes, max_correlation=-1.0, compute_area_similarity = False , metric = "CLIP", normalization_values=None):
    """
        computes best template according to given metric

        Args:
            crop (str): line from crop file
            templates (list(str)): names of all templates
            correlations (np.array): all correlation values
            max_correlation (float): maximal found correlation value
            compute_area_similarity (bool) = False: apply restrictions to area and side length when comparing to templates
            metric : either "LPIPS" or "CLIP" can be selected (std. "CLIP"). With "LPIPS" the LPIPS between the template and the image crop is computed, with "CLIP" the cosine similarity between the CLIP features.
    """
    best_result = ""
    # set best value to worst expected value according to selected metric
    if metric == "CLIP":
        best_matching_value = 0.0
    else:
        best_matching_value = 1.0

    if max_correlation == -1.0:
        for template_key in correlations.keys():
            corr = correlations[template_key]
            if corr > max_correlation:
                max_correlation = corr

    crop_features = None

    if metric == "CLIP":
        device, model, preprocess = model_information
    elif metric == "LPIPS":
        model = model_information[0]
    
    for temp in template_features.keys():
        # check if template is above max correlation - 0.1
        if correlations[temp] > max_correlation - 0.1:            
            # do area and side length ratio checks
            # skip comparison with template if image is scaled outside of [0.5, 2] times along a side, or if the area is less than 0.25 of the template
            if compute_area_similarity:
                crop_height, crop_width = len(crop_image), len(crop_image[0])
                side_length_comp = crop_width/ crop_height * template_image_sizes[temp][0] / template_image_sizes[temp][1]
                area_comp = (crop_width * crop_height) / (template_image_sizes[temp][0] * template_image_sizes[temp][1])
                
                if side_length_comp < 0.5 or side_length_comp > 2.0 or area_comp < 0.25:
                    continue
            
            # compute metric
            if metric == "CLIP":
                if crop_features is None:
                    crop_features = clip_model.compute_clip_features(crop_image, preprocess, model, device)
                    crop_features = clip_model.normalize_vectors_min_max(crop_features, normalization_values[0], normalization_values[1])
                matching_value = cosine_similarity(crop_features, template_features[temp])[0][0]
            else:
                # only compute crop features for first template
                if crop_features is None:
                    crop_features = model.precompute_feats(crop_image)
                                
                matching_value = compute_lpips_precomputed(model, crop_features, template_features[temp])
            
            # compare with previous results
            if (metric == "CLIP" and matching_value > best_matching_value) \
                or (metric == "LPIPS" and matching_value < best_matching_value):
                    best_matching_value = matching_value
                    best_result = temp

    while best_matching_value and type(best_matching_value) is list:
        best_matching_value = best_matching_value[0]
    while best_result and type(best_result) is list:
        best_result = best_result[0]
 
    return best_result, best_matching_value

def compute_histogram_cv2(image):
    """
        computes color histogram along the RGB channels, 
    """
    channels = [0,1,2]
    hist = cv2.calcHist([image], channels, None, [50,50,50], [0,256,0,256,0,256], accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def compute_template_histograms_and_image_sizes(template_folder):
    """
    compute histograms and image sizes

    Args:
        template_folder (string): path to the folder containing the template images
    """
    
    template_data = []
    template_histograms = {}
    template_image_size = {}

    for temp in list(os.scandir(template_folder)):
        if not os.path.isfile(template_folder + temp.name):
            continue
        template_image = cv2.imread(template_folder + temp.name)
        template_data.append([temp.name.replace(".png",""), template_image])
    
    for data_entry in template_data:
        template_name, template_image = data_entry
        template_histograms[template_name] = compute_histogram_cv2(template_image)
        template_image_size[template_name] = [len(template_image), len(template_image[0])]
        
    return template_data, template_histograms, template_image_size
        
def precompute_templates_LPIPS(template_folder, model):
    features = {}
    for temp in list(os.scandir(template_folder)):
        template_image = cv2.imread(template_folder + temp.name)
        template_name = temp.name.replace(".png","")
        features[template_name] = model.precompute_feats(template_image)    
    return features
    
def hasIntersection(bbox1 , bbox2):
    """
        checks if there is an intersection between bbox1 and bbox2

        Args:
            bbox1, bbox2 (np.array(float)): bounding boxes from which to compute if there is an intersection

        return value:
            True, if there is an intersection
            False, else
    """
    # no intersection if max1 < min2
    if bbox1[2] < bbox2[0] or bbox1[3] < bbox2[1]:
        return False
    # no intersection if min1 > max2
    if bbox1[0] > bbox2[2] or bbox1[1] > bbox2[3]:
        return False
    return True

def areaRectangle(bbox):
    """
        computes the area occupied by the given bbox

        Args:
            bbox (np.array(float)) : bounding box from which to compute the area
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def computeIntersectionWithOther(bbox, other_bbox):
    if hasIntersection(bbox, other_bbox):
        intersection = areaRectangle([max(bbox[0], other_bbox[0]), max(bbox[1], other_bbox[1]), min(bbox[2], other_bbox[2]), min(bbox[3], other_bbox[3])])
        other_bbox_area = areaRectangle(other_bbox)
        return intersection / other_bbox_area
    else:
        return 0.0  
  
def non_maximum_suppression(input):
    """
        Applies non-maximum suppression by checking for the intersecting area of the predicted bounding boxes and then selecting the one with the highest metric value. 

        Args:
            input (list): contains 
                            - experiment_folder (str): path to output folder
                            - thresholds(list[float]): contains thresholds that shall be evaluated
                            - mask_type (str): can be either "inpaint" or "no_mask"
                            - scaling_type (str): should be "_minmax"
                            - metric (str): decides the features that are used, either "CLIP" or "LPIPS"
                            - overlap_threshold (float): IoU threshold, when to apply non-maximum suppression
    """
    experiment_folder, thresholds, mask_type, scaling_type, metric, overlap_threshold  = input
    
    output_folder = experiment_folder

    for threshold in range(len(thresholds)):
        threshold_val = thresholds[threshold]
        file_path = f"{output_folder}results_{metric}_{mask_type}{scaling_type}_th{threshold_val}.txt"
        file = open(file_path, "r")
        
        data = {}
        # load data from result file
        for line in file.readlines():
            crop_path = line[:line.find(" ")]
            bbox = line[line.find("["): line.find("]")+1]
            template_id = int(line[line.find("]")+1:line.rfind(" ")])
            metric_value = line[line.rfind(" "):].replace("\n","")
            
            
            image_name = crop_path[:crop_path.rfind("/")]
            image_name = image_name[image_name.rfind("/")+1:]
            image_name = image_name[:image_name.rfind("_")]
            
            if not image_name in data.keys():
                data[image_name] = {"detections":[], "other_crops":[]}
                
            data_object = {"crop_path":crop_path, "bbox":bbox, "template_id":template_id, "metric_value":metric_value}
            
                
            if template_id < 0:
                data[image_name]["other_crops"].append(data_object)
            else:
                data[image_name]["detections"].append(data_object)
                
        removements = 0
        detections_sum = 0
        for image_name in data.keys():
            remove_indices = []
            detections = data[image_name]["detections"]
            detections_sum += len(detections)

            for i in range(len(detections)):
                if i in remove_indices:
                    continue

                for j in range(len(detections)):
                    if j == i or j in remove_indices or i in remove_indices:
                        continue
                    
                    bbox1 = [float(x) for x in data[image_name]["detections"][i]["bbox"].replace("[","").replace("]","").split(",")]
                    bbox2 = [float(x) for x in data[image_name]["detections"][j]["bbox"].replace("[","").replace("]","").split(",")]
                    
                    if computeIntersectionWithOther(bbox1, bbox2) > overlap_threshold or computeIntersectionWithOther(bbox2, bbox1) > overlap_threshold:
                        # constraint best metric score
                        metric_value1 = data[image_name]["detections"][i]["metric_value"]
                        metric_value2 = data[image_name]["detections"][j]["metric_value"]
                        
                        if (metric == "LPIPS" and metric_value1 < metric_value2) or (metric == "CLIP" and metric_value1 > metric_value2):
                            remove_indices.append(j)
                            removements += 1
                        else:
                            remove_indices.append(i)
                            removements += 1
            
            detections_updated = []
            for i in range(len(detections)):
                crop_path = detections[i]["crop_path"]
                bbox = detections[i]["bbox"]
                template_id = detections[i]["template_id"]
                metric_value = detections[i]["metric_value"]
                
                if i in remove_indices:
                    data[image_name]["other_crops"].append({"crop_path":crop_path, "bbox":bbox, "template_id":-1, "metric_value":0.0})
                else:
                    detections_updated.append({"crop_path":crop_path, "bbox":bbox, "template_id":template_id, "metric_value":metric_value})
            data[image_name]["detections"] = detections_updated
            
        new_file = open(file_path[:-4] + f"_nms_{overlap_threshold}.txt", "w")
        for image_name in data.keys():
            for detection in data[image_name]["detections"]:
                output = detection["crop_path"] + " " + str(detection["bbox"]) + " " + str(detection["template_id"]) + " " + str(detection["metric_value"]) + "\n"
                new_file.write(output)
            
            for other_crop in data[image_name]["other_crops"]:
                output = other_crop["crop_path"] + " " + str(other_crop["bbox"]) + " " + str(other_crop["template_id"]) + " " + str(other_crop["metric_value"]) + "\n"
                new_file.write(output)
                
        print(f"removed {removements} icons")
        print(f"of before {detections_sum} detections")

def compute_comparison(image_folder, crop_data, thresholds, metric, template_features, template_histograms, template_sizes, normalization_values, model_information, use_correlation, correlation_threshold, do_area_comp):
    """
        Gathers all crop paths and bboxes and calls compute_metric for each crop. Computes the output strings containing the crop path, the predicted template and the metric value.

        Args:
            crop_data (list(str)): contains paths to computed image crops
            thresholds (list[float]): contains thresholds that shall be evaluated
            metric (str): decides the features that are used, either "CLIP" or "LPIPS"
            template_features (dict[str : np.array(float)]): contains either CLIP or LPIPS features
            template_histograms (dict[str : np.array(float)]): color histogram to each template
            template_sizes (dict[str : list(float)]): image size of each template
            normalization_values (list(np.array(float)) | None): (CLIP only) contains min and max values of CLIP template features
            model_information (list) : contains device, model and preproces method in case of CLIP features, and the model in case of LPIPS features
    """
    output_data = []
    for i in range(len(thresholds)):
        output_data.append([])
    
    image_path_prev = ""
    image = None
    
    for crop in crop_data:
        # split crop paths into single components
        image_name, crop_idx, bbox = crop
        
        image_path = image_folder + image_name + ".png"
        crop_name = image_name + "_" + crop_idx

        # if image has changed: open new bbox file and read data
        if image_path != image_path_prev:
            image_path_prev = image_path
            image = cv2.imread(image_path)

        assert len(bbox) > 2, f"bbox wrong: {bbox}"
        
        assert len(image) > 0 and len(image[0]) > 0, "invalid image"
        
        # compare histogram 
        bbox = [int(x) for x in bbox]
        bbox[0] = max(0,min(bbox[0], len(image[0])))
        bbox[1] = max(0,min(bbox[1], len(image)))
        bbox[2] = max(0,min(bbox[2], len(image[0])))
        bbox[3] = max(0,min(bbox[3], len(image)))
        
        image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        hist_crop = compute_histogram_cv2(image_crop)
        
        # compute color histogram correlation
        best_correlation = 0.0
        correlations = {}
        for template in template_histograms.keys():
            correlation = cv2.compareHist(hist_crop, template_histograms[template], 0)
            correlations[template] = correlation
            if correlation > best_correlation:
                best_correlation = correlation


        # if color histogram correlation below threshold continue with next candidate
        if use_correlation and best_correlation < correlation_threshold:
            for i in range(len(thresholds)):
                output_data[i].append(f"{crop_name} {bbox} -1 0.0\n")
            continue
        
        # compute metric (cosine similarity of CLIP features or LPIPS)
        best_result, metric_value = compute_metric(image_crop, template_features, correlations, model_information, template_image_sizes=template_sizes, compute_area_similarity=do_area_comp, metric=metric, normalization_values=normalization_values)
        
        # evaluate for each given threshold if metric value is above it
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            if (not use_correlation or best_correlation >= correlation_threshold) and \
                (metric == "CLIP" and metric_value > threshold) or (metric == "LPIPS" and metric_value < threshold):
                if best_result.__contains__("_"):
                    best_result = best_result[:best_result.find("_")]
                output_data[i].append(f"{crop_name} {bbox} {best_result} {metric_value}\n")
            else:
                output_data[i].append(f"{crop_name} {bbox} -1 0.0\n")
    return output_data

def run(run_name, image_folder, thresholds, mask_type, metric, crop_bbox_folder, template_features, template_folder, scaling_type="_minmax", normalization_values=None, model_information=None, do_nms=False, overlap_threshold=0.1):
    """
        Args:
            run_name (str): name of to experiment folder (results will be saved to \"output/feature_matching/[experiment_folder]\")
            thresholds(list[float]): contains thresholds that shall be evaluated
            mask_type (str): can be either "inpaint" or "no_mask"
            scaling_type (str): should be "_minmax"
            metric (str): decides the features that are used, either "CLIP" or "LPIPS"
            normalization_values (list[np.array[float]]) : used to normalize CLIP features by min and max of the template features, not used for LPIPS computation
            model_information (list) : list with device information, model, and preprocess method retrieved by load_CLIP_model(), in case of LPIPS only the model
            do_nms (bool) : whether to do non-maximum-suppression or not
            overlap_threshold (float) : threshold of IoU, when to apply nms
    """

    output_folder = f"output_data/feature_matching/{run_name}/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # compute metric independent template information
    template_data, template_histograms, template_sizes = compute_template_histograms_and_image_sizes(template_folder=template_folder)    
    
    # load crop data from folder structure
    crop_data = []
    image_names = os.scandir(crop_bbox_folder)
    for image_name in image_names:
        image_name = image_name.name
        bbox_file = open(crop_bbox_folder + image_name + "/bbox_file.txt", "r")
        lines = bbox_file.readlines()
        for line in lines:
            id = line[:line.find(" ")]
            bbox = line[line.find(" ")+1:].replace("\n","").replace("[","").replace("]","")
    
            bbox = [float(x) for x in bbox.split(",")] 
            
            data = [image_name, id, bbox]
            crop_data.append(data)
    
    if scaling_type is None:
        scaling_type = ""
    
    output = compute_comparison(image_folder=image_folder, crop_data=crop_data, thresholds=thresholds, metric=metric, template_features=template_features, template_histograms=template_histograms, template_sizes=template_sizes, normalization_values=normalization_values, model_information=model_information, use_correlation=True, correlation_threshold=0.5, do_area_comp=True)
        
    # write results to output files
    for threshold in range(len(thresholds)):
        threshold_val = thresholds[threshold]
        output_file_path = f"{output_folder}/results_{metric}_{mask_type}{scaling_type}_th{threshold_val}.txt"
        output_file = open(output_file_path,"w")


        for line in output[threshold]:
            output_file.write(line)

    if do_nms:
        non_maximum_suppression([output_folder, thresholds, mask_type, scaling_type, metric, overlap_threshold])
