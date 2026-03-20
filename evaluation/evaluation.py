
import numpy as np
import os
from PIL import Image
import concurrent.futures

# for all output_bboxes: check if it is detected
# and has the correct id

number_correct_bboxes = 0
number_missing_bboxes = 0
number_incorrect_bboxes = 0
number_overlapping_bboxes = 0


def convert_ncxywh_to_xyxy(bbox, image_size):
    w,h = bbox[2] * image_size[0], bbox[3] * image_size[1]
    bbox[0] = bbox[0] * image_size[0] - w/2
    bbox[1] = bbox[1] * image_size[1] - h/2
    bbox[2] = bbox[0] + w
    bbox[3] = bbox[1] + h

def convert_nxywh_to_xyxy(bbox, image_size):
    w,h = bbox[2] * image_size[0], bbox[3] * image_size[1]
    bbox[0] = bbox[0] * image_size[0]
    bbox[1] = bbox[1] * image_size[1]
    bbox[2] = bbox[0] + w
    bbox[3] = bbox[1] + h

def convert_xywh_to_xyxy(bbox):
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]


def hasIntersection(bbox1 , bbox2):
    # no intersection if max1 < min2
    if bbox1[2] < bbox2[0] or bbox1[3] < bbox2[1]:
        return False
    # no intersection if min1 > max2
    if bbox1[0] > bbox2[2] or bbox1[1] > bbox2[3]:
        return False
    return True

def areaRectangle(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def computeIoU(bbox1, bbox2):
    intersection = areaRectangle([max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])])
    union = areaRectangle(bbox1) + areaRectangle(bbox2) - intersection
    return intersection / union

def computeIoverGT(bbox, gt):
    if hasIntersection(bbox, gt):
        intersection = areaRectangle([max(bbox[0], gt[0]), max(bbox[1], gt[1]), min(bbox[2], gt[2]), min(bbox[3], gt[3])])
        gt_area = areaRectangle(gt)
        return intersection / gt_area
    else:
        return 0.0


def checkCenterInBbox(bbox1, bbox2):
    if hasIntersection(bbox1,bbox2):
        center = (bbox1[0] + (bbox1[2]-bbox1[0])/2, bbox1[1] + (bbox1[3]-bbox1[1])/2)

        if center[0] >= bbox2[0] and center[0] <= bbox2[2] and center[1] >= bbox2[1] and center[1] <= bbox2[3]:
            return True
    return False

# correct bounding boxes / all predicted bounding boxes
def computePrecision(predictions, ground_truth, checkForLabelClass, all_ground_truth, compute_font_coverage, font_coverage_correct, font_coverage_all):
    number_all_predicted_bboxes = len(predictions)
    bboxes_correct = np.full((len(predictions)), False)
    mismatches = 0.0
    
    # if no predicted boxes return 1.0
    if number_all_predicted_bboxes == 0:
        return 1.0, 0, bboxes_correct, mismatches
    
    number_correct_bboxes = 0
    # check for matching gt for each prediction
    for i in range(len(predictions)):
        bbox = predictions[i]
        
        if compute_font_coverage:
            font_coverage_all[bbox[5]] += 1
        
        found_bbox = False
        for j in range(len(ground_truth)):
            bbox_gt = ground_truth[j]

            if (checkCenterInBbox(bbox, bbox_gt) or checkCenterInBbox(bbox_gt, bbox)):
                if not (checkForLabelClass and bbox[4] != bbox_gt[4]):
                    if len(bbox) == 5:
                        bbox.append(bbox_gt[4])
                    number_correct_bboxes += 1.0
                    bboxes_correct[i] = True
                    found_bbox = True
                    if compute_font_coverage:
                        font_coverage_correct[bbox[5]] += 1  
                    break
                
        if not found_bbox:
            found_mismatch = False
            for id_tmp in range(len(all_ground_truth)):
                if not found_mismatch:
                    for j in range(len(all_ground_truth[id_tmp])):
                        bbox_gt = all_ground_truth[id_tmp][j]
                        if checkCenterInBbox(bbox, bbox_gt) or checkCenterInBbox(bbox_gt, bbox):
                            mismatches += 1.0
                            found_mismatch = True
                            if len(bbox) == 5:
                                bbox.append(bbox_gt[4])
                            break
                
    false_positives = number_all_predicted_bboxes - number_correct_bboxes
    return number_correct_bboxes / number_all_predicted_bboxes, false_positives, bboxes_correct, mismatches

# correct bounding boxes / all ground truth bounding boxes
def computeRecall(predictions, ground_truth, checkForLabelClass, save_images, result_folder_path, ckp_name, image_name, gt_folder_path_images, id, all_predictions, compute_font_coverage, correct_detected_font, gt_font_coverage):
    number_all_gt_bboxes = len(ground_truth)

    bboxes_found = np.full((len(ground_truth)), False)
    
    mismatches = 0.0

    # if no ground truth boxes return 1.0
    if number_all_gt_bboxes == 0:
        return 1.0, 0, bboxes_found, mismatches
    number_correct_bboxes = 0
    # check for prediction for every gt_bbox
    for i in range(len(ground_truth)):
        found_bbox = False
        bbox_gt = ground_truth[i]
        if compute_font_coverage:
            gt_font_coverage[bbox_gt[5]] += 1
        crop_idx = 0
        for j in range(len(predictions)):
            bbox = predictions[j]
            if checkCenterInBbox(bbox, bbox_gt) or checkCenterInBbox(bbox_gt, bbox):
                if not (checkForLabelClass and bbox[4] != bbox_gt[4]):
                    # correct bboxes
                    number_correct_bboxes += 1.0
                    bboxes_found[i] = True
                    found_bbox = True
                    
                    if len(bbox_gt) == 6:
                        bbox_gt.append(bbox[4])
                    
                    if compute_font_coverage:
                        correct_detected_font[bbox_gt[5]] += 1   

                    if save_images:
                        if not os.path.exists(f"{result_folder_path}/correct_bBoxes_{ckp_name}"):
                            os.makedirs(f"{result_folder_path}/correct_bBoxes_{ckp_name}")
                        if not os.path.exists(f"{result_folder_path}/correct_bBoxes_{ckp_name}/{id}"):
                            os.makedirs(f"{result_folder_path}/correct_bBoxes_{ckp_name}/{id}")
                        
                        crop_save_path = f"{result_folder_path}/correct_bBoxes_{ckp_name}/{id}/{image_name}_{crop_idx}.jpg"
                        crop_idx += 1
                        image_path = f"{gt_folder_path_images}{image_name}.png"
                        if not os.path.exists(image_path):
                            image_path = image_path.replace("/val/", "/train/")
                            assert os.path.exists(image_path), f"image file ({image_path}) not found in val or train folder"                                        
                        image = Image.open(image_path)
                        image = image.convert('RGB')
                        crop = image.crop((max(0,bbox[0]), max(0,bbox[1]), min(1800,bbox[2]), min(697,bbox[3])))
                        crop.save(crop_save_path)

                    break
        if not found_bbox:
            found_mismatch = False
            for id_tmp in range(len(all_predictions)):
                if not found_mismatch:
                    for j in range(len(all_predictions[id_tmp])):
                        bbox = all_predictions[id_tmp][j]
                        if checkCenterInBbox(bbox, bbox_gt) or checkCenterInBbox(bbox_gt, bbox):
                            mismatches += 1.0
                            found_mismatch = True
                            if len(bbox_gt) == 6:
                                bbox_gt.append(bbox[4])
                            break

    false_negatives = number_all_gt_bboxes - number_correct_bboxes
    return number_correct_bboxes / number_all_gt_bboxes, false_negatives, bboxes_found, mismatches

# compute mean intersection over union for two bboxes
def computeMeanIoU(predictions, ground_truth, checkForLabelClass):
    meanIoU = 0.0
    if len(predictions) == 0 and len(ground_truth) == 0:
        return 1.0
    elif len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0
    
    counted_predictions = 0
    for bbox in predictions:
        highest_IoU = 0.0
        for bbox_gt in ground_truth:
            if checkForLabelClass and bbox[4] != bbox_gt[4]:
                continue
            if hasIntersection(bbox, bbox_gt) and (checkCenterInBbox(bbox, bbox_gt) or checkCenterInBbox(bbox_gt, bbox)): #checkCenterInBbox(bbox, bbox_gt):
                IoU_cur = computeIoU(bbox, bbox_gt)
                if highest_IoU < IoU_cur:
                    highest_IoU = IoU_cur
        if highest_IoU != 0:
            counted_predictions += 1
        meanIoU += highest_IoU
    if counted_predictions > 0:
        meanIoU /= counted_predictions
    return meanIoU

def convert_file_to_category_id(template_file):
    category = template_file[template_file.find("p")+1:]
    if category.__contains__("_"):
        category = category[:category.find("_")]
    return category

def load_annotation_file(path):
    # load the labels from txt files
    label_path = path
    if not os.path.exists(label_path):
        if os.path.exists(label_path.replace("/val/", "/val/wrong_bboxes/")):
            return None
        label_path = label_path.replace("/val/", "/train/")
        assert os.path.exists(label_path), f"label file ({label_path}) not found in val or train folder"
    file = open(label_path,'r')
    bounding_boxes = []
    for line in file.readlines():
        content = line.replace("[","").replace("]","").replace(",","").split(" ")
        if len(content) < 4:
            print("line contains not enough data (should contain [id x_min y_min x_max y_max])")
        
        id = float(content[0])
        x_min = float(content[1])
        y_min = float(content[2])
        x_max = float(content[3])
        y_max = float(content[4])
        bounding_boxes.append([x_min, y_min, x_max, y_max, id])

        if len(content) > 5:
            print("line contains more data than expected, additional data is ignored")
        
    return bounding_boxes

def loadResultFileTxtSAM(path, ignore_template_file=False):
    # should consist of entries with filename and bounding boxes
    file = open(path,'r')
    bounding_boxes = []

    for line in file.readlines():
        first_whitespace = line.find(" ")
        template_file = line[0:first_whitespace]
        if not ignore_template_file:
            category_id = convert_file_to_category_id(template_file)
        else:
            category_id = 0
        bbox_str = line[first_whitespace+1:]
        bbox_and_id = np.concatenate((np.fromstring(bbox_str[1:-1], dtype=float, sep=", "), np.array([float(int(category_id))])))
        assert bbox_and_id.shape[0] >= 4, f"missing information in the result file {path}"
        bounding_boxes.append(np.concatenate((np.fromstring(bbox_str[1:-1], dtype=float, sep=", "), np.array([float(int(category_id))]))))
    return bounding_boxes

def loadResultFileTxt(path, image_name_searched, ignore_template_file=False):
    # should consist of entries with filepath, bounding boxes and id 
    file = open(path,'r')
    bounding_boxes = []
    for line in file.readlines():
        first_whitespace = line.find(" ")
        image_name = line[0:first_whitespace]
        
        if image_name[image_name.rfind("/")+1:].__contains__("annotated_image"):
            continue

        image_name = image_name[:image_name.rfind("/")]
        image_name = image_name[image_name.rfind("/") + 1:]
        image_name = image_name[:image_name.rfind("_")]
        
        if not image_name == image_name_searched:
            continue
        
        if not ignore_template_file:
            line = line.replace("]]","").replace("[[","")
            category_id = line[line.rfind("]")+1:line.rfind(" ")].replace(" ","").replace("\n","")
            if len(category_id) > 4 and category_id.__contains__(".png"):
                category_id = category_id[category_id.rfind("/")+1:category_id.rfind(".png")]
        else:
            category_id = 0

        bbox_str = line[first_whitespace+1:line.rfind(" ")]
        if not ignore_template_file:
            if category_id.__contains__("nf_"):
                continue
            if not category_id.__contains__(".0"):
                category_id = float(int(category_id))
            
        bbox_and_id = np.concatenate((np.fromstring(bbox_str[bbox_str.find("[")+1:bbox_str.rfind("]")], dtype=float, sep=", "), np.array([category_id])))

        assert bbox_and_id.shape[0] >= 4, f"missing information in the result file {path}"
        bounding_boxes.append(bbox_and_id)
    return bounding_boxes
    
def filterBBoxes(bboxes, filter_labels):
    # filter the bounding boxes by the labels
    bboxes = list(filter(lambda x: len(x) > 4 and not x[4] in filter_labels, bboxes))
    return bboxes

def getClassNames(file_name):
    file = open(file_name, 'r')
    class_names = {}
    for line in file.readlines():
        if line.__contains__(":"):
            data = line.split(':')
            # id has to be a number => skip if there is something else
            if not data[0].isnumeric():
                continue

            if len(data) != 2:
                assert False, "There is data missing in the id file"
            id = int(data[0])
            name = data[1].replace('\n','')
            class_names[id] = name
    return class_names

def categorize_bboxes(bboxes, num_ids, id_to_idx):
    bboxes_categorized = []
    for i in range(num_ids):
        id_array = []
        for j in range(len(bboxes)):
            if id_to_idx[str(int(bboxes[j][4]))] == i:
                id_array.append(bboxes[j])
        bboxes_categorized.append(id_array)
    return bboxes_categorized

# get the file names of the crop images of the incorrect predictions (false positives)
def get_crop_names_incorrect(result_folder_path, image_name, bboxes, bboxes_correct,id, original_indices, ckp_name, gt_folder_path_images, images_save=False):
    name_class = []
    for j in range(len(bboxes_correct)):
        if not bboxes_correct[j]:
            # save false positive images
            if images_save:     
                if not os.path.exists(f"{result_folder_path}/false_positives_ignBoxes_{ckp_name}"):
                    os.makedirs(f"{result_folder_path}/false_positives_ignBoxes_{ckp_name}")
                if not os.path.exists(f"{result_folder_path}/false_positives_ignBoxes_{ckp_name}/{id}"):
                    os.makedirs(f"{result_folder_path}/false_positives_ignBoxes_{ckp_name}/{id}")
                
                crop_save_path = f"{result_folder_path}/false_positives_ignBoxes_{ckp_name}/{id}/{image_name}_{j}.jpg"   
                
                image_path = f"{gt_folder_path_images}{image_name}.png"
                if not os.path.exists(image_path):
                    image_path = image_path.replace("/val/", "/train/")
                    assert os.path.exists(image_path), f"image not found in train or val folder, path: {image_path}"
                
                image = Image.open(image_path)
                
                image = image.convert('RGB')
                crop = image.crop((bboxes[j][0],bboxes[j][1],bboxes[j][2],bboxes[j][3]))
                crop.save(crop_save_path)

            name_class.append([bboxes[j]])

    return name_class

# create crop files and get the names of the not found ground truth bounding boxes (false negatives)
def get_crop_names_not_found(result_folder_path, gt_folder_path_images, image_name, bboxes, bboxes_found, id, original_indices, ckp_name, sam_crops, images_save=False):
    name_class = []

    for j in range(len(bboxes)):
        if not bboxes_found[j]:
            crop_idx = ""
            if original_indices[j] > 0:
                crop_idx = str(original_indices[j]+1)

            # find match from SAM crops
            highest_IoU = 0.0
            index = 0
            i = 0
            for i in range(len(sam_crops)):
                bbox_sam = sam_crops[i]
                if hasIntersection(bbox_sam, bboxes[j]) and (checkCenterInBbox(bbox_sam, bboxes[j]) or checkCenterInBbox(bboxes[j], bbox_sam)): #checkCenterInBbox(bbox, bbox_gt):
                    IoU_cur = computeIoU(bbox_sam, bboxes[j])
                    if highest_IoU < IoU_cur:
                        highest_IoU = IoU_cur
                        index = i

            if images_save:
                if not os.path.exists(f"{result_folder_path}/false_negatives_ignBoxes_{ckp_name}"):
                    os.makedirs(f"{result_folder_path}/false_negatives_ignBoxes_{ckp_name}")
                if not os.path.exists(f"{result_folder_path}/false_negatives_ignBoxes_{ckp_name}/{id}"):
                    os.makedirs(f"{result_folder_path}/false_negatives_ignBoxes_{ckp_name}/{id}")
                
                crop_save_path = f"{result_folder_path}/false_negatives_ignBoxes_{ckp_name}/{id}/{image_name}_{crop_idx}.jpg"
                
                image_path = f"{gt_folder_path_images}{image_name}.png"
                if not os.path.exists(image_path):
                    image_path = image_path.replace("/val/", "/train/")
                    assert os.path.exists(image_path), f"image not found in train or val folder, path: {image_path}"
                image = Image.open(image_path)
                image = image.convert('RGB')
                crop = image.crop((max(0,sam_crops[index][0]), max(0,sam_crops[index][1]), min(1799,sam_crops[index][2]), min(696,sam_crops[index][3])))
                crop.save(crop_save_path)

                name_class.append([bboxes[j],f"false_negatives_ignBoxes_{ckp_name}/{id}/{image_name}_{crop_idx}.jpg"])
                assert os.path.exists(f"{result_folder_path}/false_negatives_ignBoxes_{ckp_name}/{id}/{image_name}_{crop_idx}.jpg")
            else:
                name_class.append([bboxes[j]])
    return name_class


def check_for_match(bbox, gt_bboxes):
    '''
        checks for matches with gt_bboxes
        returns 0 if no match
        return 1 if match with different class
        return 2 if complete match
    '''
    for bbox_gt in gt_bboxes:
        if checkCenterInBbox(bbox, bbox_gt) or checkCenterInBbox(bbox_gt, bbox):
            if bbox[4] == bbox_gt[4]:
                return 2
            else:
                return 1
    return 0

def evaluate_result_segment(input):
    image_names_segment, variables = input
    
    output_string = ""
    output_string_detection_fails = ""
    
    SAM_folder_path, file_path, gt_folder_path_labels, num_ids, check_category, id_to_idx, no_border_bboxes, compute_font_coverage, font_mask_path, \
        number_bins, threshold_gt_bboxes, evaluate_with_sam_output, checkForLabelClass, generate_images, result_folder_path, experiment_name, gt_folder_path_images, images_save, image_size, mapping = variables
    
    missing_gt_boxes_sam_section = 0
    only_partially_visible_section = 0
    
    mismatches_predictions_section = 0
    mismatches_gt_section = 0
    false_positives_section = 0
    false_negatives_section = 0
    
    correct_detected_font_section = np.zeros(number_bins + 1)
    gt_font_coverage_section = np.zeros(number_bins + 1)
    
    correct_from_detected_font_section = np.zeros(number_bins + 1)
    all_detected_font_section = np.zeros(number_bins + 1)
    
    for image_name in image_names_segment:
        sam_crops = loadResultFileTxtSAM(os.path.join(SAM_folder_path, image_name+"/bbox_file.txt"), True)

        result_bboxes = loadResultFileTxt(file_path, image_name)

        gt_bboxes = load_annotation_file(gt_folder_path_labels + image_name+".txt")
        if gt_bboxes is None:
            continue

        # bounding boxes are converted to (x_min, y_min, x_max, y_max, id)
        gt_bboxes_scaled = []
        gt_bboxes_ignored = []
        original_indices_gt = []

        result_bboxes_scaled = []
        original_indices_results = []

        class_indices_gt = []
        class_indices = []
        for i in range(num_ids):
            original_indices_gt.append([])
            original_indices_results.append([])
            class_indices_gt.append(0)
            class_indices.append(0)
            
        if compute_font_coverage:
            font_segmentation_mask_image = np.asarray(Image.open(font_mask_path + image_name + ".png"))

        for i in range(len(gt_bboxes)):
            tmp = gt_bboxes[i][0:5]

            if not check_category:
                tmp[4] = -1
            else:
                tmp[4] = str(int(tmp[4]))
                
            tmp[4] = str(mapping[tmp[4]])
            class_indices_gt[id_to_idx[tmp[4]]] += 1.0
            tmp[4] = float(tmp[4])

            if no_border_bboxes and (tmp[0] < 0 or tmp[1] < 0 or tmp[2] > image_size[0] - 2  or tmp[3] > image_size[0] - 2):
                gt_bboxes_ignored.append(tmp)
                continue
            
            if compute_font_coverage:
                font_segmentation_mask = font_segmentation_mask_image[int(tmp[1]):int(tmp[3]), int(tmp[0]):int(tmp[2]),0].copy()
                font_segmentation_mask[font_segmentation_mask > 0.0] = 1.0 
                sum = np.sum(font_segmentation_mask)
                whole_bbox = (int(tmp[3]) - int(tmp[1])) * (int(tmp[2]) - int(tmp[0]))
                if compute_font_coverage:
                    if whole_bbox == 0:
                        coverage = 0.0
                    else:
                        coverage = sum / whole_bbox

                    if coverage == 0:
                        tmp.append(-1)
                    else:
                        tmp.append(int(coverage * number_bins))
                
                
            

            # remove elements with size 0 and elements outside of the image
            if (abs(tmp[0] - tmp[2]) < threshold_gt_bboxes or abs(tmp[1] - tmp[3]) < threshold_gt_bboxes) or \
                (tmp[0] < 1.0 or tmp[1] < 1.0 or tmp[2] > image_size[0]-2 or tmp[3] > image_size[1]-2):
                gt_bboxes_ignored.append(tmp)
            else:
                if evaluate_with_sam_output:
                    found_in_sam_results = False
                    highest_IoG = 0.0
                    for sam_crop in sam_crops:
                        if (checkCenterInBbox(sam_crop, tmp) or checkCenterInBbox(tmp, sam_crop)) and computeIoU(sam_crop, tmp) > 0.4:
                            found_in_sam_results = True
                            IoG = computeIoverGT(sam_crop, tmp)
                            if IoG > highest_IoG:
                                highest_IoG = IoG
                    if not found_in_sam_results:
                        gt_bboxes_ignored.append(tmp)
                        missing_gt_boxes_sam_section += 1
                        continue
                    if highest_IoG < 0.8:
                        only_partially_visible_section += 1

                original_indices_gt[id_to_idx[str(int(tmp[4]))]].append(class_indices_gt[id_to_idx[str(int(tmp[4]))]]-1)
                gt_bboxes_scaled.append(tmp)
            
        for i in range(len(result_bboxes)):
            tmp = [float(result_bboxes[i][0]),float(result_bboxes[i][1]),float(result_bboxes[i][2]),float(result_bboxes[i][3]), result_bboxes[i][4]]

            if tmp[4] < 0:
                continue

            if not check_category:
                tmp[4] = 0
            else:
                if type(tmp[4]) == str and tmp[4].__contains__(".0"):
                    tmp[4] = int(float(tmp[4]))
                else:
                    tmp[4] = int(tmp[4])
            
            class_indices[id_to_idx[str(tmp[4])]] += 1
            if ((no_border_bboxes and (tmp[0] < 1.0 or tmp[1] < 1.0 or tmp[2] > image_size[0]-2 or tmp[3] > image_size[1]-2)) and not check_for_match(tmp, gt_bboxes_scaled)) or check_for_match(tmp, gt_bboxes_ignored):
                continue
            else:
                result_bboxes_scaled.append(tmp)
                original_indices_results[id_to_idx[str(tmp[4])]].append(class_indices[id_to_idx[str(tmp[4])]]-1)
                
                if compute_font_coverage:
                    font_segmentation_mask = font_segmentation_mask_image[int(tmp[1]):int(tmp[3]), int(tmp[0]):int(tmp[2]),0].copy()
                    font_segmentation_mask[font_segmentation_mask > 0.0] = 1.0
                    sum = np.sum(font_segmentation_mask)
                    whole_bbox = (int(tmp[3]) - int(tmp[1])) * (int(tmp[2]) - int(tmp[0]))
                    
                    if whole_bbox == 0:
                        coverage = 0.0
                    else:
                        coverage = sum / whole_bbox
                    
                    if coverage == 0:
                        tmp.append(-1)
                    else:
                        tmp.append(int(coverage * number_bins))
                    
                    
        # sort by ids
        result_bboxes_scaled = categorize_bboxes(result_bboxes_scaled, num_ids,id_to_idx)
        gt_bboxes_scaled = categorize_bboxes(gt_bboxes_scaled, num_ids,id_to_idx)

        output_string += "image_name: " + image_name + "\n"
        output_string_detection_fails += "image_name: " + image_name + "\n"
        
        for id in range(num_ids):

            precision, false_positives, bboxes_correct, mismatches_precision = computePrecision(result_bboxes_scaled[id], gt_bboxes_scaled[id], checkForLabelClass, gt_bboxes_scaled, compute_font_coverage, correct_from_detected_font_section, all_detected_font_section)
            
            recall, false_negatives, bboxes_found, mismatches_recall = computeRecall(result_bboxes_scaled[id], gt_bboxes_scaled[id] , checkForLabelClass, generate_images, result_folder_path, experiment_name, image_name, gt_folder_path_images, id, result_bboxes_scaled, compute_font_coverage, correct_detected_font_section, gt_font_coverage_section)
            mismatches_predictions_section += mismatches_precision
            mismatches_gt_section += mismatches_recall
            false_positives_section += false_positives
            false_negatives_section += false_negatives

            crops_str_crops_incorrect = get_crop_names_incorrect(result_folder_path, image_name,result_bboxes_scaled[id], bboxes_correct, id, original_indices_results[id], experiment_name , gt_folder_path_images, images_save) 

            bboxes_crops_not_found = get_crop_names_not_found(result_folder_path, gt_folder_path_images, image_name, gt_bboxes_scaled[id], bboxes_found, id, original_indices_gt[id], experiment_name, sam_crops, images_save)

            meanIoU = computeMeanIoU(result_bboxes_scaled[id], gt_bboxes_scaled[id], checkForLabelClass)
            output_string += str(id) + " " + str(precision) + " " + str(recall) + " " + str(false_positives) + " " + str(false_negatives) + " " + str(meanIoU)  + " " + str(float(len(result_bboxes_scaled[id])))+ " " + str(float(len(gt_bboxes_scaled[id]))) + "\n"
            output_string_detection_fails += str(id) + " " + str(crops_str_crops_incorrect) + " " + str(bboxes_crops_not_found) + "\n"

        output_string += "----------------------------------\n"
        output_string_detection_fails += "----------------------------------\n"
    return [output_string, output_string_detection_fails, mismatches_predictions_section, mismatches_gt_section, false_positives_section, false_negatives_section, missing_gt_boxes_sam_section, only_partially_visible_section, correct_detected_font_section, gt_font_coverage_section, correct_from_detected_font_section, all_detected_font_section]

def get_template_ids(template_folder_path):
    template_names = []
    for temp in os.scandir(template_folder_path):
        if os.path.isfile(os.path.join(template_folder_path, temp.name)):
            template_names.append(temp.name.replace(".png","").replace(".jpg",""))
    return template_names

def evaluate_results(datasets):
    '''
        datasets (list(list)) : list containing all configurations with each configuration consisting of:
        dataset[0]  : dataset path
        dataset[1]  : dataset name
        dataset[2]  : experiment name
        dataset[3]  : number of ids
        dataset[4]  : SAM crop folder path
        dataset[5]  : evaluate only with SAM output
        dataset[6]  : result folder path ("results/" if not given)
        dataset[7]  : generate_images (bool)
        dataset[8]  : print_per_image_information (bool)
        dataset[9]  : path to font segmentation masks
        dataset[10] : compute font intersection coverage (bins in 5% steps)
        dataset[11] : check_category - check for matching categories and not only the bounding boxes
        dataset[12] : number of threads (not parallel if less than 2)
        dataset[13] : template folder path
        dataset[14] : image_size
        dataset[15] : mapping (dict that handles different ids directing to one class)
    '''
   
    for dataset in datasets:
         # if true check also for category value, else it is only checked if there is a bbox
        check_category = dataset[11]
        check_category_str = ""
        if not check_category:
            check_category_str = "noCategoryIds"
        only_SAM_results_str = ""

        evaluate_with_sam_output = dataset[5]
        if evaluate_with_sam_output:
            only_SAM_results_str = "_only_SAM_results"

        dataset_path = dataset[0]
        gt_folder_path_labels = dataset_path + "/labels/"
        gt_folder_path_images = dataset_path + "/images/"
        SAM_folder_path = f"{dataset[4]}"
        if dataset[6] != None:
            result_folder_path = dataset[6]
        else:
            result_folder_path = f"results/"
        file_path = dataset[2]
        experiment_name = dataset[2]
        experiment_name = experiment_name[:experiment_name.rfind(".")]
        if experiment_name.__contains__("/"):
            experiment_name = experiment_name[experiment_name.rfind("/")+1:]
        if not os.path.exists(f"{result_folder_path}results_converted/"):
            os.makedirs(f"{result_folder_path}results_converted/")
        result_file = f"{result_folder_path}results_converted/result_file_{experiment_name}{only_SAM_results_str}_{check_category_str}_noBorderBoxes_ignBoxes.txt"
        result_file_detection_errors = result_file[:-4] + f"_detection_fails.txt"
        experiment_name = dataset[2]
        num_ids = dataset[3]
        generate_images = dataset[7]
        print_per_image_information = dataset[8]
        font_mask_path = dataset[9]
        compute_font_coverage = dataset[10]
        number_threads = dataset[12]
        template_folder_path = dataset[13]
        image_size = dataset[14]
        mapping = dataset[15]
        
        images_save = generate_images

        checkForLabelClass = True
        threshold_gt_bboxes = 2
        no_border_bboxes = True

        image_names_iter = os.scandir(gt_folder_path_labels)
        image_names = []
        for image_name in image_names_iter:
            image_names.append(image_name.name[:-4])

        file = open(result_file, 'w')
        file_detection_errors = open(result_file_detection_errors, 'w')
        
        files = []
        res_file = open(f"{file_path}", "r")
        for line in res_file.readlines():
            image_name = line[:line.find(" ")]
            image_name = image_name[:image_name.rfind("/")]
            image_name = image_name[image_name.rfind("/")+1:]
            image_name = image_name[:image_name.rfind("_")]
            if not files.__contains__(image_name):
                files.append(image_name)
        res_file.close()

        ids =  get_template_ids(template_folder_path)
        id_to_idx = {}


        for i in range(len(ids)):
            id_to_idx[ids[i]] = i
        
        file_ids = open(result_file[:-4] + "_id-mapping.txt","w")
        for i in range(len(ids)):
            file_ids.write(f"{i} : {ids[i]} \n")

        # result variables
        mismatches_total_predictions = 0
        mismatches_total_gt = 0
        false_positives_total = 0
        false_negatives_total = 0
        
        missing_gt_boxes_sam = 0
        only_partially_visible = 0
        
        # computation of font coverage
        number_bins = 20
        
        correct_detected_font = np.zeros(number_bins + 1)
        gt_font_coverage = np.zeros(number_bins + 1)
        correct_from_detected_font = np.zeros(number_bins + 1)
        all_detected_font = np.zeros(number_bins + 1)
        
        num_files = len(files)
            
        input = []
        
        files_per_thread = int(num_files/number_threads) + 2
        variables = [SAM_folder_path, file_path, gt_folder_path_labels, num_ids, check_category, id_to_idx, no_border_bboxes, compute_font_coverage, font_mask_path, \
                    number_bins, threshold_gt_bboxes, evaluate_with_sam_output, checkForLabelClass, generate_images, result_folder_path, experiment_name, gt_folder_path_images, images_save, image_size, mapping]
        
        files_cur_count = 0
        files_count_total = 0
        files_cur = []
        for j in range(num_files):
            image_name = files[j]

            files_count_total += 1
            files_cur.append(image_name)
            files_cur_count += 1
            
            if files_cur_count >= files_per_thread or j == num_files-1:
                input.append([files_cur, variables])
                files_cur = []
                files_cur_count = 0
        
        if number_threads > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=number_threads) as ex:
                output = ex.map(evaluate_result_segment, input)
        else:
            output = []
            for i in range(number_threads):
                output.append(evaluate_result_segment(input[i]))
        
        
        for output_elem in output:
            output_str, output_str_detection_fails, mismatches_section, mismatches_gt_section, false_positives_section, false_negatives_section, missing_gt_boxes_sam_section, only_partially_visible_section, correct_detected_font_section, gt_font_coverage_section, correct_from_detected_font_section, all_detected_font_section = output_elem
            mismatches_total_predictions += mismatches_section
            mismatches_total_gt += mismatches_gt_section
            false_positives_total += false_positives_section
            false_negatives_total += false_negatives_section
            
            missing_gt_boxes_sam            +=      missing_gt_boxes_sam_section
            only_partially_visible          +=      only_partially_visible_section
            correct_detected_font[:]        +=      correct_detected_font_section[:]
            gt_font_coverage[:]             +=      gt_font_coverage_section[:]
            correct_from_detected_font[:]   +=      correct_from_detected_font_section[:]
            all_detected_font[:]            +=      all_detected_font_section[:]
            
            file.write(output_str)
            file_detection_errors.write(output_str_detection_fails)

        
        # convert txt outputs in more readable js file format
        only_SAM_results_str = ""
        evaluate_with_sam_output = dataset[5]
        if evaluate_with_sam_output:
            only_SAM_results_str = "_only_SAM_results"

        filename = result_file
        detection_fail_crops_file = filename[:-4] + "_detection_fails.txt"

        file = open(filename, 'r')

        output_file = open(filename[:-4] + ".js","w")
        output_file_fails = open(detection_fail_crops_file[:-4] + ".js","w")

        image_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        image_counter = 0.0
        mean_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mean_image_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        counter = 0.0
        per_class_statistics = []

        # array of [name, [idx]] pairs
        accumulate_statistics = [] #dataset[7]
        accumulated_statistics = []
        for i in range(len(accumulate_statistics)):
            accumulated_statistics.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        output_file.write("var results = {")
        for line in file.readlines():
            # begin of an image
            if line.__contains__("image_name:"):
                line = line.split(" ")
                continue

            # end of an image
            if line.__contains__("-------"):
                # statistics over an image - weighted mean over all classes
                if image_counter > 0.0:
                    if image_values[5] == 0:
                        image_values[0] = 1.0
                    else:
                        image_values[0] = image_values[0] / image_values[5]
                    if image_values[6] == 0:
                        image_values[1] = 1.0
                    else:
                        image_values[1] = image_values[1] / image_values[6]
                    image_values[2] = image_values[2]
                    image_values[3] = image_values[3]
                    if min(image_values[5],image_values[6]) == 0:
                        image_values[4] = 1.0
                    else:
                        image_values[4] = image_values[4] / min(image_values[5],image_values[6])
                    image_values[5] = image_values[5]
                    image_values[6] = image_values[6]
                    if print_per_image_information:
                        output_file.write(f"\"precision\": {image_values[0]},\n")
                        output_file.write(f"\"recall\": {image_values[1]},\n")          
                        output_file.write(f"\"false_positives\": {image_values[2]},\n")          
                        output_file.write(f"\"false_negatives\": {image_values[3]},\n") 
                        output_file.write(f"\"mean_IoU\": {image_values[4]},\n") 
                        output_file.write(f"\"num_predictions\": {image_values[5]},\n") 
                        output_file.write(f"\"num_labels\": {image_values[6]},\n") 
                    image_counter = 0.0
                if print_per_image_information:
                    output_file.write("}")
                    output_file.write("},\n")

                mean_image_values [0] += image_values[0]
                mean_image_values [1] += image_values[1]
                mean_image_values [2] += image_values[2]
                mean_image_values [3] += image_values[3]
                mean_image_values [4] += image_values[4]
                mean_image_values [5] += image_values[5]
                mean_image_values [6] += image_values[6]

                image_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                counter += 1.0
                continue

    
            line = line.split(" ")
            id = int(line[0])

            # append new entries for classes, if not already there
            while len(per_class_statistics) < image_counter+1:
                per_class_statistics.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            precision = float(line[1]) * float(line[6])
            recall = float(line[2]) * float(line[7])
            false_positives = float(line[3])# * float(line[6])
            false_negatives = float(line[4])# * float(line[7])
            num_predictions = float(line[6])
            num_labels = float(line[7])
            meanIoU = float(line[5]) * max(num_predictions-false_positives, num_labels-false_negatives)

            per_class_statistics[int(id)][0] += precision
            per_class_statistics[int(id)][1] += recall
            per_class_statistics[int(id)][2] += false_positives
            per_class_statistics[int(id)][3] += false_negatives
            per_class_statistics[int(id)][4] += meanIoU
            per_class_statistics[int(id)][5] += num_predictions
            per_class_statistics[int(id)][6] += num_labels
            per_class_statistics[int(id)][7] += max(num_predictions-false_positives, num_labels-false_negatives)

            image_values[0] += precision
            image_values[1] += recall
            image_values[2] += false_positives
            image_values[3] += false_negatives
            image_values[4] += meanIoU
            image_values[5] += num_predictions
            image_values[6] += num_labels

            image_counter += 1.0


        if counter != 0.0:
            # append mean values
            output_file.write(f"\"mean_image_values\" : \n") 
            output_file.write("{ \n")
            output_file.write(f"\"precision\": {mean_image_values[0] /counter},\n")
            output_file.write(f"\"recall\": {mean_image_values[1]/counter},\n")          
            output_file.write(f"\"false_positives\": {mean_image_values[2]/counter},\n")          
            output_file.write(f"\"false_negatives\": {mean_image_values[3]/counter},\n") 
            output_file.write(f"\"mean_IoU\": {mean_image_values[4]/counter},\n") 
            output_file.write(f"\"num_predictions\": {mean_image_values[5]/counter},\n") 
            output_file.write(f"\"num_labels\": {mean_image_values[6]/counter},\n") 
            output_file.write("}, \n")

            for class_id in range(len(per_class_statistics)):
                output_file.write(f"\"{class_id}\" : \n") 
                output_file.write("{ \n")
                if per_class_statistics[class_id][5] != 0:
                    output_file.write(f"\"precision\": {per_class_statistics[class_id][0] /per_class_statistics[class_id][5]},\n")
                else:
                    output_file.write(f"\"precision\": {per_class_statistics[class_id][0] /counter},\n")

                if per_class_statistics[class_id][6] != 0:
                    output_file.write(f"\"recall\": {per_class_statistics[class_id][1]/per_class_statistics[class_id][6]},\n")
                else:
                    output_file.write(f"\"recall\": {per_class_statistics[class_id][1]/counter},\n") 
                if per_class_statistics[class_id][5] > 0:         
                    output_file.write(f"\"false_positives\": {per_class_statistics[class_id][2]/per_class_statistics[class_id][5]},\n") 
                else:
                    output_file.write(f"\"false_positives\": {per_class_statistics[class_id][2]},\n")         
                if per_class_statistics[class_id][6] > 0:
                    output_file.write(f"\"false_negatives\": {per_class_statistics[class_id][3]/per_class_statistics[class_id][6]},\n") 
                else:
                    output_file.write(f"\"false_negatives\": {per_class_statistics[class_id][3]},\n")  

                output_file.write(f"\"false_positives_total\": {per_class_statistics[class_id][2]},\n")          
                output_file.write(f"\"false_negatives_total\": {per_class_statistics[class_id][3]},\n") 
                if per_class_statistics[class_id][7] != 0:
                    output_file.write(f"\"mean_IoU\": {per_class_statistics[class_id][4]/per_class_statistics[class_id][7]},\n")
                else:
                    output_file.write(f"\"mean_IoU\": {per_class_statistics[class_id][4]/counter},\n")
                output_file.write(f"\"num_predictions\": {per_class_statistics[class_id][5]},\n") 
                output_file.write(f"\"num_labels\": {per_class_statistics[class_id][6]},\n") 
                output_file.write("}, \n")

                mean_values[2] += per_class_statistics[class_id][2]
                mean_values[3] += per_class_statistics[class_id][3]
                if min(per_class_statistics[class_id][5], per_class_statistics[class_id][6]) > 1.0:
                    right_pairs = max(per_class_statistics[class_id][5]-per_class_statistics[class_id][2], per_class_statistics[class_id][6]-per_class_statistics[class_id][3])
                    
                    mean_values[4] += per_class_statistics[class_id][4] #* right_pairs
                    mean_values[7] += right_pairs
                mean_values[5] += per_class_statistics[class_id][5]
                mean_values[6] += per_class_statistics[class_id][6]
                
                if accumulate_statistics != []:
                    for i in range(len(accumulate_statistics)):
                        if accumulate_statistics[i][1].__contains__(class_id):
                            accumulated_statistics[i][2] += per_class_statistics[class_id][2]
                            accumulated_statistics[i][3] += per_class_statistics[class_id][3]
                            if per_class_statistics[class_id][7] > 1.0:
                                accumulated_statistics[i][4] += per_class_statistics[class_id][4]
                                accumulated_statistics[i][7] += per_class_statistics[class_id][7]
                            accumulated_statistics[i][5] += per_class_statistics[class_id][5]
                            accumulated_statistics[i][6] += per_class_statistics[class_id][6]

            # normalize the mean values by dividing with the total amount of (gt, result) boxes
            output_file.write(f"\"mean_values\" : \n") 
            output_file.write("{ \n")
            if mean_values[5] != 0:
                output_file.write(f"\"precision\": {1.0 - (mean_values[2] /mean_values[5])},\n")
            else:
                output_file.write(f"\"precision\": {-1.0},\n")

            if mean_values[6] != 0:
                output_file.write(f"\"recall\": {1.0 - (mean_values[3] /mean_values[6])},\n")
            else:
                output_file.write(f"\"recall\": {-1.0},\n")          
            output_file.write(f"\"false_positives\": {mean_values[2]/counter},\n")          
            output_file.write(f"\"false_negatives\": {mean_values[3]/counter},\n") 
            output_file.write(f"\"false_positives_total\": {mean_values[2]},\n")          
            output_file.write(f"\"false_negatives_total\": {mean_values[3]},\n") 
            output_file.write(f"\"mismatches_predictions_total\": {mismatches_total_predictions},\n")
            output_file.write(f"\"mismatches_ground_truth_total\": {mismatches_total_gt},\n")
            if not(mean_values[7] == 0.0):
                output_file.write(f"\"mean_IoU\": {mean_values[4]/mean_values[7]},\n")
            else:
                output_file.write(f"\"mean_IoU\": {-1.0},\n")
            output_file.write(f"\"num_predictions\": {mean_values[5]/counter},\n") 
            output_file.write(f"\"num_labels\": {mean_values[6]/counter},\n") 
            output_file.write(f"\"num_predictions_total\": {mean_values[5]},\n") 
            output_file.write(f"\"num_labels_total\": {mean_values[6]},\n") 
            output_file.write(f"\"bboxes ignored by SAM\": {missing_gt_boxes_sam},\n")
            output_file.write(f"\"bboxes only partially visible\": {only_partially_visible}\n")
            output_file.write("}, \n")

            acc_index = 0
            for classes in accumulate_statistics:
                acc_name = classes[0]
                
                output_file.write(f"\"{acc_name}\" : \n") 
                output_file.write("{ \n")
                if accumulated_statistics[acc_index][5] != 0:
                    output_file.write(f"\"precision\": {1.0 - (accumulated_statistics[acc_index][2] /accumulated_statistics[acc_index][5])},\n")
                else:
                    output_file.write(f"\"precision\": {-1.0},\n")

                if accumulated_statistics[acc_index][6] != 0:
                    output_file.write(f"\"recall\": {1.0 - (accumulated_statistics[acc_index][3] /accumulated_statistics[acc_index][6])},\n")
                else:
                    output_file.write(f"\"recall\": {-1.0},\n")        
                output_file.write(f"\"false_positives_total\": {accumulated_statistics[acc_index][2]},\n")          
                output_file.write(f"\"false_negatives_total\": {accumulated_statistics[acc_index][3]},\n")   
                output_file.write(f"\"false_positives_mean\": {accumulated_statistics[acc_index][2]/counter},\n")          
                output_file.write(f"\"false_negatives_mean\": {accumulated_statistics[acc_index][3]/counter},\n") 
                if not(accumulated_statistics[acc_index][7] == 0.0):
                    output_file.write(f"\"mean_IoU\": {accumulated_statistics[acc_index][4]/accumulated_statistics[acc_index][7]},\n")
                else:
                    output_file.write(f"\"mean_IoU\": {-1.0},\n")
                output_file.write(f"\"num_predictions\": {accumulated_statistics[acc_index][5]},\n") 
                output_file.write(f"\"num_labels\": {accumulated_statistics[acc_index][6]},\n") 
                output_file.write("}, \n")
                acc_index += 1
                
            output_file.write(f"\"found / gt bboxes sorted by font intersection\" : \n") 
            output_file.write("{ \n")
            
            output_file.write(f"\"0% found\" : {correct_detected_font[-1]}, \n")
            output_file.write(f"\"0% gt\" : {gt_font_coverage[-1]}, \n")
            
            for i in range(number_bins):
                output_file.write(f"\"{i * 5}% - {((i+1) * 5)}% found\" : {correct_detected_font[i]}, \n")
                output_file.write(f"\"{i * 5}% - {((i+1) * 5)}% gt\" : {gt_font_coverage[i]}, \n")
                
            output_file.write(f"\"0% correct_crops\" : {correct_from_detected_font[-1]}, \n")
            output_file.write(f"\"0% all_found\" : {all_detected_font[-1]}, \n")
            for i in range(number_bins):
                output_file.write(f"\"{i * 5}% - {((i+1) * 5)}% correct_crops\" : {correct_from_detected_font[i]}, \n")
                output_file.write(f"\"{i * 5}% - {((i+1) * 5)}% all_found\" : {all_detected_font[i]}, \n")
            output_file.write("}, \n")
            output_file.write("}")
            
            output_file.close()
        file.close()


        file2 = open(detection_fail_crops_file, 'r')

        output_file_fails.write("var fail_paths = {")
        for line in file2.readlines():

            # begin of an image
            if line.__contains__("image_name:"):
                line = line.split(" ")
                output_file_fails.write(f"\"{line[1][:-1]}\" : \n")                  
                output_file_fails.write("{ \n")
                continue

            if line.__contains__("-------"):
                output_file_fails.write("}, \n")
                continue
            
            id_pos = line.find(" ")
            id = line[0:id_pos]
            line = line[id_pos+1:]
            output_file_fails.write(f"\"{id}\" : ")                  
            output_file_fails.write("{ \n")
                
            # arrays consisting of arrays with bounding box and crop location
            space_pos = line.find("] [") + 1
            false_positives = line[:space_pos]
            false_negatives = line[space_pos+1:]

            output_file_fails.write("\"false_positives\":")
            output_file_fails.write(false_positives)
            output_file_fails.write(",")

            output_file_fails.write("\"false_negatives\":")
            output_file_fails.write(false_negatives)
            output_file_fails.write("}, \n")
        output_file_fails.write("}")
        
        file2.close()
        output_file_fails.close()
        
def compute_only_SAM_detections(dataset):
    """
        computes the number of icons that is covered by any of the SAM proposals and prints it

        Args:
            dataset (list) : consists of:
                dataset[0] : path to dataset
                dataset[1] : path to folder with SAM crops
                dataset[2] : image_size 
    """

    dataset_path = dataset[0]
    gt_folder_path_labels = dataset_path + "/labels/"
    SAM_folder_path = f"{dataset[1]}"
    image_size = dataset[2]

    scale_gt_bboxes = True
    
    threshold_gt_bboxes = 2
    no_border_bboxes = True

    image_names_iter = os.scandir(gt_folder_path_labels) #["Labels_30037548139500"]
    image_names = []
    for image_name in image_names_iter:
        image_names.append(image_name.name[:-4])

    files = list(os.scandir(SAM_folder_path))

    missing_gt_boxes_sam = 0
    only_partially_visible = 0
    number_ground_truth = 0
    mean_IoG_hits = 0.0
    number_files = 0

    for j in range(len(files)):
        if not os.path.isdir(files[j]):
            continue
        
        number_files += 1

        image_name = files[j].name
        
        sam_crops = loadResultFileTxtSAM(os.path.join(SAM_folder_path, image_name+"/bbox_file.txt"), True)
        
        gt_bboxes = load_annotation_file(gt_folder_path_labels + image_name+".txt")
        if gt_bboxes is None:
            continue

        # bounding boxes are converted to (x_min, y_min, x_max, y_max, id)
        gt_bboxes_ignored = []

        for i in range(len(gt_bboxes)):
            tmp = [gt_bboxes[i][0],gt_bboxes[i][1],gt_bboxes[i][2],gt_bboxes[i][3], gt_bboxes[i][4]]
            if scale_gt_bboxes:
                w, h = gt_bboxes[i][2] * image_size[0], gt_bboxes[i][3] * image_size[1]
                tmp[0] = gt_bboxes[i][0] * image_size[0] - w/2.0
                tmp[1] = gt_bboxes[i][1] * image_size[1] - h/2.0
                tmp[2] = tmp[0] + w
                tmp[3] = tmp[1] + h
            else:
                print("The size of the ground truth bounding boxes is not checked. This can lead to failures.")

            tmp[4] = -1

            if no_border_bboxes and (tmp[0] < 0 or tmp[1] < 0 or tmp[2] > image_size[0] - 2 or tmp[3] > image_size[1] -2):
                gt_bboxes_ignored.append(tmp)
                continue

            # remove elements with size 0 and elements outside of the image
            if (abs(tmp[0] - tmp[2]) < threshold_gt_bboxes or abs(tmp[1] - tmp[3]) < threshold_gt_bboxes) or \
                (tmp[0] < 1.0 or tmp[1] < 1.0 or tmp[2] > image_size[0] - 2 or tmp[3] > image_size[1] - 2):
                gt_bboxes_ignored.append(tmp)
            else:
                found_in_sam_results = False
                highest_IoG = 0.0

                number_ground_truth += 1
                for sam_crop in sam_crops:
                    if (checkCenterInBbox(sam_crop, tmp) or checkCenterInBbox(tmp, sam_crop)) and computeIoU(sam_crop, tmp) > 0.4:
                        found_in_sam_results = True
                        IoG = computeIoverGT(sam_crop, tmp)
                        if IoG > highest_IoG:
                            highest_IoG = IoG
                if not found_in_sam_results:
                    gt_bboxes_ignored.append(tmp)
                    missing_gt_boxes_sam += 1
                    continue
                mean_IoG_hits += highest_IoG
                if highest_IoG < 0.8:
                    only_partially_visible += 1
    mean_IoG_hits = mean_IoG_hits / (number_ground_truth - missing_gt_boxes_sam)
    print(SAM_folder_path)
    print("mean Intersection over Ground Truth", mean_IoG_hits)
    print("not recognized by SAM: ", missing_gt_boxes_sam)
    print("only_partially_visible (IoG below 0.8): ", only_partially_visible)
    print("total ground truth labels: ", number_ground_truth)