# easy ocr
import easyocr
import supervision as sv
import cv2
import numpy as np
import os

def ocr(input_path, output_folder = None, write_output=True):
    """
        computes ocr bounding boxes to an image

        Args:
            input_path (str)    : path to input file or folder
            output_folder (str) : path to output folder where result image is stored
            write_output (bool) : if set to true output image is saved and bounding box coordinates are written to file

        returns:
            list[str]           : names of image names
            list[list[float]]   : bbox coordinates
    """
    if input_path is None:
        assert False, "No argument was given to ocr method. Give either a path to image(s) or an image in cv2/numpy format"

    if not input_path is None and os.path.isdir(input_path):
        image_list = [x.name for x in list(os.scandir(input_path))]
    elif not input_path is None and os.path.isfile(input_path):
        folder, file = os.path.split(input_path)
        image_list = [file]
        input_path = folder
    
    if output_folder != None and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize EasyOCR reader (German language, GPU)
    reader = easyocr.Reader(['de'], gpu=True, model_storage_directory='models/',quantize=False)
    
    bboxes = []
    image_names = []

    for image_name in image_list:
        Image_path = os.path.join(input_path, image_name)

        # Load image using OpenCV
        image_in = cv2.imread(Image_path)

        for index, image in [(0, image_in)]:
            result = reader.readtext(image)
            # Prepare lists for bounding boxes, confidences, class IDs, and labels
            xyxy, confidences, class_ids, label = [], [], [], []

            bboxes_image = []
            
            # Extract data from OCR result
            for detection in result:
                bbox, text, confidence = detection[0], detection[1], detection[2]
            
                # Convert bounding box format
                x_min = int(min([point[0] for point in bbox]))
                y_min = int(min([point[1] for point in bbox]))
                x_max = int(max([point[0] for point in bbox]))
                y_max = int(max([point[1] for point in bbox]))
            
                # Append data to lists
                xyxy.append([x_min, y_min, x_max, y_max])
                label.append(text)
                confidences.append(confidence)
                class_ids.append(0)  
                
                # filter correct bboxes
                bboxes_image.append([x_min, y_min, x_max, y_max])
                
            if len(xyxy) == 0:
                annotated_image = image
            else:
                # Convert to NumPy arrays
                detections = sv.Detections(
                    xyxy=np.array(xyxy),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids)
                )

                # Annotate image with bounding boxes and labels
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator()

                annotated_image = box_annotator.annotate(scene=image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=label)
            
            # save the annotated image
            if write_output:
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                cv2.imwrite(output_folder + image_name[:-4] + ".png", annotated_image)
                
                bboxes.append(bboxes_image)
                image_names.append(image_name)
        
        if write_output:
            bbox_file = open(f"{output_folder}bbox_file.txt", "a")
            for idx, image in enumerate(bboxes):
                bbox_file.write(image_names[idx] + "\n")
                for bbox in image:
                    bbox_file.write(str(bbox) +"\n")
    return image_names, bboxes

def check_color_clusters(color_cluster_images, output_folder):
    '''
        Args:
            color_cluster_images (list[str]) : paths to masked images according to color clusters
            output_folder (str) : path to output folder
            
        returns:
            np.array[int] : number ocr bbox found in mask image
    '''
        
    # traverse all color clusters and evaluate the amount of text bounding boxes detected by OCR in each color
    amount_bboxes_per_color_cluster = []

    for cluster_image in color_cluster_images:
        assert os.path.exists(cluster_image), "No color image to be processed."

        # ocr for all images per color cluster    
        _ , bboxes_ocr  = ocr(input_path=cluster_image, output_folder=output_folder)

        assert len(bboxes_ocr) < 2
        amount_bboxes_per_color_cluster.append(len(bboxes_ocr[0]))
        
    return np.array(amount_bboxes_per_color_cluster)

def below_distance_per_channel(color1, color2, distance):
    """ checks if the difference of at least one of the color channels of color1 and color2 is below distance """
    assert len(color1) == len(color2), "length of the two different colors differs. No comparison possible."
    for i in range(len(color1)):
        if abs(color1[i]-color2[i]) >= distance:
            return False
    return True

def check_color_clusters_area_based(color_cluster_images, output_folder):
    '''
        Args:
            color_cluster_images (list[str]) : paths to masked images according to color clusters
            output_folder (str) : path to output folder
            
        returns:
            list[tuple[float]] : consists of tuples (number ocr bbox, ratio of set pixels inside ocr bboxes)
    '''
    area_ocr_bbox = []
    background_color = [0,0,0]
    counter = 0
    for cluster_image_path in color_cluster_images:      
        assert os.path.exists(cluster_image_path), f"No color image to be processed at cluster path: {cluster_image_path}."
        # ocr for all images per color cluster    
        _ , bboxes_ocr  = ocr(input_path=cluster_image_path, output_folder=output_folder)

        cluster_image = cv2.imread(cluster_image_path)
        
        bbox_negative = cluster_image.copy()
        bbox_positive = cluster_image.copy()

        for bbox in bboxes_ocr[0]:
            bbox_negative[bbox[1]:bbox[3], bbox[0]:bbox[2]][:,:] = background_color

        pixels_ocr = 0
        pixels_total = 0

        for y in range(len(bbox_positive)):
            for x in range(len(bbox_positive[y])):
                if not below_distance_per_channel(bbox_negative[y,x], background_color, 3):
                    bbox_positive[y,x][0:3] = background_color
                    pixels_total += 1
                if not below_distance_per_channel(bbox_positive[y,x], background_color, 3):
                    pixels_ocr += 1
                    pixels_total += 1
        if pixels_total > 0:
            area = float(pixels_ocr) / float(pixels_total)
        else:
            area = 0

        assert len(bboxes_ocr) < 2

        area_ocr_bbox.append((len(bboxes_ocr[0]), area))
        
        counter += 1
    return np.array(area_ocr_bbox)
