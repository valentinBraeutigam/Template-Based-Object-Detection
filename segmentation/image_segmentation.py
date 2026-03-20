import torch
import numpy as np
import time
import os
import cv2

np.random.seed(3)

def clamp_bbox(bbox, min_x, min_y, max_x, max_y):
    bbox[0] = max(min(max_x, bbox[0]), min_x)
    bbox[1] = max(min(max_y, bbox[1]), min_y)
    bbox[2] = max(min(max_x, bbox[2]), min_x)
    bbox[3] = max(min(max_y, bbox[3]), min_y)
    return bbox

def hasIntersection(bbox1 , bbox2):
    if bbox1[2] < bbox2[0] or bbox1[3] < bbox2[1]:
        return False
    if bbox1[0] > bbox2[2] or bbox1[1] > bbox2[3]:
        return False
    return True

def areaRectangle(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def computeIoU(bbox1, bbox2):
    if not hasIntersection(bbox1,bbox2):
        return 0.0
    intersection = areaRectangle([max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])])
    union = areaRectangle(bbox1) + areaRectangle(bbox2) - intersection
    return intersection / union

def checkCenterInBbox(bbox1, bbox2):
    if hasIntersection(bbox1,bbox2):
        center = (bbox1[0] + (bbox1[2]-bbox1[0])/2, bbox1[1] + (bbox1[3]-bbox1[1])/2)
        if center[0] >= bbox2[0] and center[0] <= bbox2[2] and center[1] >= bbox2[1] and center[1] <= bbox2[3]:
            return True
    return False

def segment_images_sam2(image_folder, output_folder, image_shape):
    import supervision as sv
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("segment images")
    # load SAM from checkpoint
    checkpoint = "./sam-2_checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # assert os.path.exists(checkpoint), "checkpoint file not found"

    sam = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=True)

    image_width, image_height = image_shape

    print_information = False # print status and runtime information

    # thresholds for filtering predicted segmentations
    stability_score_offset = 0.7
    stability_score_thresh = 0.7
    box_nms_thresh = 0.7
    pred_iou_thresh = 0.5

    save_annotated_image = True

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_crops = True # create image of each crop
    resize_image = False # fit the image to maximum side length of 1024 while keeping ratio

    r = np.min([1024 / image_height, 1024 / image_width]) # scaling factor of the longer side

    points_per_side = 64 # number of points sampled along the x-direction
    offset = 1 / (2 * points_per_side)
    points_side_x = np.linspace(offset, 1 - offset, points_per_side)

    vertical_factor = 1
    if image_width / image_height > 1:
        vertical_factor = image_height/image_width

    if not resize_image: 
        points_side_y = np.linspace(offset, 1 - offset, int(points_per_side * vertical_factor))
    else:
        # scale number of points in y direction to have the same distance as the points in x-direction
        points_side_y = np.linspace(offset, (image_height/1024 * r) - offset, int(points_per_side * vertical_factor))

    xv, yv = np.meshgrid(points_side_x,points_side_y)
    points = np.stack([xv, yv], axis = 0)
    points = [np.stack([xv, yv], axis=-1).reshape(-1, 2)]

    # set thresholds to filter predicted segmentations
    mask_generator = SAM2AutomaticMaskGenerator(sam, stability_score_offset=stability_score_offset, stability_score_thresh=stability_score_thresh, box_nms_thresh=box_nms_thresh, pred_iou_thresh=pred_iou_thresh,points_per_side=None, point_grids=points) #, min_mask_region_area=50.0) # #, points_per_side=None, point_grids=points)
    mask_annotator = sv.MaskAnnotator()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image_name = ""
        image_completed = 0
        for image_name in list(os.scandir(image_folder)):
            image_name = image_name.name[:-4]
            print(image_name)
            
            if print_information:
                print(f"load {image_folder}{image_name}.png")

            image_bgr = cv2.imread(f"{image_folder}{image_name}.png")
            number_crops = 1 # if you like to split the image in multiple parts (1 => use the whole image)
            for img_crop_idx in range(number_crops):
                if print_information:
                    print("SAM2.1 : image_bgr shape" ,image_bgr.shape)

                image_bgr_crop = image_bgr[:,int(img_crop_idx*image_bgr.shape[1]/number_crops):int((img_crop_idx+1) *image_bgr.shape[1]/number_crops) ,:]
                image = cv2.cvtColor(image_bgr_crop, cv2.COLOR_BGR2RGB)

                image_in = image.copy()

                if resize_image:
                    image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
                    if image.shape[0]<1024:
                        image = np.concatenate([image,np.zeros([1024 - image.shape[0], image.shape[1],3],dtype=np.uint8)],axis=0)
                    if image.shape[1]<1024:
                        image = np.concatenate([image, np.zeros([image.shape[0] , 1024 - image.shape[1], 3], dtype=np.uint8)],axis=1)
                

                begin = time.time()
                masks = mask_generator.generate(image)
                detections = sv.Detections.from_sam(masks)

                # save crops to file
                crop_folder = f"{output_folder}{image_name}"

                if not os.path.exists(crop_folder):
                    os.makedirs(crop_folder)

                crop_file = open(crop_folder+"/bbox_file.txt", 'w')
                
                duration = time.time() - begin
                if print_information:
                    print(f"SAM2.1 : segmenting step took {duration} seconds")

                removeDetections = []
                idx = 0
                for i in range(len(detections)):

                    if resize_image:
                        bbox = detections[i].xyxy[0] / r
                    else:
                        bbox = detections[i].xyxy[0]
                        
                    bbox = clamp_bbox(bbox,0,0,image_in.shape[1],image_in.shape[0])
                    
                    # other restrictions to remove outliers:
                    if  detections[i].box_area > 90 * 90 or abs(bbox[0] - bbox[2]) < 2 or abs(bbox[1] - bbox[3]) < 2 or abs(bbox[0] - bbox[2]) > 100 or abs(bbox[1] - bbox[3]) > 120 or detections[i].box_area < 20 * 20:
                        removeDetections.append(i)
                        continue
                    
                    # write bbox information of each crop in text file
                    bbox_str = np.array2string(bbox, precision=3, separator=", ")
                    crop_file.write(f"{idx} {bbox_str} \n")
                    
                    # save image per crop
                    if output_crops:
                        image_crop = image_in[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), ::-1]
                        cv2.imwrite(f"{crop_folder}/crop_{idx}.png", image_crop)
                    idx += 1
                
                # create list with all bounding boxes that fulfill the restriction criteria
                detections_list = []
                for i in range(len(detections)):
                    if i not in removeDetections:
                        detections_list.append(detections[i])

                detections_cleared = detections.merge(detections_list)

                image[:, :, :] = image[:, :, ::-1]
                detections_cleared.class_id = [i for i in range(len(detections_cleared))]
                annotated_image = mask_annotator.annotate(image, detections_cleared)

                duration = time.time() - begin
                if print_information:
                    print(f"SAM2.1 : Processing the image took {duration} seconds in total")

                # save image
                if save_annotated_image:
                    cv2.imwrite(crop_folder + f"/annotated_image_{img_crop_idx}.png", annotated_image)
                image_completed += 1
                print(f"{image_completed} images completed")
    
    # clear cuda memory
    sam = None
    torch.cuda.empty_cache()
    
def segment_images_sam3(image_folder, output_folder, icon_image_path=None):
    import PIL.Image as Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox
    
    
    # Load the model
    checkpoint_path = "./sam-3_checkpoints/sam3.pt"
    model = build_sam3_image_model(bpe_path="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", checkpoint_path=checkpoint_path,load_from_HF=False)
    processor = Sam3Processor(model,confidence_threshold=0.3)

    if output_folder != "" and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if not icon_image_path is None:
        assert os.path.exists(icon_image_path), f"no file found at location given for icon image ({icon_image_path})"
    
    icon_image = Image.open(icon_image_path)  
        
    image_names = [x.name for x in list(os.scandir(image_folder))]
    write_bboxes = True
    print_information = False # print status and runtime information
    text_prompt = "icon"
    
    durations = []

    for i,image_name in enumerate(image_names):
        if not (image_name.__contains__(".png") or image_name.__contains__(".jpg") or image_name.__contains__(".jpeg")):
            print("skipped", image_name, "as the file extension is not recognized as image")
            continue
        
        print(image_name)
        
        image_name_short = image_name[:image_name.rfind(".")]
        crop_folder_path = f"{output_folder}{image_name_short}"

        image_in = Image.open(image_folder + image_name)
        image = image_in.convert('RGB')

        if not icon_image is None:
            # add searched icon to image
            composed_image = Image.new(mode="RGB", size=(image.width + icon_image.width, image.height))
            composed_image.paste(image, (0,0))
            composed_image.paste(icon_image, (image.width,0))
            
            start_time = time.time()
            inference_state = processor.set_image(composed_image)
            if print_information:
                print("SAM3 : feature extraction from the image took ", time.time() - start_time, "seconds")
                
            start_time2 = time.time()
            # Apply text prompt
            inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
            masks, bboxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]
            if print_information:
                print("SAM3 : first prompt took ", time.time() - start_time2, "seconds")
            
            # visual prompt with icon added to image
            box_input_xywh = torch.tensor([image_in.width, 0, icon_image.width, icon_image.height]).view(-1, 4)
            box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

            norm_box_cxcywh = normalize_bbox(box_input_cxcywh, composed_image.width, composed_image.height).flatten().tolist()

            start_time3 = time.time()
            processor.reset_all_prompts(inference_state)
            output = processor.add_geometric_prompt(
                state=inference_state, box=norm_box_cxcywh, label=True
            )
            if print_information:
                print("SAM3 : second prompt took ", time.time() - start_time3, "seconds")
            masks2, bboxes2, scores2 = output["masks"], output["boxes"], output["scores"]
        
        else:
            start_time = time.time()
            inference_state = processor.set_image(composed_image)
            if print_information:
                print("SAM3 : feature extraction from the image took ", time.time() - start_time, "seconds")
            
            # Apply text prompt
            start_time2 = time.time()
            inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
            masks, bboxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]
            if print_information:
                print("SAM3 : feature extraction from the image took ", time.time() - start_time2, "seconds")
        
        boxes_combined = []
        
        if not icon_image is None:
            # remove duplicates if there are multiple prompts
            boxes2_filtered = []
            for j in range(len(bboxes2)):
                if not bboxes2[j][0] > image_in.width and not bboxes2[j][1] > image_in.height:
                    boxes_combined.append(bboxes2[j])
                    boxes2_filtered.append(bboxes2[j])
            for j in range(len(bboxes)):
                found_intersection = False
                for k in range(len(boxes2_filtered)):
                    if computeIoU(bboxes[j], bboxes2[k]) > 0.95:
                        found_intersection = True
                        break
                if not found_intersection:
                    boxes_combined.append(bboxes[j])
            
        else:
            boxes_combined = bboxes
        
        total_time = time.time() - start_time
        durations.append(total_time)
        if print_information:
            print("SAM3 : segmenting the image took ", time.time() - start_time, "seconds")
        
        if write_bboxes:
            if not os.path.exists(crop_folder_path):
                os.makedirs(crop_folder_path)
            crop_file = open(crop_folder_path+"/bbox_file.txt", 'w')
            
            bbox_counter = 0
            for bbox in boxes_combined:
                if bbox[0] > image_in.width or bbox[1] > image_in.height:
                    continue
                crop_file.write(f"{bbox_counter} {bbox.tolist()} \n")
                bbox_counter += 1
        print(f"{i+1} images completed")