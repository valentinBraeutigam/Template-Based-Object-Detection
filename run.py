import segmentation.image_segmentation as segment
import segmentation.generate_crops as generate_crops

import font_segmentation.run as font_segmentation
import font_segmentation.inpaint_images as inpaint_images

import classification.feature_matching as feature_matching
import classification.clip_model as clip_model
import classification.lpips_precompute as lpips_precompute

import evaluation.result_visualizer as result_visualizer
import evaluation.evaluation as evaluation

import input_data.mapping as mapping

import os

def get_all_paths_visualization(classification_methods, LPIPS, CLIP, output_folder, classification_result_folder):
    paths = []
    for method in classification_methods:
        if method == "CLIP":
            thresholds = CLIP[0]
            mask_types = CLIP[1]
            scaling_type = CLIP[2]
        elif method == "LPIPS":
            thresholds = LPIPS[0]
            mask_types = LPIPS[1]
            scaling_type = "" # LPIPS[2]

        for threshold in thresholds:
            for mask_type in mask_types:
                result_folder_tmp = f"{classification_result_folder}/results_{method}_{mask_type}{scaling_type}_th{threshold}.txt"
                output_folder_tmp = output_folder + f"{method}_th_{threshold}"

                paths.append([result_folder_tmp, output_folder_tmp])
    return paths

if __name__ == "__main__":
    ## Data Configuration ##
    segment_images = True
    generate_masks = True
    generate_inpainted_images = True
    generate_image_crops = True
    generate_inpainted_image_crops = True
    do_classification = True
    do_visualization = True
    do_evaluation = True # requires ground truth annotations
    
    # general settings
    number_threads = 1 # some parts can be parallelized (with value < 2 nothing is parallelized)
    dataset_name = "my_dataset"
    dataset_path = f"input_data/dataset/{dataset_name}"
    image_folder = f"{dataset_path}/images/" # path to original images
    template_folder = f"{dataset_path}/templates/"
    image_shape = [1800, 697] # [image_width, image_height]
        
    # segmentation settings
    sam_model = "SAM3" # "SAM2.1", "SAM3"
    SAM_output_name = f"{sam_model}_test_run_2026_03/"
    sam3_icon_image_path = "input_data/dataset/my_dataset/templates/318_150.png" # either image path or None, only for sam3 (additional visual prompt for icon)

    # font_masking and inpainting settings
    cluster_image_name = [x.name for x in os.scandir(image_folder)][0] # if first image does not contain font, select another one manually
    masking_output_folder = f"output_data/masks/{dataset_name}/"
    inpaint_image_folder = f"output_data/inpainted_images/{dataset_name}/"
        
    # crop generation settings
    crop_output_path = f"output_data/crops/{dataset_name}/"
    crop_output_path_inpaint = f"output_data/crops/{dataset_name}_inpaint/"
    bbox_path = f"output_data/SAM/{SAM_output_name}"
    
    # classfication settings
    classification_methods = ["LPIPS"] # "LPIPS", "CLIP"
    LPIPS_thresholds = [0.7] 
    CLIP_thresholds = [0.8]
    run_name = "test_run"
    mask_type = "no_mask" # either "no_mask" or "inpaint"
    crop_folder_classification = "output_data/SAM/" + SAM_output_name # folder to folder per image with bbox file
    do_nms = True
    overlap_threshold = 0.1

    # visualization settings
    classification_result_folder = f"output_data/feature_matching/{run_name}/"
    output_folder_name = f"{dataset_name}_vis/"

    # change these paths if you changed other configs
    result_paths_to_visualize = [['output_data/feature_matching/test_run/results_LPIPS_no_mask_th0.7_nms_0.1.txt', \
                                  'output_data/feature_matching/result_visualization/my_dataset_vis/LPIPS_th_0.7_nms_0.1/']] # in format [annotation_text_file, output_folder_for_images]

    # evaluation configuration
    number_ids = len(list(os.scandir(template_folder)))
    experiment_folder = f"output_data/feature_matching/{run_name}/"
    experiment_name = f"{run_name}_{mask_type}"
    render_crops = False
    mask_folder_eval = masking_output_folder + "complete_results/"
    bbox_path_eval = f"output_data/SAM/{SAM_output_name}"
    

    # other variables - no settings
    inpaint_image_folder_output = inpaint_image_folder + "resulting_images/"

    # activate environment
    os.system("conda activate templateMatchingSAM")

    # image segmentation
    # images are saved to output_data/SAM/[SAM_output_name]
    if segment_images:
        print("start segmenting images with SAM")
        if sam_model == "SAM2.1":
            segment.segment_images_sam2(image_folder=image_folder, output_folder=bbox_path, image_shape=image_shape)
        elif sam_model == "SAM3":
            assert os.path.exists("sam-3_checkpoints/sam3.pt"), "SAM 3 checkpoint not found. Have you downloaded it and put it in the right place (sam3-checkpoints)?"
            segment.segment_images_sam3(image_folder=image_folder, output_folder=bbox_path, icon_image_path=sam3_icon_image_path)
        print("segmenting images done")
    
    # generate text segmentation masks
    if generate_masks:
        print("start generating text masks")
        font_segmentation.run_font_segmentation(input_image_folder=image_folder, output_folder=masking_output_folder, cluster_image=cluster_image_name,number_threads=number_threads)
        print("generating text masks done")
        
    # generate inpainted images
    if generate_inpainted_images:
        print("start inpainting images")
        inpaint_images.inpaintImages(images_path=image_folder, masks_path=masking_output_folder + "complete_results/", output_path=inpaint_image_folder)
        print("inpainting images done")

    # generate crops
    if generate_image_crops:
        print("start generating crops")
        generate_crops.generate_crops(image_folder, crop_output_path, bbox_path, number_threads)
        print("generating crops done")
    
    # generate crops for inpainted images
    if generate_inpainted_image_crops:
        print("start generating inpainted crops")
        generate_crops.generate_crops(inpaint_image_folder_output, crop_output_path_inpaint, bbox_path, number_threads)
        print("generating inpainted crops done")

    # classification
    if do_classification:
        print("start classification")
        for classification_method in classification_methods:
            templates = {}
            normalization_values = None
            if classification_method == "LPIPS":
                model_information = [lpips_precompute.LPIPS_pre(net='alex')]
                templates = feature_matching.precompute_templates_LPIPS(template_folder, model_information[0])
                thresholds = LPIPS_thresholds
                scaling_type = None
            elif classification_method == "CLIP":
                # compute CLIP features for templates 
                model_information = clip_model.load_CLIP_model()
                templates, normalization_values = clip_model.precompute_templates_CLIP(template_folder, model_information)                              
                thresholds = CLIP_thresholds
                scaling_type = "_minmax"
            else:
                assert False, "unknown classification method selected"
                
            image_folder_classification = image_folder
            
            if mask_type == "inpaint":
                image_folder_classification = inpaint_image_folder_output   
             
            feature_matching.run(run_name=run_name, image_folder=image_folder, thresholds=thresholds, mask_type=mask_type, metric=classification_method, \
                    crop_bbox_folder=crop_folder_classification, template_features=templates, template_folder=template_folder, scaling_type=scaling_type, \
                    normalization_values=normalization_values, model_information=model_information, \
                    do_nms=do_nms, overlap_threshold=overlap_threshold)
        print("classification done")

     # visualization
    if do_visualization:
        print("start visualize results")
        for paths in result_paths_to_visualize:
            result_file_path_vis = paths[0]
            output_folder_path_vis = paths[1]
            result_visualizer.visualize_result_image(result_file_path=result_file_path_vis, image_folder_path=image_folder, template_folder_path=template_folder, output_folder_path=output_folder_path_vis)
        print("visualizing results done")
        
    # evaluation
    if do_evaluation:
        datasets = []
        print("start evaluation")
        for paths in result_paths_to_visualize:
            datasets.append([dataset_path, dataset_name, paths[0], number_ids, bbox_path_eval, True, experiment_folder, render_crops, False, mask_folder_eval, True, True, number_threads, template_folder, image_shape, mapping.mapping_a])
        evaluation.evaluate_results(datasets)
        print("evaluation done")
