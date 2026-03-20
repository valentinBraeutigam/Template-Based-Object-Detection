import sys
sys.path.append("./font_segmentation/")
import ocr_clustering
import pixel_analysis
import numpy as np
import cv2 
import os
import time

def convolution_1(image, kernel_size):
    """
        own convolution variant

        Args:
            image (np.array[int])   : input image with shape (height, width, channels), only first channel is used
            kernel_size (int)       : size of the kernel that is taken into account
        return:
            np.array[int] : array of shape (height, width) containing the number of elements per kernel position
    """
    output = np.zeros(image.shape[0:2])

    for y in range(len(image)):
        y_min, y_max = int(max(y-np.floor(kernel_size/2),0)), int(min(y+np.floor(kernel_size/2),len(image)-1))
        for x in range(len(image[y])):
            if image[y,x,0] == 0:
                continue
            x_min, x_max = int(max(x-np.floor(kernel_size/2),0)), int(min(x+np.floor(kernel_size/2),len(image[0])-1))
            output[y,x] = np.sum(image[y_min:y_max, x_min:x_max,0])
    return output

def process_image(input):
    """
    Args:
        input (list): 
            input_image_folder (list[str])          - folder with input images
            output_folder      (str)                - path to output folder
            image_name         (str)                - image file name
            color_cluster      (list[list[float]])  - color cluster array 
            required_colors    (list[list[float]])  - array containing required colors
            time_file          (file)               - txt file for method is written down
    """
    input_image_folder, output_folder, image_name, color_cluster, required_colors, time_file = input
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(input_image_folder + image_name)

    time_begin = time.time()

    masked_image = pixel_analysis.mask_cluster(image, color_cluster, required_colors, time_file)

    if not os.path.exists(output_folder + "/image_bit_masks/"):
        os.makedirs(output_folder + "/image_bit_masks/")

    cv2.imwrite(output_folder + "/image_bit_masks/" + image_name, masked_image*255) 
    
    masked_image = cv2.imread(output_folder + "/image_bit_masks/" + image_name)

    if len(masked_image.shape) < 3:
        masked_image = np.expand_dims(masked_image, axis=2)
        masked_image = np.repeat(masked_image, 3, axis=2)
    print("masked image shape:", masked_image.shape)
    
    background_color = [0.0,0.0,0.0]
    background = np.all(np.logical_and(masked_image > np.array(background_color) - np.array([0.1,0.1,0.1]), masked_image < np.array(background_color) + np.array([0.1,0.1,0.1])), axis=2)

    if np.sum(background) > 0:
        masked_image[background] = np.array([0,0,0])
        masked_image[np.logical_not(background)] = np.array([1,1,1])
        
    # postprocessing
    if time_file != None:
        time_begin = time.time()
    
    kernel_size = 5
    threshold = 3
    neighbors = convolution_1(masked_image, kernel_size)
    print("neighbors shape", neighbors.shape)
    masked_image[neighbors < threshold] = np.array([0,0,0])
    masked_image[neighbors >= threshold] = np.array([1,1,1])
    
    kernel_size = 21
    threshold = 31
    neighbors = convolution_1(masked_image, kernel_size)
    masked_image[neighbors < threshold] = np.array([0,0,0])
    masked_image[neighbors >= threshold] = np.array([1,1,1])
    
    kernel_size = 5
    threshold = 1
    # dilation
    kernel = np.ones((3, 3), np.uint8) 
    masked_image = cv2.dilate(masked_image, kernel, iterations=3)

    img_dilation = masked_image

    if time_file != None:
        time_file.write(f"time_postprocessing: {time.time() - time_begin}\n")

    if not os.path.exists(output_folder +f"complete_results/"):
        os.makedirs(output_folder +f"complete_results/")
    cv2.imwrite(output_folder +f"complete_results/{image_name}", img_dilation * 255)
    print("postprocess: current file amount:", len(list(os.scandir(output_folder +f"complete_results/"))))

def run_font_segmentation(input_image_folder, output_folder, cluster_image, number_threads, clustering_type="Birch", num_clusters=3, num_bboxes=5, measure_time=False):
    """
        run the complete font segmentation process. 
        
        Args:
            input_image_folder (str) : path to input image folder
            output_folder (str) : path to output folder
            cluster_image (str) : name of the image used to compute the color clusters
            number_threads (int) : number of threads to use for the parallel parts
            clustering_type = "Birch" (str) : which clustering algorithm to use
            num_clusters = 3 (int) : number of clusters to compute
            num_bboxes = 5 (int) : number of ocr bounding boxes to use for computing the color clusters
            measure_time = False (bbol) : whether to measure the time or not
    """
    assert num_bboxes > 0 and num_clusters > 0, "num_bboxes and num_clusters have to be non-negative"

    if not input_image_folder[-1] == "/":
        assert False, "input folder has to end with /"
    cluster_image_path = input_image_folder + cluster_image
    run_images = [x.name for x in os.scandir(input_image_folder)] # if x.name != "Labels_10925072085500.png"]
    
    print("do ocr")
    # do ocr and get bboxes
    image_names, bboxes = ocr_clustering.ocr(input_path = cluster_image_path, output_folder=output_folder + "ocr/")
    
    bboxes = bboxes[0]
    bboxes.sort(key=lambda x: abs(x[2] - x[0]) * abs(x[3] - x[1]), reverse=True)
    bboxes = bboxes[:num_bboxes]
    
    print("do ICA and clustering")
    # ica + color clustering
    clusters = pixel_analysis.run([cluster_image], [bboxes], clustering_type, num_clusters=num_clusters, input_folder=input_image_folder, output_folder=output_folder + "ica/")
    
    # get best matching cluster by found boxes ocr
    color_cluster_names = []
    for i in range(num_clusters):
        color_cluster_names.append(output_folder + "ica/" + cluster_image[:-4] + f"_cluster_{i}.png")
    
    print("check color clusters: ")
    print(color_cluster_names)

    # compute the number of ocr bboxes found in per mask generated from the color clusters
    ocr_results_segmentations = ocr_clustering.check_color_clusters_area_based(color_cluster_names, output_folder=output_folder+ "ocr/color_cluster/")
    print(ocr_results_segmentations)
    
    # get best cluster by multiplying the number of found ocr bboxes with the ratio of color from that cluster inside ocr bounding boxes
    best_cluster_idx = np.argmax(ocr_results_segmentations[:, 0] * ocr_results_segmentations[:,1])
    print("found clusters")
    print(f"best cluster index: {best_cluster_idx}")
    print("clusters:" , clusters)
    color_cluster = clusters[0][best_cluster_idx]
    
    if not os.path.exists(output_folder + "masked_images/"):
        os.makedirs(output_folder + "masked_images/")
    
    color_cluster_image = np.zeros(shape=(len(color_cluster) * 5, len(color_cluster) * 5,3), dtype=float)
    
    for idx, color in enumerate(color_cluster):
        color_cluster_image[idx*5:(idx +1)*5, :] = color
    
    cv2.imwrite(output_folder + "/masked_images/color_cluster_image.png", color_cluster_image*255)    

    color_cluster = []

    for idx in range(int(len(color_cluster_image)/5)):
        color_cluster.append(color_cluster_image[idx*5 + 2, 1])

    # compute the two colors (if there is more than one color in the cluster) appearing the most in the ocr bounding boxes
    colored_pixels = pixel_analysis.compute_counts_per_color(image_path=cluster_image_path, bboxes=bboxes, color_cluster=color_cluster)
    colored_pixels = [[color_cluster[x], colored_pixels[x]] for x in range(len(colored_pixels))]
    colored_pixels.sort(key=lambda x: -x[1])
    required_colors = [colored_pixels[0][0]]
    if len(colored_pixels) > 1:
        required_colors.append(colored_pixels[1][0])

    if measure_time:
        time_file_name = "time_file.txt"
        if os.path.exists(time_file_name):
            assert False
        time_file = open(time_file_name, "w")
    else:
        time_file = None

    required_colors = np.array(required_colors)
    required_colors.astype(int)
    color_cluster = np.array(color_cluster)
    color_cluster.astype(int)


    # process the images by creating bit masks and dilate them
    import concurrent.futures
    input_processing = [[input_image_folder, output_folder, x, color_cluster, required_colors, time_file] for x in run_images]

    if number_threads > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_threads) as ex:
            ex.map(process_image, input_processing)
    else:
        for input_data in input_processing:
            process_image(input_data)

        
        
    