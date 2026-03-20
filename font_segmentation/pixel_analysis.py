import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize
import cv2
import math
import os, sys
import time

from sklearn import cluster
from skimage.segmentation import flood_fill

class ColorPatch:
    def __init__(self, position=None, xyxy = None, reflectance_spectrum=None, name=None, patch_radius=5):
        self.name = name
        self.xyxy = xyxy
        self.position = position
        self.reflectance_spectrum = reflectance_spectrum
        self.patch_radius = patch_radius

def scatterplot_from_pixels(in_pixels):  
    pixels = in_pixels[::100]  
    x = pixels[:, 0]
    y = pixels[:, 1]
    z = pixels[:, 2]
    colors = pixels*50
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=np.clip(colors, 0, 1), marker='o')

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()    

def ica_from_pixels(in_pixels, show=True):
    pixels = in_pixels
    colors = pixels
    # Perform ICA
    ica = FastICA(n_components=2, random_state=42)
    transformed_pixels = ica.fit_transform(pixels)
    print(transformed_pixels.shape) 

    # Normalize ICA components (L2 norm per row)
    ica_components = ica.components_
    normalized_components = normalize(ica_components, axis=1, norm='l2')

    # Alternative: Min-Max normalization (scale to [0,1])
    min_val = ica_components.min()
    max_val = ica_components.max()
    normalized_components_01 = (ica_components - min_val) / (max_val - min_val)
    
    if show==True:
        # 3D Scatter Plot
        fig = plt.figure(figsize=(12, 5))

        # 2D Projection Scatter
        ax2 = fig.add_subplot(111)
        ax2.scatter(transformed_pixels[:, 0], transformed_pixels[:, 1], c=colors, alpha=0.7, marker='o')
        ax2.set_title("2D Projection (IC1 vs IC2)")

        print("ICA Components:")
        print(ica.components_)
        
        # Print results
        print("Original ICA Components:")
        print(ica_components)
        print("\nL2 Normalized ICA Components:")
        print(normalized_components)
        print("\nMin-Max Normalized ICA Components (0 to 1):")
        print(normalized_components_01)
        
        ax2.set_xlabel(f"IC 1: {normalized_components_01[0]}")
        ax2.set_ylabel(f"IC 2: {normalized_components_01[1]}")
        
        plt.tight_layout()
        plt.show()
    
    reconstructed_pixels = ica.inverse_transform(transformed_pixels)
    return reconstructed_pixels, transformed_pixels, ica_components, normalized_components_01


def mask_cluster(image, color_cluster, required_colors=None, time_file=None): 
    '''
        expects image in [0:255] range, outputs image in [0:255] range
        Args:
            image (np.array) : image opened with opencv
            color_cluster (list(tuple)) : list containg all colors as tuples of the color cluster
            required_colors (list(tuple)) : list of colors that need to be present in the image to be part of the color cluster
            time_file (file) : file where the measured times are saved
    '''
    image_mask = np.zeros(image.shape[0:2], dtype=int)
    
    if not time_file is None:
        last_time = time.time()

    color_cluster = np.array(color_cluster)
    # set all pixels inside of color cluster         
    for i in range(len(color_cluster)):
        if required_colors is None:
            image_mask[np.all(np.logical_and(image > color_cluster[i] - [4,4,4], image < color_cluster[i] + [4,4,4]), axis=2)] = 255
        else:
            image_mask[np.all(np.logical_and(image > color_cluster[i] - [4,4,4], image < color_cluster[i] + [4,4,4]), axis=2)] = 230
    
    if not time_file is None:
        time_file.write(f"set_pixels: {time.time()- last_time}\n")
        print("set pixels", time.time()- last_time)
        last_time = time.time()
    
    if not required_colors is None:
        # get all pixels of required color
        pixel_coordinates = []
        for color_idx in range(len(required_colors)):
            pixel_coordinates_color = np.where(np.all(np.logical_and(image > required_colors[color_idx] - [4,4,4], image < required_colors[color_idx] + [4,4,4]), axis=2))
            pixel_coordinates_color = list(zip(pixel_coordinates_color[0], pixel_coordinates_color[1]))
            pixel_coordinates += pixel_coordinates_color

        if not time_file is None:
            print("get main color pixels", time.time()- last_time)
            time_file.write(f"get_main_color_pixels: {time.time()- last_time}\n")
            last_time = time.time()

        # flood fill
        for coordinates in pixel_coordinates:
            if image_mask[coordinates[0],coordinates[1]] < 254.0:
                image_mask = flood_fill(image_mask, (coordinates[0],coordinates[1]), new_value=255)
        
    image_mask[image_mask < 250] = 0
    if not time_file is None:
        print("flood fill", time.time()- last_time)
        time_file.write(f"flood_fill: {time.time()- last_time}\n")
    return image_mask


def unify_clusters(cluster1, cluster2):
    new_cluster = []
    
    for cl2 in cluster2:
        new_cluster.append(cl2)
    
    for cl1 in cluster1:
        found_cl1 = False
        for cl2 in cluster2:
            if  cl1[0] - 0.01 < cl2[0] < cl1[0] + 0.01 and \
                cl1[1] - 0.01 < cl2[1] < cl1[1] + 0.01 and \
                cl1[2] - 0.01 < cl2[2] < cl1[2] + 0.01: 
                    found_cl1 = True
                    break
        if not found_cl1:
            new_cluster.append(cl1)
    return new_cluster

def intersect_clusters(cluster1, cluster2):
    new_cluster = []
    
    for cl1 in cluster1:
        found_cl1 = False
        for cl2 in cluster2:
            if  cl1[0] - 0.01 < cl2[0] < cl1[0] + 0.01 and \
                cl1[1] - 0.01 < cl2[1] < cl1[1] + 0.01 and \
                cl1[2] - 0.01 < cl2[2] < cl1[2] + 0.01: 
                    found_cl1 = True
                    break
        if found_cl1:
            new_cluster.append(cl1)
    return new_cluster
    
def get_similarities(cluster1, cluster2):
    count = 0
    for cl1 in cluster1:
        found_cl1 = False
        for cl2 in cluster2:
            if  cl1[0] - 0.01 < cl2[0] < cl1[0] + 0.01 and \
                cl1[1] - 0.01 < cl2[1] < cl1[1] + 0.01 and \
                cl1[2] - 0.01 < cl2[2] < cl1[2] + 0.01: 
                    found_cl1 = True
                    break
        if found_cl1:
            count += 1
    return count
            
def check_similarity(clusters1, clusters2):
    combined_clusters = []
    for cluster1 in clusters1:
        best_index = 0
        best_value = 0
        for ind2, cluster2 in enumerate(clusters2):
            value = get_similarities(cluster1, cluster2)
            if value > best_value:
                best_index = ind2
                best_value = value
        combined_clusters.append(intersect_clusters(cluster1, clusters2[best_index])) # unify_clusters(cluster1, clusters2[best_index]))
    
    return combined_clusters

def save_clusters(clusters, file_path):
    file = open(file_path, "w")
    for idx, cluster in enumerate(clusters):
        file.write(str(idx) + " : " + str(cluster) + "\n")
            
def ica_multiple_patches(image, patches:list[ColorPatch], savepath, clustering_type, num_clusters, output_folder, input_folder, image_name="untitled.png"):
    """
        compute ica on multiple image patches and convert color clusters from them
    """
    image = image / 255.0

    np.set_printoptions(precision=2)
    
    all_clusters = []
    
    for i, patch in enumerate(patches):
        mask = np.zeros(image.shape[:2], dtype=np.float16)
        if not patch.xyxy is None:
            pt1 = patch.xyxy[0:2]
            pt2 = patch.xyxy[2:4]
            image_width = len(image[0])
            image_height = len(image)
            
            if pt1[0] < 0:
                pt1[0] = 0
            if pt1[1] < 0:
                pt1[1] = 0
            
            if pt2[0] >= image_width:
                pt2[0] = image_width - 1
                
            if pt2[1] >= image_height:
                pt2[1] = image_height - 1
        
            if pt1[0] >= pt2[0] or pt1[1] >= pt2[1]:
                continue
        else:
            half_width =  patch.patch_radius / math.sqrt(2)
            width = int(round(half_width*2))
            topleft = np.astype((np.array(patch.position) - half_width).round(), int)
            pt1 = list(topleft)
            pt2 = list(topleft+width)
        mask = cv2.rectangle(mask, pt1, pt2, color=1, thickness=-1)
        width = abs(pt2[0]-pt1[0])
        height = abs(pt2[1]-pt1[1])
        patch_pixels = image[mask>0]
        print(pt1, pt2)
        print(width, height, width*height)
        print("shape:", patch_pixels.shape)
        height, width = width, height
        reconstructed_pixels, transformed_pixels, ica_components, normalized_components = ica_from_pixels(patch_pixels, show=False)
        print(ica_components)
        
        colors =  np.clip(patch_pixels,0,1)
        print("len colors unique: ", len(np.unique(np.unique(colors, axis=0),axis=0)))
        print("colors: ", np.unique(np.unique(colors, axis=0),axis=0))
        labeled_data = clustering(transformed_pixels[:,0:2], clustering_type, num_clusters)
        np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)
        labeled_data = labeled_data.astype(float)
        max = np.max(labeled_data)
        labeled_data /= max
        clusters = []
        cluster_values = np.unique(labeled_data)
        for x in range(len(labeled_data)):
            for index, x_2 in enumerate(cluster_values):
                if labeled_data[x] == x_2:
                    while not len(clusters) > index:
                        clusters.append([])
                    found_similar = False
                    for index_sim in range(len(clusters[index])):
                        if clusters[index][index_sim][0] - 0.01 < colors[x][0] < clusters[index][index_sim][0] + 0.01 and \
                           clusters[index][index_sim][1] - 0.01 < colors[x][1] < clusters[index][index_sim][1] + 0.01 and \
                           clusters[index][index_sim][2] - 0.01 < colors[x][2] < clusters[index][index_sim][2] + 0.01: 
                               found_similar = True
                               break
                    if not found_similar:
                        clusters[index].append((int(colors[x][0]*255.0),int(colors[x][1]*255.0),int(colors[x][2]*255.0)))
                    break
        for index in range(len(clusters)):
            clusters[index] = [[x[0], x[1], x[2]] for x in list(set(clusters[index]))]
        all_clusters.append(clusters)
    
    print("start matching clusters")
    clusters_combined = all_clusters[0]
    for i in range(1,len(patches)):    
        clusters_combined = check_similarity(clusters_combined, all_clusters[i])

    print("matching clusters finished")
    #print(len(clusters_combined))
    print("The clusters have the lengths: ")
    for i in range(len(clusters_combined)):
        print(len(clusters_combined[i]))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for cluster_idx, cluster in enumerate(clusters_combined):
        image_new = cv2.imread(os.path.join(input_folder, image_name))
        print("ica multiple patches min-max files:", np.min(image_new), np.max(image_new))
        assert not image_new is None, "loading the image went wrong"
        masked_image = mask_cluster(image_new, cluster)
        assert not masked_image is None, "masking image with color cluster has no result"
        cv2.imwrite(output_folder + image_name[:-4] + f"_cluster_{cluster_idx}.png", masked_image) 
        print(f"generated mask image with each color cluster ({cluster_idx + 1}/{len(clusters_combined)})")

        color_cluster_image = np.zeros(shape=(len(cluster) * 5, 20,3), dtype=float)
        if len(cluster) > 0:
            for idx, color in enumerate(cluster):
                color_cluster_image[idx*5:(idx +1)*5, :] = color
            cv2.imwrite(f"font_segmentation/color_cluster_image_{cluster_idx}.png", color_cluster_image)
            assert os.path.exists(f"font_segmentation/color_cluster_image_{cluster_idx}.png")
        
    # get correct cluster
    save_clusters(clusters_combined, output_folder + image_name[:-4] + "_computed_clusters.txt")
    
    return clusters_combined
    
def clustering(data, clustering_type, num_clusters):
    if clustering_type == "Birch":
        clustering = cluster.Birch(n_clusters=num_clusters)#mixture.GaussianMixture(n_components=2, covariance_type="full", random_state=0) # cluster. AgglomerativeClustering(n_clusters=5, linkage="ward") # DBSCAN(eps=0.1, min_samples=4) # KMeans(n_clusters=2, random_state=0, n_init="auto") #SpectralClustering(n_clusters=2, assign_labels='discretize',random_state=0)
    elif clustering_type == "Agglomerative":
        clustering = cluster.AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
    elif clustering_type == "DBSCAN":
        clustering = cluster.DBSCAN(eps=0.1, min_samples=10)
    elif clustering_type == "KMeans":
        clustering = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
    elif clustering_type == "Spectral":
        clustering = cluster.SpectralClustering(n_clusters=num_clusters, assign_labels='discretize',random_state=0)
    labels = clustering.fit_predict(data)
    return labels


def run(image_names, bboxes, clustering_type, num_clusters, input_folder, output_folder):
    """
        For each image given compute the color clusters from the given bboxes and return for each image the computed color clusters.
    """
    clusters_per_image = []
    
    assert clustering_type in ["Birch", "Agglomerative", "DBSCAN", "KMEANS", "Spectral"], "The clustering type has to be one of the implemented ones"
    
    for image_idx, image_name in enumerate(image_names):
        print(image_name)
        if not os.path.exists(os.path.join(input_folder + image_name)):
            print(os.path.join(input_folder + image_name) + " ")
            assert False, "Path not found"
        image = cv2.imread(os.path.join(input_folder + image_name))
        patch_list = []
        for bbox in bboxes[image_idx]:
            patch_list.append(ColorPatch(xyxy=bbox))
            
        clusters = ica_multiple_patches(image, patches=patch_list, savepath=output_folder + image_name[:-4] + "_ica.png", image_name = image_name, clustering_type="Birch", num_clusters=num_clusters, input_folder=input_folder, output_folder=output_folder)  
        clusters_per_image.append(clusters)
    
    return clusters_per_image

def compute_counts_per_color(image_path, bboxes, color_cluster):
    """
    Args:
        image_path (str): path to input image
        bboxes (list[float]): bounding boxes of the regions where to count the pixels
        color_cluster (np.array[np.array[float]] | list[np.array[float]]): list of color clusters

    Returns:
        np.array[int]: number of pixels with each color from color_cluster
    """
    
    image = cv2.imread(image_path)
    
    color_cluster = np.array(color_cluster,dtype=int)
    
    pixel_counts = np.zeros(len(color_cluster),dtype=int)
    
    for bbox in bboxes:
        print(bbox)
        for i in range(len(color_cluster)):            
            # image is given with [height, width, channels]
            pixels = np.array(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            
            # filter pixels belonging to the color cluster with a threshold of +/- 2
            filtered_pixels = np.where(np.all(np.logical_and(pixels > color_cluster[i] - [2,2,2], pixels < color_cluster[i] + [2,2,2]), axis=2))

            num_pixels = len(filtered_pixels[0])
            
            pixel_counts[i] += num_pixels
    return pixel_counts