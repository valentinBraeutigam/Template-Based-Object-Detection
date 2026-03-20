# Template-based Object Detection Using a Foundation Model

# Installation

Please make sure to have conda or miniconda on your machine. Check in the script if the path to your conda installation matches the default path, otherwise adjust it to your conda path. 

The prerequisits will be downloaded by running the script "setup_environment.sh". It will download the InpaintAnything repository and also the model checkpoints provided by the authors of InpaintAnything from Google Drive. The script will also create conda environments containing the required python packages. 

! The checkpoint for SAM3 can not be automatically downloaded as it is not public, but has to be manually downloaded from: "https://huggingface.co/facebook/sam3" after applying for access. !

# Running 

To run the method please go to the script "run.py", select all the steps you want to run by setting them to true, adjust the other settings if needed and the paths in such way that they direct to your data. Then run the code by activating the conda environment "templateMatchingSAM" by running "conda activate templateMatchingSAM" and then start the script with "python run.py". 

# The Code

The code is splitted into the individual parts, which are:
- image segmentation: The input images will be fed into SAM to segment icon/object proposals. 
- inpaint images: Generates font masks by identifying color clusters and searching for the ones that fits the ocr bounding boxes in the selected image best. Then the masks will be used to generate inpainted iamges without the font. 
- generate crops and generate inpainted image crops: This part can be used to generate the crops of the SAM results again and also the crops from the inpainted images. 
- classification: In this section the icon proposals from SAM will be classified by the correlation of the color histograms followed by comparison of the LPIPS or CLIP features. The results will be written to a text file. A non maximum suppression will be applied to reduce results that intersect with each other more than a specified threshold. 
- visualization: In this part the results can be visualized by painting the found bounding boxes and labeling them with the resulting id.
- evaluation: In this part the results will be compared to the ground truth labels and precision, recall, and the mean intersection over union will be returned. 

# The Data

The data is expected in the following folder structure:  
input_data  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- mapping.py  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- dataset  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- my_dataset  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- images  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- some_image.png  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- ...  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- labels   
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- some_image.txt  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- ...  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- templates  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- 0.png  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- 1.png  
 ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎  ‎‏‏‎ ‎‏‏‎ ‏‏‎ \ -- 2.png  

The label files are expected to have the same name as the corresponding image. They consist of a line per ground truth bounding box in the format: "template_id [x_min, y_min, x_max, y_max]".
The template file names are expected to have the numerical names. As there is the possibility that there are multiple examples of one class to detect the file mapping.py contains a dict which maps each template id to a class id (if there is only one example this can be same number for all examples). 

    
