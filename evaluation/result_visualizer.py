from PIL import ImageFont, Image, ImageDraw
import os

text_colors = [(50,50,50),(0,0,255),(255,0,0),(255,0,255),(255,255,0)]

def draw_bbox(new_image, bbox, label="", color_nr=0):  
    # draw bboxes
    new_image.rectangle(bbox, outline=text_colors[color_nr], width=3)
    
    font = ImageFont.load_default(size=14.0)
    font_length = 10 * len(label)
    if bbox[1] > 20:
        new_image.rectangle((bbox[0],bbox[1]-15,bbox[0]+font_length, bbox[1]), fill=(255,255,255))
        new_image.text((bbox[0], bbox[1]), label, fill=text_colors[color_nr], font=font, anchor='lb')
    else:
        new_image.rectangle((bbox[0],bbox[3]+1, bbox[0]+font_length, bbox[3]+16), fill=(255,255,255))
        new_image.text((bbox[0], bbox[3]+16), label, fill=text_colors[color_nr], font=font, anchor='lb')
    
def add_bbox_below(new_image, cur_image, template_images, current_templates):
    new_image_draw = ImageDraw.Draw(new_image)
    current_templates.sort()
    cur_width = 5
    cur_height = 700
    font = ImageFont.load_default(size=14.0)
    new_image.paste(cur_image, (0,0))
    for template in current_templates:
        if template < 0:
            continue
        template = str(template)
        new_image.paste(template_images[template], (cur_width, cur_height))
        new_image_draw.text((cur_width, template_images[template].height + cur_height + 5), template, fill=(200,200,200), font=font, anchor='lt')
        cur_width += template_images[template].width

def visualize_result_image(result_file_path, image_folder_path, template_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # crops from one image should be consecutive => image is saved and next image is loaded when name changes
    cur_image_name = ""
    cur_image = None
    current_templates = []
    
    template_images = {}
    for template in os.scandir(template_folder_path):
        template_name = template.name
        template_images[template_name[:-4]] = Image.open(template_folder_path + template_name)

    # open Image 
    for line in open(result_file_path, "r"):
        # expects result file in format: crop_path [bbox] category_id metric_value
        crop_path = line[:line.find(" ")]
        line = line[line.find(" ")+1:]

        image_name = crop_path[:crop_path.rfind("/")]
        image_name = image_name[image_name.rfind("/")+1:]
        image_name = image_name[:image_name.rfind("_")]
        
        print(image_name)
        bbox = [float(x) for x in line[:line.find("]")].replace("[","").split(", ")]

        id = int(line[line.find("]")+1:line.rfind(" ")].replace(" ",""))

        # if image is switching
        if cur_image_name != image_name:
            # save current image
            if not cur_image is None:
                new_image = Image.new("RGB", (cur_image.width, cur_image.height + 90))
                add_bbox_below(new_image, cur_image, template_images, current_templates)
                new_image.save(output_folder_path + cur_image_name + ".png")
            
            # create new image
            cur_image = Image.open(image_folder_path + image_name + ".png")
            cur_image_draw = ImageDraw.Draw(cur_image)
            cur_image_name = image_name
            current_templates = []
        
        if id < 0:
            continue
        
        draw_bbox(cur_image_draw, bbox, label=str(id))
        if not id in current_templates:
            current_templates.append(id)
    
    # save last image
    if not cur_image is None:
        new_image = Image.new("RGB", (cur_image.width, cur_image.height + 90))
        add_bbox_below(new_image, cur_image, template_images, current_templates)
        new_image.save(output_folder_path + cur_image_name + ".png")