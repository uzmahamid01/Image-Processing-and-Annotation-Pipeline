#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml

labels_dict = {7:'road',
               8:'sidewalk', 
               11:'building', 
               12:'wall',
               13:'fence',
               17:'pole',
               19:'traffic light',
               20:'traffic sign',
               21:'vegetation', 
               22:'terrain',
               23:'sky', 
               24:'person', 
               25:'rider', 
               26:'car', 
               27:'truck', 
               28:'bus', 
               31:'train', 
               32:'motorcycle',
               33:'bicycle'}

print(yaml.dump(labels_dict, sort_keys=False, default_flow_style=False))


# In[2]:


import cv2

def show_image(img_np_array):
    img_with_label = Image.fromarray(img_np_array.astype('uint8'))
    print(img_with_label)
    plt.imshow(img_with_label, cmap='gray')
    plt.show()


# In[3]:


class StructBoundaryDetails:
    def __init__(self, instance, total_points, boundary_polygon, class_label, class_label_name, img_height, img_width):
        self.instance = instance
        self.total_points = total_points
        self.boundary_polygon = boundary_polygon
        self.class_label = class_label
        self.class_label_name = class_label_name
        self.img_height = img_height
        self.img_width = img_width


# In[4]:


from typing import List
import json 

class ValueClass:
    def __init__(self, points: List,  polygonlabels: List):  
        self.points = points 
        self.polygonlabels = polygonlabels
        
        
        
class ResultClass:
    def __init__(self, original_width:int, original_height:int, image_rotation:int, 
                 value: ValueClass, id:str, type: str, 
                 origin:str = "manual", from_name:str = "label", to_name:str = "image" ):  
        self.original_width = original_width
        self.original_height = original_height
        self.image_rotation = image_rotation
        self.value = value
        self.id = id
        self.from_name = from_name
        self.to_name = to_name
        self.type = type
        self.origin = origin
    
def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


# In[5]:


# Annotation Class to create the Annotation instance for the json format

class AnnotationClass:
    def __init__(self, was_cancelled:bool , ground_truth:bool, created_at:str, 
                 updated_at: str, lead_time:float, prediction: dict, result: List[ResultClass],
                 result_count:int, task:int, parent_prediction= None, parent_annotation=None ):  
        self.was_cancelled = was_cancelled
        self.ground_truth = ground_truth
        self.created_at = created_at
        self.updated_at = updated_at
        self.lead_time = lead_time
        self.result = result
        self.prediction = {}
        self.task = task
        self.parent_prediction = parent_prediction
        self.parent_annotation = parent_annotation


# In[6]:


class DataImage:
    def __init__(self, label_studio_path: str, image_name:str):
        self.image = label_studio_path + image_name
        
    
    
class LabelStudioFormat:
    def __init__(self, id: int, annotations: List[AnnotationClass], file_upload: str, 
                 drafts: List, predictions: List, data: DataImage, meta:dict, 
                 created_at: str, updated_at:str, project:int):  
        self.id = id
        self.annotations = annotations
        self.file_upload = file_upload
        self.drafts = drafts
        self.predictions = predictions
        self.data = data
        self.meta = meta
        self.created_at = created_at
        self.updated_at = updated_at
        self.project = project


# In[7]:


def get_selected_label_boundaries(label_type_to_show, npdata, height, width, min_area = 100, epsilon_per=0.00001):
    cur_id = label_type_to_show
    if cur_id in labels_dict:
        print(f"Checking the object id: {cur_id} and name is: {labels_dict[cur_id]}")
        print(f"Label {cur_id} s present in KITTI-2015")
    else:
        print("Ignoring the label", str(cur_id) , "since it is not present in KITTI-2015")
        return [], []
        
    
    # Assign pixel 255 for the object class, and 0 for background
    curr_image_label = np.where(npdata == cur_id, 255, 0)
    # showing only the current object types
    show_image(curr_image_label)

    curr_image_label_draw = curr_image_label
    #print(curr_image_label_draw)
    contours,_= cv2.findContours(curr_image_label, cv2.RETR_CCOMP,
                                cv2.CHAIN_APPROX_SIMPLE)

    print(f"Total number of contours : {len(contours)}")
    acutal_contour_count= 0;
    boundary_details_all = []
    value_points_list_all = []
    
    curr_image_label_draw_boundary_only = np.zeros((height, width, 3), dtype=np.uint8)
    #print(curr_image_label_draw_boundary_only)
    for cnt in contours:
        curr_image_label_draw_temp = curr_image_label_draw
        area = cv2.contourArea(cnt)
        # Shortlisting the regions based on there area.
        
        if area < min_area:
            continue

        acutal_contour_count += 1;

        print(f"Area is: {area}")
        epsilon = epsilon_per * cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(cnt, epsilon, True)
        larger_cont=0
        boundary_details = StructBoundaryDetails(
            larger_cont,
            len(approx), 
            approx,
            cur_id,
            labels_dict[cur_id],
            height, 
            width
        )
        
        boundary_details_all.append(boundary_details)
        point_value_list = approx.reshape(approx.shape[0], (approx.shape[1]*approx.shape[2]))
        width_normalization_factor = width/100
        height_normalization_factor = height/100
        normalization_array = np.array([width_normalization_factor, height_normalization_factor])
        point_value_list_normalized = np.divide(point_value_list, normalization_array).tolist()
        value_points_list_all.append(point_value_list_normalized)
       
    print(f"Number of contours used are:  {acutal_contour_count}")
    return boundary_details_all, value_points_list_all


# def show_contours():
    # Plotting only
#     plt.savefig("/Users/master-node/Desktop/Research/data_semantics/training_debug/semantic/new_img.png") 
#     cv2.drawContours(curr_image_label_draw_boundary_only, [approx], 0, (255,0,0), 2)
#     plt.imshow(curr_image_label_draw_boundary_only)
#     plt.show()
#     plt.savefig("/Users/master-node/Desktop/Research/data_semantics/training_debug/semantic/nnew_img.png")
    
# Showing the image along with outlined vegetation


# In[8]:


import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import asarray
from datetime import datetime
import uuid

# LABEL STUDIO INFORMATION
project_id = 6
label_studio_upload_prefix = "/data/upload/" + str(project_id) + "/"
last_label_task_id = 1217 + 1
# 000047_10.png"

def process_image(image_name: str):
    img = PIL.Image.open(image_name)
    img1 = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_name, cv2.IMREAD_COLOR)
    print(img1)
    # fetching the dimensions
    width, height = img.size
    sz = img.size  
    # displaying the dimensions
    print(str(width) + " x " + str(height))
    print(type(img))
    print(img.format)
    print(img.mode)
    print(img.size)
    
    npdata = asarray(img) 
    unique_ids = np.unique(npdata)
    print(unique_ids)
    
    result_instance_list = []
    
    # Data Image Class
    data_image = DataImage(label_studio_upload_prefix, "4b5d2856-000047_10.png")
        
    for class_id in unique_ids:
        boundary_details_all, point_value_list_all = get_selected_label_boundaries(class_id, npdata, height, width)
        
        for point_value_list in point_value_list_all:
            # Get the polygon points
            point_value_instance =  ValueClass(point_value_list, [labels_dict[class_id]])
            
            # Create Result Class
            rotation = 0
            polygon_id = str(uuid.uuid1())
            result_instance = ResultClass(width, height, rotation, point_value_instance, polygon_id, "polygonlabels")
            result_instance_list.append(result_instance)

    date_time_now = datetime.utcnow().isoformat() + "Z"
    annotation_instance = AnnotationClass(False, False, date_time_now, date_time_now, 0, {}, 
                                          result_instance_list, 0, last_label_task_id)
    final_label_studio_format = final_label_studio = LabelStudioFormat(last_label_task_id, [annotation_instance], 
                                                                       "4b5d2856-000047_10.png", [], [], 
                                                                       data_image, {}, date_time_now, date_time_now, project_id)
    json_file_contents = to_dict(final_label_studio_format)
    out_file = open("myfile.json", "w")
   
    json.dump(json_file_contents, out_file, indent = 4)
   
    out_file.close()
        


# In[11]:


#image_path = "/Users/master-node/Desktop/Research/data_semantics/training_debug/semantic/000047_10.png"
image_path = "/Users/master-node/Desktop/Prep/000047_10.png"
process_image(image_path)


# In[ ]:




