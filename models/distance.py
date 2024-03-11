import numpy as np
import random
import torch
import torch.nn as nn
import cv2
from models.common import Conv, DWConv
from utils.google_utils import attempt_download

# function that get bounding boxes from the model output and return the distance of object from the camera
def DistanceGen(bbox, img_width=640, img_height=360, human_height=165, human_length=45, focal_length=35):
    # bbox = [x1, y1, x2, y2]
    # human_height = height of the human in meters
    # human_length = length of the human in meters
    # focal_length = focal length of the camera in pixels
    # img_width = width of the image in pixels
    # img_height = height of the image in pixels
    # return distance of the object from the camera in meters

    # get the center of the bounding box
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    # get the width and height of the bounding box
    bbox_width =  bbox[2] - bbox[0] 
    bbox_height =  bbox[3] - bbox[1] 
    print(bbox_width, bbox_height)
    
    # get the distance of the object from the camera
    distance_height = (human_height * focal_length) / bbox_height
    distance_length = (human_length * focal_length) / bbox_width
    distance_height = distance_height / 100
    distance_length = distance_length / 100
    distance_height = distance_height * img_height
    distance_length = distance_length * img_width
    #distance = (distance_height + distance_length) / 2
    distance =  distance_length ** 2 + distance_height ** 2
    distance = distance ** 0.5
    return distance


def DistanceGen2(bbox, path=None, img_width=640, img_height=360, human_height=165, human_length=45, focal_length=35):
    # bbox = [x1, y1, x2, y2]
    # human_height = height of the human in meters
    # human_length = length of the human in meters
    # focal_length = focal length of the camera in pixels
    # img_width = width of the image in pixels
    # img_height = height of the image in pixels
    # return distance of the object from the camera in meters

    # get the center of the bounding box
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    # get the width and height of the bounding box
    bbox_width =  bbox[2] - bbox[0] 
    bbox_height =  bbox[3] - bbox[1] 
    #print(bbox_width, bbox_height)
    
    # get the distance of the object from the camera
    distance_height = (human_height * focal_length) / bbox_height
    distance_length = (human_length * focal_length) / bbox_width
    distance_height = distance_height / 100
    distance_length = distance_length / 100
    distance_height = distance_height * img_height
    distance_length = distance_length * img_width
    #distance = (distance_height + distance_length) / 2
    distance =  distance_length ** 2 + distance_height ** 2
    distance = distance ** 0.5

    number_of_points = 0
    # read the radar image
    import PIL
    
    rpath= path.replace('image', 'Test')
    #Radar_img = PIL.Image.open(rpath)
    Radar_img = cv2.imread(rpath)
    depthi = 0
    
    #Radar_img = cv2.cvtColor(Radar_img, cv2.COLOR_BGR2GRAY)
    # get the number of pixels in the bounding box area in the radar image that are not zero
    imr = Radar_img
    imr = np.zeros((imr.shape[0], imr.shape[1], 3))
    for i in range(int(bbox[0]), int(bbox[2])):
        for j in range(int(bbox[1]), int(bbox[3])):
            if Radar_img[j][i][0] != 0:
                imr[j][i]  = Radar_img[j][i] 
                depthi += Radar_img[j][i][0]
                number_of_points += 1
    # remove the radar points that are not in the bounding box area and make other points zero
    

    if number_of_points == 0:
        depth = 0
        imr = np.zeros((imr.shape[0], imr.shape[1], 3))
    else:
        depth = (float(depthi) / (number_of_points * 255)) * 20

    return distance , number_of_points , depth , imr