# codes that read the results from the output files and plot the results on the image and save the image

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourceA', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--sourceB', type=str, default='inference/lidars', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--outputA', default='results', help='save results to project/name')
    parser.add_argument('--outputB', default='results', help='save results to project/name')
    opt = parser.parse_args()
    print(opt)
    #files_results_list = os.listdir(opt.sourceA)
    files_results_listB = os.listdir(opt.sourceB)
    #files_results_list.sort()
    for file_name in tqdm.tqdm(files_results_listB):
            lidar_name = file_name 
            #file_name = file_name.replace('lidars', 'images')
            #image = cv2.imread(opt.sourceA+file_name)
            lidar = cv2.imread(opt.sourceB+lidar_name) 
            (h, w) = lidar.shape[:2]
            r = 640. / float(w)
            dim = (640, int(h * r))
            #resized = cv2.resize(image, dim)
            resizedL = cv2.resize(lidar, dim)
            #cv2.imwrite(opt.outputA+file_name, resized)
            cv2.imwrite(opt.outputB+lidar_name, resizedL)

