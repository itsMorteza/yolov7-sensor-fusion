# codes that read the results from the output files and plot the results on the image and save the image

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-txt', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--split', default='trainped', help='save results to project/name')
    opt = parser.parse_args()
    print(opt)
    Class = ['Pedestrian' ]
    # line format class x1 y1 x2 y2 confidence distance
    # x1 y1 x2 y2 are normalized by image width and height

    files_results_list = os.listdir(opt.source_txt)
    only_pedestrian_list = []
    files_results_list.sort()
    for file_name in files_results_list:
        if file_name.endswith(".txt"):
            
            with open(opt.source_txt + '/'+file_name, 'r') as f:
                    lines = f.readlines()
            content = [line.strip().split(' ') for line in lines]

            CLASSES = ('Car', 'Pedestrian', 'Cyclist')
            #CLASSES = ('Pedestrian')
            cat2label = {cat_id: i for i, cat_id in enumerate(CLASSES)}
            #file_name = file_name.split('.')[0
            for x in content:
                if x[0] == 'Pedestrian':
                    print(file_name)
                    only_pedestrian_list.append(file_name[-10:-4])
                    break
    print(len(only_pedestrian_list))
    with open(opt.split + '.txt', 'w') as f:
        for item in only_pedestrian_list:
            f.write("%s\n" % item)
            