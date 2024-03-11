# codes that read the results from the output files and plot the results on the image and save the image

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source-txt', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', default='results', help='save results to project/name')
    opt = parser.parse_args()
    print(opt)
    Class = ['Pedestrian' ]
    # line format class x1 y1 x2 y2 confidence distance
    # x1 y1 x2 y2 are normalized by image width and height
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    

    files_results_list = os.listdir(opt.source_txt)
    files_results_list.sort()
    for file_name in files_results_list:
        if file_name.endswith(".txt"):
            print(file_name)
            file_name = file_name.split('.')[0]
            image = cv2.imread(os.path.join(opt.source, file_name + '.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            with open(os.path.join(opt.source_txt, file_name + '.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    cl = int(line[0])
                    label_cl = Class[cl]
                    x1 = float(line[1])  
                    y1 = float(line[2])
                    x2 = float(line[3])
                    y2 = float(line[4]) 
                    print(x1, y1, x2, y2)
                    confidence = float(line[5])
                    distance = float(line[6]) * 100
                    if confidence < 0.2:
                        continue
                    ax.text(x1, y1, label_cl + ' ' + str(round(confidence, 2)) + ' ' + str(round(distance, 2)), color='white', fontsize=12)
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
            plt.axis('off')
            #plt.show()
            print(os.path.join(opt.output, file_name + '.png'))
            
            plt.savefig(os.path.join(opt.output, file_name + '.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
    