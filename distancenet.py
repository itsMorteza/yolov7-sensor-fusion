import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import yaml
import logging

from tqdm import tqdm

def get_bbox_result(result_path):
    # read the lines that are only have pedestrian which is label id 1
    with open(result_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    content = [x for x in content if x[0] == '1']
    return content

def get_labeles(label_path, imagesize):
 
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]

    #CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    CLASSES = ('Pedestrian')
    SupportedCLASS = ('Van', 'Car', 'Person_sitting', 'Pedestrian' , 'Cyclist')
    cat2label = {cat_id: i for i, cat_id in enumerate(CLASSES)}
    classname = np.array([[cat2label[x[0]]] for x in content if x[0] in CLASSES ])
    bboxes =  np.array([convertbbox2yolo(imagesize,[float(info) for info in x[4:8]]) for x in content if x[0] in CLASSES])
    distance = np.array([[float(x[14])] for x in content if x[0] in CLASSES])
    return np.concatenate((classname, bboxes), 1), distance

def convertbbox2yolo(size, box):
    dw = 1./float(size[0])
    dh = 1./float(size[1])
    x = abs(box[0] + box[2])/2.0
    y = abs(box[1] + box[3])/2.0
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

class DistanceModel(nn.Module):
    def __init__(self, boxnode=4):
        # bbouning box node is 4 and we have 4 layers and 1 output
        super(DistanceModel, self).__init__()
        self.boxnode = boxnode
        self.lin1 = nn.Linear(boxnode, 8)
        self.lin2 = nn.Linear(8, 16)
        self.lin3 = nn.Linear(16, 16)
        self.lin4 = nn.Linear(16, 1)
    def forward(self, x):
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        x = self.lin4(x)
        return x
    
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print(data.shape)
        #print(target.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #print(data.shape)
            #print(target.shape)
            output = model(data)
            test_loss += criterion(output, target).item()
    return test_loss / len(test_loader.dataset)

# get input arguments
def get_args():
    parser = argparse.ArgumentParser('Train a distance model')
    parser.add_argument('--data', type=str, default='data', help='dataset directory')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--save', type=str, default='distance.pt', help='save path')
    parser.add_argument('--log', type=str, default='log.txt', help='log path')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval')
    return parser.parse_args()
def main():


