'''
Original src for 'Object Detection' : https://github.com/ayooshkathuria/pytorch-yolo-v3.git

Implemented Multiple-Object-Tracking by Editing built-in Object_Detection theme

This src can only run through CPU, NOT GPU

>>>--- the Algorithm for tracking objects is simple.
>>>--- Using only previous frame's information of objects and its position and class,
>>>--- compare it to the current frame's information of objects and its position and class.
>>>--- if the object has (short enough) minimum euclidean distance between previous frame and current frame
>>>--- and their classes are identical, then consider it as a same object.
>>>--- if not, they are different object


> yolov3.cfg, yolov3.weight are used for the network model,
> coco.names is use for using 80 classes

<for video file Execution>
python trackVideo.py --video 'PATH_to_the_video_file'

<for camera execution>
python trackVideo.py --camera True

Edited by Sangheon-Yang on May, 31st, 2020
'''


from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import argparse


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable 
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
    

def write_person(x, batches, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[7])
    if cls == 0:
        label = "{0}".format(classes[cls])
        label = label + ' ' + str(int(x[8]))
        color = colors[int(x[8]) % 14]
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 1)
    return img
        
        
def write_(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[7])
    label = "{0}".format(classes[cls])
    label = label + ' ' + str(int(x[8]))
    color = colors[int(x[8]) % 14]
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 1)
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--camera", dest = 'cam_use', help =
    "run camera",
    default = False)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.6)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.5)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()

colors = [(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255), (200,100,100),(100,100,200),(100,200,100),(200,200,100),(200,100,200),(100,200,200),(100,100,100)]


if __name__ == '__main__':

    initial_start = time.time()
    
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    cam_use = args.cam_use

    num_classes = 80
    classes = load_classes('data/coco.names')
    class_load = time.time()
    
    #Set up the network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    
    #Set the model in evaluation mode
    model.eval()
    
    
    videofile = args.video
    
    if cam_use: # Option For CAMERA
        cap = cv2.VideoCapture(0)
    
    else: # Option for Video file
        cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    
    n_of_frames = 0
    start = time.time()
    
    obj_id = 0
    previous_output_with_obj_id = 0
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        n_of_frames += 1
        
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
            if n_of_frames == 1:
                print(dim)
            
            with torch.no_grad():
                output = model(Variable(img), False)
                
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int: # NO DETECTION made
                previous_output_with_obj_id = 0
                print("Frame: " + str(n_of_frame))
                print("Could Not Detect Any Object ")
                print("FPS of the video is {:5.2f}".format( n_of_frames / (time.time() - start)))
                print("--------------------------------------------")
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            # calculate videos size (dimension)
            if n_of_frames < 5:
                scaling_factor = min(inp_dim/dim[0], inp_dim/dim[1])
            
            
            # resize to original size
            output[:,[1,3]] -= (inp_dim - scaling_factor*dim[0])/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*dim[1])/2
            output[:,1:5] /= scaling_factor
            
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, dim[0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, dim[1])
            
            # to check Object id for tracking
            obj_flag = torch.zeros(output.shape[0],1)
            
            
            # change coordinate to midpoint, width, height
            # -> for calculating euclidean distance
            output_w_mid_coord = torch.clone(output)
            # (x1, y1, x2, y2) -> (x_mid, y_mid, width, height)
            for i in range(output.shape[0]):
                output_w_mid_coord[i, 1] = int((output[i, 1] + output[i, 3]) / 2)
                output_w_mid_coord[i, 2] = int((output[i, 2] + output[i, 4]) / 2)
                output_w_mid_coord[i, 3] = output[i, 3] - output[i, 1]
                output_w_mid_coord[i, 4] = output[i, 4] - output[i, 2]
            
                
            #NO detection made in previous frame
            if type(previous_output_with_obj_id) == int:
                for n in range(output_w_mid_coord.shape[0]):
                    obj_id += 1
                    obj_flag[n, 0] = obj_id # grant new object_ID
                
            #some detections were made in prevous frame
            else:
                for i in range(output_w_mid_coord.shape[0]): # check current frame's detection
                    min_dist_sqr = (max(dim[0], dim[1]))**2
                    temp_obj_index = -1
                        
                    for j in range(previous_output_with_obj_id.shape[0]): # check previous frame's detection
                        if output_w_mid_coord[i,7] == previous_output_with_obj_id[j, 7]: # two objectd have same class
                            #calculate euclidean distance square
                            temp_dist = (output_w_mid_coord[i,1] - previous_output_with_obj_id[j,1])**2 + (output_w_mid_coord[i,2] - previous_output_with_obj_id[j,2])**2
                            
                            # if the distance is minimum and short enough, and the object is not matched yet,
                            if min_dist_sqr > temp_dist and temp_dist < (dim[1] * 0.1)**2 and float(previous_output_with_obj_id[j, 6]) > 0:
                                min_dist_sqr = temp_dist
                                temp_obj_index = j
                    
                    # current frame's object did not match with any object in previous frame
                    if temp_obj_index == -1:
                        obj_id += 1
                        obj_flag[i,0] = obj_id
                    
                    # an object in previous frame matched with current frame's object
                    else:
                        obj_flag[i,0] = previous_output_with_obj_id[temp_obj_index, 8]
                        previous_output_with_obj_id[temp_obj_index, 6] = -1
                        
            # add information of object_id to output tensor
            output_w_mid_coord = torch.cat((output_w_mid_coord, obj_flag), 1)
            previous_output_with_obj_id = torch.clone(output_w_mid_coord)

            # output for drawing bound boxes
            output = torch.cat((output, obj_flag), 1)
            
            # draw bound box in current frame image
            list(map(lambda x: write_(x, frame), output))
            
            # show the video in the CV2 window
            cv2.imshow("tracking", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
    
    
            print("Frame: " + str(n_of_frames))
            print("total " + str(output.shape[0]) +" Objects Detected")
            print("FPS of the video is {:5.2f}".format( n_of_frames / (time.time() - start)))
            print("--------------------------------------------")
            
        else:
            break
    

    
    

