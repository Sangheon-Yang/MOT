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

<for Execution>

python trackMOT.py --images 'PATH_TO_IMAGE_DIRECTORY' --det 'PATH_TO_DESTINATION' --count "the number of images you want to track"

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
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import itertools

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    
    parser.add_argument("--count", dest = 'count', help = " the number of sequence images in Directory", default = 30)
    
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
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
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()


def write_person(x, results):
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

colors = [(255, 255, 255), (255, 100, 100), (0, 255, 0), (110, 110, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 150), (255, 150, 255), (150, 255, 255), (150, 255, 150), (150, 150, 255), (255, 150, 150), (150, 150, 150)]

def track_result(frame, file, frame_no):
    line = frame.numpy()
    line_count = 0

    while line_count < line.__len__():
        output_line = str(frame_no+1) + ","
        output_line += str(line[line_count][1]) + ","
        output_line += str(line[line_count][2]) + ","
        output_line += str(line[line_count][3]) + ","
        output_line += str(line[line_count][4]) + ","
        output_line += str(line[line_count][8]) + "\n"
        file.write(output_line)
        line_count += 1

    file.flush()

if __name__ ==  '__main__':
    result_file = open('trackResult.txt', 'w')
    initial_start = time.time()
    
    args = arg_parse()
    scales = args.scales
    images = args.images
    print(images)
    count_limit = int(args.count)
    
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    num_classes = 80
    classes = load_classes('data/coco.names')
    class_load = time.time()
    
    #Set up the neural network
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
    
   
    #check the path whether it is valid or not
    img_num = count_limit
    img_num_str = str(img_num).zfill(6) + ".jpg"
    
    try:
        tmp_img_path = osp.join(osp.realpath('.'), images, img_num_str)
        
    except NotADirectoryError:
        print ("No directory with the name {}".format(images))
        exit()
    
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists(args.det):
           os.makedirs(args.det)
       
   

    ### calculate input image's size (dimension) & Scaling Factor
    print(tmp_img_path)
    temp = cv2.imread(tmp_img_path)
    
    orig_inp_dim_w = int(temp.shape[1])
    orig_inp_dim_h = int(temp.shape[0])
    
    print("original image size: ")
    print(temp.shape)
    
    scaling_factor_w = float(inp_dim) / float(orig_inp_dim_w)
    scaling_factor_h = float(inp_dim) / float(orig_inp_dim_h)
    scaling_factor = min(scaling_factor_w, scaling_factor_h)
    
    print("scale factor")
    print(scaling_factor)

    #-----start for loop -----#
    obj_id = 0
    previous_output_with_obj_id = 0
    
    for img_id in range(count_limit):
        #load the image 
        frame_start = time.time()
        
        # img_id & img_path process
        curr_img_num_str = str(img_id+1).zfill(6) + ".jpg"
        curr_img_path = osp.join(osp.realpath('.'), images, curr_img_num_str)
        print(curr_img_path)
        
        processed_img, curr_img, dim  = prep_image(curr_img_path, inp_dim)
        
        with torch.no_grad():
            prediction = model(Variable(processed_img), False)

        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        if type(prediction) == int: # NO DETECTION made
            previous_output_with_obj_id = 0
            print("in image : " + curr_img_num_str)
            print("Could Not Detect Any Object ")
            print("--------------------------------------------")
            i += 1
            continue
        
        # ****** detect only person ********************
        prediction = prediction[prediction[:, 7] == 0]
        
        # detection were made
        print("in image : " + curr_img_num_str)
        print("total " + str(prediction.shape[0]) +" Objects Detected")
        print("--------------------------------------------")
    
        
        # resize to original size
        output = torch.clone(prediction)
        output[:,[1,3]] -= (inp_dim - scaling_factor*dim[0])/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*dim[1])/2
        output[:,1:5] /= scaling_factor
        
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, orig_inp_dim_w)
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, orig_inp_dim_h)
        
        
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
                min_dist_sqr = orig_inp_dim_w**2
                temp_obj_index = -1
                
                for j in range(previous_output_with_obj_id.shape[0]): # check previous frame's detection
                    if output_w_mid_coord[i,7] == previous_output_with_obj_id[j, 7]: # two objectd have same class
                        #calculate euclidean distance square
                        temp_dist = (output_w_mid_coord[i,1] - previous_output_with_obj_id[j,1])**2 + (output_w_mid_coord[i,2] - previous_output_with_obj_id[j,2])**2
                        
                        # if the distance is minimum and short enough, and the object is not matched yet,
                        if min_dist_sqr > temp_dist and temp_dist < (orig_inp_dim_h * 0.1)**2 and float(previous_output_with_obj_id[j, 6]) > 0:
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

        track_result(output_w_mid_coord, result_file, img_id)
        # draw bound box in original image
        list(map(lambda x: write_(x, curr_img), output))

        # write a new image in destination_path
        det_names = args.det + "/det_" + curr_img_num_str
        cv2.imwrite(det_names, curr_img)
        
        # resize a image to show it in the window with appropriate size
        window_resize = cv2.resize(curr_img, ( int(orig_inp_dim_w*0.6), int(orig_inp_dim_h*0.6) ), cv2.INTER_CUBIC)
        cv2.imshow("tracking" , window_resize)

        output = torch.cat((output, obj_flag), 1)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit()
        
        # calculate elapsed time
        frame_end = time.time()
        print("time-elapsed for this frame: " + str(frame_end - frame_start))
        frame_start = frame_end
    
    #----- for loop ended----------------#

    end = time.time()
    print("done!! ")
    print("Total-elapsed time: " + str(end - initial_start))

    result_file.close()
    
