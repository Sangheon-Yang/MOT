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
import math
import pandas as pd
import random
import itertools
#HI

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)

    parser.add_argument("--count", dest='count', help=" the number of sequence images in Directory", default=30)

    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.5)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


def write_person(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[7])
    if cls == 0:
        label = "{0}".format(classes[cls])
        label = label + ' ' + str(int(x[8]))
        color = colors[int(x[8]) % 14]

        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
    return img


def write_(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[7])
    label = "{0}".format(classes[cls])
    label = label + ' ' + str(int(x[8]))
    color = colors[int(x[8]) % 14]

    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
    return img


def write_track(x, img):

    for object_index in range(1, object_data_base.shape[0]):
        detected_object = object_data_base[object_index]
        point_count = detected_object[15]
        curr_point = detected_object[16]
        loop_count = 0

        while loop_count < point_count:
            curr_point = curr_point % (point_count+1)

            if curr_point == 0:
                curr_point += 1

            next_point = (curr_point+1) % (point_count+1)

            if next_point == 0:
                next_point += 1

            c1 = tuple([detected_object[curr_point], detected_object[curr_point+6]])
            c2 = tuple([detected_object[next_point], detected_object[next_point+6]])
            color = colors[int(detected_object[0]) % 14]

            if loop_count + 1 != point_count:
                cv2.line(img, c1, c2, color, 1)

            cv2.circle(img, c1, 3, color, -1)

            loop_count += 1
            curr_point = next_point

    return img


colors = [(255, 255, 255), (255, 100, 100), (0, 255, 0), (110, 110, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 150), (255, 150, 255), (150, 255, 255), (150, 255, 150), (150, 150, 255), (255, 150, 150), (150, 150, 150)]


def track_result(result, frame_num):
    line = result.numpy()
    line_count = 0

    while line_count < line.__len__():
        output_line = str(frame_num) + ","
        output_line += str(line[line_count][8]) + ","
        output_line += str(line[line_count][1]) + ","
        output_line += str(line[line_count][2]) + ","
        output_line += str(float(line[line_count][3]) - float(line[line_count][1])) + ","
        output_line += str(float(line[line_count][4]) - float(line[line_count][2])) + "\n"
        result_file.write(output_line)
        line_count += 1

    result_file.flush()


if __name__ == '__main__':
    result_file = open('MOT_result/trackResult.txt', 'w')
    initial_start = time.time()

    args = arg_parse()
    scales = args.scales
    images = args.images
    count_limit = int(args.count)

    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    num_classes = 80
    classes = load_classes('data/coco.names')
    class_load = time.time()

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # Set the model in evaluation mode
    model.eval()

    # check the path whether it is valid or not
    img_num = count_limit
    img_num_str = str(img_num).zfill(6) + ".jpg"

    try:
        tmp_img_path = osp.join(osp.realpath('.'), images, img_num_str)

    except NotADirectoryError:
        print("No directory with the name {}".format(images))
        exit()

    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
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

    # -----start for loop -----#
    obj_id = 0
    #previous_output_with_obj_id = 0
    min_score_standard = (max(orig_inp_dim_w, orig_inp_dim_h) ** 2) * 10

    object_ID_Cache = torch.zeros(1, 5, dtype=torch.int)

    for img_id in range(count_limit):
        # load the image
        frame_start = time.time()

        frame_num = img_id+1
        # img_id & img_path process
        curr_img_num_str = str(img_id + 1).zfill(6) + ".jpg"
        curr_img_path = osp.join(osp.realpath('.'), images, curr_img_num_str)
        #print(curr_img_path)

        processed_img, curr_img, dim = prep_image(curr_img_path, inp_dim)

        with torch.no_grad():
            prediction = model(Variable(processed_img), False)

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:  # NO DETECTION made
            #previous_output_with_obj_id = 0
            print("in image : " + curr_img_num_str)
            print("Could Not Detect Any Object ")
            print("--------------------------------------------")
            i += 1
            continue

        # ****** detect only person ********************
        #prediction = prediction[prediction[:, 7] == 0]
        
        # detection were made
        print("in image : " + curr_img_num_str)
        print("total " + str(prediction.shape[0]) + " Objects Detected")
        print("--------------------------------------------")
    
    
        # resize to original size
        output = torch.clone(prediction)
        output[:, [1, 3]] -= (inp_dim - scaling_factor * dim[0]) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * dim[1]) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, orig_inp_dim_w)
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, orig_inp_dim_h)

        # to check Object id for tracking
        obj_flag = torch.zeros(output.shape[0], 1)

        # change coordinate to midpoint, width, height
        # -> for calculating euclidean distance
        output_w_mid_coord = torch.clone(output)
        # (x1, y1, x2, y2) -> (x_mid, y_mid, width, height)
        for i in range(output.shape[0]):
            output_w_mid_coord[i, 1] = int((output[i, 1] + output[i, 3]) / 2)
            output_w_mid_coord[i, 2] = int((output[i, 2] + output[i, 4]) / 2)
            output_w_mid_coord[i, 3] = output[i, 3] - output[i, 1]
            output_w_mid_coord[i, 4] = output[i, 4] - output[i, 2]

        #track_result(output_w_mid_coord, result_file)

        # when the object_data_base in empty.
        # This code is for only first frame of the video
        if object_ID_Cache.shape[0] == 1 :
            for n in range(output_w_mid_coord.shape[0]):
                obj_id += 1
                obj_flag[n, 0] = obj_id  # grant new object_ID
                tempObjInfo = torch.zeros(1, 5, dtype=torch.int)
                tempObjInfo[0, 0] = obj_id # object_id that granted to objects uniquely
                tempObjInfo[0, 1] = int(output_w_mid_coord[n, 1]) # first x_position of detected object
                tempObjInfo[0, 2] = int(output_w_mid_coord[n, 2]) # first y_position of detected object
                tempObjInfo[0, 3] = int(0.5*(max(output_w_mid_coord[n,3], output_w_mid_coord[n,4])))**2 # Radius sqr of range of further detection
                tempObjInfo[0, 4] = frame_num # for checking whether it is matched or not
                # concating new obj_id to object_data_base
                object_ID_Cache = torch.cat((object_ID_Cache, tempObjInfo), 0)


        # some detections were made in prevous frame
        # object_ID_Cache is not empty
        # we should grant the obj_id to newly detected objects in the current frame
        else:
            for i in range(output_w_mid_coord.shape[0]):  # check the newly detected objects
                temp_obj_index = -1
                local_min_dist = min_score_standard

                # position of the object that newly detected in this new frame
                new_position_x = int(output_w_mid_coord[i, 1])
                new_position_y = int(output_w_mid_coord[i, 2])
                # RAdius threshold related to new detected Bbox Size (radius of max)
                radius_threshold = int(0.5 * max(output_w_mid_coord[i, 3], output_w_mid_coord[i, 4])) ** 2

                for j in range(object_ID_Cache.shape[0]):  # check the database of detected objects before

                    if j == 0 or object_ID_Cache[j, 4] == frame_num:
                        continue

                    # last location of object (current position that detected)
                    last_position_x = object_ID_Cache[j, 1]
                    last_position_y = object_ID_Cache[j, 2]

                    # direction vector between last and new position
                    dist_x = new_position_x - last_position_x
                    dist_y = new_position_y - last_position_y

                    # distance between new and last position
                    dist_between = dist_x**2 + dist_y**2

                    if radius_threshold > dist_between and local_min_dist > dist_between:
                        # if the distance is short enough and this is not matched yet,
                        local_min_dist = dist_between
                        temp_obj_index = j

                # current frame's object did not match with any object in previous frame
                if temp_obj_index == -1:
                    obj_id += 1
                    obj_flag[i, 0] = obj_id  # grant new object_ID
                    tempObjInfo = torch.zeros(1, 5, dtype=torch.int)
                    tempObjInfo[0, 0] = obj_id  # object_id that granted to objects uniquely
                    tempObjInfo[0, 1] = int(output_w_mid_coord[i, 1])  # first x_position of detected object
                    tempObjInfo[0, 2] = int(output_w_mid_coord[i, 2])  # first y_position of detected object
                    tempObjInfo[0, 3] = int(0.5 * (max(output_w_mid_coord[i, 3], output_w_mid_coord[
                        i, 4]))) ** 2  # Radius sqr of range of further detection
                    tempObjInfo[0, 4] = frame_num  # for checking whether it is matched or not
                    # concating new obj_id to object_data_base
                    object_ID_Cache = torch.cat((object_ID_Cache, tempObjInfo), 0)


                # an object  matched with object database
                else:
                    object_ID_Cache[temp_obj_index, 1] = new_position_x
                    object_ID_Cache[temp_obj_index, 2] = new_position_y
                    object_ID_Cache[temp_obj_index, 3] = radius_threshold
                    object_ID_Cache[temp_obj_index, 4] = frame_num
                    obj_flag[i, 0] = temp_obj_index

        # output for drawing bound boxes
        output = torch.cat((output, obj_flag), 1)

        track_result(output, frame_num)
        # draw bound box in original image
        list(map(lambda x: write_(x, curr_img), output))

        #list(map(lambda x: write_track(x, curr_img), object_data_base))

        # write a new image in destination_path
        det_names = args.det + "/det_" + curr_img_num_str
        cv2.imwrite(det_names, curr_img)

        # resize a image to show it in the window with appropriate size
        window_resize = cv2.resize(curr_img, (int(orig_inp_dim_w * 0.6), int(orig_inp_dim_h * 0.6)), cv2.INTER_CUBIC)
        cv2.imshow("tracking", window_resize)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit()

        # calculate elapsed time
        frame_end = time.time()
        print("time-elapsed for this frame: " + str(frame_end - frame_start))
        frame_start = frame_end

    # ----- for loop ended----------------#


    end = time.time()
    print("done!! ")
    print("Total-elapsed time: " + str(end - initial_start))

    result_file.write("#," + str(obj_id))
    result_file.flush()
    result_file.close()

