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
    result_file = open('./trackResult.txt', 'w')
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

    object_data_base = torch.zeros(1, 19, dtype=torch.int)

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
        prediction = prediction[prediction[:, 7] == 0]
        
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
        if object_data_base.shape[0] == 1 :
            for n in range(output_w_mid_coord.shape[0]):
                obj_id += 1
                obj_flag[n, 0] = obj_id  # grant new object_ID
                tempObjInfo = torch.zeros(1, 19, dtype=torch.int)
                tempObjInfo[0, 0] = obj_id # object_id that granted to objects uniquely
                tempObjInfo[0, 1] = int(output_w_mid_coord[n, 1]) # first x_position of detected object
                tempObjInfo[0, 7] = int(output_w_mid_coord[n, 2]) # first y_position of detected object
                tempObjInfo[0, 13] = 0 # x instance of direction vector
                tempObjInfo[0, 14] = 0 # y instance of direction vector
                tempObjInfo[0, 15] = 1 # count of objects that are in 'object_data_base'
                tempObjInfo[0, 16] = 1 # index of current object's postion
                tempObjInfo[0, 17] = int(0.5*(max(output_w_mid_coord[n,3], output_w_mid_coord[n,4])))**2 # Radius sqr of range of further detection
                tempObjInfo[0, 18] = frame_num # for checking whether it is matched or not
                # concating new obj_id to object_data_base
                object_data_base = torch.cat((object_data_base, tempObjInfo), 0)


        # some detections were made in prevous frame
        # object_data_base is not empty
        # we should grant the obj_id to newly detected objects in the current frame
        else:
            for i in range(output_w_mid_coord.shape[0]):  # check the newly detected objects
                temp_obj_index = -1
                local_min_score = min_score_standard

                for j in range(object_data_base.shape[0]):  # check the database of detected objects before

                    if j == 0 or object_data_base[j, 18] == frame_num:
                        continue

                    # calculate euclidean distance square
                    radius_threshold = object_data_base[j, 17]

                    # last location of object (current position that detected)
                    last_position_x = object_data_base[j, object_data_base[j, 16]]
                    last_position_y = object_data_base[j, (object_data_base[j, 16] + 6)]

                    # position of the object that newly detected in this new frame
                    new_position_x = int(output_w_mid_coord[i, 1])
                    new_position_y = int(output_w_mid_coord[i, 2])

                    # new detected Bbox Size (radius of max)
                    detected_radius = int(0.5 * max(output_w_mid_coord[i, 3], output_w_mid_coord[i, 4]))**2

                    # direction vector between last and new position
                    distVector_x = new_position_x - last_position_x
                    distVector_y = new_position_y - last_position_y

                    # distance between new and last position
                    dist_between = distVector_x**2 + distVector_y**2

                    ratio_of_Box = float(detected_radius) / float(radius_threshold)

                    if radius_threshold > dist_between and 2 > ratio_of_Box and 0.5 < ratio_of_Box :
                        # if the distance is short enough and this is not matched yet,
                        #calculate dist * 8 + direction * 2 < local_min

                        # estimated direction vector at the last position of the object

                        size_of_dV = math.sqrt(object_data_base[j, 13]**2 + object_data_base[j, 14]**2)
                        pure_dist = math.sqrt(dist_between)

                        if size_of_dV < 0.000000001:
                            dirVector_x = 0
                            dirVector_y = 0
                        else:
                            dirVector_x = int(pure_dist * (object_data_base[j, 13] / size_of_dV))
                            dirVector_y = int(pure_dist * (object_data_base[j, 14] / size_of_dV))

                        # estimated position using the estimated direction vector and the last position
                        estimated_x = last_position_x + dirVector_x
                        estimated_y = last_position_y + dirVector_y

                        # distnace between estimated and real new position
                        between_estimated_detected = (estimated_x - new_position_x)**2 + (estimated_y - new_position_y)**2

                        temp_min_score = 8*dist_between + 2*between_estimated_detected

                        if temp_min_score < local_min_score:
                            local_min_score = temp_min_score
                            temp_obj_index = j

                # current frame's object did not match with any object in previous frame
                if temp_obj_index == -1:
                    obj_id += 1
                    obj_flag[i, 0] = obj_id
                    # create new object information in database.
                    tempObjInf = torch.zeros(1, 19, dtype=torch.int)
                    tempObjInf[0, 0] = obj_id  #object_id that granted to objects uniquely
                    tempObjInf[0, 1] = int(output_w_mid_coord[i, 1])  # first x_position of detected object
                    tempObjInf[0, 7] = int(output_w_mid_coord[i, 2])  # first y_position of detected object
                    tempObjInf[0, 13] = 0  # x instance of direction vector
                    tempObjInf[0, 14] = 0  # y instance of direction vector
                    tempObjInf[0, 15] = 1  # count of objects that are in 'object_data_base'
                    tempObjInf[0, 16] = 1  # index of current object's postion
                    tempObjInf[0, 17] = int(0.5 * (max(output_w_mid_coord[i, 3], output_w_mid_coord[i, 4]))) ** 2  # Radius sqr of range of further detection
                    tempObjInf[0, 18] = frame_num
                    # concating new obj_id to object_data_base

                    object_data_base = torch.cat((object_data_base, tempObjInf), 0)

                # an object  matched with object database
                else:
                    last_point_index = object_data_base[temp_obj_index, 16]
                    new_point_index = (last_point_index % 6) + 1
                    object_data_base[temp_obj_index, 16] = new_point_index

                    # setting start point of direction vector
                    if object_data_base[temp_obj_index, 15] < 6:
                        starting_point = 1
                        object_data_base[temp_obj_index, 15] += 1
                    else:
                        starting_point = (new_point_index % 6) + 1

                    # adding new detected point
                    object_data_base[temp_obj_index, new_point_index] = int(output_w_mid_coord[i, 1])
                    object_data_base[temp_obj_index, new_point_index+6] = int(output_w_mid_coord[i, 2])

                    # calculating accumulated vector of direction
                    acc_vector_x = object_data_base[temp_obj_index, new_point_index] - object_data_base[temp_obj_index, starting_point]
                    acc_vector_y = object_data_base[temp_obj_index, new_point_index+6] - object_data_base[temp_obj_index, starting_point+6]

                    object_data_base[temp_obj_index, 13] = acc_vector_x
                    object_data_base[temp_obj_index, 14] = acc_vector_y

                    # setting new radius sqr of range of further detection
                    object_data_base[temp_obj_index, 17] = int(0.5 * (max(output_w_mid_coord[i, 3], output_w_mid_coord[
                        i, 4]))) ** 2

                    object_data_base[temp_obj_index, 18] = frame_num

                    obj_flag[i, 0] = temp_obj_index
                    #previous_output_with_obj_id[temp_obj_index, 6] = -1

        # output for drawing bound boxes
        output = torch.cat((output, obj_flag), 1)

        track_result(output, frame_num)
        # draw bound box in original image
        list(map(lambda x: write_(x, curr_img), output))
        list(map(lambda x: write_track(x, curr_img), object_data_base))

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

    result_file.write("#,"+str(object_data_base.shape[0]))
    result_file.flush()
    result_file.close()

