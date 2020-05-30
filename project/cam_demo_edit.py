from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
    
    
    
    
def map_objs_2_prev(output, previous_output, obj_id): # 
    n_obj_curr = output.shape[0]
    output_w_mid_coord = torch.clone(output)
    obj_flag = torch.zeros(n_obj_curr,1)
    for i in range(n_obj_curr):
    # (x1, y1, x2, y2) -> (x_mid, y_mid, width, height)
        output_w_mid_coord[i, 1] = (output[i, 1] + output[i, 3]) // 2
        output_w_mid_coord[i, 2] = (output[i, 2] + output[i, 4]) // 2
        output_w_mid_coord[i, 3] = output[i, 3] - output[i, 1]
        output_w_mid_coord[i, 4] = output[i, 4] - output[i, 2]
           
    if type(previous_output) == int:
        for n in range(n_obj_curr):
            obj_flag[n, 0] = ++obj_id
       
    else:
        n_obj_prev = previous_output.shape[0]  # previous (img num, x_mid, y_mid, width, height, confidenc, confidenc->matched, cls, obj_id)
        
        for i in range(n_obj_curr):
            min_dist_sqr = 4000000
            temp_obj_index = -1
                
            for j in range(n_obj_prev):
                if output[i,7] == previous_output[j, 7]:
                    temp_dist = torch.dot(output_w_mid_coord[i, 1:3], previous_output[j, 1:3])
                    if min_dist_sqr > temp_dist and temp_dist < 10000 and previous_output[j, 6] > 0:
                        min_dist_sqr = temp_dist
                        temp_obj_index = j
                               
            if temp_obj_index == -1:
                obj_flag[i,0] = ++obj_id
                       
            else:
                obj_flag[i,0] = previous_output[temp_obj_index, 8]
                previous_output[temp_obj_index, 6] = -1
    new_prev_mid_coord_output = torch.cat((output_w_mid_coord, obj_flag), 1)
    new_output_for_draw = torch.cat((output, obj_flag), 1)
       
    return new_output_for_draw, new_prev_mid_coord_output, obj_id
       


    




def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[7])
    
    if cls == 0:
        label = "{0}".format(classes[cls])
        label = label + ' ' + str(int(x[8]))
        #color = random.choice(colors)
        color = colors[int(x[8]) % 7]
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()



if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    
    #num_classes = 80
    num_classes = 1
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    #videofile = 'sample.mp4'
    
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()
    
    classes = load_classes('data/coco.names')
    #colors = pkl.load(open("pallete", "rb"))
    colors = [(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    
    
    
    
    
    
    
    obj_id = 0
    previous_output_with_obj_id = 0
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
#            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                previous_output_with_obj_id = 0
                
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            
            
            
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]
            
            ##############################
            #output_, new_previous_output_with_obj_id, obj_id_ = map_objs_2_prev(output, previous_output_with_obj_id, obj_id)
            
            
            
            n_obj_curr = output.shape[0]
            
            print("n_obj_cur")
            print(n_obj_curr)
            
            output_w_mid_coord = torch.clone(output)
            
            obj_flag = torch.zeros(n_obj_curr,1)
            
            for i in range(n_obj_curr):
            # (x1, y1, x2, y2) -> (x_mid, y_mid, width, height)
                output_w_mid_coord[i, 1] = (output[i, 1] + output[i, 3]) // 2
                output_w_mid_coord[i, 2] = (output[i, 2] + output[i, 4]) // 2
                output_w_mid_coord[i, 3] = output[i, 3] - output[i, 1]
                output_w_mid_coord[i, 4] = output[i, 4] - output[i, 2]
                   
            
            print("output_w_mid_coord")
            print(output_w_mid_coord)
            
            if type(previous_output_with_obj_id) == int:
                for n in range(n_obj_curr):
                    if output_w_mid_coord[n, 7] == 0:
                        obj_id += 1
                        obj_flag[n, 0] = obj_id
                    #print("obj_id fuck it is not showinf")
                    #print(obj_id)
               
            else:
                n_obj_prev = previous_output_with_obj_id.shape[0]  # previous (img num, x_mid, y_mid, width, height, confidenc, confidenc->matched, cls, obj_id)
                
                for i in range(n_obj_curr):
                    min_dist_sqr = 400000000
                    temp_obj_index = -1
                        
                    for j in range(n_obj_prev):
                    
                        if output_w_mid_coord[i,7] == previous_output_with_obj_id[j, 7] and output_w_mid_coord[i,7] == 0:
                            #temp_dist = torch.(output_w_mid_coord[i, 1:3], previous_output_with_obj_id[j, 1:3])
                            
                            temp_dist = (output_w_mid_coord[i,1] - previous_output_with_obj_id[j,1])**2 + (output_w_mid_coord[i,2] - previous_output_with_obj_id[j,2])**2
                            '''
                            print("dotproduction")
                            print(output_w_mid_coord[i,1:3])
                            print(previous_output_with_obj_id[j,1:3])
                            temp_dist = int(temp_dist)
                            print("temp_dist")
                            print(temp_dist)
                            '''
                            if min_dist_sqr > temp_dist and temp_dist < 40000 and previous_output_with_obj_id[j, 6] > 0:
                                min_dist_sqr = temp_dist
                                temp_obj_index = j
                                       
                    if temp_obj_index == -1 and output_w_mid_coord[i, 7] == 0:
                        obj_id += 1
                        obj_flag[i,0] = obj_id
                               
                    else:
                        obj_flag[i,0] = previous_output_with_obj_id[temp_obj_index, 8]
                        previous_output_with_obj_id[temp_obj_index, 6] = -1
                        
                        
            new_prev_mid_coord_output = torch.cat((output_w_mid_coord, obj_flag), 1)
            new_output_for_draw = torch.cat((output, obj_flag), 1)
            
            
            
            
            
            
            
            previous_output_with_obj_id = new_prev_mid_coord_output
            output_ = new_output_for_draw


            
            
            print("output for draw")
            print(output_)
            print("prev output w obj id")
            print(previous_output_with_obj_id)
            
            
            ##################
            
            list(map(lambda x: write(x, orig_im), output_))
            
            cv2.imshow("frame", orig_im)
            
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
            
            
        else:
            break
    

    
    
   
