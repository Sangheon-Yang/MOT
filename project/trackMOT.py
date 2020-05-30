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
import pickle as pkl
import itertools

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, image_path, CUDA):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_



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
   

colors = [(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255), (200,100,100),(100,100,200),(100,200,100),(200,200,100),(200,100,200),(100,200,200),(100,100,100)]


if __name__ ==  '__main__':

    initial_start = time.time()
    
    # arg_parse()
    args = arg_parse()
    scales = args.scales
    images = args.images
    count_limit = int(args.count)
    #batch_size = int(args.bs)
    
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    
    #CUDA = torch.cuda.is_available()

    num_classes = 80 # only detect person class
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


    '''
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    '''
    #Set the model in evaluation mode
    model.eval()
    
    
    read_dir = time.time()
    
    #Detection phase
    
    #___________check the path________________--------
    
    img_num = count_limit
    #img_num_str = "{:06d}".format(img_num)
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
       
    #------------------------------------

    ### calculate input image's size (dimension)
    print(tmp_img_path)
    temp = cv2.imread(tmp_img_path)
    
    #cv2.imshow("temp",temp)
    #key = cv2.waitKey(1000)
    
    print(type(temp))
    orig_inp_dim_w = int(temp.shape[1])
    orig_inp_dim_h = int(temp.shape[0])
    
    
    print("original image size: ")
    print(temp.shape)
    
    #print("resized: ")
    #print(inp_dim)
    
    
    scaling_factor_w = float(inp_dim) / float(orig_inp_dim_w)
    scaling_factor_h = float(inp_dim) / float(orig_inp_dim_h)
    
    scaling_factor = min(scaling_factor_w, scaling_factor_h)
    
    print("scale factor")
    print(scaling_factor)
    

    i = 0
    

    #write = False
    
    #model(get_test_input(inp_dim, tmp_img_path, CUDA), CUDA)
    model(get_test_input(inp_dim, tmp_img_path, False), False)
    start_det_loop = time.time()
    
    objs = {}


#-----start for loop -----#


    obj_id = 0
    previous_output_with_obj_id = 0


    frame_start = time.time()
    
    for img_id in range(count_limit):
        #load the image 
        start = time.time()
        
        '''
        if CUDA:
            batch = batch.cuda()'''
        
        
        # img_id_process
        curr_img_num_str = str(img_id+1).zfill(6) + ".jpg"
        curr_img_path = osp.join(osp.realpath('.'), images, curr_img_num_str)
        print(curr_img_path)
        
        
        processed_img, curr_img, dim  = prep_image(curr_img_path, inp_dim)
        
        
        
        #print("processed image")
        #print(processed_img)
        
        
        with torch.no_grad():
            #prediction = model(Variable(processed_img), CUDA)
            prediction = model(Variable(processed_img), False)

        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        if type(prediction) == int:
            previous_output_with_obj_id = 0
            print("in image : " + curr_img_num_str)
            print("Could Not Detect Any Object ")
            print("--------------------------------------------")
            i += 1
            continue
        
        print("in image : " + curr_img_num_str)
        print("total " + str(prediction.shape[0]) +" Objects(person) Detected")
        print("--------------------------------------------")
       
        
        
        #print("raw prediction")
        #print(prediction)
        
        output = torch.clone(prediction)
        
        # scale to original size
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*dim[0])/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*dim[1])/2
        output[:,1:5] /= scaling_factor
        
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, orig_inp_dim_w)
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, orig_inp_dim_h)
        
        '''
        for i in range(prediction.shape[0]):
            output[i, 1] = int(scaling_factor_h * prediction[i, 1])
            output[i, 2] = int(scaling_factor_w * prediction[i, 2])
            output[i, 3] = int(scaling_factor_h * prediction[i, 3])
            output[i, 4] = int(scaling_factor_w * prediction[i, 4])
        '''
        '''
        # scale to original size
        for i in range(prediction.shape[0]):
            output[i, 1] = scaling_factor_w * prediction[i, 1]
            output[i, 2] = scaling_factor_h * prediction[i, 2]
            output[i, 3] = scaling_factor_w * prediction[i, 3]
            output[i, 4] = scaling_factor_h * prediction[i, 4]
         '''
        
        #print("resized output")
        #print(output)
        
        obj_flag = torch.zeros(output.shape[0],1)
    
        output_w_mid_coord = torch.clone(output)
        
        
        # change coordinate to midpoint
        for i in range(output.shape[0]):
        # (x1, y1, x2, y2) -> (x_mid, y_mid, width, height)
            output_w_mid_coord[i, 1] = int((output[i, 1] + output[i, 3]) / 2)
            output_w_mid_coord[i, 2] = int((output[i, 2] + output[i, 4]) / 2)
            output_w_mid_coord[i, 3] = output[i, 3] - output[i, 1]
            output_w_mid_coord[i, 4] = output[i, 4] - output[i, 2]
        
        #print("mid coord output")
        #print(output_w_mid_coord)
        
        
        if type(previous_output_with_obj_id) == int: #이전 프레임에서 detect된 obj가 없음
            for n in range(output_w_mid_coord.shape[0]):
                obj_id += 1
                obj_flag[n, 0] = obj_id ## 새로운 obj id부여
        
        else: #이전 프레임에서 detect 된게 있을경우
            for i in range(output_w_mid_coord.shape[0]):
                min_dist_sqr = orig_inp_dim_w**2
                temp_obj_index = -1
                
                for j in range(previous_output_with_obj_id.shape[0]):
                       
                    if output_w_mid_coord[i,7] == previous_output_with_obj_id[j, 7]: #same class
                        #dostance square
                        temp_dist = (output_w_mid_coord[i,1] - previous_output_with_obj_id[j,1])**2 + (output_w_mid_coord[i,2] - previous_output_with_obj_id[j,2])**2
                        
                        if min_dist_sqr > temp_dist and temp_dist < 400 and float(previous_output_with_obj_id[j, 6]) > 0:
                            min_dist_sqr = temp_dist
                            temp_obj_index = j
                                          
                if temp_obj_index == -1:
                    obj_id += 1
                    obj_flag[i,0] = obj_id
                                  
                else:
                    obj_flag[i,0] = previous_output_with_obj_id[temp_obj_index, 8]
                    previous_output_with_obj_id[temp_obj_index, 6] = -1
                
    
        output_w_mid_coord = torch.cat((output_w_mid_coord, obj_flag), 1)
        #print("output_w_mid_coord after adding id")
        #print(output_w_mid_coord)
        
        previous_output_with_obj_id = torch.clone(output_w_mid_coord)
        #previous_output_with_obj_id = 0

        output = torch.cat((output, obj_flag), 1)
        #print("output after adding id")
        #print(output)
        

        list(map(lambda x: write_(x, curr_img), output))

        det_names = args.det + "/det_" + curr_img_num_str
        
        cv2.imwrite(det_names, curr_img)
        
        
        window_resize = cv2.resize(curr_img, (1200, 675), cv2.INTER_CUBIC)
        cv2.imshow("tracking" , window_resize)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit()
        
        
        frame_end = time.time()
        
        print("time- elapsed for a frame: " + str(frame_end - frame_start))
        
        frame_start = frame_end
        
        
       
        
    #############################


    end = time.time()
    
    print("done!! ")
    print("ToTal-elapsed time: " + str(end - initial_start))
    
    #torch.cuda.empty_cache()
    
    
        
        
    
    
