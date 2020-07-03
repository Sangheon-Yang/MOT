import argparse
import cv2
import os.path as osp
import numpy as np
import torch
from preprocess import prep_image, inp_to_image

colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (200, 100, 100), (100, 100, 200), (100, 200, 100), (200, 200, 100), (200, 100, 200), (100, 200, 200),
              (100, 100, 100)]

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='Estimation')
    parser.add_argument("--threshold", dest='threshold', help="Estimation distance threshold", default=1/5, type=float)
    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",default="./MOT17-03-DPM/img1/", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)
    return parser.parse_args()


def write_result(x, img):
    write_object = x[1]
    left_x = float(write_object[1]) - float(write_object[3])/2
    left_y = float(write_object[2]) - float(write_object[4])/2
    right_x = float(write_object[1]) + float(write_object[3])/2
    right_y = float(write_object[2]) + float(write_object[4])/2
    id = int(float(x[1][5]))

    track_object = torch.from_numpy(np.array([left_x, left_y, right_x, right_y]))

    c1 = tuple(track_object[0:2].int())
    c2 = tuple(track_object[2:4].int())

    label = "track"+str(int(id))

    color = colors[id % 14]

    cv2.rectangle(img, c1, c2, (255,255,255), 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 1)

    write_object = x[0]
    left_x = float(write_object[1]) - float(write_object[3]) / 2
    left_y = float(write_object[2]) - float(write_object[4]) / 2
    right_x = float(write_object[1]) + float(write_object[3]) / 2
    right_y = float(write_object[2]) + float(write_object[4]) / 2
    id = int(float(x[1][5]))

    track_object = torch.from_numpy(np.array([left_x, left_y, right_x, right_y]))

    c1 = tuple(track_object[0:2].int())
    c2 = tuple(track_object[2:4].int())

    label = "MOT"+str(int(id))

    color = colors[id % 14]

    cv2.rectangle(img, c1, c2, (255, 255, 255), 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
    return img



# Frame 출력 함수
def printFrame(frame):

    frameSize = frame.__len__()
    frameCount = 0

    while frameCount < frameSize:
        print("frame"+str(frame[frameCount][0][0])+": ")

        objSize = frame[frameCount].__len__()
        objCount = 0

        while objCount < objSize:
            print("          "+str(frame[frameCount][objCount]))
            objCount += 1

        frameCount += 1


# change line
# [frameNo , ID , leftCoordinate , topCoordinate , width , height , conf , x , y , z]
# to
# [frameNo, midPointX, midPointY , width, height]
def convert_to_midPoint(line):

    midPointX = float(line[2]) + (float(line[4]) / 2)
    midPointY = float(line[3]) + (float(line[5]) / 2)

    parsedLine = [line[0], midPointX, midPointY, line[4], line[5]]
    return parsedLine


# matching_result = [
#                     [ [MOT result[1]] , [track_result[0]] ]
#                     [ [MOT result[3]] , [track_result[4]] ]
#                                       ...
#                    ]
def compare_results(track_result, MOT_result, threshold):
    matching_result = []
    matched_list = []

    # frame loop
    for i in range(1, MOT_result.__len__()):
        matching_result.append([])
        # MOT object loop
        for MOT_object in MOT_result[i]:
            # track object loop
            for track_object in track_result[i]:
                distance_X = abs(float(track_object[1]) - float(MOT_object[1]))
                distance_Y = abs(float(track_object[2]) - float(MOT_object[2]))

                if float(MOT_object[3]) * threshold >= distance_X and float(MOT_object[4]) * threshold >= distance_Y:
                    matching_result[i-1].append([MOT_object, track_object])
                    matched_list.append(track_object)
                    break

            for pop_object in matched_list:
                track_result[i].remove(pop_object)
            matched_list.clear()

    return matching_result

if __name__ == '__main__':

    args = arg_parse()
    threshold = args.threshold
    images = args.images

####################### Track MOT Parsing #######################
    trackFile = open('trackResult.txt', 'r')
    track_result = [[], []]

    curFrame = 1

    while True:
        line = trackFile.readline()

        if line == '':
            break

        line = line.split("\n")[0]
        parsed_line = line.split(",")

        if curFrame != float(parsed_line[0]):
            curFrame += 1
            track_result.append([])

        track_result[curFrame].append(parsed_line)

    trackFile.close()
    trackFrame = curFrame

####################### MOT 17 Det Parsing #######################
    detFile = open('./MOT17-03-DPM/det/det.txt', 'r')
    mot_result = [[], []]

    curFrame = 1

    while True:
        line = detFile.readline()

        if line == '':
            break

        line = line.split("\n")[0]
        parsed_line = line.split(",")
        converted_line = convert_to_midPoint(parsed_line)

        if curFrame != float(parsed_line[0]):
            curFrame += 1
            # track_result의 Frame 개수 초과 시 break
            if curFrame > trackFrame:
                break

            mot_result.append([])

        mot_result[curFrame].append(converted_line)

    detFile.close()

########################## Compare Result ##########################
    output_list = compare_results(track_result, mot_result, threshold)

########################## Write Result ##########################

    for img_id in range(trackFrame):

        inp_dim = int(args.reso)
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        curr_img_num_str = str(img_id + 1).zfill(6) + ".jpg"
        curr_img_path = osp.join(osp.realpath('.'), images, curr_img_num_str)
        processed_img, curr_img, dim = prep_image(curr_img_path, inp_dim)

        list(map(lambda x: write_result(x, curr_img), output_list[img_id]))

        # write a new image in destination_path
        det_names = "est_output/est_" + curr_img_num_str
        cv2.imwrite(det_names, curr_img)