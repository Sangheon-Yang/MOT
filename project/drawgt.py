import os.path as osp
import cv2
import numpy as np
import torch
from preprocess import prep_image, inp_to_image

colors = [(255, 255, 255), (255, 100, 100), (0, 255, 0), (110, 110, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 150), (255, 150, 255), (150, 255, 255), (150, 255, 150), (150, 150, 255), (255, 150, 150), (150, 150, 150)]

def write_(x, img):
    track_object = torch.from_numpy(np.array([float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]))
    c1 = tuple(track_object[2:4].int())
    c2 = tuple(track_object[4:6].int())
    label = str(int(track_object[1]))

    color = colors[int(track_object[1]) % 14]

    cv2.rectangle(img, c1, c2, color, 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
    return img

def convert_to_drawPoint(line):

    rightPointX = float(line[2]) + float(line[4])
    rightPointY = float(line[3]) + float(line[5])

    parsedLine = [line[0], line[1], line[2], line[3], rightPointX, rightPointY]
    return parsedLine

path = './train/MOT17-04-DPM/'

if __name__ == '__main__':
    gtFile = open(path + 'gt/gt.txt', 'r')
    gt_result = [[]]
    images = 1050

    for frame in range(0, images):
        gt_result.append([])

    while True:
        line = gtFile.readline()

        if line == '':
            break

        line = line.split("\n")[0]
        parsed_line = line.split(",")

        if int(parsed_line[0]) > images:
            continue

        converted_line = convert_to_drawPoint(parsed_line)
        gt_result[int(converted_line[0])].append(converted_line)

    gtFile.close()

    for img_id in range(images):

        inp_dim = 416

        curr_img_num_str = str(img_id + 1).zfill(6) + ".jpg"
        curr_img_path = osp.join(osp.realpath('.'), path + "img1/", curr_img_num_str)
        processed_img, curr_img, dim = prep_image(curr_img_path, inp_dim)

        list(map(lambda x: write_(x, curr_img), gt_result[img_id]))

        # write a new image in destination_path
        det_names = "gt_output/gt_" + curr_img_num_str
        cv2.imwrite(det_names, curr_img)