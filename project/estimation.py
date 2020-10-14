import argparse
import cv2
import os.path as osp
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
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
    parser.add_argument("--images", dest='images', help="Image / Directory containing count to perform detection upon",default="./MOT17-03-DPM/img1/", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)
    return parser.parse_args()


def write_result(x, img):
    write_object = x[0]
    id = int(float(write_object[1]))

    track_object = torch.from_numpy(np.array([float(write_object[2]), float(write_object[3]), float(write_object[4]), float(write_object[5])]))

    c1 = tuple(track_object[0:2].int())
    c2 = tuple(track_object[2:4].int())

    label = "GT"+str(int(id))

    color = colors[id % 14]

    cv2.rectangle(img, c1, c2, (255,255,255), 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 1)

    write_object = x[1]
    id = int(float(write_object[1]))

    track_object = torch.from_numpy(np.array([float(write_object[2]), float(write_object[3]), float(write_object[4]), float(write_object[5])]))

    c1 = tuple(track_object[0:2].int())
    c2 = tuple(track_object[2:4].int())

    label = "Track"+str(int(id))

    color = colors[id % 14]

    cv2.rectangle(img, c1, c2, (255, 255, 255), 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
    return img


# [frameNo , ID , left_X , left_Y , right_X , right_Y , mid_X , mid_Y , width , height]
# [   0    , 1  ,   2    ,   3    ,    4    ,    5    ,   6   ,   7   ,   8   ,   9   ]
def convert_to_track_array(line):
    right_X = float(line[2]) + float(line[4])
    right_Y = float(line[3]) + float(line[5])
    mid_X = float(line[2]) + (float(line[4]) / 2)
    mid_Y = float(line[3]) + (float(line[5]) / 2)
    width = float(line[4])
    height = float(line[5])

    parsedLine = [line[0], line[1], line[2], line[3], right_X, right_Y, mid_X, mid_Y, width, height]
    return parsedLine


# matching_result = [
#                     [ [MOT result[1]] , [track_result[0]] ]
#                     [ [MOT result[3]] , [track_result[4]] ]
#                                       ...
#                    ]
# def compare_results(track_result, MOT_result, threshold):
#     matching_result = []
#     matched_list = []
#
#     # frame loop
#     for i in range(1, MOT_result.__len__()):
#         matching_result.append([])
#         # MOT object loop
#         for MOT_object in MOT_result[i]:
#             # track object loop
#             for track_object in track_result[i]:
#                 distance_X = abs(float(track_object[6]) - float(MOT_object[6]))
#                 distance_Y = abs(float(track_object[7]) - float(MOT_object[7]))
#
#                 if float(MOT_object[8]) * threshold >= distance_X and float(MOT_object[9]) * threshold >= distance_Y:
#                     matching_result[i-1].append([MOT_object, track_object])
#                     matched_list.append(track_object)
#                     break
#
#             for pop_object in matched_list:
#                 track_result[i].remove(pop_object)
#             matched_list.clear()
#
#     return matching_result

def compare_results(track_result, MOT_result, threshold):
    matching_result = []
    iou_sum = 0
    object_sum = 0

    # frame loop
    for i in range(1, track_result.__len__()):
        matching_result.append([])
        result = 0
        GT_object_count = MOT_result[i].__len__()

        # MOT object loop
        for track_object in track_result[i]:
            # track object loop
            max_iou = 0
            max_iou_object = None

            for MOT_object in MOT_result[i]:
                xA = max(float(MOT_object[2]), float(track_object[2]))
                yA = max(float(MOT_object[3]), float(track_object[3]))
                xB = min(float(MOT_object[4]), float(track_object[4]))
                yB = min(float(MOT_object[5]), float(track_object[5]))

                interArea = max(0, xB - xA) * max(0, yB - yA)

                if interArea == 0:
                    continue

                boxAArea = MOT_object[8] * MOT_object[9]
                boxBArea = track_object[8] * track_object[9]

                iou = interArea / float(boxAArea + boxBArea - interArea)

                if iou > max_iou:
                    max_iou = iou
                    max_iou_object = MOT_object

            if max_iou_object != None:
                result += max_iou
                matching_result[i - 1].append([max_iou_object, track_object])
                MOT_result[i].remove(max_iou_object)

                iou_table[int(max_iou_object[1])][i][0] = int(float(track_object[1]))
                iou_table[int(max_iou_object[1])][i][1] = max_iou

        if matching_result[i - 1].__len__() != 0:
            iou_sum += result / matching_result[i - 1].__len__()
            object_sum += matching_result[i - 1].__len__() / GT_object_count

    print("Average IOU: " + str((iou_sum / matching_result.__len__())*100) + "%")
    print("Average Detection Object: " + str((object_sum / matching_result.__len__()) * 100) + "%")

    return matching_result

def track_table_result():
    line = track_table.numpy()

    for i in range(line.__len__()):
        for j in range(line[i].__len__()):
            track_table_file.write(str(line[i][j]) + " ")
        track_table_file.write("\n")
        track_table_file.flush()

# gt_id , frame , track_id , iou
def iou_table_result():
    chart_color = ['g', 'r', 'b', 'c', 'y', 'm']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(1, iou_table.__len__()):
        train_id = 0
        color_id = -1
        iou_sum = 0

        ax.set_xlabel('Frame')
        ax.set_ylabel('IOU')
        ax.axis([0, iou_table[i].__len__(), 0, 1])

        for j in range(1, iou_table[i].__len__()):
            color = 'k'

            if iou_table[i][j][1] != 0:
                if train_id != iou_table[i][j][0]:
                    color_id += 1
                    plt.text(j-1.15, iou_table[i][j][1]+0.03, "{}".format(iou_table[i][j][0]), fontsize=7, color=chart_color[color_id % 6])

                color = chart_color[color_id % 6]

            ax.plot([j-1, j], [iou_table[i][j-1][1], iou_table[i][j][1]], '-', color=color)
            ax.plot([j], [iou_table[i][j][1]], 'o', color=color, markersize=4)

            if iou_table[i][j][0] != 0:
                train_id = iou_table[i][j][0]

            iou_sum += iou_table[i][j][1]
            iou_table_file.write(str(i) + " " + str(j) + " " + str(iou_table[i][j][0]) + " " + str(iou_table[i][j][1]) + "\n")

        iou_sum /= iou_table[i].__len__()
        fig.suptitle(path.split('/')[2])
        ax.set_title("GT_ID : " + str(i) + "    /    IOU : " + str(round(iou_sum, 2)))

        if iou_sum != 0:
            plt.savefig("./chart/fig" + str(i) + ".png", dpi=300)
        plt.cla()


    iou_table_file.close()

path = './train/MOT17-05-DPM/'

if __name__ == '__main__':

    args = arg_parse()
    threshold = args.threshold
    images = args.images

####################### Track MOT Parsing #######################
    trackFile = open('./MOT_result/trackResult.txt', 'r')
    track_result = [[], []]
    max_object_id = -1

    curFrame = 1

    while True:
        line = trackFile.readline()

        if line == '':
            break

        if line.startswith('#'):
            max_object_id = int(line.split(",")[1])
            break

        line = line.split("\n")[0]
        parsed_line = line.split(",")

        if curFrame != float(parsed_line[0]):
            curFrame += 1
            track_result.append([])

        converted_line = convert_to_track_array(parsed_line)
        track_result[curFrame].append(converted_line)

    trackFile.close()

####################### MOT 17 GT Parsing #######################
    gtFile = open(path + 'gt/gt.txt', 'r')
    gt_result = [[]]
    count = curFrame
    max_gt_id = 0

    for frame in range(0, count):
        gt_result.append([])

    while True:
        line = gtFile.readline()

        if line == '':
            break

        line = line.split("\n")[0]
        parsed_line = line.split(",")

        if int(parsed_line[0]) > count:
            continue

        converted_line = convert_to_track_array(parsed_line)
        gt_result[int(converted_line[0])].append(converted_line)
        max_gt_id = int(converted_line[1])

    gtFile.close()

    print(path)
########################## Compare Result ##########################
    # GT_id Frame_num Track_id IOU
    iou_table = [[]]
    for gt_id in range(1, max_gt_id + 1):
        iou_table.append([])
        for frame_num in range(0, count + 1):
            iou_table[gt_id].append([0, 0])

    output_list = compare_results(track_result, gt_result, threshold)
    iou_table_file = open('./MOT_result/iouTable.txt', 'w')
    iou_table_result()

    track_table = torch.zeros(count+1, max_object_id+1, dtype=torch.int)

    for i in range(0, count+1):
        track_table[i][0] = i

    for i in range(0, max_object_id+1):
        track_table[0][i] = i

    for pair in output_list:
        for matched_object in pair:
            track_table[int(float(matched_object[0][0]))][int(float(matched_object[1][1]))] = int(float(matched_object[0][1]))

    track_table_file = open('./MOT_result/trackTable.txt', 'w')
    track_table_result()

########################## Estimate Accuracy ##########################
    # 프레임 바이 프레임
    previous_id_array = np.zeros(max_object_id + 1)
    result = 0.0

    for i in range(1, count + 1):
        id_count = 0
        matched_id_count = 0

        for j in range(1, max_object_id + 1):
            if track_table[i][j] == 0:
                continue

            if previous_id_array[j] == 0:
                previous_id_array[j] = track_table[i][j]
                continue

            if track_table[i][j] == previous_id_array[j]:
                matched_id_count += 1

            id_count += 1
            previous_id_array[j] = track_table[i][j]

        if id_count != 0:
            result += float(matched_id_count) / float(id_count)

    result = (result / (count-1)) * 100

    print("Frame By Frame: "+str(result)+"%")

    # 이니셜 아이디 키핑 레이트
    initial_id_array = np.zeros(max_object_id + 1)
    result = 0.0

    for i in range(1, max_object_id + 1):
        id_count = 0
        matched_id_count = 0

        for j in range(1, count + 1):
            if track_table[j][i] == 0:
                continue

            if initial_id_array[i] == 0:
                initial_id_array[i] = track_table[j][i]
                continue

            id_count += 1
            if track_table[j][i] == initial_id_array[i]:
                matched_id_count += 1

        if id_count != 0:
            result += float(matched_id_count) / float(id_count)

    result = (result / max_object_id) * 100

    print("Initial ID Keeping: " + str(result) + "%")

    # 처음과 끝
    start_id_array = np.zeros(max_object_id + 1)
    end_id_array = np.zeros(max_object_id + 1)
    id_count = 0
    matched_id_count = 0

    for i in range(1, max_object_id + 1):
        for j in range(1, count + 1):
            if track_table[j][i] != 0:
                if start_id_array[i] == 0:
                    start_id_array[i] = track_table[j][i]
                else:
                    end_id_array[i] = track_table[j][i]

    for i in range(1, max_object_id + 1):
        if start_id_array[i] == 0 or end_id_array[i] == 0:
            continue

        id_count += 1
        if start_id_array[i] == end_id_array[i]:
            matched_id_count += 1

    print("Initial Last Matched: " + str((matched_id_count / id_count)*100) + "%")

    # 메이저 아이디
    result = 0
    for i in range(1, max_object_id + 1):
        major_id = {}
        id_count = 0
        max_val = 0

        for j in range(1, count + 1):
            if track_table[j, i] == 0:
                continue

            id_count += 1
            if major_id.__contains__(int(track_table[j, i])):
                major_id[int(track_table[j, i])] += 1
            else:
                major_id[int(track_table[j, i])] = 1

        for value in major_id.values():
            if max_val < value:
                max_val = value

        if id_count != 0:
            result += max_val / id_count

    print("Major ID: " + str((result / max_object_id) * 100) + "%")

    kind_of_id = {}
    for i in range(1, max_object_id + 1):
        for j in range(1, count + 1):
            if kind_of_id.__contains__(int(track_table[j, i])):
                continue

            kind_of_id[int(track_table[j, i])] = 1

    print("Error: " + str(((max_object_id - kind_of_id.__len__()) / kind_of_id.__len__()) * 100) + "%")
    print("Max ID: " + str(max_object_id) + "   ID 종류: " + str(kind_of_id.__len__()))
########################## Write Result ##########################

    # for img_id in range(count):
    #     inp_dim = 416
    #
    #     curr_img_num_str = str(img_id + 1).zfill(6) + ".jpg"
    #     curr_img_path = osp.join(osp.realpath('.'), path + "img1/", curr_img_num_str)
    #     processed_img, curr_img, dim = prep_image(curr_img_path, inp_dim)
    #
    #     list(map(lambda x: write_result(x, curr_img), output_list[img_id]))
    #
    #     # write a new image in destination_path
    #     det_names = "est_output/est_" + curr_img_num_str
    #     cv2.imwrite(det_names, curr_img)