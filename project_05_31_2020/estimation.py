def printFrame(frame):
    # Frame 출력 함수

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


def convert_to_midPoint(line):
    """
    change line
    [frameNo , ID , leftCoordinate , topCoordinate , width , height , conf , x , y , z]
    to
    [frameNo, midPointX, midPointY , width, height]
    """

    midPointX = float(line[2]) + (float(line[4]) / 2)
    midPointY = float(line[3]) + (float(line[5]) / 2)

    parsedLine = [line[0], midPointX, midPointY, line[4], line[5]]
    return parsedLine

def compare_results(track_result, MOT_result, threshold):
    # matching_result = [
    #                     [ [MOT result[1]] , [track_result[0]] ]
    #                     [ [MOT result[3]] , [track_result[4]] ]
    #                                       ...
    #                    ]
    matching_result = []
    matched_list = []

    # frame loop
    for i in range(1, MOT_result.__len__()):
        # MOT object loop
        for MOT_object in MOT_result[i]:
            # track object loop
            for track_object in track_result[i]:
                distance_X = abs(float(track_object[1]) - float(MOT_object[1]))
                distance_Y = abs(float(track_object[2]) - float(MOT_object[2]))

                if float(MOT_object[3]) * threshold >= distance_X and float(MOT_object[4]) * threshold >= distance_Y:
                    matching_result.append([MOT_object, track_object])
                    matched_list.append(track_object)
                    break

            for pop_object in matched_list:
                track_result[i].remove(pop_object)
            matched_list.clear()

    printFrame(matching_result)
    return matching_result

if __name__ == '__main__':

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
    compare_results(track_result, mot_result, 1/5)