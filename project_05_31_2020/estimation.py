
def printFrame(frame):
    # Frame 출력 함수

    frameSize = frame.__len__()
    frameCount = 0

    while frameCount < frameSize:
        print("frame"+str(frameCount)+": ")

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
    [midPointX, midPointY , width, height]
    """

    midPointX = float(line[2]) + (float(line[4])/2)
    midPointY = float(line[3]) + (float(line[5]) / 2)

    parsedLine = [midPointX, midPointY, line[4], line[5]]
    return parsedLine


if __name__ == '__main__':
    detFile = open('./MOT17-03-DPM/det/det.txt', 'r')
    curFrame = 1
    frame = [[], []]

    while True:

        line = detFile.readline()

        if line == '':
            break

        parsed_line = line.split(",")
        converted_line = convert_to_midPoint(parsed_line)

        if curFrame != float(parsed_line[0]):
            curFrame += 1
            frame.append([])

        frame[curFrame].append(converted_line)

    printFrame(frame)