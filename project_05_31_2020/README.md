[May, 31, 2020] Simple Multiple-Object-Tracking for Sequential images and Video(including camera)
=======

How To Execute
-------------------
**[ for sequential images ]**

- ```python trackMOT.py --images 'PATH_TO_IMG_DIRECTORY' --det 'PATH_TO_DESTINATION' --count 'THE_NUMBER_OF_IMGS'```

- ex )   ```python trackMOT.py --images ./MOT17-03-DPM/img1/ --det MOT_output --count 300```

**[ for video file ]**

- ```python trackVideo.py --video 'PATH_TO_VIDEOFILE'```

- ex ) ```python trackMOT.py --video ./sq.mp4```

**[ for camera capture ]**

- ```python trackVideo.py --camera True```


Network & Weights & Classes used for Object Detection
----------------

- Network : yolov3.cfg 

- Weights : yolov3.weights

- classes : coco.names ( total 80 classes exist : 'person', 'bicycle', 'car', 'motorbike' ... etc )


About Object Detection
---------------
- Objects are classified to total 80 classes (can fix it to lower number by editing main function)

- The Result of Object-detection in a frame is a PyTorch Tensor with (```n```, 8 ) dimension, ```n``` is the number of Detected-Objects

- The Result of Object-detection contains the information of 4-point of (x,y) coordinate position, confidence, class

- We edited the Result of Object-detection format to  (```n```, 9 ) dimension adding ```Object_id``` and changing the 4-point of (x,y) coordinate position to mid-point, width, height coordinate in order to make it easier to compare objects' location and check whether Tracking is well performing. 


About Tracking Algorithm Used in this Update
--------------

[ENGLISH]

- The Algorithm for Tracking Objects is Simple.

- Using only previous frame's information of detected objects and its position((x,y) coordinate) and class, compare it to the current frame's information of objects and its position and class.

- If the object has (short enough) minimum euclidean distance between previous frame and current frame and their classes are identical, then consider it as a same object.

- If not, they are different object.

- When calculating euclidean distance, we set the boundary of maximum distance so that only those objects in short-enough distance can be considered as a same object.

- Also grant unique id number to all object So that we can easily find out whether the Object-Tracking is well performing or not.


[KOREAN]

- 매우 간단한 알고리즘이 적용되었다.

- 직전 프레임에서 detect된 물체들의 (x,y)좌표위치정보와 class 정보만을 사용하여, 현재 프레임에서 detect된 물체들의 (x,y)좌표위치정보와 class 정보를 비교한다.

- 만약 현재 프레임의 특정한 물체가 이전프레임의 특정한 물체와 좌표상으로 (충분히 짧은) 가장 짧은 거리에 위치하고, 이 두 물체들의 class가 서로 같다면, 이 두물체를 같은 하나의 물체로 인식한다.

- 만약 그렇지 않은 경우, 이 두 물체는 다른 물체로 인식한다.

- 좌표상 거리 비교시에는 최고치 기준값을 정해주어 충분히 짧은 거리에 위치해야만 같은 물체로 인식하도록 한다.

- 또한 각 물체마다 고유의 id 번호를 부여하여 실제로 Tracking이 잘 이루어지는지 확인하기 쉽도록 했다.



Result
--------


Conclusion & Further Step
----------------

cpu만 사용,  속도, 정확성 문제가 너무 심함

cpu->gpu

ref Frame의 갯수를 더 늘려서 정확도를 개선하는 방향으로 구현해볼 예정




