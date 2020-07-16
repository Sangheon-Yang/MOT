[May, 31, 2020] Simple Multiple-Object-Tracking for Sequential images & Video
=======
Installation
-------------------
**[ Project Source ]**
- ```git clone https://github.com/Sangheon-Yang/MOT.git```

**[ Yolo Weights ]**
- Install [yolov3.weight](https://pjreddie.com/media/files/yolov3.weights)

- Move to ```project_05_31_2020``` folder

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

**ENGLISH**

- The Algorithm for Tracking Objects is Simple.

- Using only previous frame's information of detected objects and its position((x,y) coordinate) and class, compare it to the current frame's information of objects and its position and class.

- If the object has (short enough) minimum euclidean distance between previous frame and current frame and their classes are identical, then consider it as a same object.

- If not, they are different object.

- When calculating euclidean distance, we set the boundary of maximum distance so that only those objects in short-enough distance can be considered as a same object.

- Also grant unique id number to all object So that we can easily find out whether the Object-Tracking is well performing or not.


**KOREAN**

- 매우 간단한 알고리즘이 적용되었다.

- 직전 프레임에서 detect된 물체들의 (x,y)좌표위치정보와 class 정보만을 사용하여, 현재 프레임에서 detect된 물체들의 (x,y)좌표위치정보와 class 정보를 비교한다.

- 만약 현재 프레임의 특정한 물체가 이전프레임의 특정한 물체와 좌표상으로 (충분히 짧은) 가장 짧은 거리에 위치하고, 이 두 물체들의 class가 서로 같다면, 이 두물체를 같은 하나의 물체로 인식한다.

- 만약 그렇지 않은 경우, 이 두 물체는 다른 물체로 인식한다.

- 좌표상 거리 비교시에는 최고치 기준값을 정해주어 충분히 짧은 거리에 위치해야만 같은 물체로 인식하도록 한다.

- 또한 각 물체마다 고유의 id 번호를 부여하여 실제로 Tracking이 잘 이루어지는지 확인하기 쉽도록 했다.



Result of Sample Test
----------------

- dataset: ```./MOT17-03-DPM/img1/```    ```000001.jpg ~ 000200.jpg``` , 200 sequential imgs.

- result img files  in ```./MOT_out/ ```,  gif files in ```./gif_result/ ```



Result Analysis
-------------

**ENGLISH**

- Implementation was done by CPU Programming, So that it takes about 1 to 1.5 second per frame when processing Image. Since there are dozens of frames in a second of video, it will take a lot of times to handle video files.

- Since we only use the information of Objects Deteced in Previous frame to grant the Object_id to the Objects Detected in Current frame, the accuracy of Tracking is tend to be really low. 

**KOREAN**

- CPU만 사용하는 방법으로 구현하였기 때문에 한개의 프레임당 1초~1.5초 정도의 처리 시간이 소모된다. 초당 수십개의 프레임을 처리해야 하는 동영상을 처리할 시 실제 동영상의 길이보다 수십배 더 긴 시간이 소모된다. 

- 직전프레임에서 Detect된 물체들의 정보만 사용하여 현재 프레임에서 Detect된 물체들의 id를 부여하기 때문에 Tracking의 정확성이 매우 낮은 편이다.


Further Step
-------------

**ENGLISH**

- We are going to make it faster by using GPU-Programming(CUDA).

- We are going to use not only Previous frame's Detection, but also few frames before it, so that we make some meaningful data out of them. This method would improve the accuracy of granting Object_id to appropriate Object, and it would be helpful for improving overall accuracy of tracking.

**KOREAN**

- GPU 프로그래밍 코드(CUDA) 사용을 통해 프레임 처리 속도를 향상시키는 방향으로 개발한다.

- 여러개의 이전 프레임에서 Detect된 물체들의 정보를 적절히 혼합하여 현재 프레임에서 Detect된 물체들의 id를 부여할때 정확도를 더 높여 Tracking의 정확성을 개선하는 방향으로 개발한다.





