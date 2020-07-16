
Installation
================
**[ Project Source ]**
- ```git clone https://github.com/Sangheon-Yang/MOT.git```

**[ Yolo Weights ]**
- Move to ```project_05_31_2020``` folder
- DownLoad [yolov3.weight](https://pjreddie.com/media/files/yolov3.weights)

How To Execute
================

Version 01
----------
**[ for sequential images ]**

- ```python trackMOT.py --images 'PATH_TO_IMG_DIRECTORY' --det 'PATH_TO_DESTINATION' --count 'THE_NUMBER_OF_IMGS'```

- ex )   ```python trackMOT.py --images ./MOT17-03-DPM/img1/ --det MOT_output --count 300```

**[ for video file ]**

- ```python trackVideo.py --video 'PATH_TO_VIDEOFILE'```

- ex ) ```python trackMOT.py --video ./sq.mp4```

**[ for camera capture ]**

- ```python trackVideo.py --camera True```

Version 02
----------

**[ for sequential images ]**

- ```python trackMOT_4.py --images 'PATH_TO_IMG_DIRECTORY' --det 'PATH_TO_DESTINATION' --count 'THE_NUMBER_OF_IMGS'```

- ex )   ```python trackMOT_4.py --images ./MOT17-03-DPM/img1/ --det MOT_output --count 300```


Network & Weights & Classes used for Object Detection
===============

- Network : yolov3.cfg 

- Weights : yolov3.weights

- classes : coco.names ( total 80 classes exist : 'person', 'bicycle', 'car', 'motorbike' ... etc )


About Object Detection
================
- Objects are classified to total 80 classes (can fix it to lower number by editing main function)

- The Result of Object-detection in a frame is a PyTorch Tensor with (```n```, 8 ) dimension, ```n``` is the number of Detected-Objects

- The Result of Object-detection contains the information of 4-point of (x,y) coordinate position, confidence, class

- We edited the Result of Object-detection format to  (```n```, 9 ) dimension adding ```Object_id``` and changing the 4-point of (x,y) coordinate position to mid-point, width, height coordinate in order to make it easier to compare objects' location and check whether Tracking is well performing. 


Tracking Algorithm
=================

[Version 01](https://github.com/Sangheon-Yang/MOT/wiki/Implementation-Version-01)
----------

[Version 02](https://github.com/Sangheon-Yang/MOT/wiki/Implementation-Version-02)
----------

Further Step
==============

Quantitative evaluation and Qualitative evaluation are needed. 

The Standard Measurement of those evaluations should be established as soon as possible.




