# CNNFace

This project aims to implement the MTCNN from "Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks" [1], using TensorFlow 2.x. It is still in early stage, 
although the code for face classification and bounding box regression is ready.

## Objective

Main objective is to provide learning ground about face detection using deep learning with an easy to use TensorFlow 2.x implementation.

## Status

Pnet, Rnet, Onet implemented for classifying faces and bounding boxes regression. Landing marks are not supported yet.

## Instructions

All configuration is available through config.json. I have added configurations which i achieved the best results
in my machine. You can change as you wish. Some of then were taken from [1], others i found empirically.

You will need tensorflow 2 (tested with 2.4.1) and matplotlib (tested with matplotlib 3.4.0) installed.

Firstly, you need to extract widerface files to preprocessing.widerface.*.
Then, run the following scripts:
1. preprocess_pnet.py
2. train_pnet.py
3. preprocess_rnet.py
4. train_rnet.py
5. preprocess_onet.py
6. train_onet.py

without parameters. It is straightforward, but it will take a while.
After training all stages, you can run detect.py passing the path of the image.

## TODO

* Implement facial landmark localization;

## Other TF Implementations

* https://github.com/AITTSMD/MTCNN-Tensorflow
* https://github.com/ipazc/mtcnn

## References

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao, "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," IEEE Signal Processing Letters. https://kpzhang93.github.io/MTCNN_face_detection_alignment/
