# CNNFace

This project aims to implement the MTCNN from "Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks" [1], using TensorFlow 2.x. It is still in early stage, 
although the code for face classification and bounding box regression is ready.

## Objective

Main objective is providing learning ground about face detection using deep learning with an easy to use TensorFlow 2.x implementation.

## Status

Pnet, Rnet, Onet implemented for classifying faces and bounding boxes regression. Landing marks are not supported yet.

## TODO

* Improve accuracy of each stage of the classifier;
* Improve performance during extraction and inference;
* Add proper documentation;
* Implement facial landmark localization;

## References

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao, "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," IEEE Signal Processing Letters. [(Github)](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)
2. [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)
