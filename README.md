# Multi_face_detection_et_track
This is part of my thesis project, it will be comparing with other two of my multi face detection and tracking projects

# Overview
This projects aims at for multi face detection and tracking, simply just for multi-face detection, you can run detector.py. For multi-face tracking, you can run the main.py that will call detector.py and tracker 

# Detection
The detectors mainly use 5 different detection models with different Model(Meta_Architecture), different backbone(Feature Extractors) and different dataset to fine-tuning the exists pre-trained models on COCO dataset with either WIDERFACE or FDDB dataset. Here only share the results of what I trained SSD Inception V2 with WIDERFACE dataset (will give you the bounding box of the face) and SSD Mobilenet V1 with FDDB dataset (will give you the bounding box of the Face and labels of eyes). Faster R-CNN inception v2 and R-FCN ResNet101 detection training result could be obtained by emailing jwang212@uottawa.ca

# Tracking

The tracking part is to use Kalman filter for the state estimation and Hungarian Algorithm to associate each frames for different Face ID. They implement on those 5 different detection models. Before face tracking offline from detectors, you can run the test script in main.py by set debug == True, then use the videos frames from the folder 2 to see if the detector work or not, and then set debug back to false (debug = False) to start track faces offline.
The tracking part is to use Kalman filter for the state estimation and Hungarian Algorithm to associate each frames for different Face ID. They implement on those 5 different detection models. Before face tracking offline from 

