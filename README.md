# Multi_face_detection_et_track
This is part of my thesis project, it will be comparing with other two of my multi face detection and tracking projects (Will be released soon)

# Overview
This projects aims at multi face detection and tracking. First for multi-face detection, you can run detector.py; for multi-face tracking, you can run the main.py that will call detector.py and tracker.py 

# Detection
The detectors mainly use 5 different detectors with different models(Meta_Architecture), different backbone(Feature Extractors) and different dataset to fine-tuning the exists pre-trained models on COCO dataset with either WIDERFACE or FDDB dataset. Here only share the results of what I trained SSD Inception V2 with WIDERFACE dataset (will give you the bounding box with the label of faces) and SSD Mobilenet V1 with FDDB dataset (will give you the bounding box with label of faces and eyes). Faster R-CNN inception v2 with WIDERFACE, SSD Inception V2 with FDDB and R-FCN ResNet101 with WIDERFACE plus MTCNNN detection training result could be obtained by emailing jwang212@uottawa.ca

# Tracking

The tracking part is to use Kalman filter for the state estimation and Hungarian Algorithm to associate each frames for different Face ID. They implement on those 5 different detection models. Before face tracking offline from detectors, you can run the test script in main.py by set debug == True, then use the videos frames from the folder 2 to see if the detector work or not, if it works without errors, then you can set debug back to false (debug = False) to start tracking faces offline.


