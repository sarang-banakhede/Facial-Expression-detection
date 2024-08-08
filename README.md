# Face Detection and Emotion Classification 
(see demo :- " https://www.linkedin.com/posts/sarang-banakhede-79327823a_ai-machinelearning-computervision-activity-7226737131547717632-t8gH?utm_source=share&utm_medium=member_desktop " )

# Overview
This project presents a comprehensive pipeline for detecting faces in video frames and classifying the emotional expressions of the detected faces. The pipeline leverages the power of the YOLOv7 object detection model for face detection and a custom VGG16-based classifier for emotion classification.

# Face Detection
The face detection component of the pipeline utilizes the "Face Detection Dataset" from Kaggle, available at https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset. This dataset provides a diverse collection of labeled face images, which are used to train the YOLOv7 object detection model to accurately locate and identify faces within video frames.

# Emotion Classification
The emotion classification component of the pipeline involves the creation of a custom dataset, consisting of images representing the following seven emotional categories: 'Cry', 'Surprise', 'angry', 'confuse', 'happy', 'neutral', and 'sad'. This dataset is used to train a VGG16-based classification model, which is capable of accurately identifying the emotional state of the detected faces.

# Pipeline Flow
The overall pipeline flow is as follows:
-> Video frames are processed using the trained YOLOv7 object detection model to locate and extract the detected faces.
-> The extracted face images are then passed through the VGG16-based emotion classification model, which assigns an emotional label to each face.
-> The processed video frames, along with the detected faces and their corresponding emotional labels, are then presented as the final output of the pipeline.

# How to use
### To use this model, first install Requirments 
### Then train the expression model (expression_train.py).
### Then, download and install the dependencies of YOLOv7 and train the YOLOv7 model(yolo_train.py).
### Finally, execute the main.ipynb file.
