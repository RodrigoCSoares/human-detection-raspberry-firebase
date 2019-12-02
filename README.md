# Human detection with Firebase connection for Raspberry PI
 
## Introduction
First of all, I would like to thank EdjeElectronics for making his repository available by implementing object detection with TensorFlow from a Raspberry PI, this project is an adaptation of his project.
The purpose of this system is to detect how many people are in any environment from a webcam (connected or via IP) and send this information to Firebase Cloud Firestore.
 
## Requirements
- Firebase project created
- Firebase Credentials that will be used to access Cloud Firestore
- Raspberry PI 3 || Raspberry PI 4
- Webcam
 
## How to use
- Clone this repository into any system folder.
- Copy your Firebase credentials file into the database folder.
- Access the project's folder via terminal
- Create and access a virtual environment with Python 3 (optional)
- Run the requirements.sh file (bash requirements.sh)
- Run: python3 detector.py
- Some flags are available for the above command, which are:
    - --modeldir: Folder in which the .tflite template is located, REQUIRED
    - --graph, Filename .tflite, if different from detect.tflite
    - --labels, Filename labelmap, if different from labelmap.txt
    - --threshold, minimum confidence threshold for object detection
    - --sleep, Set the number of seconds between detections
    - --cameraip, camera's IP
    - --showlog, True to show log or False to do not show
