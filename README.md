Hand Gesture Recognition Using MediaPipe

Author: Akio Araki
Institution: Kyushu University
License: Apache 2.0

This project demonstrates real-time hand pose estimation and gesture recognition using MediaPipe and Python. The system recognizes hand signs and finger gestures using a simple Multi-Layer Perceptron (MLP) model trained on detected hand keypoints. This project was developed as part of my university coursework.

Features

Real-time detection of hand keypoints using MediaPipe.

Recognition of hand signs and finger gestures using pre-trained TFLite models.

Data collection and model retraining support for custom gestures.

Lightweight Python implementation with OpenCV for webcam capture.

Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>

Requirements

Python 3.7 or higher

MediaPipe
 0.8.1 or later

OpenCV 3.4.2 or later

TensorFlow 2.3.0 or later (tf-nightly 2.5.0.dev if using LSTM models)

scikit-learn 0.23.2 or later (optional, for confusion matrix display)

matplotlib 3.3.2 or later (optional, for confusion matrix display)

How to Run

Run the demo using your webcam:

python app.py

Optional arguments:

--device : Camera device number (default 0)

--width : Capture width (default 960)

--height : Capture height (default 540)

--use_static_image_mode : Enable static image mode in MediaPipe

--min_detection_confidence : Detection confidence threshold (default 0.5)

--min_tracking_confidence : Tracking confidence threshold (default 0.5)

Training
Hand Sign Recognition

Collect keypoint data by pressing "k" to enter logging mode.

Assign class IDs (0, 1, 2, etc.) for each gesture.

Run keypoint_classification.ipynb to train the MLP model.

Finger Gesture Recognition

Collect fingertip coordinate history by pressing "h".

Assign class IDs for each gesture type.

Run point_history_classification.ipynb to train the model.

Notes

The project supports retraining with custom gestures and additional data.

Models are exported as TFLite for efficient inference.

LSTM-based sequence recognition is available by setting use_lstm = True in the training notebooks (requires tf-nightly).
