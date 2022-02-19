## Introduction

In this project we implement a custom object detection machine using a method called "Transfer Learning" based on YOLOv3([1]).

More on transfer learning: https://en.wikipedia.org/wiki/Transfer_learning

The project is provided via Goolge Colab notebook as Google offers free and powerful GPU support which is necessary for the training phase.

The provided example is a model that can detect the 6 original Avengers (and extras...) but the code is very flexible and with just a tiny bit of changes you can train a
machine to identify any group of objects. Follow the instructions to understand how. 

## Instructions

1. Sign up to https://www.kaggle.com/ and download an API key.
2. Open `Avengers_Detector.ipynb` via Google Colab.
3. Change runtime type settings of Google Colab to 'GPU'.
4. Drag and drop your API key in the home directory of your Colab session.
5. Drag and drop an input video for detection (call it input.mp4) in the home directory of your Colab session.
6. (If you would like to train on your own custom dataset, upload it to Kaggle and download this dataset instead of the avengers dataset. all other steps remain the same)
7. Run all cells but the last and wait for the training to conclude.
8. The result will be generated in the 'res' folder in the home directory.

[1]: https://pjreddie.com/darknet/yolo/
