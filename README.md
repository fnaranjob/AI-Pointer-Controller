# Computer Pointer Controller

This is an AI computer pointer controller written in Python, it allows you to move your mouse pointer using your gaze, it uses Intel's OpenVino Toolkit to perform inference on video feed from a webcam, however, running it is also possible using pre recorded video.

*WARNING*: notice that OpenVino uses OpenCV to process video feed, so, not all video formats might be compatible with your system, and this varies between different OSs (Windows, for example, only supports .AVI videos)



## Project Set Up and Installation



### 1 - Install OpenVino Toolkit

Follow the instructions for your OS here:

* [Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
* [Windows](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)
* [MacOS](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)



### 2 - Download models

This app requires 4 pretrained models to run, you can find the used models docs here:

* [Face Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

You can use OpenVino's [Model Downloader](https://docs.openvinotoolkit.org/latest/_tools_downloader_README.html) to get these models, as they all are part of [Intel's Open Model Zoo](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)



### 3 - Clone or download source code

This repo contains all source code required to run this application, create a folder to contain all required files and either clone this repo in it or download the source files as a ZIP file and extract them



### 4 - If not already installed, install Python 3

To run this application, your system has to have Python3 installed and added to your PATH, see instructions [here](https://www.tutorialspoint.com/python/python_environment.htm)



### 5 - Install all dependencies

Inside your source folder you will find a requirements.txt file that contains a list of all modules required to run the application, to install the required modules, open a terminal/command prompt, navigate you the folder that contains all the source files and run:

`pip install requirements.txt`



## Demo

*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
