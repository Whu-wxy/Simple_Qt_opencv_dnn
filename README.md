# Qt_opencv_dnn
Using deep learning model with opencv in Qt

# Configuration
Qt5.8-mingw,opencv-mingw

# File structure
Img_Object_detect-----detect a image

Object_detect-----detect a video

ssd_mobilenet_v1_coco_11_06_2017------saved model


# Usage
1.import opencv to Qt
INCLUDEPATH += D:\OpenCVMinGW3.4.1\include
LIBS += D:\OpenCVMinGW3.4.1\bin\libopencv_*.dll

2.Build your project

3.Put the saved model(.pb, .pbtxt) in the Debug/Release file

4.Change file path in Qt code

5.Run

# Result
![Result Image](/result.png)
