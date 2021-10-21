# Shop-floor-Monitoring
Shop-floor monitoring system using YOLOv4-DeepSort. YOLOv4-DeepSort is a object tracking algorithm that uses deep convolutional neural networks. System calculates factory indicators using ID and coordinates of the target object.



## Resulting Video
![stable](https://user-images.githubusercontent.com/78286605/138114240-bd7a6440-f9b3-49cd-a2d7-e30cd0b7edec.gif)

## Getting Started
Cloning [Yolov4-DeepSort repository](https://github.com/theAIGuysCode/yolov4-deepsort) and Shop-floor monitoring repository in your computer.   
Move 'Object_tracker_SF.py' file into YoloV4-deepsort repository file.

Install the proper dependencies in repository path via Anaconda.   

**Conda**   
```
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```   
**Pip**
```
# TensorFlow GPU
pip install -r requirements-gpu.txt
```   

### Step 3
Downloading Pre-trained Weights(Custom Object)   
링크 삽입하기   
Copy and paste yolov4.weights from your downloads folder into the 'data' folder of Yolov4 repository.


