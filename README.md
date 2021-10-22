# Shop-floor-Monitoring
Shop-floor monitoring system using YOLOv4-DeepSort. YOLOv4-DeepSort is a object tracking algorithm that uses deep convolutional neural networks. System calculates factory indicators using ID and coordinates of the target object.



## Resulting Video
![stable](https://user-images.githubusercontent.com/78286605/138114240-bd7a6440-f9b3-49cd-a2d7-e30cd0b7edec.gif)

## Getting Started
Cloning [Yolov4-DeepSort repository](https://github.com/theAIGuysCode/yolov4-deepsort) and Shop-floor monitoring repository in your computer.   
Move 'Object_tracker_SF.py' file into YoloV4-deepsort repository file.

Install the proper dependencies in repository path via Anaconda.   
I recommend using GPU for real time.


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

## Downloading Pre-trained Weights(Custom object)
[yolov4.weights](https://drive.google.com/file/d/1pVvB2SQoSqM3zq2s_xUmbke-1G7OR5iD/view?usp=sharing)
Copy and paste yolov4.weights from your downloads folder into the 'data' folder of Yolov4 repository.

## Running the Tracker with YOLOv4-DeepSort
First, convert .weights into TensorFlow model.   
Then run the object_tracker_SF.py using camera.   
```
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on camera (set video flag to 1)
python object_tracker_SF.py --video 1 --model yolov4
```  
--video flag number can be differ.   
It will be helpful if you refer to Yolov4-Deepsort repository's README.

