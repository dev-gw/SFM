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

## Difference from the Original Code   
You can initialize variables and change values in ShopFloor.py modules.   

I summarize the added parts of the object_tracker_SF.py   
`line 105~120` - Initialize indicator variables   
`line 122~123` - Connect GUI client   
`line 197~218` - Calculate indicators for each class.   
`line 239~242` - Draw start and end line   
`line 251~256` - Set boundingbox color   
`line 262~263` - Calculate the center point   
`line 265~269` - Send indicators to DB(Updating)   
`line 271~281` - Track object through the start line   
`line 284~301` - Check object through the end line and change variables   
`line 303~315` - Calculate cycletime   
`line 317~331` - Check Error point   
`line 333~335` - Function that draws indicators   
`line 341` - Send indicators to GUI

