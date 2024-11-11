# YOLOScope

# The code is ready for use 
But I plan to add more functionalities to make it more handy for use :)

## Important first step!! 

1. Download the YOLOv4 weights and model 
```
# The folder where the weights will be saved
%mkdir model

%cd model
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
!wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
!wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

2. Install libraries in your venv

3. Run the command to start the app :
```
python run_yolo.py
```

# User Interface overview


![image](https://github.com/user-attachments/assets/180e22dc-70af-4c2b-9e47-ab44fc098ff3)


# Predicted image


![image](https://github.com/user-attachments/assets/76e091ae-541f-4d88-923d-46fdc7364094)



