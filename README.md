# vision_ros2_node

 This program depends on [Ar-Ray-code/bbox_ex_msgs](//github.com/Ar-Ray-code/bbox_ex_msgs.git)

## Installation

```
mkdir -p {your_ws}/src
cd {your_ws}/src
git clone https://github.com/Ar-Ray-code/bbox_ex_msgs.git
```

## Demo
```
mkdir -p {your_ws}
source ./install/setup.bash
ros2 launch vision_ros2_node img_process_launch.py
```

## Requirements
- ROS2 
- OpenCV 4
- PyTorch
- bbox_ex_msgs
