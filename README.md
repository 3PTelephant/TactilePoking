# Where shall I touch? Vision-Guided Tactile Poking for Transparent Objects Grasping 
Repository for our T-Mech paper ["Where shall I touch? Vision-Guided Tactile Poking for Transparent Objects Grasping"](https://arxiv.org/abs/2208.09743).


## Installation
This code is tested with Ubuntu 20.04, Python3.7 and Pytorch 1.8.1, and CUDA 11.0.

Driver for [Intel RealSense](https://github.com/kevindehecker/librealsense).  

Driver for [UR5 robotic arm](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver). 

Driver for [Robotiq Gripper](https://github.com/ros-industrial/robotiq/pull/184).

## Hardware
[GelSight](https://www.mdpi.com/1424-8220/17/12/2762/htm) or other optical tactile sensors, such as [GelTip](http://ras.papercept.net/images/temp/IROS/files/2214.pdf), [GelSight Wedge](https://arxiv.org/abs/2106.08851) are required. 

## Usage
### System Calibration 
(1) Using the Tsai hand-eye calibration method to calibrate the transformation between RealSense Camera and UR5 robotic arm.

(2) You can also calibrate the transformation by putting a ArUco Marker on the table, calculate the transformation between RealSense camera and marker first, and then get the translation between marker and robotic base. 


### Running
```bash
python3 contact_trigger.py
python3 grasp_node_network.py
```
Pretrained model can be found [here]().

If you want to know more details, you can visit our [website](https://sites.google.com/view/tactilepoking) or send email to me (jiaqi.1.jiang@kcl.ac.uk).

If you use our [rendering code](https://github.com/3PTelephant/TransparentObjectRender) or this repository, please cite our paper:

```
@article{jiang2022shall,
  title={Where Shall I Touch? Vision-Guided Tactile Poking for Transparent Object Grasping},
  author={Jiang, Jiaqi and Cao, Guanqun and Butterworth, Aaron and Do, Thanh-Toan and Luo, Shan},
  journal={IEEE/ASME Transactions on Mechatronics},
  year={2022},
  publisher={IEEE}
}

```
