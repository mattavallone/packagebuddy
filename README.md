# Package Buddy

Package Buddy is a project team at New York University. The goal of the project is to design and implement a robot that can enter a building, traverse different floors, and successfully deliver a package to a given room. The robot has no prior knowledge of the building floor plans, nor access to GPS signal. It must rely solely on onboard sensors, including camera vision, ultrasonic range sensors, depth sensor, an IMU, and odometry measurements. It can use a speaker to send auditory messages to humans around it, if it needs assistance with something (e.g. opening a door). The team must integrate hardware and create software to perform five main robot tasks. 

## Tasks
|         Robot Tasks           |
|:--------------------------|
| **1**.   Roam a space without the aid of a predefined map or GPS signal |
| **2**.   Avoid collisions with stationary and moving objects |
| **3**.   Read door numbers to locate a room |
| **4**.   Find elevators and enter and exit them on the correct floor |
| **5**.   Collaborate with other robots in the area |

## Goals
There are several goals for the project, each building upon the prior with greater complexity. On the most basic level, the robot will be able to traverse a single floor to deliver a package, while avoiding stationary objects. Once it is able to achieve that, we will add dynamically moving objects for the robot to avoid colliding with. After that, we will have the robot deliver a package on a different floor using an elevator. The final goal is to have multiple robots operating simultaneously and communicating with each other.

| Number of Robots	| Objects	| Number of Floors|
|:-------------------:|:---------:|:-----------------:|
|One | Static | One |
|One | Static & Dynamic | One |
|One | Static & Dynamic | Two |
|Multiple | Static & Dynamic | Two |

## Application
The main application of Package Buddy would be in the logistics and transportation sector.  Companies like UPS and FedEx are investing millions of dollars into AVs in order to reduce costs and meet demands of the 21st century supply chain.  Other companies, like Dominos and Amazon, are using robots to bring products straight to people’s doorsteps.  Currently, indoor deliveries require a human to bring the parcel directly to the correct location inside the building.  Mail carriers may need special access to these places, creating possible security vulnerabilities.  Delivering packages one-by-one also lacks efficiency.  This makes shipping times longer and slows down progress.

Package Buddy seeks to solve these issues and improve the way parcels are delivered in the future.  The vision is for a delivery person to unload a fleet of these robots from a delivery truck, each with a payload to transport.  They will be programmed with the desired rooms to bring the packages to inside the building, and autonomously navigate in an efficient manner to successfully deliver them.

## Hardware
* iRobot Create 2 Robot
* NVIDIA Jetson TX2 Development Kit
* Intel Realsense D435i Camera
* Logitech c270 Web Camera
* Arduino Uno
* HC-SR04 Ultrasonic Range Sensors
* Servo Motors SG90
* 2W 8 Ohm Magnet Speaker
* 4 Port USB Hub
* 3D Printed Camera Mount

## Software
Robot Operating System (ROS) is used to integrate the software. ROS is an open-source collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior.  It serves as a framework for developing robot software that is easily maintainable, scalable, and extendable for many different types of applications.

### ROS Packages Used
* create_autonomy
* realsense-ros
* cv_camera
* rosserial
