

But generally, the main structure of the project is illustrated in Fig 13 of the paper. 
There 2 main parts: computing the value function of the game (independent of Gazebo and ROS) and controlling the defender drone using the value function computed (which is a ROS node).  The ROS node would then just send velocity commands to low-level controller. For low-level control, I used PX4 as a low-level controller.  For Gazebo simulated drone, I would just recommend PX4.

You can divide the work into 2 parts: 1. understand the general method and figure out how to model the drone and compute the value function,  2.  Figure out how to write ROS2 node to control simulated drone in Gazebo. 

These two should be happening at the same time, since you need to test the drone in simulation to tune the parameters of the high-level model used for the game modeling. 


You can refer to this basic 1vs1 reach-avoid (RA) game for single integrator dynamics: MARAG/MRAG/hjvalue1v1_basement.py at master · Hu-Hanyang/MARAG. In my paper, this RA game are computed for the vertical and horizontal separately. for hessian jacobian stuiff

https://docs.px4.io/main/en/sim_gazebo_gz/index
https://docs.px4.io/main/en/sim_gazebo_gz/index





Gazebo simulation: Getting floor and walls as obstacles (super easy) half a day to get up. The hard part is definitely the drone. Figure out if there is an easy way to get crazyflie into a gazebo simulator. Minh’s simulator in ros1, go with:

https://www.bitcraze.io/2024/09/crazyflies-adventures-with-ros-2-and-gazebo/

Make some environment in gazebo and then put in drone 

Problem we design shouldnt be too hard. Dont make a challenging environment. For the simulator, have some obstacles, for the real world, use empty room 

For testing, break into parts: Just vertical & just horizontal so that if we dont end up finishing we have something. 

One person can do horizontal capture and tracking, second do vertical capture and tracking, if both work independently, should also work together.

Don’t drive the drone too fast, decouple horizontal and vertical assume that theyre independent, but when slow its easier to go specified speed in both dimensions. 

Find ways to make problem easier while keeping scope the same. 

Lab for crazyflie drones.

Gazebo harmonic, check versions before we do anything substantial. He thinks gazebo harmonic is what we are using for class/course. Check all the versions and make sure it’s all compatible. 

For the same reason, get robot right away.





[4:40 PM]I followed the Crazyswarm1 at that time.
[4:40 PM]Crazyswarm2 is based on ROS 2 which is compatible with what you are using I think. (edited)
https://imrclab.github.io/crazyswarm2/installation.html