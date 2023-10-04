# Particle-filtered Visual Teach and Repeat Navigation

## Overview

This ROS package allows a robot to be taught a path through an environment using a monocular uncalibrated camera.<br>
PFVTR exploits the foundations of [Bearnav](https://github.com/gestom/stroll_bearnav).<br>
The whole method is described in [paper](https://ieeexplore.ieee.org/document/10042995). <br>
The neural network used by the PFVTR is described [here](https://github.com/Zdeeno/Siamese-network-image-alignment).


## First start

Once built, to run the system use `roslaunch pfvtr jackal.launch`. <br>
Inside the launch file, set required variables at the top of the file, pointing to topics which are required by the framework (odometry, camera, controller, cmd_vel).

## Usage

We recommend using tmux and we prepared a tmux launch script `start_session.sh` for you which sources the package in multiple windows. <br>
Our package requires input images of width 512 so there is a window `resize` in the tmux which uses standart resizing ROS node. You just need to run it with parameters based on your camera resolution. <br>
The first window is made for the main launch file. The second and third windows are for `mapping` and `repeating`. <br>

Once the package is running, you can begin mapping by publishing a message to the mapmaker module: <br>
`rostopic pub /pfvtr/mapmaker/goal [tab][tab]` <br>
There are multiple parameters in the action command:
- `sourceMap` - this is for continual mapping you can leave it empty
- `mapStep` - distance between map images, if you keep 0.0 it is automatically set to 1.0m
- `start` - set to True to start mapping, when your mapping is finished you have to publish this action again with start = false to save the map
- `mapName` - name of the map under which it is saved into `home/.ros` folder
- `SaveImgsForViz` - by default the framework saves only the image representations in latent space, if you want to save also jpg images set this to true

To replay a map, run: <br>
`rostopic pub /pfvtr/repeater/goal [tab][tab]` <br>
This action has also multiple parameters:
- `startPos` - how far in the map you want to start repeating - usually keep 0.0 beacause you want to repeat from start
- `endPos` - currently not working keep as is
- `traversals` - currently not working keep as is
- `imagePub` - lookaround window tells how many image forwards and backwards should be filtering done, we use 1 or 2.
- `mapName` - name of the map you want to repeat (you can fill in multiple maps of the SAME trajectory divided by "," - see in the article)

You can use the visualisation script `src/gui/particle_viz.py`, which publishes images showing position of the particles.

## Important

- We use time synchronization for odometry and camera topics! You won't be able to record a map if your topics have a very different timestamp.
- The robot is repeating the action commands from mapping phase so at the initial position there is command with 0 velocity, you have to push the robot little bit to start repeating. You also should not stop completelly during mapping because of this.
- Keep in mind that this is not production code it can yield some errors eventhough it is working properly. We also suggest to relaunch the code between the traversals.
