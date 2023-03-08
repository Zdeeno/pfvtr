# Bearnav2

## Overview

This ROS package allows a robot to be taught a path through an environment using a camera.
The robot can then retrace this path at a future point in time, correcting any error.
The theory can be found in the [linked paper.](http://eprints.lincoln.ac.uk/12501/7/surfnav_2010_JFR.pdf)
The system works by replaying the robot commands from during the training phase, and applying slight adjustments to them according to how the camera looks compared to during the teaching phase.
As long as the error is reasonable, the robot will converge to the original path.

## Installation

Clone the repository into a ROS workspace and build.


## Usage

Once built, to run the system use `roslaunch bearnav2 bearnav2-gui.launch`. May run slower, but provides more feedback using gui interface.
You can optionally run `roslaunch bearnav2 bearnav2-no-gui.launch` if these aren't required. Faster as no additional computations are done.

Inside these launch files, set the three required variables at the top of the file, pointing to your robot's camera topic, cmd\_vel topic, and odometry topic.

Don't forget to source your workspace!

Once the package is running, you can begin mapping by publishing a message to the mapmaker module:

`rostopic pub /bearnav2/mapmaker/goal [tab][tab]`
Fill in the mapName, for loading the map later!
Set start to `true`
Publish the message (enter) to start mapping (you can Ctrl+C this publish and it will not stop the mapping).
After you finnish your path, publish the same message with same mapName, but change start to `false`.

Note: Every line of the message above mapName is internal stuff for ROS, so do not worry about it.

To replay a map, run:

`rostopic pub /bearnav2/repeater/goal [tab][tab]`

Simply fill in the mapname field and your robot will begin to re-trace the path.

You can start following the map at different point by setting startPos to how far (in meters) from the beginning you want to start.
endPos is how far (in meters) from the beginning you want to end.

Note: Every line of the message above mapName is internal stuff for ROS, so do not worry about it.




## Improvements

 - Maps are now streamed from disk, therefore don't require waiting for them to be loaded, and can be much larger
 - Can cover strafe movement

## Principles
### Mapping
![Mapping](images/Mapping.svg)
### Replaying
![Navigating](images/Navigating.svg)
