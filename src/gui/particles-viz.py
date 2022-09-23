#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from sensor_msgs.msg import Image
from bearnav2.msg import FloatList
import matplotlib.pyplot as plt
import numpy as np
import ros_numpy

pub = None


def callback(msg):
    if pub.get_num_connections() == 0:
        return

    print("particles received")

    plt.clf()
    fig = plt.figure()
    ax = plt.axes()
    ax.title.set_text("Particle filter")
    msg_size = len(msg.data) - 2
    part_size = msg_size//3
    distances = np.array(msg.data[:part_size])
    displacements = np.array(msg.data[part_size:part_size*2])
    maps = np.array(msg.data[part_size*2:-2])
    estimate_x = msg.data[-1]
    estimate_y = msg.data[-2]
    ax.plot(displacements[maps == 0], distances[maps == 0], "yo", alpha=0.4)
    ax.plot(displacements[maps == 1], distances[maps == 1], "bo", alpha=0.4)
    ax.plot(displacements[maps == 2], distances[maps == 2], "ko", alpha=0.4)
    ax.plot(estimate_x, estimate_y, "rx")
    ax.set_xlim([-1.0, 1.0])
    curr_window = np.mean(distances)//3
    ax.set_ylim([3*curr_window, 3*(curr_window+1)])
    # ax.set_ylim([np.min(distances), np.max(distances)])
    ax.grid()
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    msg = ros_numpy.msgify(Image, img, "rgb8")
    pub.publish(msg)


if __name__ == "__main__":

    rospy.init_node("pf_viz")
    pub = rospy.Publisher("pf_viz", Image, queue_size=0)
    rospy.Subscriber("/bearnav2/particles", FloatList, callback)
    print("PF viz ready...")
    rospy.spin()
