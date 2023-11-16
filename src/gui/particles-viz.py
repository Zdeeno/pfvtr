#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from sensor_msgs.msg import Image
from pfvtr.msg import FloatList
import matplotlib.pyplot as plt
import numpy as np
import ros_numpy

pub = None
index = 1


def callback(msg):
    global index
    if pub.get_num_connections() == 0:
        return

    # print("particles received")

    plt.clf()
    fig = plt.figure()
    ax = plt.axes()
    ax.title.set_text("Particles")
    msg_size = len(msg.data) - 2
    part_size = msg_size//3
    distances = np.array(msg.data[:part_size])
    displacements = np.array(msg.data[part_size:part_size*2])
    maps = np.array(msg.data[part_size*2:-2])
    estimate_x = msg.data[-1]
    estimate_y = msg.data[-2]
    ax.plot(displacements[maps == 0], distances[maps == 0], "yo", alpha=0.3)
    ax.plot(displacements[maps == 1], distances[maps == 1], "bo", alpha=0.3)
    # ax.plot(displacements[maps == 2], distances[maps == 2], "ko", alpha=0.2)
    ax.plot(estimate_x, estimate_y, "rx")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([estimate_y - 3.0, estimate_y + 3.0])
    ax.legend(["particles"]) # , "faulty map", "estimate"])
    ax.set_xlabel("Displacement [%]")
    ax.set_ylabel("Distance [m]")
    # ax.set_ylim([np.min(distances), np.max(distances)])
    ax.grid()
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.savefig("images/" + str(index) + ".png")
    # index += 1
    # plt.close()

    # print("map0:",np.sum(maps == 0))
    # print("map1:", np.sum(maps == 1))
    msg = ros_numpy.msgify(Image, img, "rgb8")
    pub.publish(msg)


if __name__ == "__main__":
    
    rospy.init_node("pf_viz")
    pub = rospy.Publisher("pf_viz", Image, queue_size=0)
    rospy.Subscriber("/pfvtr/particles", FloatList, callback)
    print("PF viz ready...")
    rospy.spin()
