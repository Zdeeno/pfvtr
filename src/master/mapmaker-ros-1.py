#!/usr/bin/env python
import time
import rospy
import rostopic
import os
import actionlib
import cv2
import rosbag
import roslib
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Header
from bearnav2.msg import MapMakerAction, MapMakerResult, SensorsOutput, SensorsInput, ImageList, DistancedTwist, Features, FeaturesList
from bearnav2.srv import SetDist, Alignment, Representations 
import numpy as np
from copy import deepcopy
import ros_numpy
from message_filters import ApproximateTimeSynchronizer, Subscriber


TARGET_WIDTH = 512


def numpy_to_feature(array):
    return Features(array.flatten(), array.shape)


def save_img(img_repr, image, header, filename, save_img):
    time_str = str(header.stamp.secs).zfill(10)[-4:] + str(header.stamp.nsecs).zfill(9)[:4]
    filename = filename + "_" + time_str
    with open(filename + ".npy", 'wb') as fp:
        np.save(fp, img_repr, allow_pickle=False, fix_imports=False)
    if save_img:
        if "rgb" in image.encoding: 
            cv2.imwrite(filename + ".jpg", cv2.cvtColor(ros_numpy.numpify(image), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(filename + ".jpg", ros_numpy.numpify(image))

class ActionServer:

    def __init__(self):

        self.isMapping = False
        self.img_msg = None
        self.img_features = None
        self.last_img_features = None
        self.mapName = ""
        self.mapStep = 1.0
        self.nextStep = 0
        self.bag = None
        self.lastDistance = 0.0
        self.visual_turn = True
        self.max_trans = 0.3
        self.curr_trans = 0.0
        self.last_saved_dist = None
        self.save_imgs = False
        self.header = None

        self.additionalTopics = rospy.get_param("~additional_record_topics")
        self.additionalTopics = self.additionalTopics.split(" ")
        self.additionalTopicSubscribers = []
        if self.additionalTopics[0] != "":
            rospy.loginfo(f"Recording the following additional topics: {self.additionalTopics}")
            for topic in self.additionalTopics:
                msgType = rostopic.get_topic_class(topic)[0]
                s = rospy.Subscriber(topic, msgType, self.miscCB, queue_size=1)
                self.additionalTopicSubscribers.append(s)

        rospy.loginfo("Waiting for services to become available...")
        rospy.wait_for_service("teach/set_dist")
        rospy.loginfo("Starting...")

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("teach/set_dist", SetDist)
        self.align_reset_srv = rospy.ServiceProxy("repeat/set_align", SetDist)
        self.distance_reset_srv(0.0)
        self.align_reset_srv(0.0)

        rospy.logdebug("Subscibing to commands")
        self.joy_topic = rospy.get_param("~cmd_vel_topic")
        self.joy_sub = rospy.Subscriber(self.joy_topic, Twist, self.joyCB, queue_size=100, buff_size=250000)

        rospy.logdebug("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("mapmaker", MapMakerAction,
                                                   execute_cb=self.actionCB, auto_start=False)
        self.server.start()

        if self.visual_turn:
            rospy.wait_for_service("teach/local_alignment")
            self.local_align = rospy.ServiceProxy("teach/local_alignment", Alignment)
            rospy.logwarn("Local alignment service available for mapmaker")

        # here it subscribes necessary topics
        rospy.logdebug("Subscibing to cameras")
        self.camera_topic = rospy.get_param("~camera_topic")
        # self.cam_sub = rospy.Subscriber(self.camera_topic, Image, self.imageCB, queue_size=1, buff_size=20000000)
        # self.distance_sub = rospy.Subscriber("teach/output_dist", SensorsOutput, self.distanceCB, queue_size=10)
        # synchronize necessary topics!
        repr_sub = Subscriber("live_representation", FeaturesList)
        distance_sub = Subscriber("teach/output_dist", SensorsOutput)
        cam_sub = Subscriber(self.camera_topic, Image)
        synced_topics = ApproximateTimeSynchronizer([repr_sub, distance_sub, cam_sub], queue_size=2, slop=0.2)
        synced_topics.registerCallback(self.distance_imgCB)

        rospy.logwarn("Mapmaker started, awaiting goal")

    def miscCB(self, msg, args):
        if self.isMapping:
            topicName = args
            rospy.logdebug(f"Adding misc from {topicName}")
            self.bag.write(topicName, msg)

    def distance_imgCB(self, repr_msg, dist_msg, img):
        self.img_features = np.array(repr_msg.data[0].values).reshape(repr_msg.data[0].shape)
        self.img_msg = img
        self.header = repr_msg.header
        dist = dist_msg.output
        self.lastDistance = dist

        if not self.isMapping:
            return

        # obtain displacement between prev and new image --------------------------------------
        if self.visual_turn and self.last_img_features is not None:
            # create message
            srv_msg = SensorsInput()
            srv_msg.map_features = [numpy_to_feature(self.last_img_features)]
            srv_msg.live_features = [numpy_to_feature(self.img_features)]
            try:
                resp1 = self.local_align(srv_msg)
                hist = resp1.histograms[0].data
                half_size = np.size(hist)/2.0
                self.curr_trans = -float(np.argmax(hist) - (np.size(hist)//2.0)) / half_size  # normalize -1 to 1
                # rospy.logwarn(self.curr_trans)
            except Exception as e:
                rospy.logwarn("Service call failed: %s" % e)
        else:
            self.curr_trans = 0.0

        # eventually save the image if conditions fulfilled ------------------------------------
        if dist > self.nextStep or abs(self.curr_trans) > self.max_trans:
            self.nextStep = dist + self.mapStep
            filename = os.path.join(self.mapName, str(dist) + "_" + str(self.curr_trans))
            save_img(self.img_features, self.img_msg, self.header, filename, self.save_imgs)  # with resizing
            rospy.loginfo("Saved waypoint: " + str(dist) + ", " + str(self.curr_trans))
            self.cum_turn = 0.0
            self.last_img_features = self.img_features

        self.checkShutdown()

    def joyCB(self, msg):
        if self.isMapping:
            rospy.logdebug("Adding joy")
            save_msg = DistancedTwist()
            save_msg.twist = msg
            save_msg.distance = self.lastDistance
            self.bag.write("recorded_actions", save_msg)

    def actionCB(self, goal):

        self.save_imgs = goal.saveImgsForViz
        result = MapMakerResult()
        if self.img_features is None:
            rospy.logerr("WARNING: no features coming through, ignoring")
            result.success = False
            self.server.set_succeeded(result)
            return

        if goal.mapName == "":
            rospy.logwarn("Missing mapname, ignoring")
            result.success = False
            self.server.set_succeeded(result)
            return

        if goal.start == True:
            self.isMapping = False
            self.img_msg = None
            self.last_img_msg = None
            self.distance_reset_srv(0.0)
            self.mapStep = goal.mapStep
            if self.mapStep <= 0.0:
                rospy.logwarn("Record step is not positive number - changing to 1.0m")
                self.mapStep = 1.0
            try:
                os.mkdir(goal.mapName)
                with open(goal.mapName + "/params", "w") as f:
                    f.write("stepSize: %s\n" % (self.mapStep))
                    f.write("odomTopic: %s\n" % (self.joy_topic))
            except:
                rospy.logwarn("Unable to create map directory, ignoring")
                result.success = False
                self.server.set_succeeded(result)
                return
            rospy.loginfo("Starting mapping")
            self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "w")
            self.mapName = goal.mapName
            self.nextStep = 0.0
            self.lastDistance = 0.0
            self.isMapping = True
            result.success = True
            self.server.set_succeeded(result)
        else:
            rospy.logdebug("Creating final wp")
            # filename = os.path.join(self.mapName, str(self.lastDistance) + ".jpg")
            # cv2.imwrite(filename, self.img)
            filename = os.path.join(self.mapName, str(self.lastDistance) + "_" + str(self.curr_trans))
            save_img(self.img_features, self.img_msg, self.header, filename, self.save_imgs)  # with resizing
            rospy.logwarn("Stopping Mapping")
            rospy.loginfo(f"Map saved under: '{os.path.join(os.path.expanduser('~'), '.ros', self.mapName)}'")
            time.sleep(2)
            self.isMapping = False
            result.success = True
            self.server.set_succeeded(result)
            self.bag.close()

    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.isMapping = False
        if self.bag is not None:
            self.bag.close()

if __name__ == '__main__':
    rospy.init_node("mapmaker_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()
