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
from bearnav2.msg import MapMakerAction, MapMakerResult, SensorsOutput, SensorsInput, ImageList, DistancedTwist, \
    Features, FeaturesList
from bearnav2.srv import SetDist, Alignment, Representations
import numpy as np
from copy import deepcopy
import ros_numpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import shutil


TARGET_WIDTH = 512


def get_map_dists(mappath):
    tmp = []
    for file in list(os.listdir(mappath)):
        if file.endswith(".npy"):
            tmp.append(file[:-4])
    rospy.logwarn(str(len(tmp)) + " images found in the map")
    tmp.sort(key=lambda x: float(x))
    if not tmp:
        rospy.logwarn("source map empty")
        raise Exception("Invalid source map")
    tmp = np.array(tmp, dtype=float)
    return tmp


def numpy_to_feature(array):
    return Features(array.flatten(), array.shape)


def save_img(img_repr, image, header, map_name, curr_dist, curr_hist, curr_align, source_map, save_img):
    # time_str = str(header.stamp.secs).zfill(10)[-4:] + str(header.stamp.nsecs).zfill(9)[:4]
    filename = str(map_name) + "/" + str(curr_dist)
    struct_save = {"representation": img_repr, "timestamp": header.stamp, "diff_hist": None, "source_map_align": None}
    if curr_hist is not None:
        struct_save["diff_hist"] = curr_hist
    if curr_align is not None and source_map is not None:
        struct_save["source_map_align"] = (source_map, curr_align)
    with open(filename + ".npy", 'wb') as fp:
        np.save(fp, struct_save, fix_imports=False)
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
        self.nextStep = 0.0
        self.bag = None
        self.visual_turn = True
        self.max_trans = 0.3
        self.curr_trans = 0.0
        self.curr_hist = None
        self.last_saved_dist = None
        self.save_imgs = False
        self.header = None
        self.target_distances = None
        self.collected_distances = None
        self.dist = 0.0
        
        rospy.loginfo("Waiting for services to become available...")
        rospy.wait_for_service("teach/set_dist")
        rospy.loginfo("Starting...")

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("teach/set_dist", SetDist)
        self.align_reset_srv = rospy.ServiceProxy("teach/set_align", SetDist)
        self.distance_reset_srv(0.0, 1)
        self.align_reset_srv(0.0, 1)

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

        # synchronize necessary topics!
        rospy.logwarn("Mapmaker starting subscribers")
        repr_sub = Subscriber("live_representation", FeaturesList)
        cam_sub = Subscriber(self.camera_topic, Image)
        distance_sub = Subscriber("teach/output_dist", SensorsOutput)
        self.synced_topics = ApproximateTimeSynchronizer([repr_sub, distance_sub, cam_sub], queue_size=50, slop=2.5)
        self.synced_topics.registerCallback(self.distance_imgCB)

        self.curr_alignment = None
        self.source_map = None

        rospy.logwarn("Mapmaker started, awaiting goal")

    def miscCB(self, msg, args):
        if self.isMapping:
            topicName = args
            rospy.logdebug(f"Adding misc from {topicName}")
            self.bag.write(topicName, msg)

    def distance_wrapper(self, repr_msg, dist_msg, align_msg, img):
        self.curr_alignment = align_msg.output
        self.distance_imgCB(repr_msg, dist_msg, img)

    def distance_imgCB(self, repr_msg, dist_msg, img):
        if self.img_features is None:
            rospy.logwarn("Mapmaker successfuly received images")
        self.img_features = np.array(repr_msg.data[0].values).reshape(repr_msg.data[0].shape)
        self.img_msg = img
        self.header = repr_msg.header
        dist = dist_msg.output
        self.dist = dist
        if not self.isMapping:
            return

        # obtain displacement between prev and new image --------------------------------------
        if self.visual_turn and self.last_img_features is not None and dist:
            # create message
            srv_msg = SensorsInput()
            srv_msg.map_features = [numpy_to_feature(self.last_img_features)]
            srv_msg.live_features = [numpy_to_feature(self.img_features)]
            try:
                resp1 = self.local_align(srv_msg)
                hist = resp1.histograms[0].data
                half_size = np.size(hist) / 2.0
                self.curr_hist = hist
                self.curr_trans = -float(np.argmax(hist) - (np.size(hist) // 2.0)) / half_size  # normalize -1 to 1
                # rospy.logwarn(self.curr_trans)
            except Exception as e:
                rospy.logwarn("Service call failed: %s" % e)
        else:
            self.curr_trans = 0.0
            self.curr_hist = None

        # eventually save the image if conditions fulfilled ------------------------------------
        if self.target_distances is not None and self.curr_hist is not None:
            # when source map is provided
            desired_idx = np.argmin(abs(dist - np.array(self.target_distances)))
            self.last_img_features = self.img_features
            if self.collected_distances[desired_idx] == 0 and self.target_distances[desired_idx] <= dist:
                self.collected_distances[desired_idx] = 1
                save_img(self.img_features, self.img_msg, self.header, self.mapName, dist, self.curr_hist,
                         self.curr_alignment, self.source_map, self.save_imgs)  # with resizing
                rospy.loginfo("Saved waypoint: " + str(dist) + ", " + str(self.curr_trans))

        if self.target_distances is None and (dist > self.nextStep or abs(self.curr_trans) > self.max_trans):
            # save after fix distance
            self.nextStep = dist + self.mapStep
            self.last_img_features = self.img_features
            save_img(self.img_features, self.img_msg, self.header, self.mapName, dist, self.curr_hist,
                     self.curr_alignment, self.source_map, self.save_imgs)  # with resizing
            rospy.loginfo("Saved waypoint: " + str(dist) + ", " + str(self.curr_trans))

            # used for taking source map remapping
        if self.last_img_features is None:
            self.last_img_features = self.img_features
        self.checkShutdown()

    def joyCB(self, msg):
        if self.isMapping:
            rospy.logdebug("Adding joy")
            save_msg = DistancedTwist()
            save_msg.twist = msg
            save_msg.distance = self.dist
            self.bag.write("recorded_actions", save_msg)

    def actionCB(self, goal):

        if goal.sourceMap != "":
            # sync different topics when repeating using source map
            self.target_distances = []
            self.source_map = goal.sourceMap
            self.target_distances = get_map_dists(self.source_map)
            self.collected_distances = np.zeros_like(self.target_distances)
            distance_sub = Subscriber("repeat/output_dist", SensorsOutput)
            align_sub = Subscriber("repeat/output_align", SensorsOutput)
            repr_sub = Subscriber("live_representation", FeaturesList)
            cam_sub = Subscriber(self.camera_topic, Image)
            self.synced_topics = ApproximateTimeSynchronizer([repr_sub, distance_sub, align_sub, cam_sub], queue_size=10,
                                                             slop=0.5)
            self.synced_topics.registerCallback(self.distance_wrapper)
            rospy.logwarn("mapmaker listening to distance callback of map " + goal.sourceMap)

        self.save_imgs = goal.saveImgsForViz
        result = MapMakerResult()
        """
        if self.img_features is None:
            rospy.logerr("WARNING: no features coming through, ignoring")
            result.success = False
            self.server.set_succeeded(result)
            return
        """

        if goal.mapName == "":
            rospy.logwarn("Missing mapname, ignoring")
            result.success = False
            self.server.set_succeeded(result)
            return

        if goal.start == True:
            self.isMapping = False
            self.img_msg = None
            self.last_img_msg = None
            self.distance_reset_srv(0.0, 1)
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
            self.isMapping = True
            result.success = True
            self.server.set_succeeded(result)
        else:
            if self.target_distances is None: # and self.nextStep - self.dist > self.mapStep/4:
                save_img(self.img_features, self.img_msg, self.header, self.mapName,
                         self.dist, self.curr_hist, self.curr_alignment, self.source_map,
                         self.save_imgs)  # with resizing
                rospy.loginfo("Creating final wp at dist: " + str(self.dist))

            rospy.logwarn("Stopping Mapping")
            rospy.loginfo(f"Map saved under: '{os.path.join(os.path.expanduser('~'), '.ros', self.mapName)}'")
            time.sleep(2)
            self.isMapping = False
            result.success = True
            self.server.set_succeeded(result)
            self.bag.close()
            if self.target_distances is not None:
                rospy.logwarn("Removing and copying action commands")
                # use action commands from source map!!!
                os.remove(os.path.join(goal.mapName, goal.mapName + ".bag"))
                shutil.copy(os.path.join(goal.sourceMap, goal.sourceMap + ".bag"),
                            os.path.join(goal.mapName, goal.mapName + ".bag"))


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
