#!/usr/bin/env python

import rospy
from bearnav2.srv import Alignment, AlignmentResponse, Representations, RepresentationsResponse
from sensor_processing import BearnavClassic, PF2D, VisualOnly
from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.crosscorrelation.crosscorr import CrossCorrelation
from sensor_msgs.msg import Image
from bearnav2.msg import FeaturesList, ImageList, Features, SensorsInput
import ros_numpy
import numpy as np


# Network hyperparameters
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


class RepresentationMatching:

    def __init__(self):
        rospy.init_node("sensor_processing")
        rospy.loginfo("Sensor processing started!")
        camera_topic = rospy.get_param("~camera_topic")

        # Choose sensor method

        self.align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W)
        self.pub = rospy.Publisher("live_representation", FeaturesList, queue_size=1)
        self.pub_match = rospy.Publisher("matched_repr", SensorsInput, queue_size=1)
        self.sub = rospy.Subscriber(camera_topic, Image,
                                    self.image_parserCB, queue_size=1, buff_size=50000000)
        self.map_sub = rospy.Subscriber("map_representations", SensorsInput,
                                        self.map_parserCB, queue_size=1, buff_size=50000000)

        self.last_live = None
        self.sns_in_msg = None
        rospy.spin()

    def parse_camera_msg(self, msg):
        img = ros_numpy.numpify(msg)
        if "bgr" in msg.encoding:
            img = img[..., ::-1]  # switch from bgr to rgb
        img_msg = ros_numpy.msgify(Image, img, "rgb8")
        return img_msg, img

    def image_parserCB(self, image):
        img_msg, _ = self.parse_camera_msg(image)
        msg = ImageList([img_msg])
        live_feature = self.align_abs._to_feature(msg)
        tmp_sns_in = self.sns_in_msg

        if self.last_live is None:
            self.last_live = live_feature[0]
        out = FeaturesList(image.header, [live_feature[0]])
        self.pub.publish(out)

        if tmp_sns_in is None:
            return

        # match live vs. live map, live vs last live, live vs maps
        ext_tensor = [*tmp_sns_in.map_features, self.last_live]
        align_in = SensorsInput()
        align_in.map_features = ext_tensor
        align_in.live_features = live_feature
        out = self.align_abs.process_msg(align_in)

        # decode these
        align_out = SensorsInput()

        live_hist = np.array(out[-1])  # all live map distances vs live img
        map_hist = np.array(out[:-1])

        # create publish msg
        align_out.header = image.header
        align_out.live_features = [Features(live_hist.flatten(), live_hist.shape)]  # now it is list of histogram, not features
        align_out.map_features = [Features(map_hist.flatten(), map_hist.shape)]     # this too
        align_out.map_distances = tmp_sns_in.map_distances
        align_out.map_transitions = tmp_sns_in.map_transitions                      # also list of histograms
        align_out.map_timestamps = tmp_sns_in.map_timestamps
        align_out.map_num = tmp_sns_in.map_num
        align_out.map_similarity = tmp_sns_in.map_similarity    # TODO: this is not received from repeater yet!
        align_out.map_offset = tmp_sns_in.map_offset

        # rospy.logwarn("sending: " + str(hists.shape) + " " + str(tmp_sns_in.map_distances))
        self.pub_match.publish(align_out)
        self.last_live = live_feature[0]

    def map_parserCB(self, sns_in):
        self.sns_in_msg = sns_in


if __name__ == '__main__':
    r = RepresentationMatching()
