#!/usr/bin/env python

import rospy
from bearnav2.srv import Alignment, AlignmentResponse, Representations, RepresentationsResponse
from sensor_processing import BearnavClassic, PF2D, VisualOnly
from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.crosscorrelation.crosscorr import CrossCorrelation


# Network hyperparameters
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


def start_subscribes(fusion_class,
                     abs_align_topic, abs_dist_topic, rel_dist_topic, prob_dist_topic,
                     rel_align_service_name, repr_service_name):
    # --------------- TOPICS ------------
    # subscribers for images and other topics used for alignment and distance estimation
    if fusion_class.abs_align_est is not None and len(abs_align_topic) > 0:
        rospy.Subscriber(abs_align_topic, fusion_class.abs_align_est.supported_message_type,
                         fusion_class.process_abs_alignment, queue_size=1, buff_size=50000000)
    if fusion_class.abs_dist_est is not None and len(abs_dist_topic) > 0:
        rospy.Subscriber(abs_dist_topic, fusion_class.abs_dist_est.supported_message_type,
                         fusion_class.process_abs_distance, queue_size=1)
    if fusion_class.rel_dist_est is not None and len(rel_dist_topic) > 0:
        rospy.Subscriber(rel_dist_topic, fusion_class.rel_dist_est.supported_message_type,
                         fusion_class.process_rel_distance, queue_size=1)
    if fusion_class.prob_dist_est is not None and len(prob_dist_topic) > 0:
        rospy.Subscriber(prob_dist_topic, fusion_class.prob_dist_est.supported_message_type,
                         fusion_class.process_prob_distance, queue_size=1)
    # -------------- SERVICES -------------
    # service for rel alignment
    relative_image_service = None
    if fusion_class.rel_align_est is not None and len(rel_align_service_name) > 0:
        relative_image_service = rospy.Service(fusion_class.type_prefix + "/" + rel_align_service_name,
                                               Alignment, fusion_class.process_rel_alignment)

    return relative_image_service


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")
    odom_topic = rospy.get_param("~odom_topic")

    # Choose sensor method
    align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W)
    align_rel = CrossCorrelation(padding=PAD, network_division=NETWORK_DIVISION, resize_w=RESIZE_W)
    dist_abs = OdometryAbsolute()
    dist_rel = OdometryRelative()

    # Set here fusion method for teaching phase -------------------------------------------
    # BearnavClassic is currently only supported
    teach_fusion = BearnavClassic("teach", align_abs, dist_abs, align_abs, align_abs)
    teach_handlers = start_subscribes(teach_fusion,
                                      "", odom_topic, "", "",
                                      "local_alignment", "get_repr")

    # Set here fusion method for repeating phase ------------------------------------------
    # 1) Bearnav classic - this method also needs publish span 0 in the repeater !!!
    # repeat_fusion = BearnavClassic("repeat", align_abs, dist_abs, align_abs, None)
    # repeat_handlers = start_subscribes(repeat_fusion,
    #                                    "matched_repr", odom_topic, "", "",
    #                                    "", "")
    # 2) Particle filter 2D - parameters are really important
    repeat_fusion = PF2D("repeat", 400, 0.1, 1.0, 0.1, 0.3, 1, True,
                         align_abs, align_rel, dist_rel, align_abs)
    repeat_handlers = start_subscribes(repeat_fusion,
                                       "matched_repr", "", odom_topic, "",
                                       "local_alignment", "")
    # 3) Visual Only
    # repeat_fusion = VisualOnly("repeat", align_abs, align_abs, align_abs)
    # repeat_handler = start_subscribes(repeat_fusion,
    #                                   "matched_repr", "", "", "sensors_input",
    #                                   "", "")

    rospy.spin()
