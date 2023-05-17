#!/usr/bin/env python
import rospy
import cv2
from dynamic_reconfigure.server import Server
from pfvtr.cfg import LiveFeaturesConfig
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

br = CvBridge()
pub = None

rectSize = 8
rectCol = [255, 0, 0] #BGR

currentType = 0
try:
    featureTypes = [
        cv2.SIFT_create(),
        cv2.xfeatures2d.SURF_create(),
        cv2.KAZE_create(),
        cv2.AKAZE_create(),
        cv2.BRISK_create(),
        cv2.ORB_create()
    ]
except Exception as e:
    featureTypes = [
        cv2.SIFT_create(),
        cv2.KAZE_create(),
        cv2.AKAZE_create(),
        cv2.BRISK_create(),
        cv2.ORB_create()
    ]

def dr_callback(config, level):
    global currentType
    print("FT: %i" %(config.feature_type))
    currentType = config.feature_type
    return config

def img_cb(msg):
    if pub.get_num_connections():
        detectImg(msg, 0)
def img_compressed_cb(msg):
    if pub.get_num_connections():
        detectImg(msg, 1)

def detectImg(msg, compressed):
    img = None
    if compressed:
        img = br.compressed_imgmsg_to_cv2(msg)
    else:
        img = br.imgmsg_to_cv2(msg)
    kps, des = featureTypes[currentType].detectAndCompute(img, None)
    for i in kps:
        x = int(i.pt[0])
        y = int(i.pt[1])
        cv2.rectangle(img, (x-rectSize, y-rectSize), (x+rectSize, y+rectSize), rectCol, 1)     
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    msg = br.cv2_to_imgmsg(img, encoding="rgb8")
    pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("live_features")
    pub = rospy.Publisher("live_viz", Image, queue_size=0)
    srv = Server(LiveFeaturesConfig, dr_callback)
    camera_topic = rospy.get_param("~camera_topic")
    rospy.Subscriber(camera_topic, Image, img_cb)
    #rospy.Subscriber("/camera_2/image_raw/compressed", CompressedImage, img_compressed_cb)
    rospy.spin()
