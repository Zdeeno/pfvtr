#!/usr/bin/env python
import os
import rospy
import cv2
#import alignment
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pfvtr.msg import Alignment

mapDir = "/home/george/tmp/coffee"
repeatDir = "/home/george/tmp/coffee/replays/imgs"

pub = None
#a = alignment.Alignment("./config.yaml")
br = CvBridge()
imgABuf = None

def callbackA(msg):
        global imgABuf
        imgABuf = br.imgmsg_to_cv2(msg)

def callbackB(msg):
        #global imgABuf
        """
        if imgABuf is None:
                print("Still haven't rec'd cam!!")

        imgB = br.imgmsg_to_cv2(msg)
        print("Getting alignment from imgs")
        alignment, uncertainty = a.process(imgABuf, imgB)
        m = Alignment()
        m.alignment = alignment
        m.uncertainty = uncertainty
        print("Sending corrections!")
        pub.publish(m)
        """

def getMapFiles():
        mapFiles = []
        for files in os.walk(mapDir):
            if files[0] == mapDir:
                mapFiles = files[2]
                break
        def jpgfilter(a):
            if "jpg" in a:
                return True
            return False
        mapFiles = filter(jpgfilter, mapFiles)
        mapFiles = [int(x.split(".")[0]) for x in mapFiles]
        print("Map Files:")
        print(mapFiles)
        return mapFiles

if __name__ == "__main__":

        mapFiles = getMapFiles()
        
        for files in os.walk(repeatDir):
            
            


        rospy.init_node("alignment_tester")
        rospy.Subscriber("/alignment/output", Alignment, cb)
        pub = rospy.Publisher("/alignment/output", Alignment, queue_size=0)
        pub = rospy.Publisher("/alignment/output", Alignment, queue_size=0)


