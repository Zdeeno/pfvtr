import cv2
import rosbag
from sensor_msgs.msg import Image
import cv_bridge
import os

fn = "/home/george/tmp/coffee/replays/2021-03-16-02-15-34.bag"
outputdir = "/home/george/tmp/coffee/replays/imgs"

br = cv_bridge.CvBridge()

distance = 0

with rosbag.Bag(fn, "r") as bag:
    for topic, msg, ts in bag.read_messages():

        if topic == "/camera_2/image_rect_color":
            img = br.imgmsg_to_cv2(msg)
            filename = os.path.join(outputdir, str(distance) + "-" + str(ts) + ".jpg")
            print("Writing: " + filename)
            cv2.imwrite(filename, img)

        if topic == "/distance":

            distance = str(int(msg.data))
            
print("Done.")
