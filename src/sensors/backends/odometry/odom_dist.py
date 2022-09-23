from base_classes import RelativeDistanceEstimator, AbsoluteDistanceEstimator
from nav_msgs.msg import Odometry
import tf
import geometry_msgs
import rospy


def pose_to_angle(pose_msg):
    quaternion = (
        pose_msg.pose.orientation.x,
        pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z,
        pose_msg.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]
    return yaw


class OdometryAbsolute(AbsoluteDistanceEstimator):

    def __init__(self):
        super(OdometryAbsolute, self).__init__()
        self.supported_message_type = Odometry
        self.last_odom = None
        rospy.loginfo("Odometry absolute distance estimator successfully initialized!")

    def _abs_dist_message_callback(self, msg: Odometry) -> float:
        if self.last_odom is None:
            self.last_odom = msg
            return self._distance
        dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
        self._distance += (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5 
        # add very slight distance during turning to avoid similar images
        yaw1 = pose_to_angle(self.last_odom.pose)
        yaw2 = pose_to_angle(msg.pose)
        dturn = min((2 * 3.16) - abs(yaw1 - yaw2), abs(yaw1 - yaw2))
        # self._distance += abs(dturn)
        self.last_odom = msg
        self.header = msg.header
        return self._distance

    def _set_dist(self, dist) -> float:
        self.last_odom = None
        self._distance = 0.0
        return dist

    def health_check(self) -> bool:
        return True


class OdometryRelative(RelativeDistanceEstimator):

    def __init__(self):
        super(OdometryRelative, self).__init__()
        self.supported_message_type = Odometry
        self.last_odom = None
        rospy.loginfo("Odometry relative distance estimator successfully initialized!")

    def _rel_dist_message_callback(self, msg: Odometry) -> float:
        if self.last_odom is None:
            self.last_odom = msg
            return None
        dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
        # add very slight distance during turning to avoid similar images
        yaw1 = pose_to_angle(self.last_odom.pose)
        yaw2 = pose_to_angle(msg.pose)
        dturn = min((2 * 3.16) - abs(yaw1 - yaw2), abs(yaw1 - yaw2))
        self.last_odom = msg
        ret = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        if ret > 7.0: # for rosbag switching
            return 1.0
        return ret  # + abs(dturn)

    def health_check(self) -> bool:
        return True
