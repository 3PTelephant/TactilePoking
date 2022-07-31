import rospy
import copy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import UInt16

from data_collection_ur5.msg import Robotiq2FGripper_robot_output as outputMsg
from data_collection_ur5.msg import Robotiq2FGripper_robot_input as inputMsg

class GripperController:
    def __init__(self):
        self.force = 150
        self.speed = 25
        self.position = 0
        self.cur_status = None
        self.status_sub = rospy.Subscriber('Robotiq2FGripperRobotInput', inputMsg, self._status_cb)
        self.cmd_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg, queue_size = 10)
        # rospy.init_node("gripper")
        print("Successful Initialization")

    def _status_cb(self, msg):
        self.cur_status = msg


    def wait_for_connection(self, timeout=-1):
        rospy.sleep(0.1)
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout):
                return False
            if self.cur_status is not None:
                return True
            r.sleep()
        return False

    def activate_gripper(self,timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = 0
        cmd.rSP = 255
        cmd.rFR = 150
        self.cmd_pub.publish(cmd)
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if timeout >= 0. and rospy.get_time() - start_time > timeout:
                return False
            if self.is_ready():
                return True
            r.sleep()
        return False

    def reset(self):
        cmd = outputMsg()
        cmd.rACT = 0
        self.cmd_pub.publish(cmd)

    # if timeout is negative, wait forever
    def wait_until_stopped(self, timeout=-1):
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
                return False
            if self.is_stopped():
                return True
            r.sleep()
        return False

    def wait_until_moving(self, timeout=-1):
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
                return False
            if not self.is_stopped():
                return True
            r.sleep()
        return False

    # Goto position with desired force and velocity
    # @param pos Gripper width in meters. [0, 0.087]
    # @param vel Gripper speed in m/s. [0.013, 0.100]
    # @param force Gripper force in N. [30, 100] (not precise)
    def goto(self, pos, vel, force, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = int(np.clip((13.-230.)/0.087 * pos + 230., 0, 255))
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        self.cmd_pub.publish(cmd)
        print("Successful Publish")
        rospy.sleep(1)
        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True


def main():  #for test
    gp = GripperController()
    if gripper.is_reset():
        gripper.reset()
        gripper.activate()
    print(gripper.close(block=True))
#    while not rospy.is_shutdown():
#        print(gripper.open(block=False))
#        rospy.sleep(0.11)
#        gripper.stop()
#        print(gripper.close(block=False))
#        rospy.sleep(0.1)
#        gripper.stop()




if __name__ == '__main__':
        main()
