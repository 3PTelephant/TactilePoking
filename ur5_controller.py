#!/usr/bin/env python
#-*- coding: utf-8 -*-
import rospy
import copy
from std_msgs.msg import String
from std_msgs.msg import UInt16
class Ur5Controller:
    def __init__(self):
        # parameters for ur5 movement; i.e. movel(pose, a=1.2, v=0.25, t=0, r=0); tool denotes the endpoint (I think)
        self.vel = 0.25  #tool speed
        self.ace = 0.2  #tool acceleration
        self.t = 0.5  # the time (seconds) to make the move
        self.r = 0  #blend radius
        self.move_type = "movel"    # movel:move to position(linear in tool-space); movej:linear in joint-space
        self.move_vel_type = "speedl"
        self.cont = 10
        self.pos = [-0.137, -0.297, -0.213, 2.813, -1.267, 0.08]  # ur pose       (either joint positions q=[Base,Shoulder, Elbow, Wrist1, Wrist2, Wrist3],
                                                            #                       or pose[x,y,z,Rx,Ry,Rz], it holds q==p(x,y,z,Rx,Ry,Rz))
        self.pos_type = 0       # ur pose type  (0:pose 1:joint positions)
        self.vel_type = 0       # ur velocity type (0:tool space 1:joint space)
        #pose = p[0.2,0.3,0.5,0,0,3.14] → position in base frame of x = 200 mm, y = 300 mm, z = 500 mm, rx = 0, ry = 0, rz = 180 deg
        self.pub = None
        self.rate = None
        self.ratet = 1

    def Init_node(self):
        rospy.init_node("move_ur5")
        pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size = 10)
        self.pub = pub
        self.rate = rospy.Rate(self.ratet)

    def Pos2UrMoveCmd(self, pos):    #return the command for ur5 to move to pos
        pose = pos
        if self.pos_type == 0:
            cmd = self.move_type + "(p[" + str(pose[0]) + "," + str(pose[1]) + "," + str(pose[2]) + "," + str(
            pose[3]) + "," + str(pose[4]) + "," + str(pose[5]) + "]," + "a=" + str(self.ace) + "," + "v=" + str(
            self.vel) + "," + "t=" + str(self.t) +  "," + "r=" + str(self.r) + ")"
        else:
            cmd = self.move_type + "([" + str(pose[0]) + "," + str(pose[1]) + "," + str(pose[2]) + "," + str(
            pose[3]) + "," + str(pose[4]) + "," + str(pose[5]) + "]," + "a=" + str(self.ace) + "," + "v=" + str(
            self.vel) + "," + "t=" + str(self.t) +  "," + "r=" + str(self.r) + ")"
        print("----cmd to ur5:", cmd)
        return cmd

    def Vel2UrMoveCmd(self, vel, time):    #return the command for ur5 to move to pos
        velocity = vel
        self.t = time
        if self.pos_type == 0:
            cmd = self.move_vel_type + "([" + str(velocity[0]) + "," + str(velocity[1]) + "," + str(velocity[2]) + "," + str(
            velocity[3]) + "," + str(velocity[4]) + "," + str(velocity[5]) + "]," + str(self.ace) + "," + "t=" + str(self.t) + ")"
        print("----cmd to ur5:", cmd)
        return cmd

    def move2Pos(self, pos):
        cmd = self.Pos2UrMoveCmd(pos)
        self.pos = pos  #update current pos of ur
        self.pub.publish(cmd)
        self.rate.sleep()

    def move2Vel(self, vel, time):
        cmd = self.Vel2UrMoveCmd(vel, time)
        # self.vel = vel
        self.pub.publish(cmd)
        self.rate.sleep()

    def setRate(self, rate):
        self.ratet = rate

    # def get_ur_pos(self): 


def main(): #for test

    try:
        urc = Ur5Controller()
        urc.ratet = 4 # this frequence should be higher than the parameter "t" in control command, otherwise, the gripper will be jitter
        urc.Init_node()

        cn = 0
        # pos = [0.446, -0.305, 0.142, 2.505, 1.968, 0]
        pos = [-0.432, 0.310, 0.444, 2.879, 1.269, -0.0162]
        tempPos = copy.deepcopy(pos)
        vel = [0,0,-0.01,0,0,0]
        urc.move2Vel(vel)
        urc.move2Vel(vel)
        urc.move2Vel(vel)
#        urc.move2Pos(tempPos)

    except KeyboardInterrupt:
        rospy.signal_shutdown('KeyboardInterrupt')
        raise


if __name__ == '__main__':
    main()
