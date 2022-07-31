import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
import copy
import numpy as np
import cv2
from gripper_controller import GripperController
from ur5_controller import Ur5Controller
import sys
import time
import math


def to_euler_angles(qw, qx, qy, qz):
    '''
    Math function for tranforming q to euler angles
    '''
    r = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    p = math.asin(2 * (qw * qy - qz * qz))
    y = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))

    return [r, p, y]

#
def d2r(angle):
    ''' degree to radius'''
    return angle/180*np.pi

def rpy2rv(rpy):
    '''
    euler angles to rotation vector
    '''
    alpha = d2r(rpy[2])
    beta = d2r(rpy[1])
    gamma = d2r(rpy[0])

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)

    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg

    theta = np.arccos((r11 + r22 + r33 - 1) / 2)
    sth = np.sin(theta)
    kx = (r32 - r23) / (2 * sth)
    ky = (r13 - r31) / (2 * sth)
    kz = (r21 - r12) / (2 * sth)

    return [(theta * kx), (theta * ky), (theta * kz)]

if __name__ == '__main__':

    ur5 = Ur5Controller()
    ur5.ratet = 1
    ur5.Init_node()

    print('Would you like to go back to home 1:Yes 0:No')
    home_index = int(input())
    while ((home_index != 1 and home_index != 2 and home_index != 3)):
        print('Would you like to go back to home 1:Yes 0:No')
        home_index = int(input())
    if home_index == 1:
        theta = np.pi/4
        dx_ef2gel = 0
        dy_ef2gel = -0.03
        base_dx_ef2gel = -math.cos(theta) * dx_ef2gel + math.cos(theta) * dy_ef2gel
        base_dy_ef2gel = -math.cos(theta) * dx_ef2gel - math.cos(theta) * dy_ef2gel
        base_dz_ef2gel = 0.08  # the distance from
        home_pos = [-0.24, 0.085, 0.295, 0.6905, -1.7642, -1.7023]
        # home_pos = [-0.4, 0.085, 0.295, 0.6905, -1.7642, -1.7023]
        # home_pos = [-0.4-base_dx_ef2gel, 0.085-base_dy_ef2gel, 0.295, 1.21, -2.846, 0.001]

        ur5.t = 2
        temp_homePos = copy.deepcopy(home_pos)
        ur5.move2Pos(temp_homePos)
    if home_index == 2:
        theta = np.pi / 4
        dx_ef2gel = 0
        dy_ef2gel = -0.03
        base_dx_ef2gel = -math.cos(theta) * dx_ef2gel + math.cos(theta) * dy_ef2gel
        base_dy_ef2gel = -math.cos(theta) * dx_ef2gel - math.cos(theta) * dy_ef2gel
        base_dz_ef2gel = 0.08  # the distance from
        home_pos = [-0.451, -0.156, 0.355, 1.21, -2.846, 0.001]
        # home_pos = [-0.4, 0.085, 0.295, 0.6905, -1.7642, -1.7023]
        # home_pos = [-0.4-base_dx_ef2gel, 0.085-base_dy_ef2gel, 0.295, 1.21, -2.846, 0.001]

        ur5.t = 2
        temp_homePos = copy.deepcopy(home_pos)
        ur5.move2Pos(temp_homePos)
    if home_index ==3:
        print('Input the yaw angle you want:')
        yaw = float(input())
        rpy = [178.94, 1.68, yaw]
        rotation_vector = rpy2rv(rpy)
        home_pos = [-0.424, 0.181, 0.321, rotation_vector[0],rotation_vector[1],0.01]

        ur5.t = 8
        temp_homePos = copy.deepcopy(home_pos)
        ur5.move2Pos(temp_homePos)


