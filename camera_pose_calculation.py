import copy
import numpy as np
import cv2
import sys
import time
import math
sys.path.append('..')
from realsense import camera

def camera_pose_estimation():
    print (cv2.getVersionMajor())
    print (cv2.getVersionMinor())
    # Initialize Camera
    print('Running live demo of robotic grasping. Make sure realsense camera is streaming.\n')
    rcamera = camera.Camera()
    camera_intrinsics = rcamera.color_intr
    realsense_fx = camera_intrinsics[0, 0]
    realsense_fy = camera_intrinsics[1, 1]
    realsense_cx = camera_intrinsics[0, 2]
    realsense_cy = camera_intrinsics[1, 2]
    time.sleep(1)  # Give camera some time to load data
    print(realsense_fx)
    print(realsense_cx)
    print(realsense_cy)

    while True:
        color_img, input_depth = rcamera.get_data()
        color_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("test", color_img)
        cv2.waitKey()
        gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        arucoParams = cv2.aruco.DetectorParameters_create()
        res = cv2.aruco.detectMarkers(gray, arucoDict,
                parameters=arucoParams)
        if len(res[0]) > 0:
            cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])
        markerLength = 0.2
        dist_coeff = np.zeros(5)
        cv2.imshow("test", gray)
        cv2.waitKey()
        # dist_coeff = [0,0,0,0,0]
        # the rotation and translation is the one that transforms points from
        # each marker coordinate system to the camera coordinate system.
        rvec, tvec, _= cv2.aruco.estimatePoseSingleMarkers(res[0], markerLength, camera_intrinsics, dist_coeff)
        print(rvec)
        R, _ = cv2.Rodrigues(rvec)

        print("Rotation Matrix :" , R)
        R_inv = np.linalg.inv(R)
        print("Inverse Rotation Matrix :" , R_inv)
        print("R vec :", rvec)
        print("T vec :" , tvec)
        cv2.imshow("aruco_marker", gray)
        cv2.waitKey(30)

    return 0



if __name__ == '__main__':
    camera_pose_estimation()

