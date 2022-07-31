import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
import geometry_msgs

import copy
import numpy as np
import cv2
from gripper_controller import GripperController
from ur5_controller import Ur5Controller
import sys
import time
import math
sys.path.append('..')
from realsense import camera


import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model.mask_rcnn import MaskRCNNPredictor
from model.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

import utils
import transforms as T

from grasp_generation import touch_generation, grasp_generation, touch_generation_bbox

contact = False
end_position_msg = geometry_msgs.msg.TransformStamped()
end_pos = [0,0,0]


# Initialize some parameter for frame transformation
base2marker_x = 0.23
base2marker_y = -0.72
base2marker_z = -0.04

# Initialize the camera parameters
fx = 1387.7
cx = 960.0
cy = 524.0


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

def contact_cb(msg):
    '''
    callback function for contact data
    '''
    global contact
    contact = msg.data
    # print(contact)
    return 0

def eepose_cb(msg):
    '''
    callback function for end-effector pose
    '''
    global end_position_msg
    global end_pos
    end_position_msg = msg
    # print(end_position_msg.transforms[0].header)
    if end_position_msg.transforms[0].child_frame_id == "tool0_controller":
        ee_x = end_position_msg.transforms[0].transform.translation.x
        ee_y = end_position_msg.transforms[0].transform.translation.y
        ee_z = end_position_msg.transforms[0].transform.translation.z

        # ee_r_x = end_position_msg.transforms[0].transform.rotation.x
        # ee_r_y = end_position_msg.transforms[0].transform.rotation.y
        # ee_r_z = end_position_msg.transforms[0].transform.rotation.z
        # ee_r_w = end_position_msg.transforms[0].transform.rotation.z

        end_pos = [ee_x, ee_y, ee_z]
        # rotation = to_euler_angles(ee_r_x, ee_r_y, ee_r_z, ee_r_w)
        # print(end_pos)
        # print(rotation)

    return 0

# Initialization of subscribers
contact_sub = rospy.Subscriber('contact_status', Bool, contact_cb)
eepose_sub = rospy.Subscriber('/tf', TFMessage, eepose_cb)
update_pub = rospy.Publisher('contact_update', Bool, queue_size=10)


def get_model_instance_segmentation(num_classes):
    # get the pretrained model
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    print(model)
    return model

def touch_mask_inference(img):
    '''
    inference of touch region
    '''

    img_tmp = copy.deepcopy(img)
    img_tmp = F.to_tensor(img_tmp)
    img_tmp = torch.unsqueeze(img_tmp, 0)

    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(num_classes)

    # move model to the right d1evice
    model.to(device)

    model.eval()
    # model = torch.load('model_RGB_original.pkl')

    # model = torch.load('model_RGB_2.pkl')
    # model = torch.load('model_RGB_40.pkl')
    model = torch.load('model_RGB_6.pkl')
    # model = torch.load('model_RGB_10.pkl')
    # model = torch.load('model_RGB_10_2.pkl')
    model.eval()

    # notice: due to the update of torchvision, we have to manually add some layer parameters
    for name, layer in model.named_modules():
        if isinstance(layer, torchvision.ops.misc.FrozenBatchNorm2d):
            layer.eps = 0.
    for name, layer in model.named_modules():
        if isinstance(layer, torchvision.models.detection.rpn.RegionProposalNetwork):
            layer.score_thresh = 0.

    img_tmp = img_tmp.to(device)
    outputs = model(img_tmp)

    return outputs


def calculateInitialPositionAndVelocity(_touch_point, _R_inv, _t_inv):
    '''
    _touch_point: pixel position of touch point
    _R_inv, _t_inv: the rotation and translation matrix between marker and camera
    '''

    # calculate the reference point
    # (1) calculate the reference point in marker frame
    reference_incam = pixel2camera(_touch_point, 0.4, fx, cx, cy)

    # (2) calculate the reference point in marker frame
    reference_inmarker = np.dot(_R_inv, reference_incam) + _t_inv

    # (3) calculate the reference point in world frame
    reference_px = base2marker_x + reference_inmarker[0]
    reference_py = base2marker_y + reference_inmarker[1]

    # (4) calculate the reference point in base frame
    theta = np.pi / 4
    reference_base_px = -math.cos(theta) * reference_px + math.cos(theta) * reference_py
    reference_base_py = -math.cos(theta) * reference_px - math.cos(theta) * reference_py
    reference_base_pz = base2marker_z + reference_inmarker[2]


    # calculate the initial point
    # (1) calculate initial point in base frame
    depth_point = _t_inv[2] - 0.15
    init_target_incam = pixel2camera(_touch_point, depth_point, fx, cx, cy)

    # (2) calculate initial point in base frame
    init_target_inmarker = np.dot(_R_inv, init_target_incam) + _t_inv
    # print("init_target_inmarker", init_target_inmarker)

    # (3) calculate the reference point in world frame
    target_px = base2marker_x + init_target_inmarker[0]
    target_py = base2marker_y + init_target_inmarker[1]

    # (4) calculate the reference point in base frame
    base_px = -math.cos(theta) * target_px + math.cos(theta) * target_py
    base_py = -math.cos(theta) * target_px - math.cos(theta) * target_py
    base_pz = base2marker_z + init_target_inmarker[2]
    initial_position = [base_px, base_py, base_pz]
    ### calculate initial point ###

    # calculate the speed for control the robotic arm
    speed_dx = (base_px - reference_base_px) * 0.08
    speed_dy = (base_py - reference_base_py) * 0.08
    speed_dz = (base_pz - reference_base_pz) * 0.08
    speed = [0.0, 0.0, 0.0]
    if speed_dz >0:
        speed = [-speed_dx, -speed_dy, -speed_dz]
    else:
        speed = [speed_dx, speed_dy, speed_dz]

    return initial_position, speed


def calculateGraspPosition(_grasp_proposal, _R_inv, _t_inv):
    '''
    calculate the grasp position
    _grasp_proposal: [px, py, depth, width, angle]
    note: px, py is the position in pixel space, so we need to calculate its corresponding 3D position
    _R_inv, _t_inv: the rotation and translation matrix between marker and camera
    '''
    theta = np.pi / 4
    target_incam = [_grasp_proposal[0], _grasp_proposal[1], _grasp_proposal[2]]
    # init_target_incam = pixel2camera((_grasp_proposal[0], _grasp_proposal[1]), 0.4, fx, cx, cy)
    # target_incam[0] = init_target_incam[0]
    # target_incam[1] = init_target_incam[1]
    target_inmarker = np.dot(_R_inv, target_incam) + _t_inv
    print("init_target_inmarker", target_inmarker)

    target_px = base2marker_x + target_inmarker[0]
    target_py = base2marker_y + target_inmarker[1] - 0.02
    base_pz = base2marker_z + target_inmarker[2] + 0.14

    base_px = -math.cos(theta) * target_px + math.cos(theta) * target_py
    base_py = -math.cos(theta) * target_px - math.cos(theta) * target_py
    return [base_px, base_py, base_pz]


def pixel2camera (pixel_pos, pixel_depth, f, x0, y0):
    # get the pixel position in camera frame
    '''
    piexl_pos: pixel position of touch point
    pixel_depth: depth for touch point
    f, x0, y0: camera intrinsic parameters
    '''
    return [(pixel_pos[0] - x0) / f * pixel_depth, (pixel_pos[1] - y0) / f * pixel_depth, pixel_depth]

def end2gelsight (end_pos, theta, end_d_pos, inv_flag):
    '''
    transformation for touch movement, let gelsight follow the light path or
    calculate the end-effector position for depth and grasping

    end_pos: end-effector position
    theta: the rotation angle between base and world frame
    end_d_pos: the position difference between end-effector and gelsight
    inv_flag: inverse transformation
    '''

    base_dx_ef2gel = -math.cos(theta) * end_d_pos[0] + math.cos(theta) * end_d_pos[1]
    base_dy_ef2gel = -math.cos(theta) * end_d_pos[0] - math.cos(theta) * end_d_pos[1]
    base_dz_ef2gel = end_d_pos[2]
    if not inv_flag:
        gelsight_pos = [end_pos[0]+base_dx_ef2gel, end_pos[1]+base_dy_ef2gel, end_pos[2]+base_dz_ef2gel]
    else:
        gelsight_pos = [end_pos[0]-base_dx_ef2gel, end_pos[1]-base_dy_ef2gel, end_pos[2]-base_dz_ef2gel]

    return gelsight_pos



def calculateDepth(_end_pos, theta, end_d_pos, _R, _t):
    '''
    calculate the depth in camera frame
    _end_pos: the position of end-effector
    '''
    ee_p = end2gelsight(_end_pos, theta, end_d_pos, True)
    maker_p = base2marker(ee_p)
    camera_p = np.dot(_R, maker_p) + _t
    depth = camera_p[2]

    return depth



def base2marker(ee_pos):
    # calculate the point in marker frame
    theta = np.pi / 4
    reference_marker_px = -math.cos(theta) * ee_pos[0] - math.cos(theta) * ee_pos[1] - base2marker_x
    reference_marker_py = math.cos(theta) * ee_pos[0] - math.cos(theta) * ee_pos[1] - base2marker_y
    reference_marker_pz = -base2marker_z + ee_pos[2]

    return [reference_marker_px, reference_marker_py, reference_marker_pz]


def servo4touch():
    global contact
    global translation

    # Initialize Gripper
    # rospy.init_node("robotiq_2f_gripper_ctrl_test")


    # Initialize Ur5
    ur5 = Ur5Controller()
    ur5.ratet = 1
    ur5.Init_node()

    gp = GripperController()
    rospy.sleep(2)
    # Initialize Camera
    print('Running live demo of robotic grasping. Make sure realsense camera is streaming.\n')
    rcamera = camera.Camera()
    time.sleep(1)  # Give camera some time to load data

    # camera_intrinsics = rcamera.color_intr
    # realsense_fx = camera_intrinsics[0, 0]
    # realsense_fy = camera_intrinsics[1, 1]
    # realsense_cx = camera_intrinsics[0, 2]
    # realsense_cy = camera_intrinsics[1, 2]
    # print(realsense_fx)

    # Initialize camera pose, this can be obtained from camera_pose_calculation script
    tvec = np.array([0.16614668, 0.06902181, 0.47960351])
    # tvec = np.array([0.11902927, 0.06902181, 0.4888751])
    # 0.16614668
    # 0.06745851
    # 0.47960351
    rvec = np.array([2.79052997, -0.03803889,  0.06994612])
    # 22.5 degrees.

    R, _ = cv2.Rodrigues(rvec)
    print("Rotation Matrix :", R)
    R_inv = np.linalg.inv(R)
    print("Inverse Rotation Matrix :", R_inv)
    t_inv = -np.dot(R_inv, tvec)
    print("Inverse Translation Matrix :", t_inv)




    while not rospy.is_shutdown():
        color_img, input_depth = rcamera.get_data()
        input_depth = input_depth.astype(np.float32)
        img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # print('Input the index for saving the image:')
        # data_index = int(input())
        # real_data_name = "real_dataset/" + str(data_index) + ".png"
        cv2.imwrite("input.png", img)
        im_pil = Image.fromarray(img)
        outputs = touch_mask_inference(im_pil)

        boxes = outputs[0]['boxes'].to("cpu")
        scores = outputs[0]['scores'].to("cpu")
        masks = outputs[0]['masks'].to("cpu")


        test_flag = False


        touch_points = touch_generation(boxes, scores, masks, test_flag)
        depths = []

        # touch_points_bbox = touch_generation_bbox(boxes, scores)
        print( boxes)
        for _touch_point in touch_points:
            cv2.circle(img, _touch_point, 6, (255, 0, 0), -1)
        # for _touch_point in touch_points_bbox:
        #     cv2.circle(img, _touch_point, 6, (0, 255, 0), -1)
            # print(_touch_points)
        cv2.imwrite("touch_point.png", img)
        cv2.imshow("test", img)
        cv2.waitKey()

        # break

        masks_deta = masks.detach().numpy()
        for i in range(len(masks)):
            # if scores[i]
            # mask = masks_deta[i]
            mask = masks_deta[i].transpose((1, 2, 0))
            print(mask.shape)
            print(scores[i])
            mask_show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
            for i_index in range(mask.shape[0]):
                for j_index in range(mask.shape[1]):
                    if mask[i_index, j_index] > 0.5:
                        mask_show[i_index, j_index] = (255, 255, 0)

            cv2.imshow("test", mask_show)
            cv2.waitKey()
            # output_name = str(i)+".png"
            # cv2.imwrite(output_name, mask_show)

        # break

        for touch_point in touch_points:
            search_max = 0.7 # from the camera to table

            # calculate the base position and speed
            base_p, speed_d = calculateInitialPositionAndVelocity(touch_point, R_inv, t_inv)

            speed_dx = speed_d[0]
            speed_dy = speed_d[1]
            speed_dz = speed_d[2]

            # calculate the transform from end-effector to gelsight sensor
            theta = np.pi/4
            d_ef2gel = [0, 0, 0.08]
            inverse_flag = False
            gelsight_pos = end2gelsight(base_p, theta, d_ef2gel, inverse_flag)
            pos = [gelsight_pos[0], gelsight_pos[1], gelsight_pos[2], 0.7371, -1.762, -1.77]
            print(pos)
            ur5.t = 3
            ur5.ratet = 2
            tempPos = copy.deepcopy(pos)
            ur5.move2Pos(tempPos)
            rospy.sleep(5)


            vel = [speed_dx, speed_dy, speed_dz, 0, 0, 0]
            tempVel = copy.deepcopy(vel)

            while not rospy.is_shutdown():
                print(contact)
                if contact:
                    break
                flag_moving = False
                if not flag_moving:
                    _t = 10
                    ur5.move2Vel(tempVel, _t)
                    flag_moving = True
                else:
                    time.sleep(0.25)
                # rospy.spinOnce
            _t = 0.5
            tempVel= [0, 0, 0, 0, 0, 0]
            ur5.move2Vel(tempVel, _t)

            # save the depth information or end-effector position
            depth = calculateDepth(end_pos, theta, d_ef2gel, R, tvec)
            depths.append(depth)


            # print('Would you like to go back to home 1:Yes 0:No')
            # home_index = int(input())
            # while ((home_index != 1)):
            #     print('Would you like to go back to home 1:Yes 0:No')
            #     home_index = int(input())
            up_pos = [end_pos[0], end_pos[1], end_pos[2] + 0.05, 0.6905, -1.7642, -1.7023]
            temp_homePos = copy.deepcopy(up_pos)
            ur5.move2Pos(temp_homePos)
            rospy.sleep(2)

            home_pos = [-0.24, 0.085, 0.295, 0.6905, -1.7642, -1.7023]
            # home_pos = [-0.3, 0.20, 0.367,  0.7371, -1.762, -1.77]
            ur5.t = 2
            ur5.ratet = 1
            temp_homePos = copy.deepcopy(home_pos)
            ur5.move2Pos(temp_homePos)
            ur5.move2Pos(temp_homePos)
            rospy.sleep(2)
            rospy.sleep(2)

            # publish the command for updating the background
            cmd = Bool()
            cmd = True
            update_pub.publish(cmd)

        print(depths)
        # print('Would you like to switch to grasp mode 1:Yes 0:No')
        # home_index = int(input())
        # while ((home_index != 1)):
        #     print('Would you like to switch to grasp mode 1:Yes 0:No')
        #     home_index = int(input())
        theta = np.pi / 4
        dx_ef2gel = 0
        dy_ef2gel = -0.0
        base_dx_ef2gel = -math.cos(theta) * dx_ef2gel + math.cos(theta) * dy_ef2gel
        base_dy_ef2gel = -math.cos(theta) * dx_ef2gel - math.cos(theta) * dy_ef2gel
        base_dz_ef2gel = 0.08
        home_pos = [-0.45-base_dx_ef2gel, 0.25-base_dy_ef2gel, 0.32, 1.21, -2.846, 0.001]
        temp_homePos = copy.deepcopy(home_pos)
        ur5.move2Pos(temp_homePos)
        rospy.sleep(2)
        # home_pos = [-0.4-base_dx_ef2gel, 0.185-base_dy_ef2gel, 0.32, 2.899, -1.2, 0.05]
        # temp_homePos = copy.deepcopy(home_pos)
        # ur5.move2Pos(temp_homePos)
        # rospy.sleep(2)

        # calculate where to grasp
        grasp_proposals = grasp_generation(boxes, scores, masks, depths, touch_points)
        # grasp_proposals = grasp_generation_depth(boxes, scores, masks, input_depth)
        for grasp_proposal in grasp_proposals:
            print(grasp_proposal)
            # control the gripper width
            # if grasp_proposal[3] == 150:
            #     gripper_width = 0.087
            # else:
            #     #TODO: calculate the gripper width
            #     gripper_width = grasp_proposal[3]/925.0*grasp_proposals[2]
            # rospy.sleep(2)
            gp.goto(grasp_proposal[3], 0.013, 100, block=False)
            rospy.sleep(2)

            # calculate grasp point
            [base_px, base_py, base_pz] = calculateGraspPosition(grasp_proposal, R_inv, t_inv)
            # move to initial grasp pose (a little higher than the grasp position)
            # rpy = [178.94, -2.6, -133.91] # 178.94, -2.6 133.91

            # 0-180 degrees in image represent -135 - 45 degrees in robotic base frame
            angle = grasp_proposal[4]
            angle_real = angle - 135

            rpy = [178.94, -1.68, angle_real] # 178.94, -2.6 133.91
            # rpy = [3.123, -0.045, -2.063] # 178.94, -2.6 133.91
            rotation_vector = rpy2rv(rpy)
            pos = [base_px, base_py, base_pz + 0.05, 1.21, -2.846, 0.01]
            # pos = [base_px, base_py, base_pz + 0.05, 1.21, -2.846, 0.001]
            ur5.t = 3
            ur5.ratet = 1
            tempPos = copy.deepcopy(pos)
            ur5.move2Pos(tempPos)
            rospy.sleep(3)

            pos = [base_px, base_py, base_pz + 0.05, rotation_vector[0], rotation_vector[1], 0.01]
            ur5.t = 3
            ur5.ratet = 1
            tempPos = copy.deepcopy(pos)
            ur5.move2Pos(tempPos)
            rospy.sleep(3)

            # move to grasp pose
            pos = [base_px, base_py, base_pz, rotation_vector[0], rotation_vector[1], 0.01]
            ur5.t = 2
            ur5.ratet = 1
            tempPos = copy.deepcopy(pos)
            ur5.move2Pos(tempPos)

            # Grasp
            rospy.sleep(3)
            gp.goto(0, 0.02, 100, block=False)
            rospy.sleep(3)

            # # move to home
            home_pos = [base_px, base_py, base_pz + 0.05, rotation_vector[0], rotation_vector[1], 0.01]
            # home_pos = [-0.451, -0.156, 0.355, 1.21, -2.846, 0.001]
            # home_pos = [-0.24, 0.085, 0.395, 1.21, -2.846, 0.001]
            # home_pos = [-0.304, 0.146, 0.32, 1.2833, -2.8331, -0.0310]
            tempPos = copy.deepcopy(home_pos)
            ur5.move2Pos(tempPos)
            #
            print('Would you like to open the gripper 1:Yes 0:No')
            open_index = int(input())
            while ( (open_index != 1) ):
                print('Would you like to open the gripper 1:Yes 0:No')
                open_index = int(input())
            gp.goto(0.087, 0.02, 30, blocks=False)
        #
        # else:
        #     continue
    return 0

if __name__ == '__main__':

    try:
        servo4touch()
    except rospy.ROSInterruptException:
        pass
