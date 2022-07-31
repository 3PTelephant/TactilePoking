import cv2
import numpy as np
import os
import copy

p_x = 960.0
p_y = 524.0
fx = 1387.7
max_width = 200

def touch_generation_bbox(obj_boxes, obj_scores):
    touch_points = []

    for i, (x1, y1, x2, y2) in enumerate(obj_boxes):
        if obj_scores[i] < 0.7:
            continue
        # [x1, y1, x2, y2] = obj_boxes[i]
        [x1, y1, x2, y2] = obj_boxes[i].detach().numpy()
        x1_np = np.uint32(x1)
        x2_np = np.uint32(x2)
        y1_np = np.uint32(y1)
        y2_np = np.uint32(y2)

        x_mean = np.uint32((x1_np+x2_np)/2)
        y_mean = np.uint32((y1_np+y2_np)/2)

        touch_points.append((x_mean, y_mean))
    return touch_points

def touch_generation(obj_boxes, obj_scores, obj_masks, flag):
    touch_points = []
    obj_masks = obj_masks.detach().numpy()

    for i, (x1, y1, x2, y2) in enumerate(obj_boxes):
        if obj_scores[i] < 0.5:
            continue
        # [x1, y1, x2, y2] = obj_boxes[i]
        [x1, y1, x2, y2] = obj_boxes[i].detach().numpy()
        x1_np = np.uint32(x1)
        x2_np = np.uint32(x2)
        y1_np = np.uint32(y1)
        y2_np = np.uint32(y2)

        box = [x1_np, y1_np, x2_np, y2_np]

        # mask_show = np.zeros(obj_masks[i].shape, dtype=np.uint8)
        # flag = True
        if flag:
            mask_tmp = obj_masks[i]
        else:
            mask_tmp = obj_masks[i].transpose((1, 2, 0))

        # mask = np.zeros(mask_tmp.shape[:-1], dtype=np.uint8)
        _mask = np.zeros(mask_tmp.shape, dtype=np.uint8)
        flag = False

        if flag is True:
            _mask = mask_tmp
        else:
            for i_index in range(mask_tmp.shape[0]):
                for j_index in range(mask_tmp.shape[1]):
                    if mask_tmp[i_index, j_index]>0.5:
                        _mask[i_index, j_index] = 255

        # find the boundary of the mask
        contours, hierarchy = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        _ellipse_x = 0
        _ellipse_y = 0

        # for i_cnt, cnt in enumerate(contours):
        #     if cv2.contourArea(cnt) < 30:
        #         continue
        (_ellipse_x, _ellipse_y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # cv2.ellipse(mask_tmp, _ellipse, (255, 255, 0), 2)
        _ellipse_y = np.uint32(_ellipse_y)
        _ellipse_x = np.uint32(_ellipse_x)
        print(_ellipse_x,_ellipse_y)
        # cv2.ellipse(_mask, [_ellipse_x, _ellipse_y], (255,255,0), 2)
        # cv2.imshow("mask", mask_tmp)
        # cv2.waitKey()
        if _mask[_ellipse_y, _ellipse_x] == 255:
            touch_points.append((_ellipse_x, _ellipse_y))
        else:
            # find the nearest positive point
            touch_point = findNearestPos((_ellipse_x, _ellipse_y), box, _mask)
            touch_points.append(touch_point)

    return touch_points


def findNearestPos(touch_point, box, touch_mask):
    [x1_tmp, y1_tmp, x2_tmp, y2_tmp] = box
    min_distance = 200.0
    _nearestPos = [(x1_tmp + x2_tmp) / 2, (y1_tmp + y2_tmp) / 2]
    for j_index in range(y1_tmp, y2_tmp):
        for i_index in range(x1_tmp, x2_tmp):
            tmp_distance = np.sqrt((i_index - touch_point[0]) ** 2 + (j_index - touch_point[1]) ** 2)
            if tmp_distance < min_distance and touch_mask[j_index, i_index]:
                _nearestPos = (i_index, j_index)
                min_distance = tmp_distance
    return _nearestPos


def grasp_generation_depth(obj_boxes, obj_scores, obj_masks, obj_depths):

    grasp_proposals = []
    obj_masks = obj_masks.detach().numpy()

    for i, (x1, y1, x2, y2) in enumerate(obj_boxes):
        if obj_scores[i] < 0.5:
            continue

        flag = False
        # if flag:
        #     mask_tmp = obj_masks[i]
        # else:
        print(obj_masks[i])
        mask_tmp = obj_masks[i].transpose((1, 2, 0))
        _mask = np.zeros(mask_tmp.shape, dtype=np.uint8)


        if flag is True:
            _mask = mask_tmp
        else:
            for i_index in range(mask_tmp.shape[0]):
                for j_index in range(mask_tmp.shape[1]):
                    if mask_tmp[i_index, j_index]>0.5:
                        _mask[i_index, j_index] = 255

        # find the boundary of the mask
        contours, hierarchy = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _ellipse_x = 0.0
        _ellipse_y = 0.0
        angle = 0.0

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # for i_cnt, cnt in enumerate(contours):
        #     if cv2.contourArea(cnt) < 30:
        #         continue
            # ellipse fitting to get the orientation and centroid position
        (_ellipse_x, _ellipse_y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # cv2.ellipse(mask_tmp, _ellipse, (255, 255, 0), 2)
        _ellipse_y = np.uint32(_ellipse_y)
        _ellipse_x = np.uint32(_ellipse_x)
        print(_ellipse_x, _ellipse_y)
        if _mask[_ellipse_y, _ellipse_x] == 255:
            # used for grasp
            x = (_ellipse_x - p_x) / fx * obj_depths[_ellipse_y, _ellipse_x]
            y = (_ellipse_y - p_y) / fx * obj_depths[_ellipse_y, _ellipse_x]
            # used for shown
            # x = _ellipse_x
            # y = _ellipse_y
            d = obj_depths[_ellipse_y, _ellipse_x]
            width = 0.087
            grasp_angle = np.int32(angle)
            grasp_proposals.append([x, y, d, width, grasp_angle])

    return


def grasp_generation(obj_boxes, obj_scores, obj_masks, obj_depths, obj_touch_points):
    # grasp proposal = [x,y,z,width,angle]
    grasp_proposals = []
    obj_masks = obj_masks.detach().numpy()

    for i, (x1, y1, x2, y2) in enumerate(obj_boxes):
        if obj_scores[i] < 0.5:
            continue

        # mask_tmp = np.zeros(obj_masks[i].shape, dtype=np.uint8)

        # flag is set to False, when mask comes from prediction
        # flag set to True, when mask comes from ground truth
        flag = False
        # if flag:
        #     mask_tmp = obj_masks[i]
        # else:
        print(obj_masks[i])
        mask_tmp = obj_masks[i].transpose((1, 2, 0))
        _mask = np.zeros(mask_tmp.shape, dtype=np.uint8)

        # flag =

        if flag is True:
            _mask = mask_tmp
        else:
            for i_index in range(mask_tmp.shape[0]):
                for j_index in range(mask_tmp.shape[1]):
                    if mask_tmp[i_index, j_index]>0.5:
                        _mask[i_index, j_index] = 255

        # find the boundary of the mask
        contours, hierarchy = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _ellipse_x = 0.0
        _ellipse_y = 0.0
        angle = 0.0

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # for i_cnt, cnt in enumerate(contours):
        #     if cv2.contourArea(cnt) < 30:
        #         continue
            # ellipse fitting to get the orientation and centroid position
        (_ellipse_x, _ellipse_y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # cv2.ellipse(mask_tmp, _ellipse, (255, 255, 0), 2)
        _ellipse_y = np.uint32(_ellipse_y)
        _ellipse_x = np.uint32(_ellipse_x)
        print(_ellipse_x, _ellipse_y)
        if _mask[_ellipse_y, _ellipse_x] == 255:
            # used for grasp
            x = (_ellipse_x - p_x) / fx * obj_depths[i]
            y = (_ellipse_y - p_y) / fx * obj_depths[i]
            # used for shown
            # x = _ellipse_x
            # y = _ellipse_y
            d = obj_depths[i]
            width = 0.087
            grasp_angle = np.int32(angle)
            grasp_proposals.append([x, y, d, width, grasp_angle])

        else:
            # check the size of the center region
            theta_t = np.arctan2(obj_touch_points[i][1]-p_y, fx)
            theta_c = np.arctan2(_ellipse_y-p_y, fx)
            deta_theta = np.fabs(theta_t - theta_c)
            alpha = 0.3736
            TO_angle = np.fabs(np.pi/2 + theta_c - alpha)
            TO = obj_depths[i]/np.cos(theta_t)
            TC = TO*np.sin(deta_theta)/np.sin(TO_angle)
            distance = TC*(np.sqrt((obj_touch_points[i][1]-p_y)**2+(obj_touch_points[i][0]-p_x)**2)/np.fabs(obj_touch_points[i][1]-p_y))
            print("The distance between touch point and center point:%f", distance)
            if distance > 0.02:
                width_big = True
            else:
                width_big = False

            if width_big:
                # used for grasp
                x = (obj_touch_points[i][0] - p_x) / fx * obj_depths[i]
                y = (obj_touch_points[i][1] - p_y) / fx * obj_depths[i]
                width = 2*distance
                # used for shown
                # (x, y) = obj_touch_points[i]
                # width = np.sqrt((x - _ellipse_x) ** 2 + (y - _ellipse_y) ** 2)
                grasp_angle = np.arctan2(y - _ellipse_y, x - _ellipse_x)
                # change the value range from [-pi,pi] to [0,180]
                if grasp_angle <0.0:
                    grasp_angle = grasp_angle/np.pi*180+180
                else:
                    grasp_angle = grasp_angle/np.pi*180
                grasp_proposal = [x, y, obj_depths[i], width, grasp_angle]
                grasp_proposals.append(grasp_proposal)

            else:
                # use the ellipse fitting and
                # grasp the center
                # used for grasp
                x = (_ellipse_x - p_x) / fx * obj_depths[i]
                y = (_ellipse_y - p_y) / fx * obj_depths[i]
                # used for shown
                # x = _ellipse_x
                # y = _ellipse_y
                d = obj_depths[i]
                width = 0.087
                grasp_angle = np.int32(angle)
                grasp_proposals.append([x, y, d, width, grasp_angle])
                # grasp_proposal = [0, 0, 0, 0, 0]

    return grasp_proposals

def touch_generation_vis(obj_boxes, obj_scores, obj_masks, flag):
    touch_points = []
    fitted_ellipse = []
    obj_masks = obj_masks.detach().numpy()

    for i, (x1, y1, x2, y2) in enumerate(obj_boxes):
        if obj_scores[i] < 0.5:
            continue
        # [x1, y1, x2, y2] = obj_boxes[i]
        [x1, y1, x2, y2] = obj_boxes[i].detach().numpy()
        x1_np = np.uint32(x1)
        x2_np = np.uint32(x2)
        y1_np = np.uint32(y1)
        y2_np = np.uint32(y2)

        box = [x1_np, y1_np, x2_np, y2_np]

        # mask_show = np.zeros(obj_masks[i].shape, dtype=np.uint8)
        # flag = True
        if flag:
            mask_tmp = obj_masks[i]
        else:
            mask_tmp = obj_masks[i].transpose((1, 2, 0))

        # mask = np.zeros(mask_tmp.shape[:-1], dtype=np.uint8)
        _mask = np.zeros(mask_tmp.shape, dtype=np.uint8)
        flag = False

        if flag is True:
            _mask = mask_tmp
        else:
            for i_index in range(mask_tmp.shape[0]):
                for j_index in range(mask_tmp.shape[1]):
                    if mask_tmp[i_index, j_index]>0.5:
                        _mask[i_index, j_index] = 255

        # find the boundary of the mask
        contours, hierarchy = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        _ellipse_x = 0
        _ellipse_y = 0

        # for i_cnt, cnt in enumerate(contours):
        #     if cv2.contourArea(cnt) < 30:
        #         continue
        (_ellipse_x, _ellipse_y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # cv2.ellipse(mask_tmp, _ellipse, (255, 255, 0), 2)
        _ellipse_y = np.uint32(_ellipse_y)
        _ellipse_x = np.uint32(_ellipse_x)
        print(_ellipse_x,_ellipse_y)
        # cv2.ellipse(_mask, [_ellipse_x, _ellipse_y], (255,255,0), 2)
        # cv2.imshow("mask", mask_tmp)
        # cv2.waitKey()
        fitted_ellipse.append((_ellipse_x, _ellipse_y, angle))
        if _mask[_ellipse_y, _ellipse_x] == 255:
            touch_points.append((_ellipse_x, _ellipse_y))
        else:
            # find the nearest positive point
            touch_point = findNearestPos((_ellipse_x, _ellipse_y), box, _mask)
            touch_points.append(touch_point)

    return touch_points,fitted_ellipse

def grasp_generation_vis(obj_boxes, obj_scores, obj_masks, obj_touch_points):
    # grasp proposal = [x,y,z,width,angle]
    grasp_proposals = []
    obj_masks = obj_masks.detach().numpy()

    for i, (x1, y1, x2, y2) in enumerate(obj_boxes):
        if obj_scores[i] < 0.5:
            continue

        # mask_tmp = np.zeros(obj_masks[i].shape, dtype=np.uint8)

        # flag is set to False, when mask comes from prediction
        # flag set to True, when mask comes from ground truth
        flag = False
        # if flag:
        #     mask_tmp = obj_masks[i]
        # else:
        print(obj_masks[i])
        mask_tmp = obj_masks[i].transpose((1, 2, 0))
        _mask = np.zeros(mask_tmp.shape, dtype=np.uint8)

        # flag =

        if flag is True:
            _mask = mask_tmp
        else:
            for i_index in range(mask_tmp.shape[0]):
                for j_index in range(mask_tmp.shape[1]):
                    if mask_tmp[i_index, j_index]>0.5:
                        _mask[i_index, j_index] = 255

        # find the boundary of the mask
        contours, hierarchy = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _ellipse_x = 0.0
        _ellipse_y = 0.0
        angle = 0.0

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # for i_cnt, cnt in enumerate(contours):
        #     if cv2.contourArea(cnt) < 30:
        #         continue
            # ellipse fitting to get the orientation and centroid position
        (_ellipse_x, _ellipse_y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # cv2.ellipse(mask_tmp, _ellipse, (255, 255, 0), 2)
        _ellipse_y = np.uint32(_ellipse_y)
        _ellipse_x = np.uint32(_ellipse_x)
        print(_ellipse_x, _ellipse_y)
        if _mask[_ellipse_y, _ellipse_x] == 255:
            # used for grasp
            x = _ellipse_x
            y = _ellipse_y
            # used for shown
            # x = _ellipse_x
            # y = _ellipse_y
            width = 125
            grasp_angle = np.int32(angle)/180*np.pi
            grasp_proposals.append([x, y, 0, width, grasp_angle])

        else:
            # check the size of the center region
            distance = np.sqrt((obj_touch_points[i][0]-_ellipse_x)**2+(obj_touch_points[i][1]-_ellipse_y)**2)
            # used for grasp
            x = obj_touch_points[i][0]
            y = obj_touch_points[i][1]
            width = 2*distance
            # used for shown
            # (x, y) = obj_touch_points[i]
            # width = np.sqrt((x - _ellipse_x) ** 2 + (y - _ellipse_y) ** 2)
            grasp_angle = np.arctan2(y - _ellipse_y, x - _ellipse_x)
            grasp_proposal = [x, y, 0, width, grasp_angle]
            grasp_proposals.append(grasp_proposal)


    return grasp_proposals

if __name__ == '__main__':
    root = "/home/jackey/TouchRegion/highball"

    # read the mask and touch region ground truth to generate the touch point
    imgs_list = list(sorted(os.listdir(os.path.join(root, "rgb"))))
    masks_list = list(sorted(os.listdir(os.path.join(root, "masks"))))
    touchs_list = list(sorted(os.listdir(os.path.join(root, "touch-region"))))

    for idx in range(len(imgs_list)):
        # read image and annotations
        img_path = os.path.join(root, "rgb", imgs_list[idx])
        mask_path = os.path.join(root, "masks", masks_list[idx])
        touch_index = np.uint8(idx / 10)
        touch_path = os.path.join(root, "touch-region", touchs_list[touch_index])

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        touch = cv2.imread(touch_path, 0)

        # copy the mask for generating the bbox
        mask_original = copy.deepcopy(mask)
        # modify the mask image based on touch region
        mask[touch == 0] = 0

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask_original == obj_ids[:, None, None]

        # generate the annotations (boxes, scores, masks)
        num_objs = len(obj_ids)
        boxes = []
        scores = np.ones(num_objs, dtype=np.float32)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        for ma_index in range(0, len(obj_ids)):
            masks[ma_index][touch == 0] = 0

        test_flag = True
        _touch_points = touch_generation(boxes, scores, masks, test_flag)

        for _touch_point in _touch_points:
            cv2.circle(img, _touch_point, 6, (255, 0, 0), -1)
        cv2.imshow("test", img)
        cv2.waitKey()

        # if get the depth information, generate grasp proposal
        depths = np.ones(masks.shape[0], dtype=np.float32)

        # _grasp_proposals = []
        _grasp_proposals = grasp_generation(boxes, scores, masks, depths, _touch_points)

        for _grasp_proposal in _grasp_proposals:
            center = (_grasp_proposal[0], _grasp_proposal[1])
            width = _grasp_proposal[3]
            angle = _grasp_proposal[4]
            pt1 = (np.uint32(_grasp_proposal[0] + width * np.cos(angle)),
                   np.uint32(_grasp_proposal[1] + width * np.sin(angle)))
            # check the boundary
            pt11 = (np.uint32(pt1[0] + 20 * np.sin(angle)), np.uint32(pt1[1] - 20 * np.cos(angle)))
            pt12 = (np.uint32(pt1[0] - 20 * np.sin(angle)), np.uint32(pt1[1] + 20 * np.cos(angle)))

            pt2 = (np.uint32(_grasp_proposal[0] - width * np.cos(angle)),
                   np.uint32(_grasp_proposal[1] - width * np.sin(angle)))
            pt21 = (np.uint32(pt2[0] + 20 * np.sin(angle)), np.uint32(pt2[1] - 20 * np.cos(angle)))
            pt22 = (np.uint32(pt2[0] - 20 * np.sin(angle)), np.uint32(pt2[1] + 20 * np.cos(angle)))
            # pt2 = np.uint32((_grasp_proposal[0] - width*np.cos(angle), _grasp_proposal[1] - width*np.sin(angle)))
            # pt21 = np.uint32((pt2[0] + 20 * np.sin(angle), pt2[1] - 20 * np.cos(angle)))
            # pt22 = np.uint32((pt2[0] - 20 * np.sin(angle), pt2[1] + 20 * np.cos(angle)))
            # pt21 = pt2 + 20 * (np.sin(angle), -np.cos(angle))
            # pt22 = pt2 - 20 * (np.sin(angle), -np.cos(angle))
            cv2.line(img, pt1, pt2, (255, 255, 0), 4)
            cv2.line(img, pt11, pt12, (255, 255, 0), 3)
            cv2.line(img, pt21, pt22, (255, 255, 0), 3)

        cv2.imshow("test", img)
        cv2.waitKey()
