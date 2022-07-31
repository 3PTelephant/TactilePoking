# This Python file uses the following encoding: utf-8
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
import cv2
import copy
import numpy as np
contact = False
contact_pub = rospy.Publisher('contact_status', Bool, queue_size = 10)

update = False
def update_cb(msg):
    '''
    callback function for update the background
    '''
    global update
    update = msg.data
    # print(contact)
    return 0

update_sub = rospy.Subscriber('contact_update', Bool, update_cb)


def main():
    rospy.init_node("contact_trigger")
    # read gelsight image
#    background_img = cv2.imread("/home/jackey/grasp_ws/src/data_collection_ur5/grasp_control_node/2021-06-08-001945.jpg")
    cap = cv2.VideoCapture(8)
    cap.set(3, 1280)
    cap.set(4, 720)
    number = 1

    while number < 50:
        ret, frame = cap.read()
        number = number + 1
        cv2.imshow("test", frame)
        cv2.waitKey(1)
        continue
    ret, frame = cap.read()
    background_img = copy.deepcopy(frame)
    cv2.imshow("background", background_img)
    background_img = background_img.astype(np.float32)
    b_b, g_b, r_b=cv2.split(background_img)
    global update
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        frame_show = copy.deepcopy(frame)
        frame = frame.astype(np.float32)
        if update:
            background_img = frame
            b_b, g_b, r_b = cv2.split(background_img)
            update = False

        b,g,r = cv2.split(frame)
#        diff = abs(frame - background_img)

#        print (background_img.size)
#        gray_diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        b_d =  abs(b-b_b)
        g_d =  abs(g-g_b)
        r_d =  abs(r-r_b)
        # cv2.imshow("b",b)
        # cv2.imshow("bb",b_b)
        # cv2.imshow("bd",b_d)
        # cv2.waitKey(1)

        # Due to the fact that blue occupy most of the image
        _, thresh_b = cv2.threshold(b_d, 30,255,cv2.THRESH_BINARY)
        # cv2.imshow("diff_b",thresh_b)
        # cv2.waitKey(1)

        _, thresh_g = cv2.threshold(g_d, 30,255,cv2.THRESH_BINARY)
        # cv2.imshow("diff_g",thresh_g)
        # cv2.waitKey(1)

        _, thresh_r = cv2.threshold(r_d, 30,255,cv2.THRESH_BINARY)
        # cv2.imshow("diff_r",thresh_r)
        # cv2.waitKey(1)

        changes = 0
        b_number = thresh_b == 255
        changes = len(b[b_number])

        g_number = thresh_g == 255
        changes = changes + len(g[g_number])

        r_number = thresh_r == 255
        changes = changes + len(r[r_number])
#        print(changes)
#        # img processing to check whether there is a contact

        if changes > 800:
            cmd = Bool()
            cmd = True
            contact_pub.publish(cmd)
            cv2.putText(frame_show, "Contact!!!", (100,100), 0, 2, 255)
            cv2.imshow("test", frame_show)
            cv2.waitKey(1)
            print ("contact")
        else :
            cv2.putText(frame_show, "No Contact!!!", (100,100), 0, 2, 255)
            cv2.imshow("test", frame_show)
            cv2.waitKey(1)
            cmd = Bool()
            cmd = False
            contact_pub.publish(cmd)


    return 0

if __name__ == '__main__':
    main()
