# title           :AUV_3_2.py
# description     :AUV巡游系统（待测试版）
# author          :Fatih and Chen YiLin
# date            :2019.8.3
# version         :0.3
# notes           :工控机 寻线过框 抓球 撞球
# python_version  :3.6



import time
import cv2
import logging
import sys
import numpy as np
import serial
import threading
from collections import deque
from scipy.spatial import distance as dist
from collections import OrderedDict

#寻球PID参数
P0 = 0.20
I0 = 0.01
D0 = 0.005
#寻球PID参数
P1 = 0.2
I1 = 0.01
D1 = 0.005
#转线PID参数
P2 = 40
I2 = 0.1
D2 = 0.5
#对框PID参数
P3 = 0.25
I3 = 0.01
D3 = 0.005
#对框PID参数
P4 = 0.1
I4 = 0.01
D4 = 0.005


#寻球可能所在的参考坐标，给予AUV一个大致的开始寻找方向
Reference_coor = (150,350)

#指令发送计数器
order_count1 = 0
order_count2 = 10
order_count3 = 20
order_count4 = 30
order_count5 = 0
count_max1 = 30
count_max2 = 35
count_max3 = 40
count_max4 = 45
count_max5 = 5

#框的数量
Rect_num = 5

#线计数器
line_count = 0

#记录已经过的框
crossed_count = 0

#中点两边边界
left_min = 280
right_max = 360

#引导线转动参数
guide_line_enable_lower = -0.2
guide_line_enable_higher = 0.2


#模式计数器
SEARCH_count = 0            #寻找模式
SEARCH_count_max = 41
SEARCH_count_time = 0
SEARCH_count_time_max = 2

ADJUST_count = 0            #调整模式
ADJUST_count_max = 200
ADJUST_count_time = 0
ADJUST_count_time_max = 2

CROSS_count = 0            #调整模式
CROSS_count_max = 300

#miniArea噪声抑制程度，越大识别框的能力越低。准确性越高
cnts2_area = 20000
cntsl0_area = 500
cntsr0_area = 500

#框信任度最大方差之
var_maxvalue = 35000

# AUV测框的距离参数
FORCAL = 600                # 距离函数设定的焦距，为定值
Know_Distance = 30.0        # 已知距离（定值）
Know_Width = 25.2

data_count = 0              # 框信任度，存储历史数据的计数器
cX_container = []
minAreax_container = []

# AUV位置列表，分别记录AUV的x,y,theta最大存储400个位置数据
x0 = deque(maxlen=400)
y0 = deque(maxlen=400)
theta0 = deque(maxlen=400)
x0.appendleft(0)
y0.appendleft(0)
theta0.appendleft(1.57)

Rect_point = []
Line_point = []


# 差速模型接口
# 暂定的AUV结构参数
# b : AUV宽度
# dl,dr : 左右推进器一次推进器前进距离
b = 0.5
dl = 0
dr = 0
AUV_dx = 0                                  #未给出
AUV_dy = 0
AUV_dtheta = 0
SL = 0
SR = 0
model = 'diff'

# 无目标信任度计数参数
Count = 0
Count1 = 0
K_COUNT = 0
X_COUNT = 0

# AUV检测球参数
buff = 64
ballLower = (29, 86, 6)
ballUpper = (64, 255, 255)
pts = deque(maxlen=buff)


#颜色抑制阈值，加一层颜色阈值提高图像分割去除噪声的能力
red_lower = 0
green_lower = 50
bule_lower = 60
red_higher = 255
green_higher = 150
bule_higher = 150
color = [([red_lower, green_lower, bule_lower], [red_higher, green_higher, bule_higher])]

red_lower_d = 0
green_lower_d = 50
bule_lower_d = 60
red_higher_d = 255
green_higher_d = 150
bule_higher_d = 150
color_d = [([red_lower_d, green_lower_d, bule_lower_d], [red_higher_d, green_higher_d, bule_higher_d])]

# AUV标志位
#Trunum信任度标志位
#Tarnum目标标志位
Trunum = None
Tarnum = 2

#进入抓球标志位
ball_flag = False

#随机运动计数位
turn_count = 0

#撞球标志位
rush_ball_flag = False

#串口通信接口
portx = 'COM3'
bps = 9600
timex = 0.01
ser = serial.Serial(portx, bps, timeout=timex)

# portx1 = '/dev/ttyUSB2'
# bps1 = 115200
# ser1 = serial.Serial(portx1, bps1, timeout=timex)


#分水岭图像分割函数
# 用于二值化图像，分割出所需要的内容
def get_fg_from_hue_watershed_saturation(img, margin):
    mask, hue = get_fg_from_hue(img, margin)

    mask_bg = cv2.inRange(hue, 60, 90)
    mask_bg = cv2.bitwise_or(mask_bg, cv2.inRange(hue, 128, 200))

    markers = np.zeros(mask.shape, np.int32)
    markers[mask == 255] = 1
    markers[mask_bg == 255] = 2

    cv2.watershed(img, markers)
    mask[markers == 1] = 255

    # img2 = img.copy()
    # img2[markers == 1] = 255
    # cv.imshow("1", img2)
    #
    # img2 = img.copy()
    # img2[markers == 2] = 255
    # cv.imshow("2", img2)
    #
    # img2 = img.copy()
    # img2[markers == -1] = 255
    # cv.imshow("3", img2)

    return mask

#HSV处理函数
def get_fg_from_hue(img, margin):
    FRACTION_AS_BLANK = 0.003
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dark = hsv[..., 2] < 32
    hsv[..., 0][dark] = 128

    dark = hsv[..., 1] < 50
    hsv[..., 0][dark] = 128

    mask = cv2.inRange(hsv[..., 0], np.array((0)), np.array((margin)))
    mask2 = cv2.inRange(hsv[..., 0], np.array((180 - margin)), np.array((180)))

    mask = cv2.bitwise_or(mask, mask2)

    if cv2.countNonZero(mask) < mask.shape[0] * mask.shape[1] * FRACTION_AS_BLANK:
        mask.fill(0)

    return [mask, hsv[..., 0]]

#引导线检测函数
def guide_line_detect(mask, area_th=5000, aspect_th=0.8):
    '''

    TODO：部分时候很靠近边框时，会检测到框
    :param img:
    :param area_th:
    :param aspect_th:
    :return:
    '''
    ASPECT_RATIO_MIN = 0.15  # 重要参数
    MAX_CONTOUR_NUM = 6  # 如果出现更多的轮廓，不进行处理。这是为了对抗白平衡

    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 目前对自动白平衡的处理，太多轮廓则直接返回
    candidates = []
    candidates_y = []
    if len(contours) < MAX_CONTOUR_NUM:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_th:  # 关键参数
                (x1, y1), (w1, h1), angle1 = cv2.minAreaRect(cnt)
                minAreaRect_area = w1 * h1
                aspect_ratio = float(w1) / h1
                if aspect_ratio > 1:
                    aspect_ratio = 1.0 / aspect_ratio
                    angle1 = np.mod(angle1 + 90, 180)

                extent = float(area) / minAreaRect_area

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area

                (x2, y2), (MA, ma), angle2 = cv2.fitEllipse(cnt)
                if angle2 > 90:
                    angle2 -= 180

                logging.debug('area %f,aspect_ratio %f,extent %f,solidity %f,angle1 %f,angle2 %f' % (
                area, aspect_ratio, extent, solidity, angle1, angle2))

                if aspect_ratio > aspect_th or aspect_ratio < ASPECT_RATIO_MIN or extent < 0.7 or solidity < 0.7 or abs(
                                angle1 - angle2) > 30:
                    break

                # img2 = img.copy()
                # contour_info(img2,area,aspect_ratio,extent,solidity,angle1,angle2,((x2, y2), (MA, ma), angle2))
                # cv.drawContours(img2, [cnt], 0, (0, 255, 0), 3)
                # show_img(img2)

                candidates.append((x1, y1, angle2))  # 目前这个组合是比较好的。
                candidates_y.append(y1)

        nc = len(candidates)
        if nc == 0:
            return None
        elif nc == 1:
            return candidates[0]
        else:
            logging.debug('multiple')

            idx = np.argmax(np.array(candidates_y))
            return candidates[idx]


#检测框距离函数
def distance_to_camera(width, forcal, perwidth):  # 距离计算
    return ((width * forcal) * 0.3048) / (12 * perwidth)


#角度转弧度函数
def angle2rad(theta):
    w = (theta * np.pi) / 180
    return w


#AUV控制，通信协议（2019版）
# def control_AUV(od_r,dis=1):
#     global dl
#     global dr
#     global AUV_dx
#     global AUV_dy
#     head = [0xaa,0x55,0x10]
#     depth_lock_bit = [0x01]
#     dir_lock_bit = [0x01]
#     Left_control_bit = [0x80]
#     Right_control_bit = [0x80]
#     depth_motion_bit = [0x00]
#     dir_motion_bit = [0x00]
#     power_value = [0x00]
#     other_bit = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
#     start_stop_bit = [0x00]
#
#     if od_r == 'left':
#         dir_lock_bit[0] = 0x02
#         Left_control_bit[0] = 0x80
#         Right_control_bit[0] = 0xb2
#         start_stop_bit[0] = 0x01
#         dl = -0.1                              #右旋浆前进0.1m
#         dr = 0.1
#         print('left')
#     if od_r == 'right':
#         dir_lock_bit[0] = 0x02
#         Right_control_bit[0] = 0x80
#         Left_control_bit[0] = 0xb2
#         start_stop_bit[0] = 0x01
#         dl = 0.1                              #左旋浆前进0.1m
#         dr = -0.1
#         print('right')
#     if od_r == 'left_translation':                               # 左平移
#         dir_lock_bit[0] = 0x02
#         Right_control_bit[0] = 0x80
#         Left_control_bit[0] = 0xb2
#         start_stop_bit[0] = 0x01                              #参数待修改
#         AUV_dx = -0.2
#         print('left_translation')
#     if od_r == 'right_translation':                              #右平移
#         dir_lock_bit[0] = 0x02
#         Right_control_bit[0] = 0x80
#         Left_control_bit[0] = 0xb2
#         start_stop_bit[0] = 0x01                              #参数待修改
#         AUV_dx = 0.2
#         print('right_translation')
#     if od_r == 'left_rotation':                              #左旋转
#         dir_lock_bit[0] = 0x02
#         start_stop_bit[0] = 0x01                              #参数待修改
#         print('left_rotation')
#     if od_r == 'right_rotation':                              #右旋转
#         dir_lock_bit[0] = 0x02
#         start_stop_bit[0] = 0x01                              #参数待修改
#         print('right_rotation')
#     if od_r == 'go':
#         dir_lock_bit[0] = 0x02
#         Left_control_bit[0] = 0xb2
#         Right_control_bit[0] = 0xb2
#         start_stop_bit[0] = 0x01
#         dl = 0.2                            #前进0.2m
#         dr = 0.2
#         print('go')
#     if od_r == 'up':
#         depth_lock_bit[0] = 0x02
#         depth_motion_bit[0] = 0x01
#         start_stop_bit[0] = 0x01
#         print('up')
#     if od_r == 'down':
#         depth_lock_bit[0] = 0x02
#         depth_motion_bit[0] = 0x02
#         start_stop_bit[0] = 0x01
#         print('down')
#     if od_r == 'stop':
#         Left_control_bit[0] = 0x80
#         Right_control_bit[0] = 0x80
#         start_stop_bit[0] = 0x02
#         dl = 0
#         dr = 0
#         print('stop')
#     if od_r == 'back':
#         Left_control_bit[0] = 0x4e
#         Right_control_bit[0] = 0x4e
#         start_stop_bit[0] = 0x01
#         print('back')
#
#     parameter = head + depth_lock_bit + dir_lock_bit + Left_control_bit + Right_control_bit + depth_motion_bit\
#                 + dir_motion_bit + power_value + other_bit + start_stop_bit
#     check_sum = sum(parameter)
#     check_sum = [check_sum & 255]
#
#     msg = head + parameter + check_sum
#     msg = bytearray(msg)
#     try:                                          #发送串口指令 与单片机通信
#         ser.write(msg)
#     except Exception as e:
#         print("--异常--:", e)
#
#     return dl,dr,AUV_dx,AUV_dy


#通信协议（2018版）
def PID_controlAUV(od_r,output):
    global model,AUV_dx,AUV_dy,AUV_dtheta,dl,dr
    if output > 32:
        output = 32
    print(output)
    head_bit = [0xaa,0x55]  # 两个字节为包头
    length_bit = [0x03]      #数据长度
    follow_bit = [0x08]      #用来选择三种模式
    control_bit = [0x00]  # 控制字节有效值：0-255
    time_level_bit = [0x00]  # 高四位为推进器动作时间，低四位为推进器推力的级数

    print(od_r)

    if od_r=='ball_down':
        if output > 32:
            output = 32
        follow_bit = [0x08]
        control_bit = [1+output]
        time_level_bit = [0x33]

    if od_r=='ball_up':
        if output > 32:
            output = 32
        follow_bit = [0x08]
        control_bit = [222+output]
        if control_bit[0]>=255:
            control_bit = [255]
        time_level_bit = [0x33]

    if od_r == 'left_translation':                      #左平移
        if output > 54:
            output = 54
        follow_bit = [0x08]
        control_bit = [35+output]
        time_level_bit = [0x34]
        model = 'trans'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = -0.01*output/30, 0, 0, 0, 0       #提供给里程表参数待修改
    elif od_r == 'right_translation':                    #右平移
        if output > 46:
            output = 46
        follow_bit = [0x08]
        control_bit = [128+output]
        time_level_bit = [0x34]
        model = 'trans'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0.01 * output/30, 0, 0, 0, 0       #提供给里程表参数待修改

    if od_r == 'left':                 #左旋转
        if output > 34:
            output = 34
        follow_bit = [0x08]
        control_bit = [91+output]
        time_level_bit = [0x34]
        model = 'dtheta'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0.026, 0, 0      # 提供给里程表参数待修改
    if od_r == 'right':                 #右旋转
        if output > 44:
            output = 44
        follow_bit = [0x08]
        control_bit = [176+output]
        time_level_bit = [0x34]
        model = 'dtheta'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, -0.026, 0, 0      # 提供给里程表参数待修改

    if od_r == 'go':
        follow_bit = [0x08]
        control_bit = [0xfe]
        time_level_bit = [0x84]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0.2/30, 0.2/30               # 提供给里程表参数待修改

    if od_r == 'down':
        follow_bit = [0x0c]
        control_bit = [0x40]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改

    if od_r == 'up':
        follow_bit = [0x0c]
        control_bit = [0x00]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改

    if od_r == 'UP':
        follow_bit = [0x0c]
        control_bit = [0x20]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改
    if od_r == 'DOWN':
        follow_bit = [0x0c]
        control_bit = [0x30]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改

    parameter = head_bit + length_bit + follow_bit + control_bit + time_level_bit
    msg = parameter
    msg = bytearray(msg)

    try:  # 发送串口指令 与单片机通信
        ser.write(msg)
    except Exception as e:
        print("--异常--:", e)

    return model,AUV_dx,AUV_dy,AUV_dtheta,dl,dr



#识别到引导线时的转向决策
def guide_line_turn(data):
    x = data[0]
    y = data[1]
    angle = data[2]
    if x < 240 and COUNT('count5'):
        output = pid_ballx(x)
        output = int(abs(output))
        PID_controlAUV('left_translation',output)
    if x > 400 and COUNT('count5'):
        output = pid_ballx(x)
        output = int(abs(output))
        PID_controlAUV('right_translation',output)
    elif y<150 and COUNT('count5'):
        output = pid_ballx(y)
        output = int(abs(output))
        PID_controlAUV('go', output)
    if x >= 240 and x <= 400:
        if angle < guide_line_enable_lower and abs(angle)<0.8 and COUNT('count5'):
            output = pid_lineturn(angle)
            output = int(abs(output))
            PID_controlAUV('left',output)
        elif angle > guide_line_enable_higher and abs(angle)<0.8 and COUNT('count5'):
            output = pid_lineturn(angle)
            output = int(abs(output))
            PID_controlAUV('right',output)
        elif (guide_line_enable_lower < angle) and angle < guide_line_enable_higher and COUNT('count5'):
            output = pid_lineturn(angle)
            output = int(abs(output))
            PID_controlAUV('go',output)


#是否要往Rect_point添加参数
#若当前过框坐标已经记录，则无需重复记录
def add_judege(X, Y):
    now_point = [X, Y]
    itertime = len(Rect_point)
    if itertime < 1:
        return True
    for i in range(itertime):
        dis = np.sqrt(np.sum(np.square(Rect_point[i] - now_point)))
        if dis < 1:
            return False
    return True

#向Rect_point添加框坐标
def Add_Rect_point(auv_local, metre, yaw, cX, Rect_width):
    bias = (0.6 * (cX - 160)) / Rect_width
    X = auv_local[0] + metre * np.cos(yaw) + bias
    Y = auv_local[1] + metre * np.sin(yaw)
    flag = add_judege(X, Y)
    now_point = [X, Y]
    if flag:
        Rect_point.append(now_point)

#向Line_point添加线坐标
def Add_Line_point(auv_local):
    X = auv_local[-1][0]
    Y = auv_local[-1][1]
    flag = add_judege(X, Y)
    now_point = [X, Y]
    if flag:
        Line_point.append(now_point)



#图像预处理，将RGB图像转换成二值图像
def Frame_Preprocess(frame0,frame1):
    # for (lower, upper) in color:
    #     lower = np.array(lower, dtype="uint8")
    #     upper = np.array(upper, dtype="uint8")
    #     mask0 = cv2.inRange(frame0, lower, upper)
    #     mask0 = cv2.bitwise_not(mask0)
    #     output0 = cv2.bitwise_and(frame0, frame0, mask=mask0)
    #cv2.imshow("test0" , frame0)
    thresh0 = get_fg_from_hue_watershed_saturation(frame0, 20)
    thresh0 = cv2.medianBlur(thresh0, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学开运算，简单滤除离框较远的干扰
    thresh0 = cv2.morphologyEx(thresh0, cv2.MORPH_OPEN, kernel)

    # for (lower, upper) in color_d:
    #     lower = np.array(lower, dtype="uint8")
    #     upper = np.array(upper, dtype="uint8")
    #     mask1 = cv2.inRange(frame1, lower, upper)
    #     mask1 = cv2.bitwise_not(mask1)
    #     output0 = cv2.bitwise_and(frame1, frame1, mask=mask1)
    cv2.boxFilter(frame1, -1, (5, 5), frame1)
    thresh1 = get_fg_from_hue_watershed_saturation(frame1, 20)

    return thresh0,thresh1


#目标识别，用于识别框线
#thresh0:前置摄像头二值化图像 frame0:前置摄像头图像
#Rect_Tarnum:是否识别到了框  data:框中点(cX,cY),外接矩形数据，框距离吗，外接矩形四点坐标
def Rect_Target_recognition(thresh0,frame0):
    global cX
    Rect_Tarnum = False                #未识别
    data = None
    cnts2 = []
    _, cnts3, hierarchy = cv2.findContours(thresh0.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # findContours寻找轮廓
    for cnt in cnts3:
        area = cv2.contourArea(cnt)
        if area > cnts2_area:
            cnts2.append(cnt)
    if not (cnts2 == []):
        for c_3 in cnts2:
            M = cv2.moments(c_3)  # 求图形的矩
            cX = int((M["m10"] + 1) / (M["m00"] + 1))
            cY = int((M["m01"] + 1) / (M["m00"] + 1))

    if not (cnts2 == []):
        c = max(cnts2, key=cv2.contourArea)
        marker = cv2.minAreaRect(c)  # 得到最小外接矩形（中心(x,y),（宽，高），选住角度）
        metre = distance_to_camera(Know_Width, FORCAL, marker[1][0] + 1)  # 距离摄像头距离
        box = cv2.boxPoints(marker)  # 获取最小外接矩形的四个顶点
        box = np.int0(box)
        cv2.drawContours(frame0, [box], -1, (0, 255, 0), 2)
        cv2.putText(frame0, "%.2fm" % (metre), (frame0.shape[1] - 200, frame0.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        Rect_Tarnum = True
        data = [cX,cY,marker, metre, box]
        return Rect_Tarnum, data, frame0
    return Rect_Tarnum,data,frame0



#识别线目标
#thresh1:下置摄像头二值化图像，frame1:下置摄像头图像
#Line_Tarnum:是否识别到了线，data:线上的两个坐标(x1,y1),线与AUV夹角angle,先的四点坐标
def Line_Target_recognition(thresh,frame):
    data = None
    guide_line = guide_line_detect(thresh)  # 检测下置摄像头是否读到引导线
    Line_Tarnum = False
    if guide_line:
        # 发现引导线，停一下
        x, y, angle = guide_line
        angle = angle / 180 * np.pi
        cv2.line(frame, (int(x), int(y)), (int(x + 100 * np.sin(angle)), int(y - 100 * np.cos(angle))),
                 (0, 255, 0), 2)
        x1 = int(x + 100 * np.sin(angle))
        y1 = int(y - 100 * np.cos(angle))
        Line_Tarnum = True
        data = [x1, y1, angle]
        return Line_Tarnum, data, frame
    return Line_Tarnum, data, frame



#框信任度判断
def Rect_Trust(data):
    if data is not None:
        cX = data[0]
        marker = data[1]
        minArea_x = marker
        cX_container.append(cX)
        minAreax_container.append(marker)
        if cX - minArea_x > 50:
            return 0
        if len(cX_container) >= 5 and len(minAreax_container) >= 5:
            var_cX = np.var(cX_container)
            var_min = np.var(minAreax_container)
            # cv2.putText(frame, "%.2f" % (var_cX), (frame.shape[1] - 200, frame.shape[0] - 20),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             2.0, (0, 255, 0), 3)
            if var_cX < var_maxvalue and var_min < var_maxvalue:  # 方差，待测量
                return 1
            else:
                return 0
    return 2

# 无目标信任度
def Estate(flag):
    global Count
    global C_L_COUNT
    global Est
    global Tarnum
    Count = Count + 1
    if flag == 0:  # 不信任框
        Est = 1
    if flag == 2:  # 无框 无线
        Est = 2
    if flag == 1:  # 有线
        Est = 3

    if Est == 2:
        C_L_COUNT = C_L_COUNT + 1  # 无框则加一
    if Est == 1:
        C_L_COUNT = C_L_COUNT + 1

    if Count == 10:
        if C_L_COUNT >= 10:
            Tarnum = 2
            return True
        Count = 0
        C_L_COUNT = 0  # 10次后清零

    elif Count < 10:
        return 0



#里程表记录函数             目前与运动效果不匹配
def Odometer(model,AUV_dx,AUV_dy,AUV_dtheta,dl,dr):                    #里程表
    global x0
    global y0
    global theta0
    if model=='diff':                               #差速模式
        x = x0[0]
        y = y0[0]
        theta = theta0[0]
        ddx = (dr+dl)*np.cos(theta+(dr-dl)/2*b)/2
        ddy = (dr+dl)*np.sin(theta+(dr-dl)/2*b)/2
        dtheta = (dr-dl)/b
        x1 = x + ddx
        y1 = y + ddy
        theta1 = theta + dtheta
        x0.appendleft(x1)
        y0.appendleft(y1)
        theta0.appendleft(theta1)
    elif model =='dtheta':
        x1 = x0[0]
        y1 = y0[0]
        theta = theta0[0]
        dtheta = AUV_dtheta
        theta1 = theta + dtheta
        x0.appendleft(x1)
        y0.appendleft(y1)
        theta0.appendleft(theta1)
    else:                                           #平移模式
        x = x0[0]
        y = y0[0]
        theta = theta0[0]
        dx = AUV_dx * np.cos(theta)
        dy = AUV_dy * np.sin(theta)
        x1 = x + dx
        y1 = y + dy
        x0.appendleft(x1)
        y0.appendleft(y1)
        theta0.appendleft(theta)
    return 0

#无目标情况下的转向函数
def Turn():
    global turn_count
    turn_count = turn_count + 1
    if turn_count == 1:
        PID_controlAUV('right',23)
    if turn_count == 2:
        PID_controlAUV('left',34)
        turn_count = 0



#无目标情况下的位置检测      暂时用不上
def search(x, y, theta):
    global x1
    global x2
    global y1
    global y2
    global SR
    global SL
    k = np.tan(theta)
    b1 = y - k * x

    Wide = 3.66
    Long = 7.3

    if 0.9 <= np.cos(theta) <= 1:  # k=0的情况
        if 0 <= b1 < Long * 0.5:
            return 'left'
        if Long * 0.5 <= b1 <= Long:
            return 'right'
    elif -1 <= np.cos(theta) <=-0.9:
        if 0 <= b1 < Long * 0.5:
            return 'right'
        if Long * 0.5 <= b1 <= Long:
            return 'left'

    elif 0 <= x <= 0.1  and 0 <= y <= 0.1 and 0 <= theta <= 0.1:
        return 'left'

    else:
        if 0 <= b1 <= Long:  # 左边交点Y轴上（三种情况）
            x1 = 0
            y1 = b1
            if 0 <= (Long - b1) / k <= Wide:
                x2 = (Long - b1) / k
                y2 = Long
                if np.cos(theta) > 0:  # 船头方向朝右
                    SL = x1 * y2 + (x2 - x1) * (y2 - y1) / 2
                    SR = Long * Wide - SL
                if np.cos(theta) <= 0:  # 船头方向朝左
                    SR = x1 * y2 + (x2 - x1) * (y2 - y1) / 2
                    SL = Long * Wide - SR
            elif 0 <= (Wide * k + b1) <= Long:
                x2 = Wide
                y2 = (Wide * k + b1)
                if np.cos(theta) > 0:  # 船头方向朝右
                    SR = x1 * y1 + x2 * y2 - x1 * y2 + (x2 - x1) * (y1 - y2) / 2
                    SL = Long * Wide - SR
                elif np.cos(theta) <= 0:  # 船头方向朝左
                    SL = x1 * y1 + x2 * y2 - x1 * y2 + (x2 - x1) * (y1 - y2) / 2
                    SR = Long * Wide - SL
            elif 0 <= (-b1 / k) <= Wide:
                x2 = -b1 / k
                y2 = 0
                if np.cos(theta) > 0:  # 船头方向朝右
                    SR = x2 * y1
                    SL = Long * Wide - SR
                elif np.cos(theta) <= 0:  # 船头方向朝左
                    SL = x2 * y1
                    SR = Long * Wide - SL
        elif 0 <= (Wide * k + b1) <= Long:  # 右边交点在x=3.66上（三减一 种情况）
            x2 = Wide
            y2 = (Wide * k + b1)
            if 0 <= (Long - b1) / k <= Wide:
                x1 = (Long - b1) / k
                y1 = Long
                if np.cos(theta) > 0:  # 船头方向朝右
                    SR = x1 * y1 + x2 * y2 - x1 * y2 + (x2 - x1) * (y1 - y2) / 2
                    SL = Long * Wide - SR
                elif np.cos(theta) <= 0:  # 船头方向朝左
                    SL = x1 * y1 + x2 * y2 - x1 * y2 + (x2 - x1) * (y1 - y2) / 2
                    SR = Long * Wide - SL
            elif 0 <= -b1 / k <= Wide:
                x1 = -b1 / k
                y1 = 0
                if np.cos(theta) > 0:  # 船头方向朝右
                    SR = (x2 - x1) * y2 * 0.5
                    SL = Long * Wide - SR
                elif np.cos(theta) <= 0:  # 船头方向朝左
                    SL = (x2 - x1) * y2 * 0.5
                    SR = Long * Wide - SL

        elif 0 <= (Long - b1) / k <= Wide and 0 <= (-b1 / k) <= Wide and k >= 0:  # 两个交点在y=0和y=13上
            x1 = -b1 / k
            y1 = 0
            x2 = (Long - b1) / k
            y2 = Long
            if np.sin(theta) >= 0:  # 船头方向朝上
                SL = (x1 + x2) * Long * 0.5
                SR = Long * Wide - SL
            elif np.sin(theta) <= 0:  # 船头方向朝下
                SR = (x1 + x2) * Long * 0.5
                SL = Long * Wide - SR
        elif 0 <= (Long - b1) / k <= Wide and 0 <= -b1 / k <= Wide and k < 0:
            x1 = (Long - b1) / k
            y1 = Long
            x2 = -b1 / k
            y2 = 0
            if np.sin(theta) >= 0:  # 船头方向朝上
                SL = (x1 + x2) * Long * 0.5
                SR = Long * Wide - SL
            elif np.sin(theta) <= 0:  # 船头方向朝下
                SR = (x1 + x2) * Long * 0.5
                SL = Long * Wide - SR

        if SL > SR:
            return 'left'
        if SL < SR:
            return 'right'

#过框模式，切换识别模式。将放行更多的噪声，以此来增加识别能力
def cross_Rect(frame0):
    global corss_Rect_flag
    global cYl
    global cYr
    global cXl
    global cXr

    # PID_controlAUV('go', 20)
    # Odometer(model, AUV_dx, AUV_dy, AUV_dtheta, dl, dr)
    # print('直走')  # 过框模式保持直走，再做差速调整

    corss_Rect_flag = 1

    cntsl = []
    cntsr = []
    framel = frame0[:, 0:319]
    framer = frame0[:, 320:640]
    thresh = get_fg_from_hue_watershed_saturation(frame0, 20)
    thresh = cv2.medianBlur(thresh, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 形态学开运算，简单滤除离框较远的干扰
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    left_area = thresh[:, 0:319]
    right_area = thresh[:, 320:640]
    _, cntsl0, hierarchy = cv2.findContours(left_area, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    _, cntsr0, hierarchy = cv2.findContours(right_area, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cntsl0:
        area = cv2.contourArea(cnt)
        if area > cntsr0_area:
            cntsl.append(cnt)

    for cnt in cntsr0:
        area = cv2.contourArea(cnt)
        if area > cntsl0_area:
            cntsr.append(cnt)

    if not (cntsl == []):
        for c_l in cntsl:
            Ml = cv2.moments(c_l)  # 求图形的矩
            cXl = int((Ml["m10"] + 1) / (Ml["m00"] + 1))
            cYl = int((Ml["m01"] + 1) / (Ml["m00"] + 1))
            cv2.circle(framel, (cXl, cYl), 7, (0, 255, 255), -1)
            cv2.drawContours(framel, [c_l], -1, (0, 255, 0), 2)
    if not (cntsr == []):
        for c_r in cntsr:
            Mr = cv2.moments(c_r)  # 求图形的矩
            cXr = int((Mr["m10"] + 1) / (Mr["m00"] + 1))
            cYr = int((Mr["m01"] + 1) / (Mr["m00"] + 1))
            cv2.circle(framer, (cXr, cYr), 7, (255, 255, 255), -1)
            cv2.drawContours(framer, [c_r], -1, (0, 255, 0), 2)

    if cntsl == [] and not (cntsr == []) and COUNT('count5'):
        PID_controlAUV('left',20)
        print('向左转1')
    elif not (cntsl == []) and cntsr == [] and COUNT('count5'):
        PID_controlAUV('right', 20)
        print('向右转1')
    elif not (cntsl == []) and not (cntsr == []):
        if abs(cYl - cYr) < 40:
            Y_flag = True
        else:
            Y_flag = False

        if Y_flag:
            if (cXl + cXr + 320) / 2 > 390 and COUNT('count5'):
                PID_controlAUV('left',20)
                print('向左转2')
            if (cXl + cXr + 320) / 2 < 250 and COUNT('count5'):
                PID_controlAUV('right',20)
                print('向右转2')
            if (cXl + cXr + 320) / 2 < 390 and (cXl + cXr + 320) / 2 > 250 and COUNT('count5'):
                PID_controlAUV('go', 20)
                print('直走1')
        else:
            if cYl > cYr and COUNT('count5'):
                PID_controlAUV('right', 20)
                print('向右转3')
            elif cYl < cYr and COUNT('count5'):
                PID_controlAUV('left', 20)
                print('向左转3')

    elif cntsl == [] and cntsr == []:
        for i in range(3):
            PID_controlAUV('go', 20)
        print('直走2')



#标志位计数函数
def flag_count():
    global data_count

    data_count = data_count + 1
    if data_count >= 5:
        data_count = 0


#AUV球识别函数
def Auv_ball_detect(frame0):
    findball_flag = False
    blurred = cv2.GaussianBlur(frame0, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ballLower, ballUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame0, (int(x), int(y)), int(radius),
                       (0, 255, 0), 2)
            cv2.circle(frame0, center, 5, (255, 255, 255), -1)
        findball_flag = True

    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(buff / float(i + 1)) * 2.5)
        cv2.line(frame0, pts[i - 1], pts[i], (255, 0, 0), thickness)



    return frame0,findball_flag,center


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

#框追踪类，用于追踪已经识别到的框
class CentroidTracker():
    def __init__(self, maxDisappeared=300):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX[0] + endX[0]) / 2.0)
            cY = int((startY[1] + endY[1]) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

#框最终函数
ct = CentroidTracker()
def Target_Tracking(frame, box):
    rects = []
    rects.append(box.astype("int"))
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return frame, objectID, centroid


#PID控制类
class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time




#工控机摄像头读取函数
#      无
#frame0:前置摄像头画面  frame1:下置摄像头画面
def Camera_Capture():
    ret, frame0 = cap1.read()
    ret, frame1 = cap.read()
    return frame0,frame1

#改变颜色阈值函数
#    无
#    无（纯全局变量操作）
def change_colorvalue():
    global red_lower
    global green_lower
    global bule_lower
    global red_higher
    global green_higher
    global bule_higher
    if SEARCH_count >= SEARCH_count_max:
        green_lower = green_lower - 5
        bule_lower = bule_lower - 5
        red_higher = red_higher + 5
        green_higher = green_higher + 5
        bule_higher = bule_higher + 5


#颜色阈值初始化
def colorvalue_back():
    global red_lower
    global green_lower
    global bule_lower
    global red_higher
    global green_higher
    global bule_higher
    red_lower = 0
    green_lower = 50
    bule_lower = 60
    red_higher = 255
    green_higher = 150
    bule_higher = 150


#用于退出整个程序
#  无
#退出标志位
def Break_flag():
    break_flag = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break_flag = True
    return break_flag

#用于控制计数。识别要快，运动要慢、准
#计数器名称
#     无
def COUNT(name):
    global order_count1
    global order_count2
    global order_count3
    global order_count4
    global order_count5
    if name=='count1':
        enable_flag = False
        order_count1 = order_count1 + 1
        if order_count1>=count_max1:
            order_count1 = 0
            enable_flag = True
        return enable_flag
    if name=='count2':
        enable_flag = False
        order_count2 = order_count2 + 1
        if order_count2 >= count_max2:
            order_count2 = 0
            enable_flag = True
        return enable_flag
    if name=='count3':
        enable_flag = False
        order_count3 = order_count3 + 1
        if order_count3 >= count_max3:
            order_count3 = 0
            enable_flag = True
        return enable_flag
    if name=='count4':
        enable_flag = False
        order_count4 = order_count4 + 1
        if order_count4 >= count_max4:
            order_count4 = 0
            enable_flag = True
        return enable_flag
    if name=='count5':
        enable_flag = False
        order_count5 = order_count5 + 1
        if order_count5 >= count_max5:
            order_count5 = 0
            enable_flag = True
        return enable_flag





#寻找模式
#frame0:前置摄像头画面 frame1:下置摄像头画面
#SEARCH_enable:Search放行标志位
def SEARCH_MODEL(frame0,frame1):
    global SEARCH_count
    global SEARCH_count_time
    global SEARCH_count_time_max
    SEARCH_enable = False
    thresh0,thresh1 = Frame_Preprocess(frame0,frame1)
    Rect_Tarnum, data0, frame0 = Rect_Target_recognition(thresh0,frame0)
    Line_Tarnum, data1, frame1 = Line_Target_recognition(thresh1, frame1)

    SEARCH_count  = SEARCH_count + 1

    # if COUNT('count1')and (not(Rect_Tarnum) and not(Line_Tarnum)):
    #     _, cnts3, hierarchy = cv2.findContours(thresh0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # findContours寻找轮廓
    #     if not (cnts3 == []):
    #         for c_3 in cnts3:
    #             M = cv2.moments(c_3)  # 求图形的矩
    #             cX = int((M["m10"] + 1) / (M["m00"] + 1))
    #             cY = int((M["m01"] + 1) / (M["m00"] + 1))
    #             cv2.drawContours(frame0, [c_3], -1, (0, 255, 0), 2)
    #
    #             if cX < 320:
    #                 output = pid_rectx(cX)
    #                 output = int(abs(output))
    #                 PID_controlAUV('left_translation', output)
    #             else:
    #                 output = pid_rectx(cX)
    #                 output = int(abs(output))
    #                 PID_controlAUV('right_translation', output)

    if COUNT('count1')and (not(Rect_Tarnum) or not(Line_Tarnum)):
        Turn()

    if SEARCH_count >= SEARCH_count_max:
        #change_colorvalue()
        SEARCH_count = 0
        SEARCH_count_time = SEARCH_count_time + 1
    if SEARCH_count_time >= SEARCH_count_time_max:
        SEARCH_count_time = 0
        PID_controlAUV('ball_up',30)
    if Rect_Tarnum or Line_Tarnum:
        SEARCH_count = 0                       #找到目标后计数清零
        SEARCH_count_time = 0
        #colorvalue_back()
        SEARCH_enable = True
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)
    break_flag = Break_flag()

    return SEARCH_enable, break_flag


def ADJUST_MODEL(frame0,frame1):
    global ADJUST_count
    global ADJUST_count_max
    global ADJUST_count_time
    global ADJUST_count_time_max
    global MODEL_section_flag
    Rect_Aim_flag = False
    Line_Aim_flag = False
    ADJUST_enable = False
    thresh0, thresh1 = Frame_Preprocess(frame0, frame1)

    Rect_Tarnum, data0, frame0 = Rect_Target_recognition(thresh0, frame0)
    Line_Tarnum, data1, frame1 = Line_Target_recognition(thresh1, frame1)

    if not(data0 is None):
        marker = data0[2]
        if (marker[1][0] / marker[1][1]) > 0.7:
            box = data0[4]
            frame0,objectID,centroid = Target_Tracking(frame0, box)
            if centroid[0]<left_min and COUNT('count1'):
                output = pid_rectx(centroid[0])
                output = int(abs(output))
                PID_controlAUV('left_translation',output)
                print('左平移')
            if centroid[0]>right_max and COUNT('count1'):
                output = pid_rectx(centroid[0])
                output = int(abs(output))
                PID_controlAUV('right_translation', output)
                print('右平移')

            # if centroid[1] < 190:
            #     PID_controlAUV('UP',20)
            # elif centroid[1] > 290:
            #     PID_controlAUV('down', 20)
            if centroid[0]>left_min and centroid[0]<right_max: #and 190 < centroid[1] < 290:
                Rect_Aim_flag = True
        Trustnum = Rect_Trust(data0)


    elif not(data1 is None):
        x = data1[0]
        angle = data1[2]
        guide_line_turn(data1)

        if guide_line_enable_lower < angle < guide_line_enable_higher and 280 <= x <= 360:
            Line_Aim_flag = True
        Trustnum = False

    if Line_Aim_flag or (Rect_Aim_flag and Trustnum):
        ADJUST_enable = True
        ADJUST_count = 0

    ADJUST_count = ADJUST_count + 1
    # if ADJUST_count >= ADJUST_count_max:
    #     print('减小调整力度')                              #待协商
    #     ADJUST_count_time = ADJUST_count_time + 1
    #     ADJUST_count = 0
    if ADJUST_count_time>=ADJUST_count_time_max:
        ADJUST_count_time = 0
        MODEL_section_flag = 'CROSS'                               #如果一直都调不好，就直接过框

    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)
    break_flag = Break_flag()
    return ADJUST_enable,break_flag


def CROSS_MODEL(frame0,frame1):
    global line_count
    global CROSS_count
    global CROSS_count_max
    CROSS_count_flag = False
    cross_Rect_flag = False
    PID_controlAUV('go',25)
    cross_Rect(frame0)

    thresh0, thresh1 = Frame_Preprocess(frame0, frame1)
    # edges = cv2.Canny(thresh1, 40, 200, apertureSize=3)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)
    # if not(lines is None):
    #     lines1 = lines[:, 0, :]                                     # 提取为二维
    #     for rho, theta in lines1[:]:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * a)
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * a)
    #         cv2.line(frame1, (x1, y1), (x2, y2), (255, 0, 0), 1)
    #         if -0.4 < theta < 0.4:
    #             line_count = line_count + 1
    #         if line_count >= 3:
    #             line_count = 0
    #             cross_Rect_flag = True

    CROSS_count = CROSS_count + 1

    if CROSS_count >= CROSS_count_max: #and line_count < 5:           #如果一直没有过框，就跳出
        PID_controlAUV('go',25)
        CROSS_count = 0
        CROSS_count_flag = True
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)
    break_flag = Break_flag()

    return cross_Rect_flag,CROSS_count_flag,break_flag


#需求模式，采用PID跟踪
#输入下置摄像头画面
def SEARCHBALL_MODEL(frame1):
    frame1,findball_flag,center = Auv_ball_detect(frame1)
    if findball_flag:
        cX = center[0]
        cY = center[1]
        if center[0]<280:
            output = pid_ballx(cX)
            output = int(abs(output))
            PID_controlAUV('left_translation',output)
        elif center[0]>360:
            output = pid_ballx(cX)
            output = int(abs(output))
            PID_controlAUV('right_translation',output)
        if center[1]<200:
            output = pid_bally(cY)
            output = int(abs(output))
            PID_controlAUV('ball_up',output)
        elif center[1]>280:
            output = pid_bally(cY)
            output = int(abs(output))
            PID_controlAUV('ball_down',output)
    else:
        dx = Reference_coor[0] - x0[0]
        dy = Reference_coor[1] - y0[0]

#多线程 设定时间进入另一个线程
def Rush_ball_time():
    rush_ball_time = 0.1  # 进入寻球状态的时间，单位是分钟
    timer = threading.Timer(60 * rush_ball_time, to_ball)  # 设置时钟。根据rush_ball_time变量以判断是否进入撞球部分
    timer.start()

#抓球函数
def Catch_ball(frame0,frame1):
    global rush_ball_flag
    frame, findball_flag, center = Auv_ball_detect(frame0)
    thresh0, thresh1 = Frame_Preprocess(frame0, frame1)
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)
    if findball_flag:
        cX = center[0]
        cY = center[1]
        if center[0] < 280 and COUNT('count1'):
            output = pid_ballx(cX)
            output = int(abs(output))
            PID_controlAUV('left_translation', output)
        elif center[0] > 360 and COUNT('count1'):
            output = pid_ballx(cX)
            output = int(abs(output))
            PID_controlAUV('right_translation', output)

        elif center[1] < 360 and COUNT('count1'):
            output = pid_bally(cY)
            output = int(abs(output))
            PID_controlAUV('ball_up',output)
        elif center[1] > 420 and COUNT('count1'):
            output = pid_bally(cY)
            output = int(abs(output))
            PID_controlAUV('ball_down',output)

        if 280 < center[0] < 360 and 360 < center[1] < 420 and COUNT('count5'):
            for i in range(3):
                PID_controlAUV('MIXDOWN', 20)
            time.sleep(10)
            for i in range(3):
                PID_controlAUV('UP',10)
            time.sleep(15)
            rush_ball_flag = True
    else:
        Turn()

    return rush_ball_flag

#撞球函数
def Rush_ball(frame0,frame1):
    frame, findball_flag, center = Auv_ball_detect(frame0)
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1', frame1)
    if findball_flag:
        cX = center[0]
        if center[0] < 280:
            output = pid_ballx(cX)
            output = int(abs(output))
            PID_controlAUV('left_translation', output)
        elif center[0] > 360:
            output = pid_ballx(cX)
            output = int(abs(output))
            PID_controlAUV('right_translation', output)

        if center[1] < 200:
            PID_controlAUV('UP', 10)
        elif center[1] > 280:
            PID_controlAUV('DOWN', 10)

        if 280 < center[0] < 360 and 200 < center[1] < 280 and COUNT('count5'):
            PID_controlAUV('go', 30)
    else:
        if COUNT('count1'):
            PID_controlAUV('right', 25)

#开抓球标志位
def to_ball():
    #### 修改部分
    global ball_flag
    print('yes!')
    ball_flag = True



#从MCU读取陀螺仪姿态参数
# yaw:偏转角
def read_fromMCU():
    msg = ser.read(3)
    yaw = msg[2]
    return yaw

#跟踪球PID函数
#球所在位置的y坐标
#改变力度
def pid_ballx(feedback):
    pid0.update(feedback_value=feedback)
    output = pid0.output
    return output


#跟踪球PID函数
#球所在位置的y坐标
#改变力度
def pid_bally(feedback):
    pid1.update(feedback_value=feedback)
    output = pid1.output
    return output

def pid_lineturn(feedback):
    pid2.update(feedback_value=feedback)
    output = pid2.output
    return output

def pid_rectx(feedback):
    pid3.update(feedback_value=feedback)
    output = pid3.output
    return output

def pid_recty(feedback):
    pid4.update(feedback_value=feedback)
    output = pid4.output
    return output


# camera = PiCamera()
# camera.resolution = (640,480)
# camera.framerate = 32
# rawCapture = PiRGBArray(camera, size=(640,480))


cap = cv2.VideoCapture(0)  # 下置摄像头
cap.set(3, 640)  # 设置分辨率
cap.set(4, 480)

cap1 = cv2.VideoCapture(1)  # 前置摄像头
cap1.set(3, 640)  # 设置分辨率
cap1.set(4, 480)

#跟踪球PID实例化
pid0 = PID(P0,I0,D0)
pid0.SetPoint=320
pid0.setSampleTime(0.5)

#跟踪球PID实例化
pid1 = PID(P1,I1,D1)
pid1.SetPoint=240
pid1.setSampleTime(0.5)

#调整线PID实例化
pid2 = PID(P2,I2,D2)
pid2.SetPoint=0
pid2.setSampleTime(0.5)

#对框PIDx实例化
pid3 = PID(P3,I3,D3)
pid3.SetPoint=320
pid3.setSampleTime(0.5)

#对框PIDy实例化
pid4 = PID(P4,I4,D4)
pid4.SetPoint=240
pid4.setSampleTime(0.5)

#初始模式为“寻找模式”
MODEL_section_flag = 'SEARCH'

#AUV定深下沉准备
# time.sleep(20)
ser.write(b'\xaa\x55\x03\x0c\x35\x02')
time.sleep(5)

if __name__ == "__main__":
    rush_ball_time = 5
    timer = threading.Timer(60 * rush_ball_time, to_ball)  # 设置时钟。根据rush_ball_time变量以判断是否进入撞球部分
    timer.start()
    while True:
        ret1, frame1 = cap.read()               #画面读取
        ret0, frame0 = cap1.read()
        if ball_flag:
            Catch_ball(frame0, frame1)
            if rush_ball_flag:
                Rush_ball(frame0, frame1)
            break_flag = Break_flag()
        else:                                    #模式选择
            if MODEL_section_flag=='SEARCH':
                print('SEARCH')
                SEARCH_enable,break_flag = SEARCH_MODEL(frame0,frame1)
                if SEARCH_enable:
                    MODEL_section_flag = 'ADJUST'
            elif MODEL_section_flag=='ADJUST':
                print('ADJUST')
                ADJUST_enable,break_flag = ADJUST_MODEL(frame0,frame1)
                if ADJUST_enable:
                    MODEL_section_flag='CROSS'
            elif MODEL_section_flag=='CROSS':
                print('CROSS')
                corss_Rect_flag, CROSS_count_flag, break_flag = CROSS_MODEL(frame0, frame1)
                if corss_Rect_flag:
                    crossed_count = crossed_count + 1
                    MODEL_section_flag='SEARCH'
                if CROSS_count_flag:
                    MODEL_section_flag='SEARCH'

        Odometer(model,AUV_dx,AUV_dy,AUV_dtheta,dl,dr)
        break_flag = Break_flag()


        if break_flag:
            break


    cv2.destroyAllWindows()
    sys.exit()