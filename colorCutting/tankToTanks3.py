#coding:utf-8
import cv2 as cv
import numpy as np
import collections
import math
import ctypes,sys


STD_INPUT_HANDLE = -5
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12
#字体颜色定义 text colors
FOREGROUND_BLUE = 0x09 # blue.
FOREGROUND_GREEN = 0x0a # green.
FOREGROUND_RED = 0x0c # red.
FOREGROUND_YELLOW = 0x0e # yellow.
 
# 背景颜色定义 background colors
BACKGROUND_YELLOW = 0xe0 # yellow.

# get handle
std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
def set_cmd_text_color(color, handle=std_out_handle):
    Bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    return Bool
 
#reset:white
def resetColor():
    set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
 
#green
def printGreen(mess):
    set_cmd_text_color(FOREGROUND_GREEN)
    sys.stdout.write(mess + '\n')
    resetColor()

#red
def printRed(mess):
    set_cmd_text_color(FOREGROUND_RED)
    sys.stdout.write(mess + '\n')
    resetColor()
  
#yellow
def printYellow(mess):
    set_cmd_text_color(FOREGROUND_YELLOW)
    sys.stdout.write(mess + '\n')
    resetColor()

#white bkground and black text
def printYellowRed(mess):
    set_cmd_text_color(BACKGROUND_YELLOW | FOREGROUND_RED)
    sys.stdout.write(mess + '\n')
    resetColor()
#################################################################################


def l_get(a,c):
        #在c方向上对坐标进行投影变换，a为坐标
        #对于C方向直线方程：y = tan(c) * x
        c1 = int(c) * math.pi / 180
        k = math.tan(c1)
        a1 = (k * a[1] + a[0]) / (k ** 2 + 1)
        b1 = k * a1
        return (a1,b1)
        
#这个函数的意思是 对 nums从下标p一直到q的全排列。
def permutation(nums, p, q , tl5):
    if p == q:#这里使用 list(nums)是因为如果直接添加 添加的都是指向nums所存数组的地址 nums变化了 tl5里面的数组内容也会跟着变化。
        tl5.append(list(nums))
    else:
        for i in range(p, q):
            nums[i], nums[p] = nums[p], nums[i]
            permutation(nums, p+1, q,tl5)
            nums[i], nums[p] = nums[p], nums[i]
        
def line_matching(t,h):
    while len(t) != 1:
        for b in t:
            for j in range(0,num - 1):
                if b[j][h] >  b[j + 1][h]:
                    t.remove(b)
                    break
                else:
                    pass

#定义字典存放颜色分量上下限
#例如：{颜色: [min分量, max分量]}
#{'red': [array([160,  43,  46]), array([179, 255, 255])]}
 
def getColorList():
    dict = collections.defaultdict(list)
 
    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list
 
    # #灰色
    # lower_gray = np.array([0, 0, 46])
    # upper_gray = np.array([180, 43, 220])
    # color_list = []
    # color_list.append(lower_gray)
    # color_list.append(upper_gray)
    # dict['gray']=color_list
 
    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list
 
    #红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red']=color_list
 
    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([5, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list
 
    #橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list
 
    #黄色
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list
 
    #绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list
 
    #青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list
 
    #蓝色
    lower_blue = np.array([50, 15, 15])
    upper_blue = np.array([130, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list
 
    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list
 
    return dict


 
#处理图片
def get_color(frame):
    print('go in get_color')
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL)
    maxsum = -120
    color = None
    color_dict = getColorList()
    for d in color_dict:
        mask = cv.inRange(hsv,color_dict[d][0],color_dict[d][1])
        #cv.imwrite(d+'.jpg',mask)
        binary = cv.threshold(mask, 90, 255, cv.THRESH_BINARY)[1]
        binary = cv.dilate(binary,None,iterations=2)
        img, cnts, hiera = cv.findContours(binary.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum+=cv.contourArea(c)
        if sum > maxsum :
            maxsum = sum
            color = d
 
    return color


    
def separate_color(frame,color):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)                              #色彩空间转换为hsv，便于分离
    cv.imshow("f",hsv)
    lower_hsv = np.array(color[0])                                                  #提取颜色的低值
    high_hsv = np.array(color[1])                                                   #提取颜色的高值
    mask = cv.inRange(hsv, lowerb = lower_hsv, upperb = high_hsv)           #下面详细介绍
    cv.namedWindow("ColorFilter", cv.WINDOW_NORMAL)
    cv.imshow("ColorFilter", mask)
    

if __name__ == '__main__':
    ###logo
    logostr = """\033[1;32;40m \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n

 $$$$$$\  $$$$$$\         $$\   $$\                       $$\                         
$$  __$$\ \_$$  _|        $$ |  $$ |                      $$ |                        
$$ /  \__|  $$ |          $$ |  $$ |$$\   $$\ $$$$$$$\  $$$$$$\    $$$$$$\   $$$$$$\  
\$$$$$$\    $$ |  $$$$$$\ $$$$$$$$ |$$ |  $$ |$$  __$$\ \_$$  _|  $$  __$$\ $$  __$$\ 
 \____$$\   $$ |  \______|$$  __$$ |$$ |  $$ |$$ |  $$ |  $$ |    $$$$$$$$ |$$ |  \__|
$$\   $$ |  $$ |          $$ |  $$ |$$ |  $$ |$$ |  $$ |  $$ |$$\ $$   ____|$$ |      
\$$$$$$  |$$$$$$\         $$ |  $$ |\$$$$$$  |$$ |  $$ |  \$$$$  |\$$$$$$$\ $$ |      
 \______/ \______|        \__|  \__| \______/ \__|  \__|   \____/  \_______|\__|      
                                                                                      
                                                                                      
                                Author:wwy(安鸾网络安全渗透团队)       Version 1.0.0  
                """
    
    printRed(logostr.decode("utf-8").encode("gbk"))

    ###高斯模糊
    filename='D://PythonPicTemplate/colorCutting/tankTemplate1.jpg'
    targetname = "D://PythonPicTemplate/colorCutting/tanks1.jpg"
    kernel_size = (15, 15)
    sigma = 1.5
    imgC = cv.imread(filename)
    imgColor = cv.GaussianBlur(imgC, kernel_size, sigma)
    
    #提取模板颜色
    imghsv = cv.cvtColor(imgColor, cv.COLOR_BGR2HSV)                   #色彩空间转换为hsv，便于分离
    cv.imshow("f0",imghsv)
    
    content = "正在提取模板颜色HAV矩阵向量......"
    printGreen(content.decode("utf-8").encode("gbk"))
    
    
    color_dict = getColorList()
    print(get_color(imgColor))
    tplColor = color_dict[get_color(imghsv)]

    #目标颜色追踪
    target = cv.imread(targetname)    
    HSV = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    #H, S, V = cv.split(HSV)
    LowerBlue = np.array(tplColor[0])
    UpperBlue = np.array(tplColor[1])
    
    content0 = "正在对目标进行颜色分割处理......"
    printGreen(content0.decode("utf-8").encode("gbk"))
    
    
    mask = cv.inRange(HSV, LowerBlue, UpperBlue)
    targetCutting0 = cv.bitwise_and(target, target, mask=mask)
    cv.imwrite('targetCutting.jpg',targetCutting0)
    cv.namedWindow('targetCutting0', cv.WINDOW_NORMAL)
    cv.imshow('targetCutting0',targetCutting0)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ###初始化处理过的目标
    filename2='D://PythonPicTemplate/colorCutting/tankTemplate2.jpg'
    targetname2 = "D://PythonPicTemplate/colorCutting/targetCutting.jpg"
    imgC = cv.imread(filename2) 
    targetCutting0 = cv.imread(targetname2) 
    imgC = cv.GaussianBlur(imgC, kernel_size, sigma)
    
    HSV2 = cv.cvtColor(imgC, cv.COLOR_BGR2HSV)
    LowerBlue = np.array(tplColor[0])
    UpperBlue = np.array(tplColor[1])    
    mask2 = cv.inRange(HSV2, LowerBlue, UpperBlue)
    imgC= cv.bitwise_and(imgC, imgC, mask=mask2)
    
    ###输入参数
    num_content = "请输入待处理的目标数量： "
    global num

    printGreen(num_content.decode("utf-8").encode("gbk"))
    num = raw_input("TargetNumber: ")
    num = int(num)
    
    methods = []
    for o in range(0,num):
        methods.append(cv.TM_CCORR_NORMED)
    
    s = "ZZZ"
    c = "Z"
    
    s_content = "What direction? (U[上下] / L[左右] / 任意键进入角度选择)"
    c_content = "请输入您想要在哪个角度上进行匹配连线（角度）："
    printGreen(s_content.decode("utf-8").encode("gbk"))
    s = raw_input("Direction: ")
    if (s != "ZZZ" and s != "U" and s != "L"):
        printGreen(c_content.decode("utf-8").encode("gbk"))
        c = raw_input("Angle: ")
    red = (0, 0, 255)
    green = (0,255,0)
    col = red
    thickness = 20 
    
    content1 = "正在生成配色模板......"
    printGreen(content1.decode("utf-8").encode("gbk") )
    cv.namedWindow('imgC1', cv.WINDOW_NORMAL)
    cv.imshow('imgC1',imgC)
    
    
    th, tw = imgC.shape[:2]
    i =0
    tl = []
    br = []
    for md in methods:
        #print(md)
        result = cv.matchTemplate(targetCutting0, imgC, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        tl.append(max_loc)
        br.append((tl[i][0]+tw, tl[i][1]+th))  
        cv.rectangle(targetCutting0, tl[i], br[i], (0, 0, 255), 2)
        i += 1
        cv.namedWindow("match-" + np.str(md), cv.WINDOW_NORMAL)
        cv.imshow("match-" + np.str(md), targetCutting0)
    
    content7= "正在按照计划绘制给定目标战略识别图形......"
    printGreen(content7.decode("utf-8").encode("gbk"))
    
    #创建一个图像,300×400大小,数据类型无符号8位
    img=np.zeros((500,500,3),np.uint8)
    print tl
    
    ###连线 
    content7= "正在按照计划绘制给定目标战略识别图形......"
    printGreen(content7.decode("utf-8").encode("gbk"))
    
    if c != "Z":
        tl5 = []
        #tl6 = []
        permutation(tl, 0, len(tl),tl5)
        #tl6 = [i for i in tl5]

        for f in tl5:
            for g in range(0,num):
                f[g] = (f[g],l_get(f[g],c))
        #print "*********************************"
        while len(tl5) != 1:
            for b in tl5:
                for j in range(0,num - 1):
                    if b[j][1][1] >  b[j + 1][1][1]:
                        tl5.remove(b)
                        break
                    else:
                        pass

        content4= "正在分析已识别目标位置关系......"
        printGreen(content4.decode("utf-8").encode("gbk"))
        print "ProjectionTransformation: ",tl5

      
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(targetCutting0, tl5[0][y][0], tl5[0][y + 1][0], col, thickness) 
                cv.line(img, (tl5[0][y][0][0] / 5, tl5[0][y][0][1] /5),(tl5[0][y + 1][0][0] / 5, tl5[0][y + 1][0][1] / 5), green, thickness)#绿色，20个像素宽度                           
        cv.namedWindow("match-m-" + c + "-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-" + c + "-Black-White", targetCutting0)


    if(s == "U"):
        tl5 = []
        permutation(tl, 0, len(tl),tl5)
        #print "*********************************"
        line_matching(tl5,1)
        content4= "正在分析已识别目标位置关系......"
        printGreen(content4.decode("utf-8").encode("gbk"))
        print "ProjectionTransformation: ",tl5
        
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(targetCutting0, tl5[0][y], tl5[0][y + 1], col, thickness) 
                cv.line(img, (tl5[0][y][0] /5, tl5[0][y][1] /5),(tl5[0][y + 1][0] / 5, tl5[0][y + 1][1] / 5), green, thickness)#绿色，20个像素宽度                           
        cv.namedWindow("match-m-U-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-U-Black-White", targetCutting0)


    if(s == "L"):
        tl5 = []
        permutation(tl, 0, len(tl),tl5)
        #print "*********************************"
        line_matching(tl5,0)
        content4= "正在分析已识别目标位置关系......"
        printGreen(content4.decode("utf-8").encode("gbk"))
        print "ProjectionTransformation: ",tl5
        
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(targetCutting0,  tl5[0][y], tl5[0][y + 1], col, thickness) 
                cv.line(img, (tl5[0][y][0] /5, tl5[0][y][1] /5),(tl5[0][y + 1][0] / 5, tl5[0][y + 1][1] / 5), green, thickness)#绿色，20个像素宽度  
        cv.namedWindow("match-m-U-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-U-Black-White", targetCutting0)
    
    cv.imwrite('FinalMatch.png',targetCutting0)
    cv.imwrite('ContrastLine.png',img)
    printGreen("Done!!!")
    cv.waitKey(0)
    cv.destroyAllWindows()