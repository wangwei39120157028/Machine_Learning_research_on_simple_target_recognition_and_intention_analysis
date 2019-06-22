#coding:utf-8
import cv2 as cv
import numpy as np
import math
import ctypes,sys
import collections
from sympy import *
from scipy import misc
import matplotlib.pyplot as plt


STD_INPUT_HANDLE = -10
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

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def len_get(x,y):
    len_xy = math.sqrt((x[0] - y[0]) ** 2 + (x[0] - y[0]) ** 2)
    return len_xy

def Angle_get(k,xy,p):
    #C = math.atan(abs((k1 - k2) / (1 + k1 * k2)))
    #return C                                   标量结果只有0 - 90度
    if p[1] >= xy[1]:
        n1 = (0,1)
        n2 = (xy[1] - p[1] ,xy[0] - p[0])
        print(n2)
        a = ((n1[0] * n2[0] + n1[1] * n2[1]))
        b = (float((n1[0] ** 2 + n1[1] ** 2) ** 0.5) * ((n2[0] ** 2 + n2[1] ** 2) ** 0.5))
        C = math.acos(a / b)
        return C
    if p[1] < xy[1]:
        n1 = (0,1)
        n2 = (xy[1] - p[1] ,xy[0] - p[0])
        print(n2)
        a = ((n1[0] * n2[0] + n1[1] * n2[1]))
        b = (float((n1[0] ** 2 + n1[1] ** 2) ** 0.5) * ((n2[0] ** 2 + n2[1] ** 2) ** 0.5))
        C = math.acos(a / b)
        return C + math.pi

def k_get(p1,p2):
    k = float(p2[1] - p1[1]) / (p2[0] - p1[0])
    return k
    
def l_get(a,c):
        #在c方向上对坐标进行投影变换，a为坐标
        #对于C方向直线方程：y = tan(c) * x
        c1 = int(c) * math.pi / 180
        k = math.tan(c1)
        a1 = (k * a[1] + a[0]) / (k ** 2 + 1)
        b1 = k * a1
        return (a1,b1)

def AngleSet(lgS):
    S15  = 0
    S30  = 0
    S45  = 0
    S60  = 0
    S75  = 0
    S90  = 0
    S105 = 0
    S120 = 0
    S135 = 0
    S150 = 0
    S165 = 0
    S180 = 0
    S195 = 0
    S210 = 0
    S225 = 0
    S240 = 0
    S255 = 0
    S270 = 0
    S285 = 0
    S300 = 0
    S315 = 0
    S330 = 0
    S345 = 0
    S360 = 0
    S    = 0

    for c in lgS:
        if c <= (math.pi / 12):
            S15 += 1
        elif c <= math.pi / 6:
            S30+= 1
        elif c <= math.pi / 4:
            S45+= 1
        elif c <= math.pi / 3:
            S60+= 1
        elif c <= 5 * math.pi / 12:
            S75+= 1
        elif c <= math.pi / 2:
            S90+= 1
        elif c <= 7 * math.pi / 12:
            S105+= 1
        elif c <= 2 * math.pi / 3:
            S120+= 1
        elif c <= 9 * math.pi / 12:
            S135+= 1
        elif c <= 10 * math.pi / 12:
            S150+= 1
        elif c <= 11 * math.pi / 12:
            S165+= 1
        elif c <= 12 * math.pi / 12:
            S180+= 1
        elif c <= 13 * math.pi / 12:
            S195+= 1
        elif c <= 14 * math.pi / 12:
            S210+= 1
        elif c <= 15 * math.pi / 12:
            S225+= 1
        elif c <= 16 * math.pi / 12:
            S240+= 1
        elif c <= 17 * math.pi / 12:
            S255+= 1
        elif c <= 18 * math.pi / 12:
            S270+= 1
        elif c <= 19 * math.pi / 12:
            S285+= 1
        elif c <= 20 * math.pi / 12:
            S300+= 1
        elif c <= 21 * math.pi / 12:
            S315+= 1
        elif c <= 22 * math.pi / 12:
            S330+= 1
        elif c <= 23 * math.pi / 12:
            S345+= 1
        elif c <= 24 * math.pi / 12:
            S360+= 1
        else:
            print("wrong0000")
    
    S =[
        S15 ,
        S30 ,
        S45 ,
        S60 ,
        S75 ,
        S90 ,
        S105,
        S120,
        S135,
        S150,
        S165,
        S180,
        S195,
        S210,
        S225,
        S240,
        S255,
        S270,
        S285,
        S300,
        S315,
        S330,
        S345,
        S360,]
        
    return S

def Img_read(input_img):
    original_pic = cv.imread(input_img)
    return original_pic

def Img_Outline(original_img):
    gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_img, (9, 9), 0)                    # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv.threshold(blurred, 140, 255, cv.THRESH_BINARY)  # 设定阈值，对识别很重要，可以手动调节识别效果（阈值影响开闭运算效果）12
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv.morphologyEx(RedThresh, cv.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)           # 开运算（去噪点）
    return original_img, gray_img, RedThresh, closed, opened
        
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
        
        
def TemplateMatching_LineWiring():
    logostr = """\033[1;32;40m \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n

 $$$$$$\  $$$$$$\         $$\   $$\                       $$\                         
$$  __$$\ \_$$  _|        $$ |  $$ |                      $$ |                        
$$ /  \__|  $$ |          $$ |  $$ |$$\   $$\ $$$$$$$\  $$$$$$\    $$$$$$\   $$$$$$\  
\$$$$$$\    $$ |  $$$$$$\ $$$$$$$$ |$$ |  $$ |$$  __$$\ \_$$  _|  $$  __$$\ $$  __$$\ 
 \____$$\   $$ |  \______|$$  __$$ |$$ |  $$ |$$ |  $$ |  $$ |    $$$$$$$$ |$$ |  \__|
$$\   $$ |  $$ |          $$ |  $$ |$$ |  $$ |$$ |  $$ |  $$ |$$\ $$   ____|$$ |      
\$$$$$$  |$$$$$$\         $$ |  $$ |\$$$$$$  |$$ |  $$ |  \$$$$  |\$$$$$$$\ $$ |      
 \______/ \______|        \__|  \__| \______/ \__|  \__|   \____/  \_______|\__|      
                                                                                      
                                                                                      
                                Author:lx       Version 1.0.0  
                """
    
    printRed(logostr)
    
    ###原图
    tpl ="D:/PythonPicTemplate/changeView/test-A/tankTemplatec2.jpg"
    target = "D:/PythonPicTemplate/changeView/test-A/combat84.3-115.7-200.jpg"
    tpl = Img_read(tpl)
    target = Img_read(target)
    
    #旋转对结果不造成影响
    angle_content = "请输入您想旋转图形的角度（顺时针，旋转对结果不造成影响，按N跳过）："
    angle = input(angle_content)
    if angle == "N":
        angle = "0"
    target=rotate_bound(target,int(angle))
    
    ###初始化
    num_content = "请输入待处理的目标数量： "
    global num

    printGreen(num_content)
    num = input("TargetNumber: ")
    num = int(num)
    
    c = "Z"
    
    c_content = "请输入您想要在哪个角度上进行匹配连线（角度）："
    printGreen(c_content)
    c = input("Angle: ")
    red = (0, 0, 255)
    green = (0,255,0)
    col = red
    thickness = 5 
    
    
    ###透视变换
    methods = []
    for o in range(0,num):
        methods.append(cv.TM_CCORR_NORMED)
    '''
    差值平方和匹配 CV_TM_SQDIFF
    标准化差值平方和匹配 CV_TM_SQDIFF_NORMED
    相关匹配 CV_TM_CCORR
    标准相关匹配 CV_TM_CCORR_NORMED
    相关匹配 CV_TM_CCOEFF
    标准相关匹配 CV_TM_CCOEFF_NORMED
    '''
    th, tw = tpl.shape[:2]
    rows, cols = target.shape[:2]
    content1 = "正在分析模板大小："
    printGreen(content1  )
    print("TemplateSize: ",th, tw)
    content2 = "正在分析目标大小："
    printGreen(content2)
    print("TargetSize: ", rows,cols)
    
    i =0
    tl0 = []
    br0 = []
    for md in methods:
        #print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        tl0.append(max_loc)
            
        br0.append((tl0[i][0]+tw, tl0[i][1]+th))  
        cv.rectangle(target, tl0[i], br0[i], (0, 0, 255),2)
        i += 1
    content3 = "正在分析目标坐标："
    printGreen( content3)
    print("TargetCoordinates: ",tl0)
    content4 ="正在分析目标相对位置："
    printGreen( content4)
    print("RelativePosition: ",br0)
    
    tl2 = []
    tl3 = []
    tl4 = []
    for q in tl0:
        tl3.append(q[0] - tw)
        tl4.append(q[1] - th)
    for p in br0:
        tl3.append(p[0])
        tl4.append(p[1])
    min_x = min(tl3)
    min_y = min(tl4)
    max_x = max(tl3)
    max_y = max(tl4)
    content5 = "正在对目标进行透视变换："
    printGreen( content5)
    print("The targets are being perspective transformed: ",min_x,min_y,max_x,max_y)

    tl2.append([min_x - 2 * tw,min_y - 2 * th])
    tl2.append([min_x - 2 * tw,max_y + 2 * th])
    tl2.append([max_x + 2 * tw,min_y - 2 * th])
    tl2.append([max_x + 2 * tw,max_y + 2 * th])
    
    # 原图中已经识别的四个角点
    pts1 = np.float32(tl2)
    # 变换后分别在左上、右上、左下、右下四个点
    #pts2 = np.float32([[0, 0],[0, 500], [500, 0],[500, 500]])
    pts2 = np.float32([[0, 0],[0, cols], [rows, 0],[rows, cols]])
    #rows, cols
    # 生成透视变换矩阵
    M = cv.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv.warpPerspective(target, M, (rows, cols))
    cv.namedWindow("match-PerspectiveTransformation", cv.WINDOW_NORMAL)
    cv.imshow("match-PerspectiveTransformation", dst)
    
    
    ###降噪、连接、模板匹配
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(tpl)
    original_imgT, gray_imgT, RedThreshT, closedT, openedT = Img_Outline(dst)
    
    cv.namedWindow("opened", cv.WINDOW_NORMAL)
    cv.imshow("opened", opened)
    cv.namedWindow("openedT", cv.WINDOW_NORMAL)
    cv.imshow("openedT", openedT)
    content6 = "正在对目标进行降噪处理......"
    printGreen(content6)
    
    th, tw = opened.shape[:2]
    i =0
    tl = []
    br = []
    for md in methods:
        #print(md)
        result = cv.matchTemplate(openedT, opened, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        tl.append(max_loc)

        br.append((tl[i][0]+tw, tl[i][1]+th))  
        cv.rectangle(openedT, tl[i], br[i], (0, 0, 255), 2)
        i += 1
        cv.namedWindow("match-" + np.str(md), cv.WINDOW_NORMAL)
        cv.imshow("match-" + np.str(md), openedT)
    
    #创建一个图像,300×400大小,数据类型无符号8位
    img=np.zeros((500,500,3),np.uint8)
    xx = 0
    yy = 0
    ###连线 
    content7= "正在按照计划绘制给定目标战略识别图形......"
    printGreen(content7)
    
    if c != "Z":
        c1 = int(0) * math.pi / 180
        tl5 = []
        permutation(tl, 0, len(tl),tl5)

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

        content8= "正在对已识别目标进行投影变换："
        printGreen(content8)
        print("ProjectionTransformation: ",tl5)
        print(tl5[0])
      
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(dst, tl5[0][y][0], tl5[0][y + 1][0], col, thickness)
                cv.line(openedT, tl5[0][y][0], tl5[0][y + 1][0], col, thickness) 
                cv.line(img, tl5[0][y][0], tl5[0][y + 1][0], green, thickness)#绿色，20个像素宽度                           

        for z1 in range(len(tl5[0])):
            xx += tl5[0][z1][0][0]
        xx = xx / len(tl5[0])
        
        for z2 in range(len(tl5[0])):
            yy += tl5[0][z2][0][1]
        yy = yy / len(tl5[0])
        print(xx,yy)
        lenSet = []
        for y in range(len(tl5[0])):
            lg = len_get((xx,yy),tl5[0][y][0])
            lenSet.append(lg)
        
        r1 = max(lenSet)
        r2 = min(lenSet)
        
        #求方程组的解
        answer1 = ((r1 + xx,yy),(-r1 + xx,yy))
        answer2 = ((xx,r1 + yy),(xx,- r1 + yy))
        answer3 = ((r2 + xx,yy),(- r2 + xx,yy))
        answer4 = ((xx,r2 + yy),(xx,- r2 +yy))
        
        #求观察点坐标
        p0 = (xx,yy)
        p1 = answer1[0]
        p2 = answer1[1]
        p3 = answer2[0]
        p4 = answer2[1]
        p5 = answer3[0]
        p6 = answer3[1]
        p7 = answer4[0]
        p8 = answer4[1]
        
        len_p0 = 0
        len_p1 = 0
        len_p2 = 0
        len_p3 = 0
        len_p4 = 0
        len_p5 = 0
        len_p6 = 0
        len_p7 = 0
        len_p8 = 0
        
        #计算目标到观察点的距离
        for y in range(len(tl5[0])):
            len_p0 += len_get(tl5[0][y][0],p0)
            len_p1 += len_get(tl5[0][y][0],p1)
            len_p2 += len_get(tl5[0][y][0],p2)
            len_p3 += len_get(tl5[0][y][0],p3)
            len_p4 += len_get(tl5[0][y][0],p4)
            len_p5 += len_get(tl5[0][y][0],p5)
            len_p6 += len_get(tl5[0][y][0],p6)
            len_p7 += len_get(tl5[0][y][0],p7)
            len_p8 += len_get(tl5[0][y][0],p8)
        
        lg0S = []
        lg1S = []
        lg2S = []
        lg3S = []
        lg4S = []
        lg5S = []
        lg6S = []
        lg7S = []
        lg8S = []
        
        ##求直线夹角1
        for y in range(len(tl5[0])):
            lg0 = Angle_get(math.tan(c1),(tl5[0][y][0][0] / len_p0,tl5[0][y][0][1] / len_p0),(p0[0] / len_p0,p0[1] / len_p0))
            print(lg0)
            lg0S.append(lg0)
        
        for y in range(len(tl5[0])):
            lg1 = Angle_get(math.tan(c1),tl5[0][y][0],p1)
            print(lg1)
            lg1S.append(lg1)
        
        for y in range(len(tl5[0])):
            lg2 = Angle_get(math.tan(c1),tl5[0][y][0],p2)
            print(lg2)
            lg2S.append(lg2)
        
        for y in range(len(tl5[0])):
            lg3 = Angle_get(math.tan(c1),tl5[0][y][0],p3)
            print(lg3)
            lg3S.append(lg3)
        
        for y in range(len(tl5[0])):
            lg4 = Angle_get(math.tan(c1),tl5[0][y][0],p4)
            print(lg4)
            lg4S.append(lg4)
        
        for y in range(len(tl5[0])):
            lg5 = Angle_get(math.tan(c1),tl5[0][y][0],p5)
            lg5S.append(lg5)
        
        for y in range(len(tl5[0])):
            lg6 = Angle_get(math.tan(c1),tl5[0][y][0],p6)
            lg6S.append(lg6)
        
        for y in range(len(tl5[0])):
            lg7 = Angle_get(math.tan(c1),tl5[0][y][0],p7)
            lg7S.append(lg7)
        
        for y in range(len(tl5[0])):
            lg8 = Angle_get(math.tan(c1),tl5[0][y][0],p8)
            lg8S.append(lg8)
        
        #lg0SA.extend(lg1SA).extend(lg2SA).extend(lg3SA).extend(lg4SA).extend(lg5SA).extend(lg6SA).extend(lg7SA).extend(lg8SA)
        
        lg0S = AngleSet(lg0S)
        lg1S = AngleSet(lg1S)
        lg2S = AngleSet(lg2S)
        lg3S = AngleSet(lg3S)
        lg4S = AngleSet(lg4S)
        lg5S = AngleSet(lg5S)
        lg6S = AngleSet(lg6S)
        lg7S = AngleSet(lg7S)
        lg8S = AngleSet(lg8S)
        
        lgSU = lg0S + lg1S + lg2S + lg3S + lg4S + lg5S + lg6S + lg7S + lg8S
        numS = np.linspace(1,216,216,endpoint=True)
        
        plt.plot(numS, lgSU, marker='o', color='green', label='')
        plt.legend() # 显示图例
        plt.xlabel('num')
        plt.ylabel('15Contain')
        plt.show()
        
        print("__________________")
        print(lgSU)
        misc.imsave('D:/PythonPicTemplate/changeView/test-A/1.png', temp1)
        
        cv.namedWindow("match-m-" + c + "-color", cv.WINDOW_NORMAL)
        cv.imshow("match-m-" + c + "-color", dst)
        cv.namedWindow("match-m-" + c + "-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-" + c + "-Black-White", openedT)
    
    cv.imwrite('FinalMatch.png',openedT)
    cv.imwrite('ContrastLine.png',img)
    
    
TemplateMatching_LineWiring()
printGreen("Done!!!")
cv.waitKey(0)
cv.destroyAllWindows()





