#coding:utf-8
import cv2 as cv
import numpy as np
import math
import ctypes,sys
import collections


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


def l_get(a,c):
        #在c方向上对坐标进行投影变换，a为坐标
        #对于C方向直线方程：y = tan(c) * x
        c1 = int(c) * math.pi / 180
        k = math.tan(c1)
        a1 = (k * a[1] + a[0]) / (k ** 2 + 1)
        b1 = k * a1
        return (a1,b1)
        
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
                                                                                      
                                                                                      
                                Author:wwy(安鸾网络安全渗透团队)       Version 1.0.0  
                """
    
    printRed(logostr.decode("utf-8").encode("gbk"))
    #printGreen('printGreen:Gree Color Text')
    #printYellow('printYellow:Yellow Color Text')
    
    ###原图
    tpl ="D://PythonPicTemplate/binarization/tankTemplate3.jpg"
    target = "D://PythonPicTemplate/binarization/tanks4.jpg"
    tpl = Img_read(tpl)
    target = Img_read(target)
    
    ###初始化
    num_content = "请输入待处理的目标数量： "
    global num

    printGreen(num_content.decode("utf-8").encode("gbk"))
    num = raw_input("TargetNumber: ")
    num = int(num)
    
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
    printGreen(content1.decode("utf-8").encode("gbk") )
    print "TemplateSize: ",th, tw
    content2 = "正在分析目标大小："
    printGreen(content2.decode("utf-8").encode("gbk"))
    print "TargetSize: ", rows,cols
    
    i =0
    tl0 = []
    br0 = []
    for md in methods:
        #print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if md == cv.TM_SQDIFF_NORMED:
            tl0.append(min_loc)
        else:
            tl0.append(max_loc)
            
        br0.append((tl0[i][0]+tw, tl0[i][1]+th))  
        cv.rectangle(target, tl0[i], br0[i], (0, 0, 255),2)
        i += 1
    #cv.rectangle(tpl,(0,0),(th,tw),(0,0,255),2)
    content3 = "正在分析目标坐标："
    printGreen( content3.decode("utf-8").encode("gbk"))
    print "TargetCoordinates: ",tl0
    content4 = "正在分析目标相对位置："
    printGreen( content4.decode("utf-8").encode("gbk"))
    print "RelativePosition: ",br0
    
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
    printGreen( content5.decode("utf-8").encode("gbk"))
    print "The targets are being perspective transformed: ",min_x,min_y,max_x,max_y

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
    
    #cv.imshow("original", original_img)    
    #cv.imshow("gray", gray_img)
    #cv.imshow("closed", closed)
    cv.namedWindow("opened", cv.WINDOW_NORMAL)
    cv.imshow("opened", opened)
    
    #cv.imshow("original", original_imgT)    
    #cv.imshow("grayT", gray_imgT)
    #cv.imshow("closedT", closedT)
    cv.namedWindow("openedT", cv.WINDOW_NORMAL)
    cv.imshow("openedT", openedT)
    content6 = "正在对目标进行降噪处理......"
    printGreen(content6.decode("utf-8").encode("gbk"))
    
    th, tw = opened.shape[:2]
    i =0
    tl = []
    br = []
    for md in methods:
        #print(md)
        result = cv.matchTemplate(openedT, opened, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if md == cv.TM_SQDIFF_NORMED:
            tl.append(min_loc)
        else:
            tl.append(max_loc)

        br.append((tl[i][0]+tw, tl[i][1]+th))  
        cv.rectangle(openedT, tl[i], br[i], (0, 0, 255), 2)
        i += 1
        cv.namedWindow("match-" + np.str(md), cv.WINDOW_NORMAL)
        cv.imshow("match-" + np.str(md), openedT)
    
    #创建一个图像,300×400大小,数据类型无符号8位
    img=np.zeros((500,500,3),np.uint8)
    print tl
    
    ###连线 
    content7= "正在按照计划绘制给定目标战略识别图形......"
    printGreen(content7.decode("utf-8").encode("gbk"))
    
    #tl1 = []
    #content9= "正在分析已识别目标位置关系："
    #printGreen(content9.decode("utf-8").encode("gbk"))
    #print " TargeLocation: ",tl1
    
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

        content8= "正在对已识别目标进行投影变换："
        printGreen(content8.decode("utf-8").encode("gbk"))
        print "ProjectionTransformation: ",tl5

      
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(dst, tl5[0][y][0], tl5[0][y + 1][0], col, thickness)
                cv.line(openedT, tl5[0][y][0], tl5[0][y + 1][0], col, thickness) 
                cv.line(img, tl5[0][y][0], tl5[0][y + 1][0], green, thickness)#绿色，20个像素宽度                           
        cv.namedWindow("match-m-" + c + "-color", cv.WINDOW_NORMAL)
        cv.imshow("match-m-" + c + "-color", dst)
        cv.namedWindow("match-m-" + c + "-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-" + c + "-Black-White", openedT)


    if(s == "U"):
        tl5 = []
        permutation(tl, 0, len(tl),tl5)
        #print "*********************************"
        line_matching(tl5,1)
        content8= "正在对已识别目标进行投影变换："
        printGreen(content8.decode("utf-8").encode("gbk"))
        print "ProjectionTransformation: ",tl5
        
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(dst, tl5[0][y], tl5[0][y + 1], col, thickness)
                cv.line(openedT, tl5[0][y], tl5[0][y + 1], col, thickness) 
                cv.line(img, tl5[0][y], tl5[0][y + 1], green, thickness)#绿色，20个像素宽度                           
        cv.namedWindow("match-m-U-color", cv.WINDOW_NORMAL)
        cv.imshow("match-m-U-color", dst)
        cv.namedWindow("match-m-U-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-U-Black-White", openedT)


    if(s == "L"):
        tl5 = []
        permutation(tl, 0, len(tl),tl5)
        #print "*********************************"
        line_matching(tl5,0)
        content8= "正在对已识别目标进行投影变换："
        printGreen(content8.decode("utf-8").encode("gbk"))
        print "ProjectionTransformation: ",tl5
        
        for z in tl5:
            for y in range(0,num - 1):
                cv.line(dst, tl5[0][y], tl5[0][y + 1], col, thickness) 
                cv.line(openedT, tl5[0][y], tl5[0][y + 1], col, thickness) 
                cv.line(img, tl5[0][y], tl5[0][y + 1], green, thickness)#绿色，20个像素宽度  
        cv.namedWindow("match-m-U-color", cv.WINDOW_NORMAL)
        cv.imshow("match-m-U-color", dst)
        cv.namedWindow("match-m-U-Black-White", cv.WINDOW_NORMAL)
        cv.imshow("match-m-U-Black-White", openedT)
    
    cv.imwrite('FinalMatch.png',openedT)
    cv.imwrite('ContrastLine.png',img)
    
    
    
    
TemplateMatching_LineWiring()
printGreen("Done!!!")
cv.waitKey(0)
cv.destroyAllWindows()