#coding:utf-8
import cv2
import  numpy as np


#64为最大值，值越大，相似度越高

#均值哈希算法
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str
 
#差值感知算法
def dHash(img):
    #缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str
 
#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #相等则n计数+1，n最终为相似度
        if hash1[i]==hash2[i]:
            n=n+1
    return n

 
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


img1=cv2.imread('D://PythonPicTemplate/pic_Compire/ContrastLine.png')
img2=cv2.imread('D://PythonPicTemplate/pic_Compire/ContrastLine11.png')

#旋转对结果不造成影响
angle_content = "请输入您想旋转第二个图形的角度（顺时针，旋转对结果不造成影响，按N跳过）："
angle = raw_input(angle_content.decode("utf-8").encode("gbk"))
if angle == "N":
    angle = "0"
imag=rotate_bound(img2,int(angle))

hash1= aHash(img1)
hash2= aHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
n_content = r'均值哈希算法相似度：'
n_percent = (float(n )/ 64)*100
print n_content.decode("utf-8").encode("gbk"), str(n_percent) + '%'

hash3= dHash(img1)
hash4= dHash(img2)
print(hash3)
print(hash4)
p=cmpHash(hash3,hash4)
p_content = r'差值哈希算法相似度：'
p_percent = (float(p )/ 64)*100
print p_content.decode("utf-8").encode("gbk"), str(p_percent) + '%'

s_content = "最终相似度为："
s_percent = (n_percent + p_percent) / 2
print s_content.decode("utf-8").encode("gbk"), str(s_percent) + '%'

cv2.imshow('figure-1',img1)
cv2.imshow('figure-2',imag)
cv2.waitKey()





















