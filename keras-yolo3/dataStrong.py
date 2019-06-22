#from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import glob

PATH = "E:/keras-yolo3/DataSet/"
SAVE_PATH = "E:/keras-yolo3/DataSetStrong/"
# 设置生成器参数,使数据集去中心化,错切变换,在长或宽的方向进行放大
datagen = image.ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,shear_range=0.5,zoom_range=0.5)
gen_data = datagen.flow_from_directory(PATH,batch_size=1,shuffle=False,save_to_dir=SAVE_PATH,save_prefix='gen',target_size=(416, 416))

# 生成9张图
for i in range(1000):
    gen_data.next()
