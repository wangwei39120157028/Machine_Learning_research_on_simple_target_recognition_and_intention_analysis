import tensorflow as tf
import os 
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
lines = tf.gfile.GFile('retrained_labels.txt').readlines()
uid_to_human ={}
#读取参数中的数据
for uid,line in enumerate(lines):
    line=line.strip('\n')
    uid_to_human[uid]=line
def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]
#创建图来存放训练好的模型参数
with tf.gfile.FastGFile('retrained_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')
#测试图片分类
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            #jpeg格式的图片
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
            #结果转为1维度
            predictions = np.squeeze(predictions)
            #打印图片信息
            image_path = os.path.join(root,file)
            print (image_path)
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            #排序
            top_k = predictions.argsort()[::-1]
            print(top_k)
            for node_id in top_k:
                human_string =id_to_string(node_id)
                #置信度
                score = predictions[node_id]
                print ('%s (score = %.5f)' % (human_string, score))
            print()
