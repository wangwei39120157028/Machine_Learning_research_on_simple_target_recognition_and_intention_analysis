# Machine_Learning_research_on_simple_target_recognition_and_intention_analysis
关于简单目标识别与意图分析的机器学习实战项目研究
** 本项目是对群体目标智能化识别进行研究，通过识别群体目标行为，有效处理并运用所获取的信息、准确判断作战群体意图。 **  
项目围绕群体目标意图识别问题展开研究，考虑了实际环境中复杂的地狱地貌、多变的目标状态以及诸多意图判断，以坦克战斗群为例，首先通过Unity3D构建了坦克群模型，模拟了群体目标的常用阵型以及行进序列。而后将敌军坦克集群图片/视频进行降噪处理，包括二值化、黑白图、以及颜色分割，而后进行初次模板匹配，找到目标的大概位置坐标之后进行透视变换，将目标拉伸到一般视角，
最后再次进行模板匹配，并根据我们之前研究过的连线算法，得到连线简图。最后，运用GoogLeNet的inceptionV3模型结合softmax函数制作深度学习分类器的方法实现了从群体目标阵型的识别到群体目标意图的判断。
根据实验所得结果，本文所提出的方法能对群体目标的阵型进行有效的识别，并对其意图进行准确的判断和分析。  
~~ 哈哈，我是抄项目里一篇文章的摘要的！！！ ~~  
![Machine_Learning_research_on_simple_target_recognition_and_intention_analysis](https://upload-images.jianshu.io/upload_images/11477676-5ed9118875e8f7a5.png?imageMogr2/auto-orient/ "项目流程图")  
 ---   
感谢CSDN上的博客大佬，还有为我补习讲解技术的老师，我的微信号是：wwy18795980897，欢迎大家对项目进行维护或者提出改进思路，我会广泛的听取大家的意见，也期待着大家的建议。  
简书链接：  
[关于简单目标识别与意图分析的机器学习实战研究（第一节 模板匹配）](https://www.jianshu.com/p/cc681104c154)  
[关于简单目标识别与意图分析的机器学习实战研究（第二节 颜色识别）](https://www.jianshu.com/p/fdce3790146a)  
[关于简单目标识别与意图分析的机器学习实战研究（第三节 降噪处理）](https://www.jianshu.com/p/0087931ab7e9)  
[关于简单目标识别与意图分析的机器学习实战研究（第四节 连线算法）](https://www.jianshu.com/p/b6ce1d8b99fe)  
[关于简单目标识别与意图分析的机器学习实战研究（第五节 透视变换）](https://www.jianshu.com/p/160dee08db59)  
[关于简单目标识别与意图分析的机器学习实战研究（第六节 神经网络目标识别——改写某博主的简单分类脚本）](https://www.jianshu.com/p/174c32e1452a)  
[关于简单目标识别与意图分析的机器学习实战研究（第七节 神经网络目标识别——tensorflow-gpu环境部署）](https://www.jianshu.com/p/798e9b4a7c21)  
[关于简单目标识别与意图分析的机器学习实战研究（第八节 神经网络目标识别——使用keras-yolo3训练自己的数据集）](https://www.jianshu.com/p/0f6e6a81e269)  
[关于简单目标识别与意图分析的机器学习实战研究（第九节 神经网络目标识别——基于inception v3模型的分类器）](https://www.jianshu.com/p/76832d7eff2f)  
[关于简单目标识别与意图分析的机器学习实战研究（第十节 项目回顾与整体修饰）](https://www.jianshu.com/p/c430da1343d8)  
