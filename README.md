# Spatial Interaction-aware Transformer-based Trajectory Prediction

## 数据集

本项目采用 Next Generation Simulation (NGSIM) 提供的高速路车辆轨迹数据 (US-101 和 I-80)，数据详情与下载请前往[这里](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)。

数据预处理采用由 Deo 提供的 MATLAB 脚本 ([preprocess_data.m](https://github.com/nachiket92/conv-social-pooling))。

你也可以直接从如下百度网盘链接中下载已预处理好的数据，并解压到 `data` 文件夹下：
> 链接：https://pan.baidu.com/s/1_Oev_vQ4g-z2MjgSsARAQA?pwd=0hc6 
提取码：0hc6

## 环境要求
- PyTorch 1.4.0+
- Transformer 4.13.0+

## 项目结构
- data: 保存原始标注数据或预处理后的数据
- outputs: 保存模型训练过程中产生的文件
    - logs: 训练日志
    - saved_models: 模型权重等文件
- layers: 保存 attention 等神经网络模块的定义代码
- plotfigs_and_analysis: 数据可视化与分析相关代码
- 其他
    - feeder_ngsim.py: 负责 “数据 -> 特征” 的处理过程
    - utils.py: 各种工具代码
    - models.py: 轨迹预测模型的定义代码
    - main.py: 程序主入口

## 模型介绍
本项目主要提供 Spatial Interaction-aware Transformer-based Trajectory Prediction (SIT-ID) 模型的实现，该模型在标准 Transformer 的基础上改进，使其能够同时获取轨迹时序依赖关系和车辆间的空间交互模式。具体介绍可参阅论文：

> @Article{SIT-ID,
> AUTHOR = {Li, Xiaolong and Xia, Jing and Chen, Xiaoyong and Tan, Yongbin and Chen, Jing},
> TITLE = {SIT: A Spatial Interaction-Aware Transformer-Based Model for Freeway Trajectory Prediction},
> JOURNAL = {ISPRS International Journal of Geo-Information},
> VOLUME = {11},
> YEAR = {2022},
> NUMBER = {2},
> ARTICLE-NUMBER = {79},
> URL = {https://www.mdpi.com/2220-9964/11/2/79},
> ISSN = {2220-9964},
> DOI = {10.3390/ijgi11020079}}

## 如何使用

先将预处理后的数据放入到 `data` 文件夹下；

再在命令行将路径切换到当前项目路径，并执行如下命令以启动模型训练：

``
python main.py --do_train
``

