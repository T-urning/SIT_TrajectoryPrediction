# Trajectory prediction on US-101 and I-80 datasets

The maximum number of vehicles in one frame is 255.

### 数据集预处理时间太长
2000 样本量下的测试

| batch_size | num_workers | data preprocessing (dp) | total (dp + model) |      |
| ---------- | ----------- | ----------------------- | ------------------ | ---- |
| 16         | 0           | 60 s (2 it/s)           | 77 s               |      |
| 16         | 2           | 35 s (3.5 it/s)         | 38 s               |      |
| 16         | 4           | 21 s (5.76 it/s)        | 24 s               |      |
|            |             |                         |                    |      |

### 如何大幅度较少数据的预处理时间？

