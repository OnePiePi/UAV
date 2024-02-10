## 远程连接 linux 主机进行开发操作

### 选用 AutoDL 作为远端算力

目前使用A4000-16G单卡训练，0.92￥/hour。半精76.7Tensor TFLOPS，单精19.17TFLOPS。计划在跑通整个流程后，换用A5000-24G单卡训练，1.23￥/hour。半精117Tensor TFLOPS，单精27.77TFLOPS。

A4000 20.84 TFLOPS/￥ 83.37 Tensor TFLOPS/￥

A5000 22.58 TFLOPS/￥ 95.12 Tensor TFLOPS/￥

### 使用 vscode 的remote插件

### 使用 百度网盘 和 AutoPannel 解决实例间迁移问题

#### 百度网盘授权

密钥信息

- AppKey:

  ZjZyHQxjQVytKk1Fmxi4KGPAl6lKTv7F

- SecretKey:

  j46A0ffGzMKhVEQS4EI5ZntU6ZHBb8VZ

- SignKey:

  89Xh9ia50A^1xZTMMaoI7eP^7ZOr6qEF

### 实例间转移流程

1. 通过百度网盘和antopannel转移代码和数据
2. 上次关机前记得保留实例镜像，通过官方镜像转移完成环境搭建（有空学一下环境备份）