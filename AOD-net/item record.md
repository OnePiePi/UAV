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

### 代码阅读

#### dataloader.py

##### new method: glob

```
glob.glob(pathname, *, recursive=False)
```

learn some english: here's a breakdown of how the function 'glob' works. 翻译为：下面是函数“glob”是如何工作的。但是breakdown通常被翻译为故障，这里怎么就成了原理了。

wildcards: 通配符

###### Wildcards

- `*`: Matches any number of characters, including none.
- `?`: Matches exactly one character.
- `[seq]`: Matches any character in `seq`.
- `[!seq]`: Matches any character not in `seq`.

some instance:

```
for filename in glob.glob('**/*.jpg', recursive=True):
    print(filename)
# 递归查找.jpg文件
```

```
for filename in glob.glob('data_?.txt'):
    print(filename)
# 返回所有data_任意一个字符.txt 的文件
```

```
for filename in glob.glob('[abc]*.txt'):
    print(filename)
# 返回所有开头字母为abc其中一个的.txt文件
```

learn some english: parenthesis 小括号	square bracket 中括号	brace 大括号

##### new method: split

```
path = "/home/user/documents/report.txt"
filename = path.split("/")[-1]
print(filename)
```

返回根据‘/’划分的list