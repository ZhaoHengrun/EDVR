# EDVR
EDVR 的pytorch复现 <br/>
参考自<br/>
https://arxiv.org/abs/1905.02716 <br/>
https://github.com/xinntao/BasicSR <br/>
https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch <br/>
https://github.com/YapengTian/TDAN-VSR-CVPR-2020 <br/>
本程序仅能在Linux环境下运行 <br/>
## Requirements
Python 3.8<br/>
PyTorch 1.6.0<br/>
Numpy 1.19.2<br/>
Pillow 7.2.0<br/>
OpenCV 4.4.0.44<br/>
Visdom 0.1.8.9<br/>
Wandb 0.10.10<br/>
## Usage:
### Build
运行`sh make.sh`，如果编译错误，重新编译前请删除`build/`目录。<br/>
### Make datasets
所用的目录需要手动创建<br/>
下载REDS数据集https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ 提取码：basr <br/>
或https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing <br/>
解压后将`REDS/train_sharp/`下的目录移动至`datasets/train/target/`下，	<br/>
`REDS/train_sharp_bicubic/X4/`下的目录移动至`datasets/train/input/`下。	<br/>
如使用REDS4数据集作为验证或测试，则需将`datasets/train/target/`和`datasets/train/input/`下的
`000/`,`011/`,`015/`,`020/`四个目录分别移动至`datasets/test/target/`和`datasets/test/input/`下。<br/>
### Train
运行`python train.py`进行训练	<br/>
模型保存在`checkpoints/`目录下	<br/>
### Test&Eval
运行`python test.py`进行测试，生成的图片保存在`results/`目录下	<br/>
使用`utils/`目录下的`compute_psnr.m`计算psnr	<br/>
## Results
使用10个残差块，64通道，裁剪尺寸64* 64，在REDS4上的平均psnr为30.8302（RGB），若有错误欢迎指正。	<br/>