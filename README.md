# FCN implementation on ECP(Ecole Centrale Paris) facades dataset, PyTorch
PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org) on facades dataset.

##Dataset download

Images: [Ecole Centrale Paris Facades Database/CVPR 2010 dataset](http://vision.mas.ecp.fr/Personnel/teboul/files/cvpr2010.zip)  
get a image names list in .txt, and delete monge_51,52,85,89, and 75bis.  
Annotations: [Updated ECP Dataset Annotations](http://martinovi.ch/datasets/ECP_newAnnotations.zip)


## Installation

see [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
```bash
git clone https://github.com/SunYW0108/ECP_FCN.git
```

## Training
```bash
./train_fcn32s.py -g 0
# FCN16s is trained on the best model of FCN32s.
./train_fcn16s.py -g 0 --pretrained-model logs/XXX/model_best.pth.tar
# FCN8s is trained on the best model of FCN16s.
./train_fcn8s.py -g 0 --pretrained-model logs/XXX/model_best.pth.tar

# or train FCN8s at once.

./train_fcn8s_atonce.py -g 0
```
```bash
./learning_curve.py logs/XXX/log.csv
./view_log logs/XXX/log.csv
./train_fcnXXs -g 0 --resume logs/XXX/checkpoint.pth.tar
```

<img src=".readme/fcn8s_iter10000.jpg" width="50%" />
Visualization of validation result of FCN8s.

