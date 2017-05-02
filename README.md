# DeepLab-v1-Tensorflow

This is an implementation of [DeepLab-LargeFOV](http://ccvl.stat.ucla.edu/deeplab-models/deeplab-largefov/) in TensorFlow for semantic image segmentation on [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

This code is based on the implementation from [tensorflow-deeplab-lfov](https://github.com/DrSleep/tensorflow-deeplab-lfov). Please check this repository for details.

## TODO

- [x] Works with TensorFLow>=1.0
- [x] Weight decay
- [x] Tracks training with TensorBoard
- [x] Fully functional evaluation code
- [x] Achieve the performance reported in [ArXiv](https://arxiv.org/abs/1606.00915)

The Post-processing step with DenseCRF

Atrous spatial pyramid pooling (ASPP)

## Requirements

TensorFlow>=1.0 is supported

To install the required python packages (except TensorFlow), run
```bash
pip install -r requirements.txt
```
or for a local installation
```bash
pip install -user -r requirements.txt
```

## Best results at present
background: 88.5654446653
aeroplane: 66.6891497821
bicycle: 27.5425855685
bird: 71.9108573055
boat: 51.4274847257
bottle: 62.3651852486
bus: 78.4051023721
car: 70.6123718826
cat: 75.3887068995
chair: 25.7829156204
cow: 53.5545469656
diningtable: 48.3028937323
dog: 69.360881792
horse: 52.9500919802
motorbike: 64.8494736002
person: 72.9275538607
pottedplant: 35.1361763734
sheep: 65.3341241448
sofa: 37.3502489281
train: 71.4086065257
tvmonitor: 53.9655908926
('mAP:', 59.22999966028858)

Optimizer: SGD
Batch Size: 10
Learning rate: 1e-3
Lr_decay_step: 5000
Total_step: 20000
Momentum: 0.9
Weight decay: 0.0005

## Caffe model

You can download two already converted models (`model.ckpt-init` and `model.ckpt-pretrained`) [here](https://drive.google.com/open?id=0B_rootXHuswsTF90M1NWQmFYelU).

