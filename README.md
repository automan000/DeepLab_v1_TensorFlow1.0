# DeepLab-v1-Tensorflow

This is an implementation of [DeepLab-LargeFOV](http://ccvl.stat.ucla.edu/deeplab-models/deeplab-largefov/) in TensorFlow for semantic image segmentation on [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

This code is based on the implementation from [tensorflow-deeplab-lfov](https://github.com/DrSleep/tensorflow-deeplab-lfov). Please check this repository for details.

## TODO

- [x]Works with TensorFLow>=1.0

- [x]Weight decay

- [x]Tracks training with TensorBoard

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

## Caffe model

You can download two already converted models (`model.ckpt-init` and `model.ckpt-pretrained`) [here](https://drive.google.com/open?id=0B_rootXHuswsTF90M1NWQmFYelU).

