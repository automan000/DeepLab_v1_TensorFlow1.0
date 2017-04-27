# DeepLab-v1-Tensorflow

This is an implementation of [DeepLab-LargeFOV](http://ccvl.stat.ucla.edu/deeplab-models/deeplab-largefov/) in TensorFlow for semantic image segmentation on [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

This code is based on the implementation from [tensorflow-deeplab-lfov](https://github.com/DrSleep/tensorflow-deeplab-lfov). Please check this repository for details.

## Differences

Works with TensorFLow>=1.0

The Weight decay should functions correctly now.


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

## Todo

The Post-processing step with DenseCRF