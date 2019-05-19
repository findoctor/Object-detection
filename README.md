# Implementation of object detection with deformable convolution


## Version: PyTorch 1.0

## Our Work
We based our work on the implementation of https://github.com/jwyang/faster-rcnn.pytorch. We used some helper function from this repo and below are what we have done:

* **testDeform.py**: Test the performance of using deformable convolution on CIFAR-10 dataset with simple network. 
* **resnet.py**: Modify the architecture of the original ResNet-101 network and apply deformable convolution layer.
* **DeformableConv2d.py**: Implement the idea of deformable convolution.
## Result
![alt text](https://raw.githubusercontent.com/findoctor/Object-detection/master/Result.jpg)





