## Adversarial Discriminative Domain Adaptation



**[WIP]** Pytorch implementation of [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464)

- code reference: https://github.com/corenel/pytorch-adda
- source & target classifier: LeNet 5 (followed implementation in Caffe)
- discriminator: 3-layer MLP (500-500-1) with ReLU activation
  - This does not converge for some reasons.




**Envirionment**

- Python 3.6
- PyTorch 0.4.0
- torchvision 0.2.1




**Result**


|                   | SVHN(Source)  | MNIST(Target)  |
| ----------------- |:-------------:| :-------------:|
| Source Classifier | 0.9168        | 0.633          |
| Target Classifier | -             | WIP            |




**LeNet 5 implementation**

![](https://shadowthink.com/images/dl/caffe_lenet_viz.png)

