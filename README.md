# Image Classification using Convolutional Neural Networks


**Source code based on lecture CS492 Introduction to DeepLearing**
___

**Block Classes**
1. MLP block(MLPBlock)
2. Convolutional block(ConvBlock)
3. Plain residual block(ResBlockPlain)
4. Residual block with bottleneck(ResBlockBottleneck)
5. Inception Block(InceptionBlock)

**Network class**
- MyNetwork

**Illustrations for Block Classes**
- [MLP block]()
- [Convolutional block]()
- [Plain residual block]()
- [Residual block with bottleneck]()
- [Inception Block]()

**Illustrations for Network**
- [MyNetwork]()

**Data Set**
- CIFAR10 dataset into 10 categoris(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Experimental results and number of model parameters**

| Method | Accuracy | #Params | Expected Acc | Expected # Params |
|:---:|:---:|:---:|:---:|:---:|
|MLP|63.04|1649354|62.6|1649354|
|Conv|81.72|510426|81.9|510426|
|resPlain|88.53|510426|81.9|510426|
|resBottleNeck|86.18|113946|86.5|113946|
|inception|83.61|124026|83.7|124026|
