# 犬物种分类

## 调参记录

* resnet101 fine tuning : val_acc 77%

* resnet101 fine tuning + freeze feature layers : val_acc 85%
  * data augmentaion ： \
      * 过强的数据增强对acc影响不大，但是loss会变差
      * 适当的数据增强使val_acc -> 87% loss -> 0.39
  
  * label smoothing ： 性能有所下降

## tricks

* eval时，对image Resize(256) -> CenterCrop(224),会使val_loss下降0.06左右
