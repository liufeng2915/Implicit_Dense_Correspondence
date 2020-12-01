
# Learning Implicit Functions for Topology-Varying Dense 3D Shape Correspondence
Advances in Neural Information Processing Systems (NeurIPS 2020). **Oral presentation**. [[Arxiv](https://arxiv.org/abs/2010.12320), [PDF](http://cvlab.cse.msu.edu/pdfs/Implicit_Dense_Correspondence.pdf), [Supp](http://cvlab.cse.msu.edu/pdfs/Implicit_Dense_Correspondence_Supp.pdf), [Project](http://cvlab.cse.msu.edu/project-implicit-dense-correspondence.html)]

**[Feng Liu](http://cvlab.cse.msu.edu/pages/people.html),   [Xiaoming Liu](http://cvlab.cse.msu.edu/pages/people.html)**

Department of Computer Science and Engineering, Michigan State University

<font color=\#008000>***The goal of this paper is to learn dense 3D shape correspondence for topology-varying objects in an unsupervised manner.*** </font>

![teaser](docs/teaser.png)

<font face="宋体" size=2.5> Figure 1: Given a shape **S**, PointNet *E* is used to extract the shape feature code **z**. Then a part embedding **o** is produced via a deep implicit function *f*. We implement dense correspondence through an inverse  function mapping from **o** to recover the 3D shape. (b) To further make the learned part embedding consistent across all the shapes, we randomly select two shapes **S**<sub>A</sub> and **S**<sub>B</sub>. By swapping the part embedding vectors, a cross reconstruction loss is used to enforce the inverse function to recover to each other. </font>

--------------------------------------

This code is developed with Python3 and PyTorch 1.1

**Dataset**

Please refer to data/README

**Pretrained models**

Please refer to models/README

**Evaluation**

Please refer to evaluation/README

**Training**

Our method is trained in three stages: 1) PointNet like encoder and implicit function are trained on sampled point-value pairs via occupancy loss. 2) encoder, implicit function and inverse implicit function are jointly trained via occupancy and self-reconstruction losses. 3) We jointly train encoder, implicit function and inverse implicit function with occupancy, self-reconstruction and cross-reconstruction losses.

```bash
## ----------------------  chair  ----------------------------- ##
# stage1
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --cate_name chair --epoch  50 --resolution 16 --batch_size 45 
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --cate_name chair --epoch 100 --resolution 32 --batch_size 25 --pretrain_model TRUE --pretrain_model_name Corr-49.pth
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --cate_name chair --epoch 300 --resolution 64 --batch_size  7 --pretrain_model TRUE --pretrain_model_name Corr-99.pth

# stage2
CUDA_VISIBLE_DEVICES=0 python train_stage2.py --cate_name chair --epoch 500 --resolution 64 --batch_size  4 --pretrain_model TRUE --pretrain_model_name stage1/Corr-299.pth
```

--------------------------------------

**Contact**

For questions feel free to post here or directly contact the author via liufeng6@msu.edu

