
# Learning Implicit Functions for Topology-Varying Dense 3D Shape Correspondence
In Proceeding of 2020 Conference on Neural Information Processing Systems (NeurIPS 2020). **Oral presentation**.

**Feng Liu,   Xiaoming Liu**

Department of Computer Science and Engineering, Michigan State University

***The goal of this paper is to learn dense 3D shape correspondence for topology-varying objects in an unsupervised manner.*** 

![teaser](https://github.com/liuf1990/Implicit_Dense_Correspondence/tree/main/img/teaser.png)

Figure 1: Given a shape **S**, PointNet *E* is used to extract the shape feature code **z**. Then a part embedding **o** is produced via a deep implicit function *f*. We implement dense correspondence through an inverse  function mapping from **o** to recover the 3D shape. (b) To further make the learned part embedding consistent across all the shapes, we randomly select two shapes **S**<sub>A</sub> and **S**<sub>B</sub>. By swapping the part embedding vectors, a cross reconstruction loss is used to enforce the inverse function to recover to each other.