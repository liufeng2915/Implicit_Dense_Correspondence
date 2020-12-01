### Datasets

- #### ShapeNet Training Data

- #### BHCP

  To evaluate semantic correspondence, Following the setting of LMVCNN, we trained on ShapeNet and test on BHCP.  We never used the training set of BHCP, only focus on the test set. 

- #### Non-Existence Detection

  Our method can build dense correspondences for 3D shapes with different topologies, and automatically declare the non-existence of correspondence. However, to the best of our knowledge, there is no benchmark providing the non-existence label between a shape pair. We thus build a dataset with 1,000 paired shapes from the chair category of ShapeNet part dataset. Within a pair, one has the arm part while the other does not. For the former, we annotate 5 arm points and 5 non-arm points based on provided part labels.  We make the dataset publicly available for research.  Please download from: https://drive.google.com/file/d/1epwLq01XeohgQ-ITntTTa8whI_U36eIx/view?usp=sharing (157.4MB)

    ```bash
  Data structure:
     -- shape1:       8192*3
     -- shape2:       8192*3
     -- Points:       10*4 (x,y,z,label)
    ```

  

  

