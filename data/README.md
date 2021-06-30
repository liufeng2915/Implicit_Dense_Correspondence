### Datasets

- #### ShapeNet Training Data

  To evaluate semantic correspondence, Following the setting of LMVCNN, we trained on ShapeNet and tested on BHCP. Please download the training data from: https://drive.google.com/file/d/1vUM3USFtBAGCSnDEoFqfZNokge0qZ9S_/view?usp=sharing (161.1MB)

  For the ShapeNet part data in segmentation experiment, Please refer to BAE-Net [project page](https://github.com/czq142857/BAE-NET) for the data and protocols.

- #### BHCP

   We never used the training set of BHCP, only focused on the test set. Please follow Vladimir G. Kim's [project page](http://www.vovakim.com/projects/CorrsTmplt/doc_data.html) for the original BHCP dataset. The processed data could be downloaded in the link: https://drive.google.com/file/d/1_5yOwIy1V9meUdxKatG8gPwnutt2leOx/view?usp=sharing.

- #### Non-Existence Detection

  Our method can build dense correspondences for 3D shapes with different topologies, and automatically declare the non-existence of correspondence. However, to the best of our knowledge, there is no benchmark providing the non-existence label between a shape pair. We thus build a dataset with 1,000 paired shapes from the chair category of ShapeNet part dataset. Within a pair, one has the arm part while the other does not. For the former, we annotate 5 arm points and 5 non-arm points based on provided part labels.  We make the dataset publicly available for research.  Please download from: https://drive.google.com/file/d/1epwLq01XeohgQ-ITntTTa8whI_U36eIx/view?usp=sharing (157.4MB)

    ```bash
  Data structure:
     -- shape1:       8192*3
     -- shape2:       8192*3
     -- Points:       10*4 (x,y,z,label)
    ```

  

  

