# Shearlet-Transform-TensorFlow
This is an implementation of Shearlet Transform (ST) [1, 2] for light field reconstruction using TensorFlow. If you find this code useful in your research, please consider citing [1, 2] and 

    @inproceedings{tfst2019,
	Author = {Gao, Yuan and Koch, Reinhard and Bregovic, Robert and Gotchev, Atanas},
	Title = {Light Field Reconstruction Using Shearlet Transform in TensorFlow},
	Booktitle = {ICME Workshops},
	Year = {2019}
    }

This code was tested on an Ubuntu 18.04 system using Tensorflow 1.13.1.

## Introduction ##
ST is designed for reconstructing a
Densely-Sampled Light Field (DSLF) from a Sparsely-Sampled
Light Field (SSLF). It typically consists of pre-shearing, shearlet system construction, sparsity regularization and post-shearing. This TensorFlow implementation of ST focuses on sparsity regularization, which is composed of analysis transform, hard thresholding, synthesis transform and double overrelaxation. A dataflow graph of these four components are illustrated as below:

![alt text](Fig/sparse_regularization.png "sparse regularization")


A demo video of iterative sparse regularization with 30 iterations is shown as below:

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/5eQ-upVniYo/0.jpg)](http://www.youtube.com/watch?v=5eQ-upVniYo "iterative sparse regularization")




## Getting started ##
### Python requirements ###
```
conda install tensorflow-gpu
conda install -c conda-forge opencv
```
### Prepare datasets ###
Prepare the pre-sheared sparsely-sampled Epipolar-Plane Images (EPIs) and masks. Put them into folders like
```
./data/ssepi/dishes_r5
```
and name them like
```
0001_rgb.png, 0002_rgb.png, ...
0001_mask.png, 0002_mask.png, ...
```

For example, "0458_rgb.png" and "0458_mask.png" are presented as follows:

![alt text](Fig/0458_rgb.png "0458_rgb.png")

![alt text](Fig/0458_mask.png "0458_mask.png")

### Sparsity Regularization ### 
```
python validate.py --validate_path=./data/ssepi/ --save_path=./data/rec_dsepi --batch_size=4 --tensorboard_path=./tensorboard --shearlet_system_path=./model
```
The reconstructed EPI corresponding to "0458_rgb.png" is presented as follows:

![alt text](Fig/0458_rgb_reconstructed.png "0458_rgb_reconstructed.png")

Note that the shearlet system for the pre-sheared sparsely-sampled EPIs should be prepared in advance. It is placed in the folder "./model" by default. How to generate a specially-tailored shearlet system can be found in this [repository](http://www.cs.tut.fi/~vagharsh/EPISparseRec.html).

### Visualization ###
The visualization of the pipline of ST is performed using TensorBoard:
```
tensorboard --logdir=./tensorboard
```
Then visit 
```
http://localhost:6006
```

The dataflow graph is like

![alt text](Fig/dataflow.png "dataflow graph")

Intermediate results are like

![alt text](Fig/prediction.png "prediction")

## Reference ##
>  [1] S. Vagharshakyan, R. Bregovic, and A. Gotchev, “Light field
reconstruction using shearlet transform,” IEEE TPAMI, vol. 40,
no. 1, pp. 133–147, 2018.

> [2] S. Vagharshakyan, R. Bregovic, and A. Gotchev, “Accelerated
shearlet-domain light field reconstruction,” IEEE J-STSP, vol.
11, no. 7, pp. 1082–1091, 2017.
