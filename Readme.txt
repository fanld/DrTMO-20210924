---------------------
Deep Reverse Tone Mapping (http://www.npal.cs.tsukuba.ac.jp/~endo/DrTMO)
README
---------------------

This code is an implementation of the deep reverse tone mapping method described in the following paper: 

Yuki Endo, Yoshihiro Kanamori, and Jun Mitani: "Deep Reverse Tone Mapping," ACM Transactions on Graphics (Proc. of SIGGRAPH ASIA 2017), 36, 6, Article 177 (November 2017), 10 pages. 

The code is written in Python2.7, and the following packages are used: 
1. Chainer (we used version 1.24.0). A flexible framework for deep learning. https://docs.chainer.org/en/v1.24.0/install.html
2. OpenCV (we used version 3.1.0). Image processing library. http://opencv.org
The other dependencies for the above libraries are also needed. 

How to use: 

$ python main.py
This command generates an HDR image from a single LDR image. The HDR images are generated from estimated bracketed images by using the two merge methods (MergeDebevec and MergeMertens), and other merge software such as Photoshop can also be used. The learned models are available from our project web page. Please put their directories in this directory with directory names "models". For more details, see the help messages using the --help option. 

$ python train_network.py
This command trains a network. The sample data for training are included in the directory named "training samples". The original training data used in the paper can be downloaded from the URLs shown in the paper. The trained model is saved with a specified name at each epoch. For more details, see the help messages using the --help option.  

You can use this code for scientific purposes only. Use in commercial projects and redistribution are not allowed without author's permission. Please cite our paper when using this code. 

=============================
Personal Contact Information
=============================
Email:
endo@cs.tsukuba.ac.jp (Yuki Endo)
kanamori@cs.tsukuba.ac.jp (Yoshihiro Kanamori)