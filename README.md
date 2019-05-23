Description
===========
More people are diagnosed with skin cancer each year in the U.S. than all other cancers combined, and with the development of neural networks, it is possible to employ machine learning techniques to process and analyze images of skin lesions to effectively detect skin diseases. In this work we will train and implement a VGG13 Convolutional Neural Network to extract features from a set of 9,758 pigmented skin lesions and correctly classify them as one of five diseases of benign or malignant cancer. The results need to be studied using various analytic metrics across iterations to determine the performance of the classifier. Creating a both robust and highly accurate architecture will give medical professionals and patients the tools to increase the survival rates for skin cancer.

Requirements
============
Install package 'keras ' as follow :
$ pip install --user keras

demo notebook along with its required files tested on DSMLP server with 'launch-tf-gpu.sh'

Code organization
=================
<pre>
demo_HAM10000_VGG13.ipynb -- Run a demo of our code 
        |
        |----- imports VGG-13 model trained from the training notebook below 
        |----- evalutes a example testset of 10 images picked from the complete dataset 
        
HAM10000_VGG13_training.ipynb -- Run the training of our VGG-13 model on complete HAM10000 dataset
        |
        |----- dataset preprocessing (reshaping, nomalization, one-hot labels)
        |----- VGG-13 model training (fitting) and plotting of training acc/loss history
        |----- LeNet-5 model training (fitting) and plotting of training acc/loss history as comparison
        
utility.py -- Implements some helper functions for training and displaying
        |
        |----- train-test separation on data and labels 
        |----- image normalization to (0,1)
        |----- convert to one-hot labels
        |----- plotting tools such as training accuracy line plots and example image displays

assets/vgg13_model.json -- Our VGG-13 network architecture definition
      /vgg13_model.h5   -- Trained parameters of VGG-13 network on HAM10000 datasets
      /images_test.npy  -- Zipped numpy binary file of arrays of test images
      /labels_test.npy  -- Zipped numpy binary file of arrays of test labels
</pre>
