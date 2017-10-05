# Convolutional Neural Networks for single-digit image recognition

At this project three different Convolutional Neural Netwroks for the hand-written single-digit image recognition problem of the MNIST dataset are developed (the MNIST dataset can be found [here](https://www.kaggle.com/c/digit-recognizer/data)). Two different implementations take place in order to discern the effect of data augmentation on each model’s classification power: (1) training on pre-processed training data set, (2) training on pre-processed and augmented training data set. The newly created training dataset of “augmented” images has an increased sample size of 84,000 images. With a stochastic gradient descent optimization method and a categorical cross entropy loss function each model is compiled and then trained with the number of training epochs and the size of batches set to 10 and 200 respectively.

*** To read full report [click here](http://www.andreasgeorgopoulos.com/cnn-image-recognition/)
