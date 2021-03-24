#### Convolutional Neural Network model for classification between 2 classes : dogs and cats 
<img width="534" alt="cnn" src="https://user-images.githubusercontent.com/48187656/112380284-e44a1780-8cf1-11eb-996d-0362c35a06b3.png">

#### This is an usual architurecture, consisting of 2 convolutional layer, each one followed by a pooling layer(MaxPool). In order to obtain good performances, the model has a flatten layer and a dense layer with an arbitrary chosen units value of 128. The activation function use in the convolutional neural network is ReLu and in the Dense layer is sigmoid, because the model will classify 2 classes.
#### The training and test set are splitted in advance in 2 separate folders and there is a 3rd folder containing images to classify
####Tools needed: -python 3.8 -numpy 1.1.5(>=) -pandas 1.2.3(>=) -scikit-learn 0.24.1(>=) -tensorflow 2.4.1(>=)
####Model was trained and tested on MacBook Pro with CPU 2,7 GHz Dual Core Intel Core i5; Memory 8GB
####Dataset can be downloaded from : https://www.kaggle.com/c/dogs-vs-cats
