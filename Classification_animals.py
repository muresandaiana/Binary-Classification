import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#trainingset and testset are in 2 separate folders
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# create the model
CNNmodel = tf.keras.models.Sequential()
# first convolutional layer with relu
CNNmodel.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# using maxpooling layer
CNNmodel.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# second convolutional layer with relu
CNNmodel.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# using maxpooling
CNNmodel.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# add flatten layer
CNNmodel.add(tf.keras.layers.Flatten())
#fully connected to get a good classification
CNNmodel.add(tf.keras.layers.Dense(units=128, activation='relu'))
#add output layer 1 unit because it will be either cat or dog
CNNmodel.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#training the model
CNNmodel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
CNNmodel.fit(x = training_set, validation_data = test_set, epochs = 50)
# make a prediction from
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/predictions/cat_or_dog_4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = CNNmodel.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)