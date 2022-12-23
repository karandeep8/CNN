"""We are gonna build and train a CNN to recognize if there is a dog or cat in the image."""

# IMPORTING THE LIBRARIES
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

# DATA PREPROCESSING
# the Training Set
# We will apply transformation on all images of the training dataset in oder to avoid overfitting

# We are gonna do image augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # FeatureScaling
    # ImageTransformation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

"""We need to connect the datagen object to our training dataset"""
# the method flow_from_directory coonects the imageaugmentation tool to the training dataset
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',  # Path leading to the training dataset
    target_size=(64, 64),  # final size of the image when they will be feed to the CNN
    batch_size=32,  # how many images you want to their in the batch
    class_mode='binary'
)

# Preprocessing the Test Set
test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'

)

# BUILDING THE CNN
# Intialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1-Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Adding a second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# input_shape parameter is only eneterd when we create our first Convolutional layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())
# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Step - 5 Output Layer
cnn.add(tf.keras.layers.Dense(units=1,
                              activation='sigmoid'))  # since we are doing binary calssification so we need only one neuron to do it
# PART 3 - TRAINING THE CNN
# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
# PART 4 - MAKING A SINGLE PREDICTION
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)

training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
