![AI Capstone Project with Deep Learning](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/artificial_intelligence_capstone.jpg "AI Capstone Project with Deep Learning")

> :bulb: Notes on "AI Capstone Project with Deep Learning"


# Keras References
https://keras.io/api/data_loading/image/


# Load data in Keras

    # Libraries
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import skillsnetwork

    from PIL import Image

    # Download
    await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", overwrite=True)

    # List
    os.listdir('Negative')

    # Load
    negative_images = os.listdir('./Negative')
    negative_images

    # Sort
    negative_images.sort()
    negative_images

    # Open
    image_data = Image.open('./Negative/{}'.format(negative_images[0]))

    # Plot
    plt.imshow(image_data)

    # Loop
    negative_images_dir = ['./Negative/{}'.format(image) for image in negative_images]
    negative_images_dir


# Data preparation in Keras

    # Libraries
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import skillsnetwork
    import keras
    from keras.preprocessing.image import ImageDataGenerator

    # Download
    await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week2.zip",path = "./", overwrite=True)

    # List
    os.listdir('concrete_data_week2')

    # Data location
    dataset_dir = './concrete_data_week2'

    # instantiate image data generator
    data_generator = ImageDataGenerator()

    # loop through the images in batches
    image_generator = data_generator.flow_from_directory(
        dataset_dir,
        batch_size=4,
        class_mode='categorical',
        seed=24
        )

    # images with labels
    first_batch = image_generator.next()
    first_batch

    # images only
    first_batch_images = image_generator.next()[0]
    first_batch_images

    # labels only
    first_batch_labels = image_generator.next()[1]
    first_batch_labels

    # instantiate custom image data generator with rescale
    data_generator = ImageDataGenerator(
        rescale=1./255
    )

    image_generator = data_generator.flow_from_directory(
        dataset_dir,
        batch_size=4,
        class_mode='categorical',
        seed=24
        )

# Visualize Batches of Images

    data_generator = ImageDataGenerator()

    image_generator = data_generator.flow_from_directory(
        dataset_dir,
        batch_size=4,
        class_mode='categorical',
        seed=24
        )

    first_batch_images = image_generator.next()[0]
    second_batch_images = image_generator.next()[0]
    third_batch_images = image_generator.next()[0]
    fourth_batch_images = image_generator.next()[0]
    fifth_batch_images = image_generator.next()[0]


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

    ind = 0
    for ax1 in axs:
        for ax2 in ax1: 
            image_data = fifth_batch_images[ind].astype(np.uint8)
            ax2.imshow(image_data)
            ind += 1

    fig.suptitle('Third Batch of Concrete Images') 
    plt.show()

# Building a Classifier with ResNet50

# Libraries
import skillsnetwork 
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_inpu

# get the data
await skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip", overwrite=True)

# global constants
num_classes = 2
image_resize = 224
batch_size_training = 100
batch_size_validation = 100

# data generator
data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

# training images
train_generator = data_generator.flow_from_directory(
    'concrete_data_week3/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

# validation images
validation_generator = data_generator.flow_from_directory(
    'concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')

# create model
model = Sequential()

# add pretrained model
model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

# add output layer
model.add(Dense(num_classes, activation='softmax'))

# access layer
model.layers
model.layers[0].layers

# modify layers
model.layers[0].trainable = False

# model summary
model.summary()

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# definitions for training
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2

# training
fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)

# save model
model.save('classifier_resnet_model.h5')




