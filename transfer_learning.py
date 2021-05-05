import tensorflow as tf
import tensorflow_datasets as tfds
#import tensorflow_hub as hub
import numpy as np
#import matplotlib.pyplot as plt

(ds_train,ds_test,ds_valid),info = tfds.load(
    'curated_breast_imaging_ddsm/patches', split=['train','test','validation'],
    shuffle_files=True,
    with_info=True)
num_classes = info.features['label'].num_classes

IMG_SIZE = (224, 224)
IMG_SHAPE = (IMG_SIZE + (3,))
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 100
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 10 #TRAIN_LENGTH // BATCH_SIZE

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMG_SIZE)
    input_image = tf.image.grayscale_to_rgb(input_image) # if using pretrained models
    return input_image,datapoint['label']

train_dataset = ds_train.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

valid_dataset = ds_valid.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(BATCH_SIZE).cache()
valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# Freeze convolution base
base_model.trainable = False

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  #tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
])
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_classes)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
#x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

#base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())

history = model.fit(train_dataset,
                    epochs=10,
                    steps_per_epoch=10,
                    validation_data=valid_dataset)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
