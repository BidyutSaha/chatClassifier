from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


f = open("custom.csv", "w")
f.write("")
f.close()


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        f = open("custom.csv", "a")
        f.write("{},{},{},{},{}\n".format(
            epoch, logs["loss"], logs["accuracy"], logs["val_loss"], logs["val_accuracy"]))
        f.close()


adam = Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam"
)

classifier = Sequential()
x, y = 128, 128

classifier.add(Convolution2D(
    32, 3, 3, input_shape=(x, y, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
classifier.add(Dropout(0.5))


classifier.add(Flatten())

classifier.add(Dense(100, activation='relu'))
classifier.add(Dropout(0.5))


# classifier.add(Dense(100, activation='relu'))
# classifier.add(Dropout(0.4))


# classifier.add(Dense(100, activation='relu'))

classifier.add(Dense(3, activation='softmax'))

classifier.compile(
    optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

bs = 32
se = int(600/bs)


#########################################

inception = InceptionV3(
    input_shape=[128, 128, 3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in inception.layers:
    layer.trainable = False

x = Flatten()(inception.output)

prediction = Dense(3, activation='softmax')(x)
# create a model object
model = Model(inputs=inception.input, outputs=prediction)


model.compile(
    optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])


# checkpoint
filepath = "best_model_weight.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, CustomCallback()]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(x, y),
    batch_size=bs,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(x, y),
    batch_size=se,
    class_mode='categorical')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# model.fit_generator(
#     training_set,
#     steps_per_epoch=se,
#     epochs=20,
#     callbacks=callbacks_list,
#     validation_data=test_set,
#     # validation_steps=132,
#     verbose=1)

model_json = classifier.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)


# classifier.save("model3.h5")
