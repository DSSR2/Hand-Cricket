from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, UpSampling2D, Activation
from keras.applications import VGG16
import numpy as np
import cv2


class Fingertips:
    def __init__(self, weights):
        self.model = model()
        self.model.load_weights(weights)

    def classify(self, image):
        image = np.asarray(image)
        image = cv2.resize(image, (128, 128))
        image = image.astype('float32')
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        probability, position = self.model.predict(image)
        self.prob = probability[0]
        self.position = position[0]
        return self.prob, self.position

def model():
    model = VGG16(include_top=False, input_shape=(128, 128, 3))
    x = model.output

    y = x
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    probability = Dense(5, activation='sigmoid', name='confidence')(x)

    y = UpSampling2D((3, 3))(y)
    y = Activation('relu')(y)
    y = Conv2D(1, (3, 3), activation='linear')(y)
    position = Reshape(target_shape=(10, 10), name='position')(y)
    model = Model(input=model.input, outputs=[probability, position])
    return model

if __name__ == '__main__':
    model = model()
    model.summary()