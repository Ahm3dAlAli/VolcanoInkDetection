import tensorflow as tf
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, concatenate, Input
from keras.models import Model

def unet_3d(input_shape=(128, 128, 64, 1), num_classes=1):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Middle
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Output layer
    output = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[output])
    return model

if __name__ == "__main__":
    model = unet_3d()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
