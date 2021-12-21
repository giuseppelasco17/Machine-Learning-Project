import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, BatchNormalization

from models.inception import Classifier_INCEPTION
from models.mcdcnn import Classifier_MCDCNN

n_timestamps, n_features, n_outputs = 315, 3, 8


# convolutional neural network
def cnn_model(hp):
    if hp is not None:
        filt = hp.Int('filt', min_value=60, max_value=128, step=8)
        filt1 = hp.Int('filt1', min_value=40, max_value=64, step=4)
        filt2 = hp.Int('filt2', min_value=40, max_value=64, step=4)
        reg2 = hp.Choice('reg2', values=[50.0, 10.0, 5.0, 1.0])
        units = hp.Int('units', min_value=450, max_value=600, step=7)
        units2 = hp.Int('units2', min_value=250, max_value=450, step=5)
        lr = hp.Choice('lr', values=[2e-5, 2.5e-5, 3e-5])
    else:
        filt = 64
        filt1 = 64
        filt2 = 64
        reg2 = 1e-2
        units = 400
        units2 = 100
        lr = 2e-03

    model = Sequential()
    model.add(Conv1D(filters=filt, kernel_size=10, activation='relu', input_shape=(n_timestamps, n_features),
                     kernel_regularizer=keras.regularizers.l2(reg2)))
    model.add(MaxPooling1D(pool_size=8, strides=5, padding='same'))
    model.add(Conv1D(filters=filt1, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=8, strides=5, padding='same'))
    model.add(Conv1D(filters=filt2, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


# multi-channel deep convolutional neural network
def mcdcnn_model(hp):
    model = Classifier_MCDCNN(output_directory="my_dir", input_shape=(n_timestamps, n_features), nb_classes=n_outputs,
                              verbose=False, build=True)
    return model


# inception
def inception_model(hp):
    if hp is not None:
        filt = hp.Int('filt', min_value=60, max_value=128, step=8)
        depth = hp.Int('depth', min_value=40, max_value=64, step=4)
        kernel_size = hp.Int('kernel_size', min_value=40, max_value=64, step=4)
        use_residual = hp.Choice('use_residual', values=[True, False])
        use_bottleneck = hp.Choice('use_bottleneck', values=[True, False])
    else:
        filt = 56
        depth = 5
        kernel_size = 41
        use_residual = True
        use_bottleneck = True

    model = Classifier_INCEPTION(output_directory="my_dir", input_shape=(n_timestamps, n_features),
                                 nb_classes=n_outputs, depth=depth, verbose=False, build=False, batch_size=64,
                                 nb_filters=filt, use_residual=use_residual, use_bottleneck=use_bottleneck,
                                 kernel_size=kernel_size, nb_epochs=1500)
    mod = model.build_model((n_timestamps, n_features), n_outputs)
    return mod


# fully connected neural network
def fcn_model(hp):
    if hp is not None:
        filt = hp.Int('filt', min_value=60, max_value=128, step=8)
        filt1 = hp.Int('filt1', min_value=40, max_value=64, step=4)
        filt2 = hp.Int('filt2', min_value=40, max_value=64, step=4)
        lr = hp.Choice('lr', values=[2e-5, 2.5e-5, 3e-5])
    else:
        filt = 128
        filt1 = 256
        filt2 = 128
        lr = 1.e-3

    input_shape = (n_timestamps, n_features)
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=filt, kernel_size=8, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv2 = keras.layers.Conv1D(filters=filt1, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv3 = keras.layers.Conv1D(filters=filt2, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(n_outputs, activation="softmax")(gap)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
