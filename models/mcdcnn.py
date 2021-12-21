import tensorflow.keras as keras
import tensorflow as tf


# mcscnn classifier
class Classifier_MCDCNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'
        if n_t < 60:
            padding = 'same'
        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t, 1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732, activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005),
                      metrics=['accuracy'])

        return model

    def prepare_input(self, x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:, :, i:i + 1])

        return new_x

    def save(self, path):
        self.model.save(path)

    def fit(self, train_x, train_y, epochs=600, batch_size=256, verbose=1,
            callbacks=None, validation_data=None):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        mini_batch_size = batch_size
        nb_epochs = epochs
        train_x = self.prepare_input(train_x)
        train_x_val = self.prepare_input(validation_data[0])
        hist = self.model.fit(train_x, train_y, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=verbose, validation_data=(train_x_val, validation_data[1]),
                              callbacks=callbacks)
        return hist

    def evaluate(self, test_x, test_y, verbose=1):
        test_x = self.prepare_input(test_x)
        loss, accuracy = self.model.evaluate(test_x, test_y, verbose)
        return loss, accuracy
