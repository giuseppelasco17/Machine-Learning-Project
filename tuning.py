import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split

from training import prepare_data
from models import models


# keras tuning
def keras_tuner_tuning(train_x, train_y, test_x, test_y):
    tuner = kt.Hyperband(models.inception_model,
                         objective='val_accuracy',
                         max_epochs=600,
                         factor=3,
                         seed=2,
                         directory='my_dir',
                         project_name='ml')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    train_x, val_test_x, train_y, val_test_y = train_test_split(train_x, train_y, test_size=.2)

    tuner.search(train_x, train_y, epochs=600, callbacks=[stop_early], validation_data=(val_test_x, val_test_y),
                 batch_size=128)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]

    # print(best_hps.get())
    # print(f"""
    # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # layer are [{best_hps.get('units')}, {best_hps.get('units1')}] and the optimal learning rate for the optimizer
    # is {best_hps.get('lr')}, reg is [{best_hps.get('reg')}, {best_hps.get('reg2')}], filters are [{best_hps.get('filt')}
    # , {best_hps.get('filt1')}, {best_hps.get('filt2')}], drop is {best_hps.get('drop')}. """)

    # fit
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_x, train_y, epochs=600, callbacks=[stop_early], validation_data=(val_test_x, val_test_y))

    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print("\n Test Loss: ", test_loss)
    print("\n Test Accuracy: ", test_accuracy)


if __name__ == '__main__':
    trainX, trainY, testX, testY = prepare_data(False)
    # tuning
    keras_tuner_tuning(trainX, trainY, testX, testY)
