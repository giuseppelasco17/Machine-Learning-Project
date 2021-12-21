import keras
import tensorflow as tf
import training
from jproperties import Properties
import numpy as np
from scaler import apply_scaler

CONFIG_PATH = 'config.properties'


def main(i, k_fold):
    p = Properties()
    with open(CONFIG_PATH, 'rb') as f:
        p.load(f, 'utf-8')
    filename_test_x = p["test_gesture_x"].data
    filename_test_y = p["test_gesture_y"].data
    filename_test_z = p["test_gesture_z"].data
    filename_test_label = p["test_label_csv"].data

    # retrieve data
    test_x, test_y = training.retrieve_data(filename_test_x, filename_test_y, filename_test_z, filename_test_label)

    # scaling test
    if not k_fold:
        scalers = np.load("results\\scaler_best_model" + ".npy", allow_pickle=True).item()
    else:
        scalers = np.load("results\\scaler_" + str(i) + ".npy", allow_pickle=True).item()

    test_x = apply_scaler(test_x, scalers)

    # one-hot encoding test
    test_y = tf.keras.utils.to_categorical(test_y, 8)

    # load best model
    if not k_fold:
        reconstructed_model = keras.models.load_model("results\\best_model.h5")
    else:
        reconstructed_model = keras.models.load_model("results\\model_"+str(i)+".h5")

    # evaluate on test
    test_loss, test_accuracy = reconstructed_model.evaluate(test_x, test_y, verbose=1)
    print("\n Test Loss: ", test_loss)
    print("\n Test Accuracy: ", test_accuracy)


if __name__ == '__main__':
    # for i in range(1, 11):
    #     print(i)
    #     main(i, True)
    main(0, False)
