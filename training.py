import timeit
import pandas as pd
from matplotlib import pyplot as plt
from numpy import dstack
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from models.models import *
import tensorflow as tf
from scaler import standard_scale, min_max_scale, max_abs_scale, \
    robust_scale

scale_type = "ROBUST"
model_n = "cnn"
filename_train_x = "train_gesture_x.csv"
filename_train_y = "train_gesture_y.csv"
filename_train_z = "train_gesture_z.csv"
filename_train_label = "train_label.csv"


# retrieve data
def retrieve_data(path_x, path_y, path_z, path_label):
    # load data
    df_x = read_csv(path_x, header=None).values
    df_y = read_csv(path_y, header=None).values
    df_z = read_csv(path_z, header=None).values
    df_label = read_csv(path_label, header=None).values

    df_x, df_y, df_z, df_label = fill_na(df_x, df_y, df_z, df_label)

    list_df = [df_x, df_y, df_z]
    dataset_x = dstack(list_df)

    return dataset_x, df_label


# fill NaN values
def fill_na(df_x, df_y, df_z, df_label):
    df_x = pd.DataFrame(df_x)
    df_y = pd.DataFrame(df_y)
    df_z = pd.DataFrame(df_z)
    df_label = pd.DataFrame(df_label)

    df_x = df_x.fillna(method="ffill", axis=1)
    df_x = df_x.fillna(method="bfill", axis=1)
    boolean_mask_x = df_x.isna()

    df_y = df_y.fillna(method="ffill", axis=1)
    df_y = df_y.fillna(method="bfill", axis=1)
    boolean_mask_y = df_y.isna()

    df_z = df_z.fillna(method="ffill", axis=1)
    df_z = df_z.fillna(method="bfill", axis=1)
    boolean_mask_z = df_z.isna()

    boolean_mask_label = df_label.isna()

    index = []
    for i in range(0, boolean_mask_x.shape[0]):
        if boolean_mask_x[0][i]:
            if i not in index:
                index.append(i)
        if boolean_mask_y[0][i]:
            if i not in index:
                index.append(i)
        if boolean_mask_z[0][i]:
            if i not in index:
                index.append(i)
        if boolean_mask_label[0][i]:
            if i not in index:
                index.append(i)

    print("Rows index eliminated: ", index)

    for i in index:
        df_x = df_x.drop(i, axis=0)
        df_y = df_y.drop(i, axis=0)
        df_z = df_z.drop(i, axis=0)
        df_label = df_label.drop(i, axis=0)

    return df_x, df_y, df_z, df_label


# scaling
def scaling(train_x, test_x, scale_type, i):
    if scale_type == "STANDARD":
        train_x, test_x = standard_scale(train_x, test_x, i)

    if scale_type == "MINMAX":
        train_x, test_x = min_max_scale(train_x, test_x, i)

    if scale_type == "MAX_ABS":
        train_x, test_x = max_abs_scale(train_x, test_x, i)

    if scale_type == "ROBUST":
        train_x, test_x = robust_scale(train_x, test_x, i)

    return train_x, test_x


# retrieve data and pre-processing
def prepare_data(k_fold):
    test_x, test_y = None, None
    train_x, train_y = retrieve_data(filename_train_x, filename_train_y, filename_train_z, filename_train_label)

    # split training, test in 80/20
    if not k_fold:
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=.2)

        # scaling
        train_x, test_x = scaling(train_x, test_x, scale_type, "no_k_fold")

        # one-hot encoding
        train_y = tf.keras.utils.to_categorical(train_y, 8, dtype='float64')
        test_y = tf.keras.utils.to_categorical(test_y, 8, dtype='float64')
    else:
        # shuffling
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=.2, shuffle=True)
        train_x = np.concatenate((train_x, test_x))
        train_y = np.concatenate((train_y, test_y))

    return train_x, train_y, test_x, test_y


def select_model(model_name, hp=None):
    if model_name == "cnn":
        return cnn_model(hp)
    elif model_name == "inception":
        return inception_model(hp)
    elif model_name == "mcdcnn":
        return mcdcnn_model(hp)
    elif model_name == "fcn":
        return fcn_model(hp)


# training models
def training(train_x, train_y, test_x, test_y, k_fold):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                                patience=30)

    # build model
    model = select_model(model_n)
    if not k_fold:
        # split train, validation in 80/20
        train_x, val_test_x, train_y, val_test_y = train_test_split(train_x, train_y, test_size=0.2)
        history = model.fit(train_x, train_y, epochs=600, batch_size=128, verbose=1,
                            callbacks=[callback], validation_data=(val_test_x, val_test_y))
    else:
        history = model.fit(train_x, train_y, epochs=600, batch_size=128, verbose=1, callbacks=[callback],
                            validation_data=(test_x, test_y))
        # evaluate model
        val_loss, val_accuracy = model.evaluate(test_x, test_y, verbose=1)
        print("\n Test Loss: ", val_loss)
        print("\n Test Accuracy: ", val_accuracy)
        return val_loss, val_accuracy, model

    # evaluate model
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=1)
    print("\n Test Loss: ", test_loss)
    print("\n Test Accuracy: ", test_accuracy)

    # save model
    model.save("results\\my_h5_model.h5")
    make_plot(history)
    return test_loss, test_accuracy, model


# plotting loss and accuracy
def make_plot(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# Stratified K Fold Validation
def k_fold_validation(x_train, y_train, n_k):
    k_fold_ = KFold(n_splits=n_k)
    scores_val = []
    models_val = []
    n_k_fold = 1
    array = np.zeros(x_train.shape[0])
    # sum of accuracy for mean
    tot = 0.0

    for train, test in k_fold_.split(array):

        array1_x = np.array(x_train[:test[0]])
        array2_x = np.array(x_train[test[-1]:])
        array_x = np.concatenate((array1_x, array2_x))

        array1_y = np.array(y_train[:test[0]])
        array2_y = np.array(y_train[test[-1]:])
        array_y = np.concatenate((array1_y, array2_y))

        train_x_, val_x = array_x, x_train[test[0]:test[-1]]
        train_y_, val_y = array_y, y_train[test[0]:test[-1]]

        train_y_encoding = tf.keras.utils.to_categorical(train_y_, 8, dtype='float64')
        test_y_encoding = tf.keras.utils.to_categorical(val_y, 8, dtype='float64')
        print("k fold: ", n_k_fold)
        train_x_as, test_x_as = scaling(train_x_, val_x, scale_type, n_k_fold)

        val_loss, val_accuracy, model_ = training(train_x_as, train_y_encoding, test_x_as, test_y_encoding, True)

        scores_val.append([val_loss, val_accuracy])
        models_val.append(model_)

        # save model
        model_.save("results\\" + get_model_name(n_k_fold))
        f = open("results\\accuracy_val.txt", "a")
        f.write(str(n_k_fold) + ' VAL: '+str(val_accuracy)+ ' LOSS: ' + str(val_loss) + '\n')
        f.close()
        n_k_fold = n_k_fold + 1
        tot = tot + val_accuracy
        keras.backend.clear_session()

    f = open("results\\accuracy_val.txt", "a")
    f.write("Mean" + ': ' + str(tot / n_k) + '\n\n')
    f.close()
    return scores_val, models_val


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


if __name__ == '__main__':
    tic = timeit.default_timer()

    k_fold = True
    if not k_fold:
        train_x, train_y, test_x, test_y = prepare_data(k_fold)
        test_loss, test_accuracy, model = training(train_x, train_y, test_x, test_y, k_fold)
    else:
        train_x, train_y, test_x, test_y = prepare_data(k_fold)
        k = 8
        scores, models = k_fold_validation(train_x, train_y, k)

    toc = timeit.default_timer()
    print("Elapsed: " + str(toc - tic))
