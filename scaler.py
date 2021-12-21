from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np


def min_max_scale(train_x, test_x, k):
    """Scaling data using minMaxScaler fitting only on train_x

    Args:
        train_x: Training set without Class Target
        test_x: Testing set without Class Target

    Returns: scaled train_x, test_x

    """
    scalers = {}
    for i in range(train_x.shape[1]):
        scalers[i] = MinMaxScaler(feature_range=(-1, 1))
        train_x[:, i, :] = scalers[i].fit_transform(train_x[:, i, :])
    np.save("results\\scaler_" + str(k) + ".npy", scalers)
    if test_x is not None:
        for i in range(test_x.shape[1]):
            test_x[:, i, :] = scalers[i].transform(test_x[:, i, :])
    return train_x, test_x


def robust_scale(train_x, test_x, k):
    """Scaling data using RobustScaler fitting only on train_x

    Args:
        train_x: Training set without Class Target
        test_x: Testing set without Class Target

    Returns: scaled train_x, test_x

    """
    scalers = {}
    print(train_x.shape)
    for i in range(train_x.shape[1]):
        scalers[i] = RobustScaler(quantile_range=(25, 75), with_centering=False)
        train_x[:, i, :] = scalers[i].fit_transform(train_x[:, i, :])
    np.save("results\\scaler_" + str(k) + ".npy", scalers)
    if test_x is not None:
        for i in range(test_x.shape[1]):
            test_x[:, i, :] = scalers[i].transform(test_x[:, i, :])
    return train_x, test_x


def apply_scaler(test_x, scalers):
    for i in range(test_x.shape[1]):
        test_x[:, i, :] = scalers[i].transform(test_x[:, i, :])
    return test_x


def max_abs_scale(train_x, test_x, k):
    """Scaling data using MaxAbsScaler fitting only on train_x

    Args:
        train_x: Training set without Class Target
        test_x: Testing set without Class Target

    Returns: scaled train_x, test_x

    """
    scalers = {}
    for i in range(train_x.shape[1]):
        scalers[i] = MaxAbsScaler()
        train_x[:, i, :] = scalers[i].fit_transform(train_x[:, i, :])
    np.save("results\\scaler_" + str(k) + ".npy", scalers)
    if test_x is not None:
        for i in range(test_x.shape[1]):
            test_x[:, i, :] = scalers[i].transform(test_x[:, i, :])
    return train_x, test_x


def standard_scale(train_x, test_x, k):
    """Scaling data using StandardScaler fitting only on train_x

    Args:
        train_x: Training set without Class Target
        test_x: Testing set without Class Target

    Returns: scaled train_x, test_x

    """
    scalers = {}
    for i in range(train_x.shape[1]):
        scalers[i] = StandardScaler()
        train_x[:, i, :] = scalers[i].fit_transform(train_x[:, i, :])
    np.save("results\\scaler_" + str(k) + ".npy", scalers)
    if test_x is not None:
        for i in range(test_x.shape[1]):
            test_x[:, i, :] = scalers[i].transform(test_x[:, i, :])
    return train_x, test_x
