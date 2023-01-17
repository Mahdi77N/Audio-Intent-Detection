from math import ceil

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint


def read_data(csv_path="./dsl_data/development.csv", data="train"):
    """Read the data, based on the the CSV file given. it can be the train or test data.
    in case of test data, only X and sample rate will return. in case of train data, X,Y, and sample rate will return.
    X and Y are in Pandas DataFrame format.
    it will not normalize the data. also will not change the data to categorical.

    Args:
        csv_path (str, optional): path to the CSV file. Defaults to "./dsl_data/development.csv".
        data (str, optional): the the data is train data, set to train. else, it will be test data. Defaults to "train".

    Returns:
        pandas dataFrame:  In case of Train data, the X has all the features, excluding "path", "action", "object" and "Id".
        the y has "action", "object", and "intention" which is a new feature by adding the other two columns.
        in case of Test data, the X will be test data without "path" column.
        int: sample rate
    """

    development_pd = pd.read_csv(csv_path)
    development_pd["Signal"] = ""

    for index, row in development_pd.iterrows():
        wave, srr = librosa.load(row["path"], mono=True, sr=None)
        development_pd.at[index, "Signal"] = wave

    if data == "train":
        x = development_pd.drop(["Id", "action", "object", "path"], axis=1)
        y = development_pd[["action", "object"]]
        y["intention"] = development_pd["action"] + development_pd["object"]
        return x, y, srr

    else:
        x = development_pd.drop(["path"], axis=1)
        return x, srr


def trim_audios(x, top_db=30, hop_length=50):
    """Trim the audios in the X.

    Args:
        x (pandas dataFrame): DataFrame consist of features, incuding the audios
        top_db (int, optional): audio's db less than this number will consider as silent. Defaults to 30.
        hop_length (int, optional): sensitivity of triming. Defaults to 50.

    Returns:
        pandas dataFrame: features DataFrame, with trimed audio in the "Signal" column.
    """
    for index, row in x.iterrows():
        wave, i = librosa.effects.trim(
            row["Signal"], top_db=top_db, hop_length=hop_length
        )
        x.at[index, "Signal"] = wave

    return x


def convert_to_mfcc(x, srr, max_audio_len=3):
    """convert the audio signal to MFCC

    Args:
        x (pandas dataFrame): our data set
        srr (int): sample rate
        max_audio_len (int, optional): maximum audio length in data. used for set the maximum width for MFCCs. Defaults to 3.

    Returns:
        pandas dataFrame: our data set
    """

    for index, row in x.iterrows():
        mfcc = librosa.feature.mfcc(row["Signal"], sr=srr)
        max_pad_len = (max_audio_len * srr) / 512
        pad_width = ceil(max_pad_len - mfcc.shape[1])

        if mfcc.shape[1] <= max_pad_len:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, 0 : ceil(max_pad_len)]
        x.at[index, "Signal"] = np.array(mfcc)

    return x


def convert_to_numpy(x, x_columns, y=None, y_columns=None, data="train"):
    """Convert our data from pandas dataFrame to numpy array. Also, will drop all columns execpt the x_columns and y_columns

    Args:
        x (pandas dataFrame): our train set's features
        y (pandas dataFrame, optinal): train set's classes. Ignore if your data is test data.
        x_columns (array): features in train set that we want to train the model with them.
        y_columns (array, optional): one of three classes, from "object", "action", and "intention". Ignore if your data is test data.
        data (str, optional): the the data is train data, set to train. else, it will be test data. Defaults to "train".

    Returns:
        numpay arrays: will return x and y in numpy arrays and only with specified columns for train data, and only X for test data.
    """
    x_temp = x[x_columns]
    x_temp_2 = np.array([x_temp.iloc[i]["Signal"] for i in range(len(x_temp))])
    print(type(y))
    if data != "train":
        return x_temp_2

    else:
        y_temp = y[y_columns]
        y_temp = y_temp.to_numpy()

        return x_temp_2, y_temp


def convert_y_to_oneHot(y):
    """encode the classes from strings to oneHot

    Args:
        y (numpy array): train set's classes

    Returns:
        numpy array: train set's classes in oneHot format.
        Label encoder Object: in order to be able to inverse oneHot to strings when we have the results.

    """
    le = LabelEncoder()
    temp = le.fit_transform(y)
    y_oneHot = to_categorical(temp, num_classes=max(temp) + 1)

    return y_oneHot, le


def get_cnn_model(input_shape, num_classes):
    model = Sequential()

    model.add(
        Conv2D(
            32,
            kernel_size=(2, 2),
            activation="relu",
            input_shape=input_shape,
            strides=(1, 2),
            padding="same",
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            48, kernel_size=(2, 2), activation="relu", strides=(1, 2), padding="same"
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            120, kernel_size=(2, 2), activation="relu", strides=(1, 2), padding="same"
        )
    )
    model.add(BatchNormalization())

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.15))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def remove_outliers(x, y, max_length=3):
    """will remove the outliers from the dataset

    Args:
        x (Pandas DataFrame): our dataset
        y (Pandas DataFrame): our classes
        max_length (int, optional): maximum audio lengths. the function will remove audios with length greated than this. Defaults to 3.

    Returns:
        Pandas DataFrame: our dataset and classes after removing outliers
    """
    t = []
    for index, row in X_trimed.iterrows():
        t.append(librosa.get_duration(row["Signal"], sr=srr))

    ouliers_index = np.where(np.array(t) > max_length)

    x_removed_ouliers = x.copy().drop(ouliers_index[0], axis=0)
    y_removed_ouliers = y.copy().drop(ouliers_index[0], axis=0)

    return x_removed_ouliers, y_removed_ouliers


def test_output_csv(
    saved_model_path,
    output_csv_path,
    evaluate_csv_path,
    le,
    max_audio_length,
    top_db=10,
    hop_length=10,
):
    """generate output csv file for the test dataset

    Args:
        saved_model_path (str): path to the saved model, like H5 or HDF5 file.
        output_csv_path (csv): path for the new output CSV file.
        evaluate_csv_path (str): path to the CSV file of the test data.
        le (Label encoder Object): Label encoder Object for reverse the encoding from str to oneHot
        max_audio_length (int): maximum audio length between our dataset in TRAIN set.
        top_db (int, optional): audio's db less than this number will consider as silent. Defaults to 30.
        hop_length (int, optional): sensitivity of triming. Defaults to 50.
    """
    model = load_model(saved_model_path)
    x_test, srr_test = read_data(csv_path=evaluate_csv_path, data="Not_train")
    X_test_trimed = trim_audios(x_test, top_db=top_db, hop_length=hop_length)
    X_mfcc_test = convert_to_mfcc(
        X_test_trimed.copy(), srr_test, max_audio_len=max_audio_length
    )
    X_n_test = convert_to_numpy(X_mfcc_test, ["Signal"], data="not train")
    predicted_y = model.predict(X_n_test)
    predicted_y = np.argmax(predicted_y, axis=1)
    predicted_y_string = le.inverse_transform(predicted_y)
    df = pd.DataFrame(predicted_y_string)
    df["Id"] = df.index
    df = df.iloc[:, [1, 0]]
    df.rename(columns={0: "Predicted"}, inplace=True)
    df.to_csv(output_csv_path, index=False)

    return 0


max_lenght = 2.5

X, Y, srr = read_data()
X_trimed = trim_audios(X, top_db=10, hop_length=10)

X_no_outliers, y_no_outliers = remove_outliers(X_trimed, Y, max_length=max_lenght)
X_mfcc = convert_to_mfcc(X_no_outliers, srr, max_audio_len=max_lenght)

X_n, Y_n = convert_to_numpy(
    x=X_mfcc,
    x_columns=["Signal"],
    y=y_no_outliers,
    y_columns=["intention"],
    data="train",
)
y_oneHot, le = convert_y_to_oneHot(Y_n)

dim_1 = X_n.shape[1]
dim_2 = X_n.shape[2]
channels = 1
classes = y_oneHot.shape[1]
X_n = X_n.reshape((X_n.shape[0], dim_1, dim_2, channels))
input_shape = (dim_1, dim_2, channels)

cnn_model = get_cnn_model(input_shape, classes)
print(cnn_model.summary())

keras_callback = keras.callbacks.TensorBoard(
    log_dir="./Graph", histogram_freq=1, write_graph=True, write_images=True
)

checkpointer = ModelCheckpoint(
    filepath="./audio_classification.hdf5", verbose=1, save_best_only=True
)

h = cnn_model.fit(
    X_n,
    y_oneHot,
    batch_size=8,
    epochs=200,
    verbose=1,
    validation_split=0.20,
    callbacks=[keras_callback, checkpointer],
)


# USE THIS FOR LOAD and GERATE OUTPUT
##########################################################
# test_output_csv(
#     saved_model_path="audio_classification.hdf5",
#     output_csv_path="output.csv",
#     evaluate_csv_path="./dsl_data/evaluation.csv",
#     max_audio_length=max_lenght,
#     le=le,
# )
############################################################
