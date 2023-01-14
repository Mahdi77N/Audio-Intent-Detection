import librosa
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def read_data(csv_path="./dsl_data/development.csv"):
    """Read the data, based on the "development.csv" file. it will read the audio files, and return two pandas dataFrame, x and y.
       it will not normalize the data. also will not change the data to categorical.

    Args:
        csv_path (str, optional): path to the "development.csv" file. Defaults to "./dsl_data/development.csv".

    Returns:
        pandas dataFrame: the x has all the features, excluding "path", "action", "object" and "Id".
        the y has "action", "object", and "intention" which is a new feature by adding the other two columns.
        int: sample rate
    """

    development_pd = pd.read_csv(csv_path)  # .drop(["Id"], axis=1)
    development_pd["Signal"] = ""

    for index, row in development_pd.iterrows():
        wave, srr = librosa.load(row["path"], mono=True, sr=None)
        development_pd.at[index, "Signal"] = wave

    x = development_pd.drop(["Id", "action", "object", "path"], axis=1)
    y = development_pd[["action", "object"]]
    y["intention"] = development_pd["action"] + development_pd["object"]

    return x, y, srr


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


def convert_to_mfcc(x, srr, max_pad_len=215):
    """convert the audio signal to MFCC

    Args:
        x (pandas dataFrame): our data set
        srr (int): sample rate
        max_pad_len (int, optional): maximum width of MFCC. all MFCCs will be in width of max_pad_len. Defaults to 215.

    Returns:
        pandas dataFrame: our data set
    """
    for index, row in x.iterrows():
        mfcc = librosa.feature.mfcc(y=row["Signal"], sr=srr)
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")

        x.at[index, "Signal"] = mfcc

    return x


def convert_to_numpy(x, y, x_columns, y_columns):
    """Convert our data from pandas dataFrame to numpy array. Also, will drop all columns execpt the x_columns and y_columns

    Args:
        x (pandas dataFrame): our train set's features
        y (pandas dataFrame): train set's classes
        x_columns (array): features in train set that we want to train the model with them.
        y_columns (array): one of three classes, from "object", "action", and "intention".

    Returns:
        numpay arrays: will return x and y in numpy arrays and only with specified columns
    """
    x_temp = x[x_columns]
    x_temp_2 = np.array([x_temp.loc[i]["Signal"] for i in range(len(x_temp))])

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

    # tt = le.inverse_transform(t)
    # print(tt)


X, Y, srr = read_data()
X_trimed = trim_audios(X.copy(), top_db=10, hop_length=10)
X_mfcc = convert_to_mfcc(X_trimed.copy(), srr)

X_n, Y_n = convert_to_numpy(X_mfcc, Y, ["Signal"], ["intention"])
y_oneHpt, le = convert_y_to_oneHot(Y_n)
