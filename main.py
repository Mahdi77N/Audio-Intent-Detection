import librosa
import pandas as pd


def read_data(csv_path="./dsl_data/development.csv"):
    """Read the data, based on the "development.csv" file. it will read the audio files, and return two pandas dataFrame, x and y.
       it will not normalize the data. also will not change the data to categorical.

    Args:
        csv_path (str, optional): path to the "development.csv" file. Defaults to "./dsl_data/development.csv".

    Returns:
        pandas dataFrame: the x has all the features, excluding "path", "action", "object" and "Id".
        the y has "action", "object", and "intention" which is a new feature by adding the other two columns.
    """

    development_pd = pd.read_csv(csv_path)  # .drop(["Id"], axis=1)
    development_pd["Signal"] = ""

    for index, row in development_pd.iterrows():
        wave, srr = librosa.load(row["path"], mono=True, sr=None)
        development_pd.at[index, "Signal"] = wave

    x = development_pd.drop(["Id", "action", "object", "path"], axis=1)
    y = development_pd[["action", "object"]]
    y["intention"] = development_pd["action"] + development_pd["object"]

    return x, y


X, Y = read_data()
