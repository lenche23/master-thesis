import numpy as np
import pandas as pd
import datetime as dt
import os
from sklearn.model_selection import train_test_split
from training import random_state
from pandas.tseries.offsets import YearEnd


datasets_directory = "./datasets"
pd.options.mode.chained_assignment = None


def load_and_combine_datasets():
    full_dataframe = None
    for filename in os.listdir(datasets_directory):
        f = os.path.join(datasets_directory, filename)
        if os.path.isfile(f):
            df = pd.read_csv(f)
            df.rename(
                columns={'First Tooltip': df["Indicator"][0]}, inplace=True)
            df = df[df['Period'] >= 2000]

            try:
                if df["Dim1"][0] == "Both sexes":
                    df = df[df['Dim1'] == "Both sexes"]
                elif df["Dim1"][0] == "Total":
                    df = df[df['Dim1'] == "Total"]

                df.drop(["Dim1"], axis=1, inplace=True)
            except KeyError:
                pass

            df.drop(["Indicator"], axis=1, inplace=True)

            if full_dataframe is None:
                full_dataframe = df
            else:
                full_dataframe = pd.merge(full_dataframe, df, on=[
                                          "Location", "Period"], how='outer')

    full_dataframe = full_dataframe.sort_values(by=["Location", "Period"])

    for i in range(len(full_dataframe)):
        for j in range(1, 22):
            data = str(full_dataframe.iloc[i, j])
            if " " in data:
                full_dataframe.iloc[i, j] = float(data.split(" ")[0])

    return full_dataframe


def split_initial_training_and_unlabeled_data(dataframe):
    first_training_dataframe = dataframe.dropna()
    unlabeled_data_dataframe = dataframe[dataframe["HALE"].isna()]

    return first_training_dataframe, unlabeled_data_dataframe


def generate_reports_for_dataframe(dataframe, nan_report_file="finalDataNaNReportRaw.txt"):
    with open("finalDataPreview.txt", "w") as f1:
        f1.write(dataframe.to_string())
    with open(nan_report_file, "w") as f2:
        f2.write(str(dataframe.isna().sum()) + "\n")
        f2.write(str(len(dataframe)))


def split_dataframe_to_train_and_test(dataframe, include_period=False):
    dataframe = dataframe.drop(["Location"], axis=1)
    dataframe["Period"] = pd.to_datetime(
        dataframe["Period"], format='%Y') + YearEnd(1)
    dataframe['Period'] = dataframe['Period'].map(dt.datetime.toordinal)

    y = dataframe["HALE"]
    X = dataframe.drop(["HALE"], axis=1)
    if not include_period:
        X = X.drop(["Period"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().reshape(-1, 1), y_test.to_numpy().reshape(-1, 1)


def expand_training_dataset_with_unlabeled_data(unlabeled_data_df, X_train, x_scaler, y_train, y_scaler, trained_regression_model, include_period=False):
    unlabeled_data_df = unlabeled_data_df.drop(["HALE"], axis=1)
    unlabeled_data_df = unlabeled_data_df.drop(["Location"], axis=1)

    if not include_period:
        unlabeled_data_df = unlabeled_data_df.drop(["Period"], axis=1)

    X_unlabeled = x_scaler.transform(unlabeled_data_df.to_numpy())
    label_predictions = trained_regression_model.predict(X_unlabeled)
    label_predictions = y_scaler.inverse_transform(
        label_predictions.reshape(-1, 1))

    X_train = np.concatenate(
        [X_train, x_scaler.transform(unlabeled_data_df.to_numpy())])
    y_train = np.concatenate([y_train, y_scaler.transform(label_predictions)])

    return X_train, y_train
