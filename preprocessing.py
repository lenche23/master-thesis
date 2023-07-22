def interpolate_missing_data(dataframe):
    hale_column = dataframe["Healthy life expectancy (HALE) at birth (years)"]
    dataframe = dataframe.drop(
        ["Healthy life expectancy (HALE) at birth (years)"], axis=1)
    dataframe = dataframe.interpolate(
        limit_direction='both', limit_area='inside', method="nearest").bfill().ffill()
    dataframe["HALE"] = hale_column

    return dataframe


def normalise_datasets(scaler, X_train, X_test, y_train, y_test):
    x_scaler = scaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = scaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler
