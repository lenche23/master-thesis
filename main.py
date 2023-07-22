from sklearn.preprocessing import MaxAbsScaler
from dataset_utilities import load_and_combine_datasets, generate_reports_for_dataframe, split_initial_training_and_unlabeled_data, split_dataframe_to_train_and_test, expand_training_dataset_with_unlabeled_data
from preprocessing import interpolate_missing_data, normalise_datasets
from cli_utilities import parse_cli_args


interpolate = True
generate_reports = True


def run_ssl_experiment(train_and_evaluate, model_name, include_period_data, skip_semi_supervised_training=True):
    full_dataframe = load_and_combine_datasets()

    if interpolate:
        nan_report_file = "finalDataNaNReport.txt"
        full_dataframe = interpolate_missing_data(full_dataframe)

    if generate_reports:
        generate_reports_for_dataframe(full_dataframe, nan_report_file)

    first_training_dataframe, unlabeled_data_dataframe = split_initial_training_and_unlabeled_data(
        full_dataframe)

    # Supervised training
    X_train, X_test, y_train, y_test = split_dataframe_to_train_and_test(
        first_training_dataframe, include_period=include_period_data)
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = normalise_datasets(
        MaxAbsScaler, X_train, X_test, y_train, y_test)
    regressor = train_and_evaluate(
        X_train, y_train, X_test, y_test, y_scaler, report_title=f"Initial training ({model_name})")

    if skip_semi_supervised_training:
        return

    # Semi-supervised training
    X_train, y_train = expand_training_dataset_with_unlabeled_data(unlabeled_data_dataframe,
                                                                   X_train, x_scaler,
                                                                   y_train, y_scaler,
                                                                   regressor,
                                                                   include_period=include_period_data)
    train_and_evaluate(X_train, y_train, X_test, y_test, y_scaler,
                       report_title=f"Semi-Supervised training ({model_name})")


if __name__ == "__main__":
    training_and_evaluation_function, model_name, skip_sst, include_period_data = parse_cli_args()

    run_ssl_experiment(training_and_evaluation_function, model_name,
                       include_period_data, skip_semi_supervised_training=skip_sst)
