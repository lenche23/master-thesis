import sys
from enum import Enum
from training import train_and_predict_decision_tree_regressor, train_and_predict_xgboost_regressor, train_and_predict_random_forest_regressor, train_and_predict_nn_regressor


class ModelType(Enum):
    decision_tree = train_and_predict_decision_tree_regressor
    neural_network = train_and_predict_nn_regressor
    xg_boost = train_and_predict_xgboost_regressor
    random_forest = train_and_predict_random_forest_regressor


def print_manual_and_exit():
    print("For regression, please choose learning method and model >> --it/--sst <decision_tree/neural_network/xg_boost/random_forest>")
    print("For time-series regression choose option >> --t")
    exit(1)


def parse_cli_args():
    if len(sys.argv) < 2:
        print_manual_and_exit()

    args = sys.argv
    skip_sst = False

    if args[1] == "--st":
        print("Running initial training only.")
        skip_sst = True
    elif args[1] != "--sst":
        print("Non-existent training method selected.")
        print_manual_and_exit()

    try:
        training_and_evaluation_function = ModelType.__dict__[args[2]]
    except KeyError:
        print("Non-existent model selected.")
        print_manual_and_exit()

    model_name = args[2].replace("_", " ").title()
    if model_name.startswith("Xg"):
        model_name = model_name.replace("Xg", "XG")

    include_period_data = False
    try:
        if args[3] == "--t":
            include_period_data = True
    except IndexError:
        pass

    return training_and_evaluation_function, model_name, skip_sst, include_period_data
