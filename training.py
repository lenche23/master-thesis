from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


random_state = 4321


def train_and_predict_decision_tree_regressor(X_train, y_train, X_test, y_test, scaler_y, report_title="DTR model results"):
    regressor = DecisionTreeRegressor(
        random_state=random_state,
        criterion="absolute_error",
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        ccp_alpha=0.0
    )
    regressor = regressor.fit(
        X_train, y_train, sample_weight=None, check_input=True)

    predict_and_print_report(regressor, X_test, y_test, scaler_y, report_title)

    return regressor


def train_and_predict_xgboost_regressor(X_train, y_train, X_test, y_test, scaler_y, report_title="XgBoost model results"):
    regressor = XGBRegressor(
        random_state=random_state,
        n_estimators=1000,
        grow_policy="lossguide",
        learning_rate=0.015,
        booster="dart"
    )
    regressor = regressor.fit(X_train, y_train)

    predict_and_print_report(regressor, X_test, y_test, scaler_y, report_title)

    return regressor


def train_and_predict_random_forest_regressor(X_train, y_train, X_test, y_test, scaler_y, report_title="Random Forest model results"):
    regressor = RandomForestRegressor(
        max_depth=30,
        random_state=random_state,
        n_estimators=500,
        criterion="poisson",
        max_features=6
    )
    regressor = regressor.fit(X_train, y_train.ravel())

    predict_and_print_report(regressor, X_test, y_test, scaler_y, report_title)

    return regressor


def train_and_predict_nn_regressor(X_train, y_train, X_test, y_test, scaler_y, report_title="FCNN model results"):
    regressor = MLPRegressor(
        max_iter=500,
        random_state=random_state,
        hidden_layer_sizes=(300, 200, 250,),
        solver="lbfgs",
        learning_rate="adaptive"
    )
    regressor = regressor.fit(X_train, y_train.ravel())

    predict_and_print_report(regressor, X_test, y_test, scaler_y, report_title)

    return regressor


def predict_and_print_report(regression_model, X_test, y_test, scaler_y, report_title):
    predictions = regression_model.predict(X_test)

    y_true = scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(predictions.reshape(-1, 1))

    print("\n" + report_title)
    print("MSE score >>", mean_squared_error(y_true, y_pred))
    print("RMSE score >>", mean_squared_error(y_true, y_pred, squared=False))
    print("R2 score >>", r2_score(y_true, y_pred))
