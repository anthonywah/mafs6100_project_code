import timeit
from utilities.data_factory import *
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras import optimizers


class Model:
    def __init__(self, input_train_x_df, input_train_y_series, input_model):
        self.train_x_df = input_train_x_df
        self.train_y_series = input_train_y_series
        self.target_model = input_model
        self.train_result = None

    def train(self):
        log_info(f'Start training on samples with dim {self.train_x_df.shape}')
        start = timeit.default_timer()
        self.train_result = self.target_model.fit(self.train_x_df, self.train_y_series)
        log_info(f'Finished training in {timeit.default_timer() - start:.2f}s')
        return

    def predict(self, input_test_x_df):
        pred = self.train_result.predict(input_test_x_df)
        log_info(f'Got {input_test_x_df.shape[0]} predictions')
        return pred


class LSTMWrapper:
    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(input_dim=1, units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(input_dim=50, units=100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(input_dim=100, units=200, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(300, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100))
        self.model.add(Dense(units=1))
        self.model.add(Activation('relu'))
        tf.config.experimental_run_functions_eagerly(True)
        self.model.compile(loss='mean_squared_error', optimizer='Adam')
        self.model.summary()

    def fit(self, input_x, input_y):
        self.model.fit(input_x, input_y,
                       epochs=2,
                       batch_size=4096,
                       verbose=1)

    def predict(self, input_x):
        return self.model.predict(input_x)


class RandomForestClassifierWrapper:
    def __init__(self):
        grid = dict()
        grid['n_estimators'] = [50]
        grid['max_features'] = ['auto']
        grid['max_depth'] = [6]
        grid['min_samples_split'] = [5]
        grid['min_samples_leaf'] = [5]
        rf = RandomForestClassifier()
        self.model = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=5)

    def fit(self, input_x, input_y):
        self.model.fit(input_x, input_y,
                       epochs=2,
                       batch_size=4096,
                       verbose=1)

    def predict(self, input_x):
        return self.model.predict(input_x)


class GradientBoostingClassifierWrapper:
    def __init__(self):
        grid = dict()
        grid['n_estimators'] = [50]
        grid['max_features'] = ['auto']
        grid['max_depth'] = [6]
        grid['min_samples_split'] = [5]
        grid['min_samples_leaf'] = [5]
        rf = GradientBoostingClassifier()
        self.model = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=5)

    def fit(self, input_x, input_y):
        self.model.fit(input_x, input_y,
                       epochs=2,
                       batch_size=4096,
                       verbose=1)

    def predict(self, input_x):
        return self.model.predict(input_x)


if __name__ == '__main__':

    # Usages:
    eg_data_fac = DataFactory()
    eg_train_df = eg_data_fac.get_train_df()
    eg_test_df = eg_data_fac.get_test_df()
    eg_features_col = get_raw_features()
    eg_labels_col = get_raw_labels()
    eg_target_label = eg_labels_col[2]
    minmax_scaler = get_scaler(eg_train_df[eg_features_col], input_scaler_type='MinMax')
    standard_scaler = get_scaler(eg_train_df[eg_features_col], input_scaler_type='Standard')

    # Lasso model
    model = Lasso(alpha=0.03)
    lasso_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                        input_train_y_series=eg_train_df[eg_target_label],
                        input_model=model)

    # Ridge model
    model = Ridge(alpha=1.0)
    ridge_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                        input_train_y_series=eg_train_df[eg_target_label],
                        input_model=model)

    # OLS model
    model = LinearRegression()
    linear_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                         input_train_y_series=eg_train_df[eg_target_label],
                         input_model=model)

    # SVM model
    model = LinearSVC()
    svm_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                      input_train_y_series=eg_train_df[eg_target_label],
                      input_model=model)

    # MLP model
    model = MLPClassifier()
    mlp_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                      input_train_y_series=eg_train_df[eg_target_label],
                      input_model=model)

    # LSTM model
    model = LSTMWrapper()
    lstm_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                       input_train_y_series=eg_train_df[eg_target_label],
                       input_model=model)

    # Random Forest model
    model = RandomForestClassifierWrapper()
    rf_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                     input_train_y_series=eg_train_df[eg_target_label],
                     input_model=model)

    # Gradient Boosting model
    model = GradientBoostingClassifierWrapper()
    gb_model = Model(input_train_x_df=eg_train_df[eg_features_col],
                     input_train_y_series=eg_train_df[eg_target_label],
                     input_model=model)
