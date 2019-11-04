import numpy as np
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import pprint
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost

from ja_model.utils import _test_metric, _rmse, powerset, simpleaxis, x_label_setter, y_label_setter
from ja_model.models import Model

class XGB(Model):
    def __init__(self, features, *args, **kwargs):
        """
        Args:
            features : list - list of desired features, these should match the headings of the training and testing data used later in the classification

        For more information, see LinReg.help()
        """
        super().__init__(*args, **kwargs)
        print("Initialising a XGBoost regression classifier")
        print("==================================================")
        print(" ")
        print("Feature Set")
        print("-----------")
        print(" ")
        pprint.pprint(features)
        print(" ")
        print("IMPORTANT: For best results using the XGBoost Regression class, input data should be normalised to lie within the same range e.g. [0,1]. Non-normalised data could lead to poor results.")
        self.input_features = features
        self.best_features = features
        self._current_features = features
        self._latest_params = {   'nthread': 4,
                                  'objective': 'reg:linear',
                                  'learning_rate': 0.02,
                                  'max_depth': 10,
                                  'min_child_weight': 4,
                                  'silent': 1,
                                  'subsample': 0.7,
                                  'colsample_bytree': 0.7,
                                  'n_estimators': 200}
        self.__result = None
        self._model = None
        self.save_count = 0

    def optimise_parameters(self, x_train=None, y_train=None):
        """
        Algorithm to optimise the XGBoost regression hyperparameters. Runs a prebuilt grid search across a suitable parameter range and sets the classifier hyperparameters to the optimum values. Note that this is carried out on the training set.

        Args:
            x_train : pd.DataFrame or similar - training data array
            y_train : pd.DataFrame or similar - output array
        """
        # Create the random grid
        random_grid = {'nthread': [4],
              'objective': ['reg:linear'],
              'learning_rate': [x for x in np.linspace(0.02, 0.3, num = 10)],
              'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]}
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        xgb = xgboost.XGBRegressor()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1001, n_jobs = -1)
        # Fit the random search model
        if x_train == None or y_train == None:
            xgb_random.fit(self.get_data('X_train'), self.get_data('Y_train'))
        else:
            try:
                if len(x_train) != len(y_train):
                    raise ValueError('ERROR: Arrays x_train and y_train of different lengths.')
                xgb_random.fit(x_train, y_train)
            except ValueError as ve:
                print(ve.args[0])
        self._latest_params = xgb_random.best_params_
        print("Finished optimising parameters, best set found:\n")
        pprint.pprint(self._latest_params)

    def get_latest_params(self):
        """
        Returns the latest optimised hyperparaneters

        Returns:
            parameters : dictionary - dictionary of hyperparameters
        """
        return self._latest_params


    def feature_selection(self, test_function='rmse'):
        """
        Runs through a feature selection algorithm which enumerates the possible subsets of the input features and attempts to minimise the test_metric error on the validation set after training the classifier on the training data. Updates the self.best_features attribute which can then be used to run the full model on the training and test data. This is only really appropriate for a relatively small number of features. To avoid computational intensity, there is no hyperparameter optimisation, instead standard parameters are calculated from the data. Once the best feature set has been identified, one can use the additional functionality in the library to tune the hyperparameters.

        Args:
            test_function : function - default is rmse testing, but others are available, see _test_metric? for more information

        Note: the input data for the features must be in the form of a pd.DataFrame
        """
        try:
            if not isinstance(self.get_data('X_train'), pd.DataFrame):
                raise TypeError("ERROR: The input training data was not in the form of a pd.DataFrame.")
            feature_set = list(powerset(self.input_features))
            print("Feature Selection")
            print("=================")
            print(" ")
            print("Running feature selection on a feature set of size: ", len(feature_set) - 1)
            print(" ")
            feature_dict = {}
            list_results = []
            counter = 0
            X_train_data = self.get_data('X_train')
            X_val_data = self.get_data('X_val')
            Y_train_data = self.get_data('Y_train')
            Y_val_data = self.get_data('Y_val')

            if self._latest_params == None:
                self.optimise_parameters()
                print("First optimising parameters over training set.")

            if (len(feature_set) < 100):
                counter_check = 10
            elif (len(feature_set) < 1000):
                counter_check = 100
            elif (len(feature_set) < 2500):
                counter_check = 250
            elif (len(feature_set) < 5000):
                counter_check = 500
            else:
                counter_check = 1000

            for _features in feature_set[1:]:
                if (counter % counter_check == counter_check - 1):
                        print('-------------------Completed ', counter + 1, ' feature sets out of ', len(feature_set) - 1, '-------------------\n')
                X_train_data_temp = X_train_data[list(_features)]
                X_val_data_temp = X_val_data[list(_features)]
                feature_dict[counter] = list(_features)
                temp_model = xgboost.XGBRegressor(**self._latest_params)
                temp_model.fit(X_train_data_temp, Y_train_data)
                val_forecast = temp_model.predict(X_val_data_temp)
                val_rmse = _test_metric(Y_val_data, val_forecast, test_function)[0]
                list_results.append(val_rmse)
                counter += 1
            print('-------------------Finished iterating through possible feature sets.-------------------\n')
            test_mse_df = pd.DataFrame({'test_mse': list_results})
            lowest_test_mse = test_mse_df.sort_values(['test_mse'])
            index = lowest_test_mse.index
            self.best_features = feature_dict[index[0]]
            X_train_data_temp = X_train_data[feature_dict[index[0]]]
            X_val_data_temp = X_val_data[feature_dict[index[0]]]
            temp_model = xgboost.XGBRegressor(**self._latest_params)
            temp_model.fit(X_train_data_temp, Y_train_data)
            val_forecast = temp_model.predict(X_val_data_temp)
            val_rmse = _test_metric(Y_val_data, val_forecast, test_function)[0]
            val_forecast = temp_model.predict(X_val_data_temp)
            final_rmse = _test_metric(Y_val_data, val_forecast, test_function)
            print('Lowest Error on validation set with feature set: ', feature_dict[index[0]], '\n\n')
            print('Set best_features attribute to this set. With this choice, the following regression results were obtained on the training data:\n\n')
            print('The RMSE on the validation set was: ', final_rmse[0])
            print('The mean percentage error is: ', final_rmse[1], '%.')
            print('\nFinished feature selection. To see list of best_features, call get_best_features() on your classifier. To access the regression parameters, call get_latest_params()')

        except TypeError as te:
            print(te.args[0])

    def train(self, features=None, params=None):
        """
        Train the model on a chosen set of features. If none are chosen, the default is to re run the model with the current best_features attribute. Note that the training is carried out on the training data, X_train, only. To access the result, use:

        Args:
            features : list - train model with list of desired features
            params : dictionary e.g:  {'nthread': 4,
                                      'objective': 'reg:linear',
                                      'learning_rate': 0.02,
                                      'max_depth': 10,
                                      'min_child_weight': 4,
                                      'silent': 1,
                                      'subsample': 0.7,
                                      'colsample_bytree': 0.7,
                                      'n_estimators': 200}

        Returns:

        Note: The input data for the RFR should be normalised such that each feature lies in the same range e.g. [0,1] to ensure that no one feature dominates. Non-normalised data will result in potentially very poor results.
        """
        if not isinstance(self.get_data('X_train'), pd.DataFrame):
            raise TypeError("ERROR: The input training data was not in the form of a pd.DataFrame.")
        print("Training - XGBoost Regression Classifier")
        print("==============================================")
        print(" ")
        print("Running XGBoost regression classifier on feauture set:")
        print(" ")
        if features == None:
            features = self.get_best_features()
        pprint.pprint(features)
        print(" ")
        self._current_features = features
        X_train_data = self.get_data('X_train')
        X_val_data = self.get_data('X_val')
        X_test_data = self.get_data('X_test')
        Y_train_data = self.get_data('Y_train')
        Y_val_data = self.get_data('Y_val')
        Y_test_data = self.get_data('Y_test')
        X_train_data_temp = X_train_data[features]
        X_val_data_temp = X_val_data[features]
        X_test_data_temp = X_test_data[features]
        if params == None:
            params = self._latest_params

        xgb_model = xgboost.XGBRegressor(**params)
        xgb_model.fit(X_train_data_temp, Y_train_data)
        self._model = xgb_model
        Y_val_pred = xgb_model.predict(X_val_data_temp)
        Y_test_pred = xgb_model.predict(X_test_data_temp)
        print("Training r-squared:", xgb_model.score(X_train_data_temp, Y_train_data))
        print("Validation r-squared:", xgb_model.score(X_val_data_temp, Y_val_data))
        print("Testing r-squared:", xgb_model.score(X_test_data_temp, Y_test_data))
        final_rmse_val  = _test_metric(Y_val_data, Y_val_pred, 'rmse')
        self._val_rmse = final_rmse_val
        final_rmse_test = _test_metric(Y_test_data, Y_test_pred, 'rmse')
        print(' ')
        print('The RMSE on the validation set was: ', final_rmse_val[0])
        print('The mean percentage error is: ', final_rmse_val[1], '%.')
        print(' ')
        print('The RMSE on the test set was: ', final_rmse_test[0])
        print('The mean percentage error is: ', final_rmse_test[1], '%.')
        print('\nFinished training. To access the most recent classifier, call get_model()')

    def get_model(self):
        """
        Gets latest instance of SVR model as an instance of the sklearn.

        Returns:
            sklearn.svm.SVR - latest model instance from training or testing
        """
        if self._model != None:
            return self._model
        else:
            print("ERROR: No model has been trained. Please train a model.")

    def test_error(self, test_function='rmse'):
        """
        Computes the error in the current model with the given metric function. The result is stored in a class variable tget_testest_error.

        Args:
            test_function : string - for options see _test_metric?
        """
        Y_test_data = self.get_data('Y_test')
        X_test_data = self.get_data('X_test')
        Y_test_forecast = self._model.predict(X_test_data[self._current_features])
        test_rmse = _test_metric(Y_test_data, Y_test_forecast, test_function)
        self._test_error = test_rmse
        print('The RMSE on the test set was: ', test_rmse[0])
        print('The mean percentage error is: ', test_rmse[1], '%.')
        print('\nTo access the results, call get_test_error()')

    def get_test_error(self):
        """
        Returns most recent test error calculation results

        Retunrs:
            test_error : tuple - RMSE error and mean percentage error on test data
        """
        return self._test_error

    def plot_forecast(self, x_data=["test"], save=False, save_name=None, _inline=False):
        """
        Plots the selected subset of data along with a pre selected confidence interval, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        data = self.get_data()
        temp_X = []
        temp_Y = []
        for _type in x_data:
            temp_X.append(data['X_' + _type])
            temp_Y.append(data['Y_' + _type])
        X_data = pd.concat(temp_X)
        Y_data = pd.concat(temp_Y)
        try:
            if self._model != None:
                X_data = X_data[self._current_features]
                Y_forecast  = self._model.predict(X_data)
                # Y_data['Y_forecast'] = Y_forecast
                # Y_forecast_df = Y_data['Y_forecast']
                temp_df = pd.DataFrame({'Y_act': Y_data})
                temp_df['Y_forecast'] = Y_forecast
                plt.figure(figsize=(15,8))
                ax = plt.subplot()
                plt.plot('Y_act', data = temp_df, color = 'red', label = 'Actuals')
                plt.plot('Y_forecast', data = temp_df, color = 'blue', label = 'Prediction')
                plt.legend()
                x_label_setter('Date', plt.gca())
                y_label_setter('Data Value', plt.gca())
            else:
                raise ValueError("ERROR: No model trained. Please train a model.")

            if not _inline:
                if (save_name == None and save):
                    save_string = 'XGB_forecast' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except ValueError as ve:
            print(ve.args[0])

    def plot_residuals(self, save=False, save_name=None, inline=False):
        """
        Plots the residuals against the input data to assess the distribution, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            conf_level : float - confidence level in the coefficients is extracted from the regression results
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            if self._model == None:
                raise RuntimeError("No model trained, can't extract residuals.")
            fig = plt.figure(figsize=(10,10))
            residuals = self.get_data(data = 'Y_train') - self._model.predict(self.get_data(data = 'X_train')[self._current_features])
            Y_train_data = self.get_data(data='Y_train')
            sns.regplot(Y_train_data, residuals, lowess=True, line_kws={'color': 'red', 'lw': 3, 'alpha': 0.8})
            x_label_setter('y training values', plt.gca())
            y_label_setter('Residuals', plt.gca())

            if not inline:
                if (save_name == None and save):
                    save_string = 'XGB_residuals' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except RuntimeError as re:
            print(re.args[0])

    def plot_autocorrelation(self, save=False, save_name=None, inline=False):
        """
        Plots the autocorrelation of the residuals, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            conf_level : float - confidence level in the coefficients is extracted from the regression results
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            if self._model == None:
                raise RuntimeError("No model trained. Please train a model.")
            fig = plt.figure(figsize=(15, 6))
            residuals = self.get_data(data = 'Y_train') - self._model.predict(self.get_data(data = 'X_train')[self._current_features])
            autocorrelation_plot(residuals)
            x_label_setter('Lag', plt.gca())
            y_label_setter('Autocorrelation', plt.gca())
            residuals = self.get_data(data = 'Y_train') - self._model.predict(self.get_data(data = 'X_train')[self._current_features])
            if not inline:
                if (save_name == None and save):
                    save_string = 'XGB_autocorrelation' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except RuntimeError as re:
            print(re.args[0])

    def help():
        helper_string = """
-----------------------------Help: XGBoost Regression Classifier-----------------------------

This is a XGBoost regression classifier for single variable output data.

To initialise a classifier, simply provide a feature set which matches the data you will use to train and test your model:

    clf = XGB(['time', 'temperature', ...])

For input data (X, Y), we can generate a training, validation, and testing split via:

    clf.generate_split(X, Y, train_size=0.7, val_size=0.15, test_size=0.15)

Alternatively, the user can generate their own data and then set it using:

    clf.set_data({'X_train' : X_train, 'Y_train' : Y_train, ...})

The data can then be accessed using:

    clf.get_data()
    or
    clf.get_data('X_train') etc.

There are a number of hyperparameters to optimise when it comes to the RFR, as such a pragmatic solution is to optimise the parameters with the full feature set and use this moving forward. To do this, perform a grid search using;

    clf.optimise_parameters()

The latest parameters from the optimisation can then be found using:

    clf.get_latest_params()

To perform the cross-validation for feature selection, run;

    clf.feature_selection()

This will make the classifiers best_features equal to this set.

To train the classifier, call:

    clf.train(features, C=100, epsilon=1.0, kernel='rbf', gamma=0.1, degree=2)

This generates an instance of an sklearn.svm.SVR model which is saved to a member variable and can be accesed using:

    clf.get_model()

The error on the test set can be generated from the latest parameters and accesed using the commands:

    clf.test_error()
    clf.get_test_error()

The data can be plotted either manually by the user or using the inbuilt:

    clf.plot_forecast(['train', 'test', ...])
    clf.plot_residuals()
    clf.plot_autocorrelation()
"""

        print(helper_string)
