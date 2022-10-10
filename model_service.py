# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree

class ModelFactory:
    def __init__(self, type) -> None:
        self.type = type
    
    def get_model(self):
        if self.type == "LOG_REG":
            return linear_model.LogisticRegression()
        elif self.type == "LIN_SVC":
            return svm.LinearSVC(tol=0.00005)
        elif self.type == "DEC_TRE":
            return  tree.DecisionTreeClassifier()


class ModelService:

    def __init__(self, model_type:str) -> None:
        model_factory = ModelFactory(model_type)
        self.model = model_factory.get_model()
        return

    def _plot_confusion_matrix(self, matrix, normalize=False):
        if normalize:
            matrix = matrix.astype("float")/matrix.sum(axis=1)[:np.newaxis]
        
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return

    def _is_valid(self, data):
        if sum(data.duplicated()):
            print("Duplicates found")
            return False
        
        if data.isnull().values.sum():
            print("Null values found")
            return False

        return True

    def init_grid(self, parameters, cv=3, verbose=1, n_jobs=-1):
        self.model = GridSearchCV(self.model, param_grid=parameters, cv=cv, verbose=verbose, n_jobs=n_jobs)

    def train(self, x_train:pd.DataFrame, y_train:pd.Series):   
        if self._is_valid(x_train):
            print(f"Model training started at {datetime.now()}")
            self.model.fit(x_train, y_train)
            print(f"Model training completed at {datetime.now()}")
            return

    def test(self, x_test:pd.DataFrame, y_test:pd.Series):        
        pred = self.model.predict(x_test)

        # Accuracy
        acc = metrics.accuracy_score(y_test, pred)
        print(f"######## Accuracy = {acc} ########")

        # Confusion matrix
        conf_mat = metrics.confusion_matrix(y_test, pred)
        #print(f"######## Confusion matrix = \n{conf_mat} ########")
        self._plot_confusion_matrix(conf_mat)

        # Classification report
        cr = metrics.classification_report(y_test, pred)
        print(f"######## Classification report = \n{cr} ########")

        return pred

    def get_grid_outputs(self):
        print(f"####### Best Estimator \n{self.model.best_estimator_}\n#######")

        print(f"####### Best parameters \n{self.model.best_params_}\n#######")

        print(f"####### Cross Validation splits \n{self.model.n_splits_}\n#######")

        print(f"####### Best score \n{self.model.best_score_}\n#######")