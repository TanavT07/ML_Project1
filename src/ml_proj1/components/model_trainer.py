import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn .ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from ml_proj1.exception import CustomException
from ml_proj1.logger import logging
from ml_proj1.utils import evaluate_models, save_obj


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Linear Regression" : LinearRegression(),
                "Cat Boost" : CatBoostRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "KNeighbours" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor()
            }

            model_report:dict = evaluate_models(X_train = x_train,X_test = x_test, y_train = y_train,y_test = y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model")

            logging.info("best model found")

            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj = best_model)
            

            predicted = best_model.predict(x_test)

            R2_score = r2_score(y_test, predicted)
            return R2_score

        except CustomException as e:
            raise CustomException(e,sys)
