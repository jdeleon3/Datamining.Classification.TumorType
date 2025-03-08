from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold


import pandas as pd
import numpy as np
from visualizer import Visualizer

class Model:

    def __init_basemodel__(self):        
        return DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state)
    
    def __split_data__(self):
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    
    def __train_model__(self, model, X_train, y_train):
        skf_results = []
        if self.use_KFold:            
            for split_train_index, split_test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state).split(X_train, y_train):
                X_train_split, X_test_split = X_train.iloc[split_train_index], X_train.iloc[split_test_index]
                y_train_split, y_test_split = y_train.iloc[split_train_index], y_train.iloc[split_test_index]
                model.fit(X_train_split, y_train_split)
                y_pred = model.predict(X_test_split)
                skf_results.append(accuracy_score(y_test_split, y_pred))        
            print('Split results: \n\n')
            print(f'{np.mean(skf_results)}\n')
            print(f'{np.std(skf_results)}\n\n')

        else:
            model.fit(X_train, y_train)

        return model

    def __init__(self, data: pd.DataFrame, use_KFold: bool = False, visual_prefix: str = ''):
        self.use_KFold = use_KFold
        self.visual_prefix = visual_prefix
        self.criterion = 'gini'
        self.max_depth = 4
        self.random_state = 42
        self.n_estimators = 100
        self.df = data
        self.dtc_model = self.__init_basemodel__()
        self.rfc_model = RandomForestClassifier(criterion=self.criterion, n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.bagging_model = BaggingClassifier(estimator=self.__init_basemodel__(), n_estimators=self.n_estimators, random_state=self.random_state, bootstrap=True)
        self.adaboost_model = AdaBoostClassifier(estimator=self.__init_basemodel__(), n_estimators=self.n_estimators, random_state=self.random_state)        
        

    def train_decisiontree_model(self):
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        
        X_test, X_train, y_test, y_train = self.__split_data__()

        #self.dtc_model.fit(X_train, y_train)
        self.dtc_model = self.__train_model__(self.dtc_model, X_train, y_train)

        #final model test
        y_pred = self.dtc_model.predict(X_test)
        
        print('Decision Tree results: \n\n')
        score = accuracy_score(y_test, y_pred)
        print(score)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        v = Visualizer(self.df)
        v.plot_decisiontree(self.dtc_model, filename=f'./images/{self.visual_prefix}decision_tree_{self.use_KFold}.svg')
        v.plot_confusion_matrix(y_test, y_pred, f'./images/{self.visual_prefix}confusion_matrix_DecisionTree_{self.use_KFold}.svg')
        return score
        

    def train_randomforest_model(self):
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        X_test, X_train, y_test, y_train = self.__split_data__()
        
        self.rfc_model = self.__train_model__(self.rfc_model, X_train, y_train)


        y_pred = self.rfc_model.predict(X_test)

        print('Random Forest results: \n\n')
        score = accuracy_score(y_test, y_pred)
        print(score)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        #visualize results
        v = Visualizer(self.df)
        v.plot_randomforest_decisiontrees(self.rfc_model)
        v.plot_confusion_matrix(y_test, y_pred, f'./images/{self.visual_prefix}confusion_matrix_RandomForest_{self.use_KFold}.svg')
        return score

    def train_bagging_model(self):
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        X_test, X_train, y_test, y_train = self.__split_data__()
        self.bagging_model = self.__train_model__(self.bagging_model, X_train, y_train)
        y_pred = self.bagging_model.predict(X_test)

        print('Bagging results: \n\n')
        score = accuracy_score(y_test, y_pred)
        print(score)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        v = Visualizer(self.df)
        v.plot_confusion_matrix(y_test, y_pred, f'./images/{self.visual_prefix}confusion_matrix_Bagging.svg')
        return score
        
    def train_adaboost_model(self):
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        X_test, X_train, y_test, y_train = self.__split_data__()
        self.adaboost_model = self.__train_model__(self.adaboost_model, X_train, y_train)
        y_pred = self.adaboost_model.predict(X_test)

        print('AdaBoost results: \n\n')
        score = accuracy_score(y_test, y_pred)
        print(score)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        v = Visualizer(self.df)
        v.plot_confusion_matrix(y_test, y_pred, f'./images/{self.visual_prefix}confusion_matrix_adaboost.svg')
        return score
    
    