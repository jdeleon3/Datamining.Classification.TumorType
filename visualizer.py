import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Visualizer:

    def __init__(self, data: pd.DataFrame):
        self.df = data
        
    def plot_histograms(self):
        self.df.hist(figsize=(20, 20))
        #plt.show()
        filename = './images/histograms.svg'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()
    
    def plot_boxplots(self):
        for col in self.df.columns:
            sns.boxplot(self.df[col])
            filename = f'./images/boxplot_{col}.svg'
            if os.path.exists(filename):
                os.remove(filename)
            plt.savefig(filename)
            plt.close()

    def plot_correlation_with_target(self):
        plt.figure(figsize=(20,20))
        temp_df = self.df.copy()
        temp_df['diagnosis'] = temp_df['diagnosis'].map({'M': 1, 'B': 0})
        temp_df.drop('diagnosis', axis=1).corrwith(temp_df['diagnosis']).plot(kind='bar')
        filename = './images/correlation.svg'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()

    def plot_heatmap(self):
        temp_df = self.df.copy()
        temp_df['diagnosis'] = temp_df['diagnosis'].map({'M': 1, 'B': 0})

        plt.figure(figsize=(20,20))
        sns.heatmap(temp_df.corr(), annot=True)
        filename = './images/heatmap.svg'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()

    def plot_randomforest_decisiontrees(self, rfc: RandomForestClassifier):
        i=0
        for dtc in rfc.estimators_:
            filename = f'./images/randomforest_decisiontree_{i}.svg'
            self.plot_decisiontree(dtc, filename)
            i+=1
    
    def plot_decisiontree(self, dtc: DecisionTreeClassifier, filename: str='./images/decisiontree.svg'):
        plt.figure(figsize=(20,20))
        plot_tree(dtc, filled=True, feature_names=self.df.columns[:-1], class_names=['B', 'M'])
        if os.path.exists(filename):
                os.remove(filename)
        plt.savefig(filename)        
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, filename: str='./images/confusion_matrix.svg'):
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm).plot()        
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()

    def plot_score_scores(self, scores: dict):
        plt.bar(scores.keys(), scores.values())
        filename = './images/scores.svg'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()
    