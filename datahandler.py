import pandas as pd
import numpy as np
import os
from visualizer import Visualizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder


class DataHandler:

    def __init__(self, data_path: str = './data/data.csv'):
        if(data_path is None or not data_path.endswith('.csv') or not os.path.exists(data_path)):
            raise ValueError('Invalid data path')
        self.df = pd.read_csv(data_path)
        self.scaler = MinMaxScaler()
    
    def get_data(self):
        return self.df
    
    def inspect_data(self):
        print(self.df.head())
        print(self.df.describe())
        print(self.df.info())
        print(f'Null Counts:\n{self.df.isnull().sum()}\n\n')
        self.print_attribute_value_counts()
        v = Visualizer(self.df)
        v.plot_histograms()
        v.plot_boxplots()
        v.plot_correlation_with_target()
        v.plot_heatmap()



    def clean_data(self):
        #Drop id column
        self.df.drop('id', axis=1, inplace=True)

        #Convert diagnosis to binary
        self.df['diagnosis'] = LabelEncoder().fit_transform(self.df['diagnosis'])
        
        #Drop highly correlated columns - ommitted as it reduces accuracy of random forest model
        #self.df.drop(['radius_worst','texture_worst','perimeter_worst','area_worst'
        #            , 'perimeter_mean', 'perimeter_se', 'area_se'], axis=1, inplace=True)

        #Drop columns with low correlation with target - ommitted as it reduces accuracy of all models
        #self.df.drop(['fractal_dimension_mean', 'texture_se', 'symmetry_se'], axis=1, inplace=True)

        #No missing data, but will drop in case data changes later
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

    def scale_data(self):
        for col in self.df.columns:
            if col != 'diagnosis':
                self.df[col] = self.scaler.fit_transform(self.df[[col]])
    
    def print_attribute_value_counts(self):
        for col in self.df.columns:
            print(self.df[col].value_counts())


if __name__ == '__main__':
    dh = DataHandler()
    dh.inspect_data()