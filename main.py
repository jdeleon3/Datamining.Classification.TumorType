from datahandler import DataHandler
from model import Model

dh = DataHandler()
dh.clean_data()
#dh.scale_data()

m = Model(dh.get_data())
m.train_decisiontree_model()
m.train_randomforest_model()
m.train_bagging_model()
m.train_adaboost_model()

m2 = Model(dh.get_data(), use_KFold=True, visual_prefix='kfold_')
m2.train_decisiontree_model()
m2.train_randomforest_model()
m2.train_bagging_model()
m2.train_adaboost_model()

