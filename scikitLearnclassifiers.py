# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:33:01 2017

@author: dolatae
"""

# data analysis and wrangling
from __future__ import division

import warnings
warnings.filterwarnings('ignore')
# ---

import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, RandomizedSearchCV, GridSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.metrics import classification_report, roc_curve ,auc
from scipy.stats import randint, expon

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import StratifiedKFold        


from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb   

def edloaddata(fdir):
    fdir = '...'
    df = pd.read_excel(open(fdir,'rb'), sheetname='final')
    
    return df
   
def recover_train_target():
    global alldata
   
    train = alldata.copy()
    targets = alldata.Lables
   
    return train, targets
    
    
class MLclassifier:
    
    def __init__(self,train, targets,model_name, scoring_= None,cv= None,feature_selection= None):
        self.data = train.as_matrix()
        self.label = targets.as_matrix()
        self.scoring_ = scoring_
        self.cv = cv
        self.feature_selection = feature_selection
        self.classifier = model_name
    
    def set_feature_selection(self):
        from sklearn.svm import SVR
        if self.feature_selection == 'SelectKBest':
            FS = SelectKBest(mutual_info_classif)
            FS_grid_param = 'FS__k'
        elif self.feature_selection == 'RFE':
            FS = RFE(SVR(kernel="linear"))
            FS_grid_param = 'FS__n_features_to_select'
        else:
            FS = None
            
        return FS,FS_grid_param
        
        
    def get_gridsearch(self):
        n_iter = 10
        n_feat = self.data.shape[1]
        model_name = self.classifier
        if self.scoring_ is None:
            self.scoring_ = 'accuracy'
        FS,FS_grid_param = self.set_feature_selection()
            
        if model_name == 'lr':  
            print ('Logistic Regression Fitting')            
            if FS is None:
                pipe = Pipeline([('ss', StandardScaler()),('lr', LogisticRegression(max_iter=1000))])
                param_grid = {'lr__C': np.logspace(-5, 15, num=20,base=2)}
            else:
                pipe = Pipeline([('ss', StandardScaler()),('FS', FS), ('lr', LogisticRegression())])
                param_grid = {'lr__C': np.logspace(-5, 15, num=20,base=2),
                          FS_grid_param:range(1,n_feat+1)}
                          
            clf = GridSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring_, n_jobs=-1, verbose=2) 
        
        elif model_name == 'svm':
            print('svm fitting')           
            if FS is None:
                pipe = Pipeline([('ss', StandardScaler()),('svc', SVC(probability=True))])
                param_grid = {'svc__C': np.logspace(-5, 15, num=20,base=2), 
                          'svc__gamma': np.logspace(-15, 5, num=20,base=2),
                          'svc__class_weight' : ['balanced', None]}
            else:
                pipe = Pipeline([('ss', StandardScaler()),('FS', FS),('svc', SVC(probability=True))])
                param_grid = {'svc__C': np.logspace(-5, 15, num=20,base=2), 
                          'svc__gamma': np.logspace(-15, 5, num=20,base=2),
                          'svc__class_weight' : ['balanced', None],
                          FS_grid_param:range(1,n_feat+1)}
                          #'FS__score_func': [mutual_info_classif,chi2,f_classif] }
            
            clf = RandomizedSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring_, n_jobs=-1, verbose=2, n_iter = n_iter)
        
            
        elif model_name == 'rf':
            if FS is None:
                pipe = Pipeline([('ss', StandardScaler()),('rf', RandomForestClassifier())])
                param_grid = {'rf__max_depth': np.concatenate((range(1,20),[None]),axis=0),
                      'rf__max_features': ['sqrt', None , 'log2'],
                      'rf__min_samples_split': np.concatenate((np.arange(2,10,3),[17,50,75,100,300]),axis=0),
                      'rf__min_samples_leaf': np.concatenate((np.arange(2,10),[17,100,300]),axis=0),
                      'rf__n_estimators': [10, 50, 100,1000],
                      'rf__bootstrap': [True, False],
                      'rf__class_weight' : ['balanced', None]}
            else:
                pipe = Pipeline([('ss', StandardScaler()),('FS',FS), ('rf', RandomForestClassifier())])
                param_grid = {'rf__max_depth': np.concatenate((range(1,20),[None]),axis=0),
                      'rf__max_features': ['sqrt', None , 'log2'],
                      'rf__min_samples_split': np.concatenate((np.arange(2,10,3),[17,50,75,100,300]),axis=0),
                      'rf__min_samples_leaf': np.concatenate((np.arange(2,10),[17,100,300]),axis=0),
                      'rf__n_estimators': [10, 50, 100,1000],
                      'rf__bootstrap': [True, False],
                      'rf__class_weight' : ['balanced', None],
                      FS_grid_param: range(1,n_feat+1)}
      
                clf = RandomizedSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring_, n_jobs=-1, verbose=2, n_iter = n_iter)       
    
        elif model_name == 'gb':  
            if FS is None:
                pipe = Pipeline([('ss', StandardScaler()),('gb', GradientBoostingClassifier())])
                param_grid = {'gb__max_features': ['sqrt', None , 'log2'],
                      'gb__min_samples_split': np.concatenate((np.arange(2,10,3),[17,50,75,100,300]),axis=0),
                      'gb__min_samples_leaf': np.concatenate((np.arange(2,10),[17,100,300]),axis=0),
                      'gb__n_estimators': [10, 50,100, 1000],
                      'gb__warm_start': [True, False],
                      'gb__max_depth': np.concatenate((range(1,20),[None]),axis=0)}
            else:
                pipe = Pipeline([('ss', StandardScaler()),('FS',FS), ('gb', GradientBoostingClassifier())])
                param_grid = {'gb__max_features': ['sqrt', None , 'log2'],
                      'gb__min_samples_split': np.concatenate((np.arange(2,10,3),[17,50,75,100,300]),axis=0),
                      'gb__min_samples_leaf': np.concatenate((np.arange(2,10),[17,100,300]),axis=0),
                      'gb__n_estimators': [10, 50,100, 1000],
                      'gb__warm_start': [True, False],
                      'gb__max_depth': np.concatenate((range(1,20),[None]),axis=0),
                      FS_grid_param: range(1,n_feat+1)}
                      
                clf = RandomizedSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring_, n_jobs=-1, verbose=2, n_iter = n_iter)
        
        elif model_name == 'dt':  
            if FS is None:  
                pipe = Pipeline([('ss', StandardScaler()),('dt', DecisionTreeClassifier())])
                param_grid = {'dt__max_depth': np.concatenate((range(1,20),[None]),axis=0), 
                      'dt__max_features': ['sqrt', None , 'log2'],#randint(1, np.floor(np.sqrt(n_feat)))
                      'dt__min_samples_split': range(2,100),
                      'dt__min_samples_leaf': range(2,100),
                      'dt__class_weight' : ['balanced', None]}
            else:  
                pipe = Pipeline([('ss', StandardScaler()),('FS',FS), ('dt', DecisionTreeClassifier())])
                param_grid = {'dt__max_depth': np.concatenate((range(1,20),[None]),axis=0), 
                      'dt__max_features': ['sqrt', None , 'log2'],#randint(1, np.floor(np.sqrt(n_feat)))
                      'dt__min_samples_split': range(2,100),
                      'dt__min_samples_leaf': range(2,100),
                      'dt__class_weight' : ['balanced', None],
                      FS_grid_param: range(1,n_feat+1)}
            
                clf = RandomizedSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring_, n_jobs=-1, verbose=2, n_iter = n_iter)    
        
        elif model_name == 'xgb':
            print('xgb fitting')  
            ind_params = {'seed': 1,
                       'silent': True,
                       'objective':'binary:logistic'} 
            if FS is None:  
                pipe = Pipeline([('ss', StandardScaler()),('xgb', xgb.XGBClassifier(**ind_params))])
                param_grid = {'xgb__max_depth': [5,100,1000],
                      'xgb__min_child_weight': [5,100,1000],
                      'xgb__learning_rate': [0.01,0.1,1],
                      'xgb__gamma': [0,0.1,1,5],
                      'xgb__n_estimators': [10,100,1000]}
            else:  
                pipe = Pipeline([('ss', StandardScaler()),('FS',FS),('xgb', xgb.XGBClassifier(**ind_params))])
                param_grid = {'xgb__max_depth': [5,100],
                      'xgb__min_child_weight': [5,100],
                      'xgb__learning_rate': [0,0.1,1],
                      'xgb__gamma': [0,0.1,5],
                      'xgb__n_estimators': [10,100],
                      FS_grid_param: range(1,n_feat+1)}
                clf = GridSearchCV(pipe, param_grid, cv=self.cv, scoring=self.scoring_, n_jobs=-1, verbose=2)   
            
        return clf
        
def ed_measureaccuracy(y_true, y_pred,group_test):
    result = pd.DataFrame()
    temp = pd.DataFrame()
    for i in group_test.unique():        
        temp['group']=[i]
        temp['accuracy']=[accuracy_score(y_true[group_test==i],y_pred[group_test==i])]
        result = result.append(temp,ignore_index=True)
    return result
    
if __name__ == "__main__":
    
    # load data
    subs = range(0,40)
    alldata = pd.DataFrame()
    for i in subs:
        print i
        fname = 'S0'+ str(i)
        df = edloaddata(fname)
        df['group'] = i
        alldata = alldata.append(df,ignore_index=True) 
        
    alldata.isnull().any()
    print(alldata.info())
    print(alldata.describe())
    
    train, targets = recover_train_target()
    print(train.columns)
    
    classifiers = ['lr','svm','rf','xgb']
    logo = LeaveOneGroupOut()
    colname = ['group','accuracy','S as S','S as R','S as L',
                         'R as S','R as R','R as L',
                         'L as S','L as R','L as L']
    cols = ['S as S','S as R','S as L',
                         'R as S','R as R','R as L',
                         'L as S','L as R','L as L']
    groups = alldata['group']
    k = 5
    group_kfold = GroupKFold(n_splits=k)
    
    #fscore = np.empty(shape=[0,k])
    #total_acc=np.empty(shape=[0,k])  
    sub_acc = pd.DataFrame()  
    sub_acc['Subject ID'] = subs
    bestmodelparam=[]
    
    
    for cl_name in classifiers:
        tempacc1=[]
        result=pd.DataFrame(columns = colname)
        finalresult = pd.DataFrame(columns = colname)
        for train_index, test_index in logo.split(train, targets, groups):
            X_train, X_test = train.iloc[train_index], train.iloc[test_index]
            y_train, y_test = targets.iloc[train_index], targets.iloc[test_index] 
            group_train,group_test = alldata.iloc[train_index]['group'],alldata.iloc[test_index]['group']
            cross_validation = group_kfold.split(X_train, y_train, group_train)
            
            mlclassifier = MLclassifier(X_train,y_train,cv = cross_validation,feature_selection = 'SelectKBest',model_name=cl_name,scoring_='accuracy')
            clf = mlclassifier.get_gridsearch()
            clf.fit(X_train, y_train)
            y_true, y_pred = y_test, clf.predict(X_test)
            yscore = clf.predict_proba(X_test)
            del mlclassifier
            del clf
               
            result['group']=[group_test.unique()]
            result['accuracy'] = accuracy_score(y_true,y_pred)
            finalresult= finalresult.append(result)
            
            tempacc1.append(accuracy_score(y_true, y_pred))
            #print('Best score: {}'.format(clf.best_score_))    
            #print('Best parameters: {}'.format(clf.best_params_))
            bestmodelparam.append(clf.best_params_)
        
        finalresult.sort_values(by='group')
        finalresult.reset_index(drop=True, inplace=True)
        finalresult.to_csv('')
        sub_acc[cl_name] =tempacc2['accuracy']
   sub_acc[['lr','svm','rf','xgb']].plot.bar(fontsize=40)
   sub_acc[['lr','svm','rf','xgb']].plot.box(fontsize=40)
 
