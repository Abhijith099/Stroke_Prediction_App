from inspect import getmodule

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from joblib import dump, load  #for saving the model library
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

def load_data():
    df=pd.read_csv('../backend/healthcare-dataset-stroke-data.csv')
    df=df.drop('id',axis=1)
    numerical=['avg_glucose_level','bmi','age']
    categorical=['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']
    y=df['stroke']
    X=df.drop('stroke',axis=1)
    return X,y, categorical, numerical

def evaluate_model(X,y,model):
  cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1) #nsplits means cross validation that has 10 folds , n repeats =3 will repeat 10 fold cross validation 3 times basically 30 times
  scores=cross_val_score(model,X,y,scoring='roc_auc',cv=cv, n_jobs=-1)#trains of 9 folds and 1 is used for testing
  return scores

X,y,cateorgorical,numerical=load_data()
print(X.shape,y.shape)

model=LinearDiscriminantAnalysis()

transformer = ColumnTransformer(
    transformers=[
        ('imp', SimpleImputer(strategy='median'), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cateorgorical)
    ])


 # ImbPipeline allows us to combine preprocessing, SMOTE oversampling, and the ML model safely
pipeline = ImbPipeline(steps=[
        ('t', transformer),
        ('p', PowerTransformer(method='yeo-johnson',standardize=True)),
        ('over', SMOTE(random_state=42)),
        ('m', model)
    ])

#to evaluate the model;
scores = evaluate_model(X, y, pipeline)
#print('LDA %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#creates a boxplot for plotting the data
plt.boxplot([scores],tick_labels=['LDA'],showmeans=True)
plt.show()

pipeline.fit(X, y)
#to sve the trained pipleline;from joblib import dump
dump(pipeline, 'stroke_prediction_model.joblib')