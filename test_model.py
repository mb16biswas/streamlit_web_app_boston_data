import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#python test_model.py
import pickle 
from sklearn.datasets import load_boston
boston = load_boston()
boston_df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
print(boston_df.head())
boston_df["target"] = pd.Series(boston["target"])
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)        
print(rf.score(X_test , y_test))    
pickle.dump(rf, open('boston_model.pkl', 'wb'))                                        
