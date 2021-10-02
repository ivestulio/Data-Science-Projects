## LIBRARIES:

 #Manipulação de dados:
import pandas as pd 
import missingno as msno 
from collections import Counter
from warnings import filterwarnings


# Visualização Gráfica:
import seaborn as sns
import matplotlib as plt
import plotly.express as px 

# Modelos de Classificação 
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import VotingClassifier
# Evolução :
from sklearn.metrics import precision_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold

# IMPORTING DATASET 

data = pd.read_csv("heart.csv")
data.info()

# Visualizations 
pip install pandas-profiling 
from pandas_profiling import ProfileReport
profile = ProfileReport(data, title='Pandas Profiling Report to Dataset')
profile

#Data Tratament 

data.info()
fig = msno.matrix(data, color=(0,0.6,0.8))

Note:
There are some columns where the values are object, we need transform this values in numerical.

# Sex
data["Sex"] = data["Sex"].map({"M":1, "F":2})
data.head()

# ChestPain Type 
data["ChestPainType"] = data["ChestPainType"].map({"TA":1, "ATA":2, "NAP":3,"ASY":4})
data.head()

# Resting ECG
data["RestingECG"] = data['RestingECG'].map({"Normal":1,"ST":2, "LVH":3 })
data.head()

# Exercise Aginine 
data["ExerciseAngina"] = data['ExerciseAngina'].map(  {"Y":1,"N":2  })
data.head()

# ST Slope
data["ST_Slope"] = data['ST_Slope'].map({"Up":1, "Flat":2, "Down": 3}  )
data.head()

# MODELING:

X = data.drop("HeartDisease", axis=1 )
X.head()
y = data["HeartDisease"]
y.head()

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state = 15)
len(x_test),len(x_train)

# Spot Checking:

filterwarnings('ignore')

models = [("LR", LogisticRegression(max_iter=1000)),
          ("SVC", SVC()),
          ("KNC", KNeighborsClassifier(n_neighbors=10)),
          ("DTC", DecisionTreeClassifier()),
          ("GNB", GaussianNB()),
          ("SGDC", SGDClassifier()),
          ("Perc", Perceptron()),
          ("NC", NearestCentroid()),
          ("Ridge", RidgeClassifier()),
          ("NuSVC", NuSVC()),
          ("BNB", BernoulliNB()),
          ("RF", RandomForestClassifier()),
          ("ADA", AdaBoostClassifier()),
          ("XGB", GradientBoostingClassifier()),
          ("PAC", PassiveAggressiveClassifier()) 
    
]

results = []
names=[]
finalresults=[]

for name, model in models:
    model.fit(x_train, y_train)
    model_results = model.predict(x_test)
    score= precision_score(y_test, model_results, average='macro')
    results.append(score)
    names.append(name)
    finalresults.append((name,score))
    
finalresults.sort(key=lambda k:k[1], reverse=True)
finalresults

# HYPERMARAMETER TUNING:

# Grid search and space:
models_params= {
    "RF":{'model':RandomForestClassifier(),
         'params':{
             'max_features': list(range(1,10)),
             'n_estimators':[10,100,1000]
         }},
    'Ridge':{'model':RidgeClassifier(),
           'params':{
               'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'],
    
           }},
    'XGB':{'model':GradientBoostingClassifier(),
           'params':{
            'learning_rate':[0.0001,0.001,0.01,0.1],
            'n_estimators':[100,200,500,1000],
            'max_features':['sqrt','log2'],
            'max_depth':list(range(11))                        
               
           }}
    
}

# Evaluation:

cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=20)

# Search:

scores=[]

for model_name, params in models_params.items(): 
    rs = RandomizedSearchCV(params['model'], params['params'], cv=cv , n_iter=10)
    rs.fit(x_train,y_train)
    scores.append([model_name,dict(rs.best_params_),rs.best_score_])
data=pd.DataFrame(scores,columns=['Model','Parameters','Score'])
data

# FINAL MODEL :

XGB = GradientBoostingClassifier(n_estimator=500,max_features = sqrt)
XGB.fit(x_train,y_train)
Heart_failure = XGB.predict(x_test)



    
