import pandas as pd
import re
import string
from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV



#applying some transformation on data like removing numbers,punctiation,stopwords then formatting
def clean_descr(recipe):

    recipe = ''.join([i for i in recipe if not i.isdigit()])
    recipe= ''.join([i for i in recipe if i not in string.punctuation and i !="®" ])
    recipe= word_tokenize(recipe.lower())
    stemmer = PorterStemmer()
    recipe=" ".join([stemmer.stem(item) for item in recipe if item not in stopwords.words('english') ])
    return recipe

#charging data then apply clean_descr
def load_data(path):
    data= pd.read_csv(path)
    data.cleanDescription= data.cleanDescription.apply(clean_descr)
    return data

def load_train(data):
    print("preprocess")
    df=data[['cleanDescription','Store Section']].dropna(axis=0)
    tf = TfidfVectorizer()
    X_train = df.cleanDescription
    y_train=df['Store Section']
    return X_train,y_train

def compile(X,y):

    print("compiling")
#--------------------------KNN-----------------------
    knn=make_pipeline(TfidfVectorizer(),KNeighborsClassifier())
    #knn.get_params().keys()
    param_knn={'kneighborsclassifier__n_neighbors':range(1,5),
            }
    grid_knn=GridSearchCV(knn,param_knn,cv=5)
    grid_knn.fit(X,y)
    knn=grid_knn.best_estimator_
    print("best score knn {:.3f} with parameters :{}".format(grid_knn.best_score_,grid_knn.best_params_))
    return knn

def load_test(data):
    X_test= data[data['Store Section'].isna()]
    return X_test




df=load_data('Sorted.csv')
X,y=load_train(df)
model=compile(X,y)
X_test=load_test(df).cleanDescription
y_pred=model.predict(X_test)
result={"id":load_test(df).id,"Store Section":y_pred}
result=pd.DataFrame(result)
result.to_csv('Store-result.csv',index=False)






