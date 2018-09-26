'''# python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))'''

'''# dataframe
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)'''

'''# Load CSV using Pandas from URL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler'''

'''url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
y=data['class']
x=data.drop('class',axis=1)
labels=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']

array=data.values
X = array[:,0:8]
Y = array[:,8]'''

#print(data.describe())
#scatter_matrix(data)
#plt.show()
'''for i in labels:
	#print(x[i])
	plt.plot(x[i],y)
	plt.show()'''

#scaler = StandardScaler().fit(X)
#print(scaler.mean_,'\n')
#print(scaler.var_,'\n')
#print(scaler.scale_,'\n')
#rescaledX = scaler.transform(X)
#print(rescaledX,'\n')
# summarize transformed data
#np.set_printoptions(precision=4)
#print(rescaledX[0:5,:],'\n')

'''from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

kfold = KFold(n_splits=10, random_state=7)
scoring = 'neg_log_loss'
#scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold,scoring=scoring)
print(results.mean(),'\n')
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

import sklearn.model_selection
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)
model=model.fit(x_train,y_train)
print("Log. Reg. Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')

scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold,scoring=scoring)'''

'''final_pred=model.predict(x_test)
#print("Log. Reg. Pred. Accuracy-->%.3f%%"%(model.score(y_test,final_pred)*100),'\n')
y_test=y_test.values
y_test=y_test.reshape(-1,1)
final_pred=final_pred.reshape(-1,1)
results = cross_val_score(model, y_test, final_pred, cv=kfold,scoring=scoring)
print(results.mean(),'\n')'''

#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import log_loss

#print(log_loss(y_test,final_pred),'\n')
'''print(classification_report(y_test,final_pred),'\n')
c_m=confusion_matrix(y_test,final_pred,sample_weight=None)
print(c_m)
tp=c_m[0:1,0:1]
fn=c_m[1:2,0:1]
fp=c_m[0:1,1:2]
tn=c_m[1:2,1:2]
print('tp=',tp,'fn=',fn,'fp=',fp,'tn=',tn,'\n')
recall=tp/(fn+tp)
precise=tp/(fp+tp)
specificity=tn/(fp+tn)
print("recall-->"recall,"precision-->",precise,"specificity-->"specificity,'\n')
plt.plot(fp_rate,tp_rate)
plt.show()'''

'''#root mean square method
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print(rmse(final_pred,y_test))'''

#Linear regression
'''from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("Linear Reg. Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')'''

#Linear Discriminant Analysis
'''from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("LDA Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')'''

#KNN regressor
'''from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_jobs=4)
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("KNN Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')'''

#SVM:svc
'''from sklearn.svm import SVC
model = SVC()
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("SVC Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')'''

#CART(Classification & Regression Trees)
'''from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='gini',random_state=0,max_depth=6,min_samples_leaf=2)
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("Tree Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("Forest Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')

#Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
print(results.mean(),'\n')
model=model.fit(x_train,y_train)
print("Grad. boost Accuracy-->%.3f%%"%(model.score(x_test,y_test)*100),'\n')'''

#COMPARING ALGORITHMS
'''from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load dataset
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare models
models = []
#models.append(('LR', LinearRegression()))
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsRegressor(n_jobs=4)))
models.append(('SVM', SVC()))
models.append(('CART', DecisionTreeClassifier(criterion='gini',random_state=0,max_depth=6,min_samples_leaf=2)))
models.append(('Forest', RandomForestClassifier()))
models.append(('Grad boost', GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=7)
for name, model in models:
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print('%s: %.4f%% (%.4f%%)' % (name, cv_results.mean()*100, cv_results.std()*100))'''
#end

'''# Grid Search for Algorithm Tuning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_,'\n');

grid = RandomizedSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_);'''

'''# Random Forest Classification(Improving Accuracy)
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print("Forest-->",results.mean())

#Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
results = cross_val_score(model, X, Y, cv=kfold)
print("Grad. Boost-->",results.mean())'''