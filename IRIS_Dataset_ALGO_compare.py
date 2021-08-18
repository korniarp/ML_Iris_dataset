import sys
import pandas
import numpy
import scipy
import sklearn
import matplotlib

from pandas.plotting import scatter_matrix
from pandas import read_csv

from matplotlib import plotter1
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




# Loading the IRIS features_output
link_url = "https://raw.githubusercontent.com/jbrownlee/features_outputs/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
features_output = read_csv(link_url, names=names)

print(features_output.shape)

print(features_output.head(20))

print(features_output.describe())


print(features_output.groupby('class').size())

plotter1.figure(0)
features_output.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plotter1.show()

plotter1.figure(1)
features_output.hist()
plotter1.show()

plotter1.figure(2)
scatter_matrix(features_output)
plotter1.show()

# Split-out validation features_output
features_out = features_output.values
X = features_out[:, 0:4]
y = features_out[:, 4]
X_practise, X_practise_validation, Y_practise, Y_practise_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_final_outcome = cross_val_score(model, X_practise, Y_practise, cv=kfold, scoring='accuracy')
	results.append(cv_final_outcome)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_final_outcome.mean(), cv_final_outcome.std()))


# Compare Algorithms
plotter1.boxplot(results, labels=names)
plotter1.title('Algorithm performace measure')
plotter1.show()

# Make estimation on validation features_output
model = SVC(gamma='auto')
model.fit(X_practise, Y_practise)
estimation = model.predict(X_practise_validation)

print(accuracy_score(Y_practise_validation, estimation))
print(confusion_matrix(Y_practise_validation, estimation))
print(classification_report(Y_practise_validation, estimation))