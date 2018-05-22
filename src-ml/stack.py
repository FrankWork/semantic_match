from sklearn import datasets  
  
iris = datasets.load_iris()  
X, y = iris.data[:, 1:3], iris.target  
  
from sklearn import model_selection  
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB   
from sklearn.ensemble import RandomForestClassifier  
from mlxtend.classifier import StackingClassifier  
import numpy as np  
  
knn = KNeighborsClassifier(n_neighbors=1)  
rf = RandomForestClassifier(random_state=1)  
nb = GaussianNB()  
lr = LogisticRegression()  
sclf = StackingClassifier(classifiers=[knn, rf, nb],
                          use_probas=True,  
                          average_probas=True,  
                          meta_classifier=lr)  
  
print('3-fold cross validation:\n')  
  
for clf, label in zip([knn, rf, nb, sclf],
                      ['KNN',
                      'Random Forest',   
                      'Naive Bayes',  
                      'StackingClassifier']):  
  
    scores = model_selection.cross_val_score(clf, X, y,   
                                              cv=3, scoring='accuracy')  
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"   
          % (scores.mean(), scores.std(), label))

# 3-fold cross validation:

# Accuracy: 0.91 (+/- 0.01) [KNN]
# Accuracy: 0.91 (+/- 0.06) [Random Forest]
# Accuracy: 0.92 (+/- 0.03) [Naive Bayes]
# Accuracy: 0.95 (+/- 0.03) [StackingClassifier]
