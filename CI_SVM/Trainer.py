from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


class Trainer():

    def __init__(self):
        self.pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', SVC())])
        self.cv_params = {'reduce_dim__n_components': [75],
                          'classification__kernel':['linear'],
                          'classification__C':[0.2]}
        self.folds = 20

    def train(self, proteins, labels):

        trained_svm = GridSearchCV(self.pipe, self.cv_params, cv=self.folds, scoring='accuracy')
        trained_svm.fit(proteins, labels)

        return trained_svm

    def cross_validate(self, proteins, labels):

        trained_svm = self.train(proteins, labels)
        best_params = trained_svm.best_params_
        print(best_params)
        unfitted_svm = svm.SVC(kernel=best_params['classification__kernel'],
                               C=best_params['classification__C'])

        return cross_val_score(unfitted_svm, proteins, labels, cv=self.folds, scoring='accuracy')