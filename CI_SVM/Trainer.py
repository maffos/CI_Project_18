from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


class Trainer():

    def __init__(self):
        self.svm = svm.SVC()
        self.cv_params = {'kernel': ['rbf'], 'C': [0.1, 0.5, 0.9, 1, 10, 100]}
        self.folds = 10

    def train(self, proteins, labels):

        trained_svm = GridSearchCV(self.svm, self.cv_params, cv=self.folds, scoring='roc_auc')
        trained_svm.fit(proteins, labels)

        return trained_svm

    def cross_validate(self, proteins, labels):

        trained_svm = self.train(proteins, labels)
        best_params = trained_svm.best_params_
        print(best_params)
        unfitted_svm = svm.SVC(kernel=best_params['kernel'], C=best_params['C'])

        return cross_val_score(unfitted_svm, proteins, labels, cv=self.folds, scoring='roc_auc')