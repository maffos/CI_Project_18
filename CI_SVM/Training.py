import Preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

'''
TODO: Description
'''


def train_model(features, labels):



def train_sparse(file_path):

    """
    Fits an svm to a sparse representation of the features.
    The regularization constant C and the kernel are determined by grid search with 10 fold cross validation.

    TODO: Tag description
    :param file_path:
    :return: best_svm:
    """

    new_svm = SVC()
    features, labels, new_sparse_encoder, new_int_encoder = Preprocessing.feature_extraction_sparse_train(file_path)

    param_grid = {'kernel': ['poly', 'linear', 'rbf'], 'C': [0.1, 0.5, 0.9, 1, 2]}
    best_svm = GridSearchCV(new_svm, param_grid, cv=10, scoring='roc_auc')
    best_svm.fit(features, labels)

    return best_svm, new_sparse_encoder, new_int_encoder


def train_dimension_reduction(file_path):

    """
    Trains an SVM by applying dimensionality reduction to the features first.
    Hyperparameters, e.g. dimensionality of features, regularization constant C and
    the kernel being used are determined by Grid search, with 10-fold CV.

    :param file_path:
    :return: best_svm:
    """

    pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', SVC())])
    param_grid = {'reduce_dim__n_components': [70, 75, 85, 100, 120],
                  'classification__kernel': ['poly', 'linear', 'rbf'], 'classification__C': [0.1, 0.5, 1, 2]}

    best_svm = GridSearchCV(pipe, param_grid, cv=10, scoring='roc_auc')
    features, labels, new_sparse_encoder, new_int_encoder = Preprocessing.feature_extraction_sparse_train(file_path)
    best_svm.fit(features, labels)

    return best_svm, new_sparse_encoder, new_int_encoder


def cross_validate():
