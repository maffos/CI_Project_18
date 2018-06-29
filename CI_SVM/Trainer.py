from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import TruncatedSVD
import scipy as sc


class Trainer:

    """
    Comprises methods to train a svm-model to input-data and evaluate its performance.

    Uses CVGridSearch of SciKit library for parameter optimization of a svm model.
    Implements cross validation to evaluate the performance of an trained svm model.
    Implements dimensionality reduction on input data.
    """

    def __init__(self, dim_reduction=False, dim_reduction_n=2,
                 model_params=None, folds=10, scoring='accuracy'):

        self.params = {}
        self.dim_reduction = dim_reduction

        # Checks if dimensionality reduction is specified.
        if self.dim_reduction:
            self.svd = TruncatedSVD(n_components=dim_reduction_n, n_iter=2, random_state=1)

        # Sets model and parameters to use for grid search.
        self.model = SVC()
        self.params.update(model_params)

        # Sets value for cross validation folds.
        self.folds = folds

        # Sets value for used scoring metric.
        self.scoring = scoring

    def train(self, proteins, labels):

        """
        Supervised learning method to train a svm model on sample data.

        'proteins' represents the features to train on,
        'labels' represents the binary labels of features to train on.

        Runs a grid search with the specified parameters.

        :param proteins: A representation of protein sequences.
        :param labels: A Integer list of labels to use for supervised learning.
        :return: model: Trained svm model.
        """

        # Instantiates a GridSearchCV object with the chosen parameters.
        model = GridSearchCV(self.model, self.params, cv=self.folds, scoring=self.scoring)

        # Runs dimensionality reduction if enabled.
        if self.dim_reduction:

            dim_red_proteins = sc.sparse.csr_matrix(self.svd.fit_transform(proteins))

            # Fits the model to dimensional reduced features.
            model.fit(dim_red_proteins, labels)

        else:

            # Fits the model to features with original dimension.
            model.fit(proteins, labels)

        # Information for user.
        print('>Training completed')
        print("Best params: ", end="", flush=True)
        print(model.best_params_)

        return model

    def cross_validate_svm(self, parameterized_model, proteins, labels):

        """
        Runs cross validation with a parameterized, unfitted model.

        Generates output for user.

        :param parameterized_model:
        :param proteins: A representation of protein sequences.
        :param labels: A Integer list of labels to use for supervised learning.
        :return: None
        """

        # Extract best estimated parameters of the passed model.
        best_params = parameterized_model.best_params_

        # Instantiates a new SVC object with best parameters.
        unfitted_svm = SVC(kernel=best_params['kernel'], C=best_params['C'],
                               gamma=best_params['gamma'], degree=best_params['degree'], coef0=best_params['coef0'])

        # Runs a cross validation on the new instantiated SVC object.
        score = cross_val_score(unfitted_svm, proteins, labels, cv=self.folds, scoring=self.scoring)

        # Information for user.
        print('>Scores')
        print("Mean:    ", end="", flush=True)
        print(score.mean())
        print("Std:     ", end="", flush=True)
        print(score.std())
        print()

        return None
