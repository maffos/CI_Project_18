import Preprocessing

'''
TODO: Description
'''


def predict(file_path, fitted_svm, fitted_sparse_encoder, fitted_int_encoder):

    """
    Predict the data from file_path.

    TODO: Tag description
    :param file_path:
    :param fitted_svm:
    :param fitted_sparse_encoder:
    :param fitted_int_encoder:
    :return:
    """

    features = Preprocessing.feature_extraction_sparse_predict(file_path, fitted_sparse_encoder, fitted_int_encoder)

    print("Prediction: " + str(fitted_svm.predict(features)))