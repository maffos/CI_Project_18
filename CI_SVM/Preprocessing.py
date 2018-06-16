from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

'''
TODO: Class Description
'''


def parse(file_path, labeled):

    """
    Returns a 2D list with shape(n_peptides,9_aas)

    TODO: Tag description
    :param file_path:
    :param labeled:
    :return:
    """

    global labels
    proteins = []

    if labeled:
        labels = []

    with open(file_path, "r") as raw_data:

        if labeled:
            raw_data.readline()

        for line in raw_data:
            proteins.append(line.split()[0])

            if labeled:
                labels.append(line.split()[2])

    if labeled:
        int_labels = list(map(int, labels))
        return proteins, int_labels

    else:
        return proteins


def int_encode_train(proteins):

    """
    Receives a 2D list of shape(n_peptides,9_aas).
    Method should be used for training a new model.

    TODO: Tag description
    :param proteins:
    :return:
    """

    # Each amino acid is a new object in the list aa_s
    aa_s = []

    for protein in proteins:
        for aa in protein:
            aa_s.append(aa)

    # Fits a new LabelEncoder on aa's.
    new_int_encoder = LabelEncoder()
    # Encode the amino acids with an integer
    aa_s_int_encoded = new_int_encoder.fit_transform(aa_s)

    return new_int_encoder, aa_s_int_encoded


def int_encode_predict(proteins, fitted_int_encoder):

    """
    Receives a 2D list of shape(n_peptides,9_aas) and returns an integer representation of the input.
    Method should be used for predicting new data.

    TODO: Tag description
    :param proteins:
    :param fitted_int_encoder:
    :return:
    """

    # Each amino acid is a new object in the list aa_s
    aa_s = []

    for protein in proteins:
        for aa in protein:
            aa_s.append(aa)

    # Encode the amino acids with an integer
    aa_s_int_encoded = fitted_int_encoder.transform(aa_s)

    return aa_s_int_encoded


def sparse_encode_train(proteins):

    """
    Receives a 2D list of shape(n_proteins,9_aas) with int encoded aa's and returns
    a fitted OneHotEncoder together with the sparse encoded representation.
    Method should be used for training a new model.

    TODO: Tag description
    :param
    :return:
    """

    # Fits a new OneHotEncoder on int encoded proteins.
    new_bin_encoder = OneHotEncoder()

    # one-hot-encode the int encoded proteins.
    proteins_bin_encoded = new_bin_encoder.fit_transform(proteins)

    return new_bin_encoder, proteins_bin_encoded


def sparse_encode_predict(proteins, fitted_bin_encoder):

    """
    Receives a 2D list of shape(n_proteins,9_aas) with int encoded proteins.
    Returns the sparse encoded representation.
    Method should be used for predicting new data.

    TODO: Tag description
    :param
    :param
    :return:
    """

    # one-hot-encode the proteins.
    proteins_bin_encoded = fitted_bin_encoder.transform(proteins)

    return proteins_bin_encoded


def recover_from_sparse(bin_encoder, int_encoder, proteins_bin_encoded, n_proteins):

    """
    Recovers sparse encoded peptides.

    TODO: Tag description
    :param bin_encoder:
    :param int_encoder:
    :param proteins_bin_encoded:
    :param n_proteins:
    :return:
    """

    int_encoded = np.array(
        [bin_encoder.active_features_[col] for col in proteins_bin_encoded.sorted_indices().indices]).reshape(
        n_proteins, 9) - bin_encoder.feature_indices_[:-1]

    recovered = int_encoder.inverse_transform([int_encoded])[0]

    return recovered


def feature_extraction_sparse_train(file_path):

    """
    Extract sparse features from the raw data and return every encoding and processing model that was used.
    Method should be used for training a new model.

    TODO: Tag description
    :param file_path:
    :return:
    """

    features, labels_ = parse(file_path, True)
    n = len(features)
    int_encoder, int_encoded = int_encode_train(features)
    int_encoded = int_encoded.reshape(n, 9)
    sparse_encoder, features = sparse_encode_train(int_encoded)

    return features, labels_, sparse_encoder, int_encoder


def feature_extraction_sparse_predict(file_path, sparse_encoder, int_encoder):

    """
    Extract sparse features from the raw data.
    Method should be used for predicting new data.

    TODO: Tag description
    :param file_path:
    :param sparse_encoder:
    :param int_encoder:
    :return:
    """

    features = parse(file_path, False)
    n = len(features)
    int_encoded = int_encode_predict(features, int_encoder)
    int_encoded = int_encoded.reshape(n, 9)
    features = sparse_encode_predict(int_encoded, sparse_encoder)

    return features
