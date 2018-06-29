"""
Stores static data used to encode amino acids and physicochemical features for amino acids.
"""

# Alphabet of amino acids
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Numerical descriptor values. Calculated with the ModlAMP library.
modlamp_features = {

    'ar': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],

    'bi': [-1.81, 14.92, 6.64, 8.72, -1.28, 5.54, 6.81, -0.94, 4.66, -4.92,
           -4.92, 5.55, -2.35, -2.98, 0.0, 3.4, 2.57, -2.33, 0.14, -4.04],

    'hy': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
           1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],

    'al': [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 390.0,
           390.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 290.0],

    'mw': [89.09, 174.2, 132.12, 133.1, 121.16, 146.15, 147.13, 75.07, 155.16, 131.17,
           131.17, 146.19, 149.21, 165.19, 115.13, 105.09, 119.12, 204.22, 181.19, 117.15]

}

# Categorical descriptor values. See README for more details.
profeat_features = {

    'hy': [1, 0, 0, 0, 2, 0, 0, 1, 1, 2, 2, 0, 2, 2, 1, 1, 1, 1, 1, 2],

    'vv': [0, 2, 1, 0, 0, 1, 1, 0, 2, 1, 1, 2, 2, 2, 0, 0, 0, 2, 2, 1],

    'pl': [1, 2, 2, 2, 0, 2, 2, 1, 2, 0, 0, 2, 0, 0, 1, 1, 1, 0, 0, 0],

    'pa': [0, 2, 1, 0, 1, 1, 1, 0, 2, 1, 1, 2, 2, 2, 1, 0, 0, 2, 2, 1],

    'ch': [1, 2, 1, 0, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],

    'sa': [0, 2, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 2, 1, 1, 0, 1, 0]

}


def export_features(features, feature_dictionary):

    """
    Instantiates a new dictionary with chosen features (value) for all amino acids (keys).

    Is used by Descriptor class for feature annotation, if wanted.

    :param features: A list of features specified by the user.
    :param feature_dictionary: A static dictionary to extract features from.
    :return: new_feature_list: Feature space as list, new_feature_dict: Composed feature dictionary.
    """

    new_feature_list = []
    new_feature_dict = {}

    for feature in features:

        # Instantiates a new feature list for chosen features. Used for sparse encoding.
        [new_feature_list.append(value) for value in feature_dictionary[feature]]

    for index in range(0, len(amino_acids)):

        new_dict_entry = []

        for feature in features:

            # Extracts all selected features for a specific amino acid.
            new_dict_entry.append(feature_dictionary[feature][index])

        # Appends the chosen features (values) and specific amino-acid (key) to dictionary.
        new_feature_dict.update({amino_acids[index]: new_dict_entry})

    return new_feature_list, new_feature_dict
