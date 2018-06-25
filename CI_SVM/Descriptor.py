import scipy as sc
from Parser import Parser
# from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from sklearn.preprocessing import OneHotEncoder


class Descriptor():

    def __init__(self):
        self.pars = Parser()
        self.feature_dictionary = {
            'A': [28, 12, 13, 17, 29, 5],
            'R': [44, 16, 13, 13, 13, 26],
            'N': [36, 12, 13, 13, 13, 23],
            'D': [37, 7, 13, 13, 13, 25],
            'C': [34, 10, 13, 17, 13, 6],
            'Q': [38, 12, 13, 13, 13, 21],
            'E': [40, 8, 13, 13, 13, 24],
            'G': [27, 12, 13, 13, 13, 9],
            'H': [42, 14, 13, 13, 13, 20],
            'I': [35, 12, 13, 17, 48, 0],
            'L': [35, 12, 13, 17, 48, 0],
            'K': [39, 16, 13, 13, 13, 22],
            'M': [41, 12, 13, 17, 13, 3],
            'F': [43, 12, 17, 17, 13, 2],
            'P': [31, 12, 13, 13, 13, 13],
            'S': [30, 12, 13, 13, 13, 19],
            'T': [33, 12, 13, 13, 13, 18],
            'W': [46, 12, 17, 13, 13, 4],
            'Y': [45, 11, 17, 13, 13, 15],
            'V': [32, 12, 13, 17, 47, 1]
        }

        self.feature_encoder = OneHotEncoder().fit(list(self.feature_dictionary.values()))

    def annotate_protein(self, protein):
        aa_features = []

        for aa in protein:
            aa_features.append(self.feature_dictionary[aa])

        encoded_features = self.feature_encoder.transform(aa_features)
        encoded_features_array = encoded_features.toarray().reshape(477)

        return sc.sparse.csr_matrix(encoded_features_array)
