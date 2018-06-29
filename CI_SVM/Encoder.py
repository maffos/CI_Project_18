from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import scipy as sc
import DataComp


class Encoder:

    """
    Comprises methods to sparse encode features.

    Uses a feature set from DataComp to fit a sparse encoder. Default is amino_acids.
    """

    def __init__(self, features=DataComp.amino_acids):
        # Fits a LabelEncoder to 'features' parameter.
        self.int_encoder = LabelEncoder().fit(features)
        # Fits a OneHotEncoder (binary encoding) to the integer representation of 'features' parameter.
        self.sparse_encoder = OneHotEncoder().fit(self.int_encoder.transform(features).reshape(len(features), 1))

    def __int_encode_feature(self, feature):

        """
        Uses the SciKit LabelEncoder to encode categorical data to integer values.

        :param feature: Amino acid sequence or a list of physicochemical features.
        :return: Feature transformed to integer values.
        """

        return self.int_encoder.transform(feature)

    def sparse_encode_feature(self, feature):

        """
        Uses the SciKit OneHotEncoder to encode integer values to binary values in sparse matrix (csr) format.

        :param feature: Amino acid sequence or a list of physicochemical features.
        :return: Feature transformed to compressed sparse row matrix format.
        """

        # Transforms feature to compressed sparse column matrix.
        # All values of a feature a stored in one column of a matrix.
        csc_representation = self.sparse_encoder.transform(self.__int_encode_feature(feature).reshape(len(feature), 1))

        # Transforms feature to array-representation.
        # Necessary for correct transforamtion into csr format.
        array_representation = csc_representation.toarray()
        dimension = len(array_representation[0])*len(array_representation)
        array_representation = array_representation.reshape(dimension)

        # Transforms feature to compressed sparse row matrix.
        # All values of a feature a stored in one row of a matrix.
        csr_representation = sc.sparse.csr_matrix(array_representation)

        return csr_representation
