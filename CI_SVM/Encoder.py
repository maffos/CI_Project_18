from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy as sc

class Encoder():

    def __init__(self):
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                            'Y', 'V']
        self.int_encoder = LabelEncoder().fit(self.amino_acids)
        self.oh_encoder = OneHotEncoder().fit(
            self.int_encoder.transform(self.amino_acids).reshape(len(self.amino_acids), 1))

    def int_encode_protein(self, protein):
        return self.int_encoder.transform(list(protein))

    def int_encode_proteins(self, proteins):

        int_encoded_proteins = []

        for protein in proteins:
            int_encoded_proteins.append(self.int_encode_protein(protein))

        return int_encoded_proteins

    def bin_encode_protein(self, protein):

        csc_representation = self.oh_encoder.transform(self.int_encode_protein(protein).reshape(len(protein), 1))
        array_representation = csc_representation.toarray().reshape(180)
        csr_representation = sc.sparse.csr_matrix(array_representation)

        return csr_representation

    def bin_encode_proteins(self, proteins):

        bin_encoded_proteins = self.bin_encode_protein(proteins[0])

        for protein in proteins[1:]:
            bin_encoded_proteins = sc.sparse.vstack((bin_encoded_proteins, self.bin_encode_protein(protein)),
                                                    format='csr')

        return bin_encoded_proteins
