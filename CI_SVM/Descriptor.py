from Encoder import Encoder
import scipy as sc
import DataComp


class Descriptor:

    """
    Comprises methods to encode proteins and annotate them, if specified.

    Uses Encoder instances to transform a String list of proteins and
    possible component wise annotations to a compressed sparse row format.
    """

    def __init__(self, modlamp=None, profeat=None):
        # Composes a new dictionary of amino acids and fits a corresponding Encoder.
        self.aa_enc = Encoder()
        self.modlamp_enc = None
        self.profeat_enc = None

        # If specified, composes a new dictionary of ModlAMP features and fits a corresponding Encoder.
        if modlamp is not None:
            modlamp_list, self.modlamp_dict = DataComp.export_features(modlamp, DataComp.modlamp_features)
            self.modlamp_enc = Encoder(features=modlamp_list)

        # If specified, composes a new dictionary of PROFEAT features and fits a corresponding Encoder.
        if profeat is not None:
            profeat_list, self.profeat_dict = DataComp.export_features(profeat, DataComp.profeat_features)
            self.profeat_enc = Encoder(features=profeat_list)

    def encode_protein(self, protein):

        """
        Uses Encoder objects to transforms a protein (String) to compressed sparse row matrix format (Binary values).
        Appends ModlAMP and/or PROFEAT features if specified.

        :param protein: Protein sequence to transform.
        :return: Protein transformed to csr matrix format.
        """

        aa_encoded = self.aa_enc.sparse_encode_feature(list(protein))
        modlamp_encoded = []
        profeat_encoded = []

        # Appends ModlAMP features if specified.
        if self.modlamp_enc is not None:

            feature = []

            for aa in list(protein):

                for value in self.modlamp_dict[aa]:

                    feature.append(value)

            modlamp_encoded = self.modlamp_enc.sparse_encode_feature(feature)

        # Appends PROFEAT features if specified.
        if self.profeat_enc is not None:

            feature = []

            for aa in list(protein):

                for value in self.profeat_dict[aa]:

                    feature.append(value)

            profeat_encoded = self.profeat_enc.sparse_encode_feature(feature)

        # Stacks all encoded features with the encoded protein representation.
        return sc.sparse.hstack((aa_encoded, modlamp_encoded, profeat_encoded), format='csr')

    def encode_proteins(self, proteins):

        """
        Applies the encode_protein method to a list of proteins.

        :param proteins: List of proteins (String).
        :return: Proteins encoded to csr matrix format.
        """

        sparse_encoded_proteins = self.encode_protein(proteins[0])

        for protein in proteins[1:]:

            sparse_encoded_proteins = sc.sparse.vstack((sparse_encoded_proteins,
                                                        self.encode_protein(protein)), format='csr')

        return sparse_encoded_proteins
