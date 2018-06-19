from Parser import Parser
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor


class Descriptor():

    def __init__(self):
        self.pars = Parser()

    def global_desc(self, proteins, labels, thold):

        desc_glo = GlobalDescriptor(proteins)
        desc_glo.calculate_all(amide=True)

        desc_pep = PeptideDescriptor(proteins)
        desc_pep.calculate_moment()

        correlated = {}
        index = 0

        for feature in desc_glo.descriptor.transpose()[1:]:
            index = index + 1
            if np.corrcoef(feature.reshape(1, len(desc_glo.descriptor)), labels)[0, 1] > thold:
                new_entry = {desc_glo.featurenames[index]: feature}
                correlated.update(new_entry)

        if np.corrcoef(desc_pep.descriptor.reshape(1, len(desc_pep.descriptor)), labels)[0, 1] > thold:
            new_entry = {'global_hydrophob': desc_pep.descriptor}
            correlated.update(new_entry)

        return correlated
