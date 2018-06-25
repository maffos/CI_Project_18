'''import libraries'''
from modlamp.descriptors import GlobalDescriptor

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

dictionary = {}

for aa in amino_acids:
    feature = []
    desc = GlobalDescriptor(aa)

    desc.calculate_charge(ph=6.0, amide=False)
    feature.append(desc.descriptor[0][0])

    desc.aromaticity()
    feature.append(desc.descriptor[0][0])

    desc.boman_index()
    feature.append(desc.descriptor[0][0])

    desc.hydrophobic_ratio()
    feature.append(desc.descriptor[0][0])

    desc.aliphatic_index()
    feature.append(desc.descriptor[0][0])

    desc.calculate_MW(amide=False)
    feature.append(desc.descriptor[0][0])

    entry = {aa: feature}
    dictionary.update(entry)

'''
Results
{'A': [-0.0, 0.0, -1.81, 1.0, 100.0, 89.09],
 'R': [1.0, 0.0, 14.92, 0.0, 0.0, 174.2],
 'N': [-0.0, 0.0, 6.64, 0.0, 0.0, 132.12],
 'D': [-0.995, 0.0, 8.72, 0.0, 0.0, 133.1],
 'C': [-0.007, 0.0, -1.28, 1.0, 0.0, 121.16],
 'Q': [-0.0, 0.0, 5.54, 0.0, 0.0, 146.15],
 'E': [-0.986, 0.0, 6.81, 0.0, 0.0, 147.13],
 'G': [-0.0, 0.0, -0.94, 0.0, 0.0, 75.07],
 'H': [0.523, 0.0, 4.66, 0.0, 0.0, 155.16],
 'I': [-0.0, 0.0, -4.92, 1.0, 390.0, 131.17],
 'L': [-0.0, 0.0, -4.92, 1.0, 390.0, 131.17],
 'K': [1.0, 0.0, 5.55, 0.0, 0.0, 146.19],
 'M': [-0.0, 0.0, -2.35, 1.0, 0.0, 149.21],
 'F': [-0.0, 1.0, -2.98, 1.0, 0.0, 165.19],
 'P': [-0.0, 0.0, 0.0, 0.0, 0.0, 115.13],
 'S': [-0.0, 0.0, 3.4, 0.0, 0.0, 105.09],
 'T': [-0.0, 0.0, 2.57, 0.0, 0.0, 119.12],
 'W': [-0.0, 1.0, -2.33, 0.0, 0.0, 204.22],
 'Y': [-0.0, 1.0, 0.14, 0.0, 0.0, 181.19],
 'V': [-0.0, 0.0, -4.04, 1.0, 290.0, 117.15]}
'''