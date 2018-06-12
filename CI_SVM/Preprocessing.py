from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#returns a sparse matrix with a binary representation for the peptides together with the associated labels
def parse(filename):
	proteins=[]
    	labels = []
#read the peptides from the file	
	with open(filename,"r") as raw_data:
		raw_data.readline()
    		for line in raw_data:
        		proteins.append(line.split()[0])
        		labels.append(line.split()[2])
	encoded_proteins = onehot_encode(proteins)
	return encoded_proteins,labels

def onehot_encode(proteins):
	n_proteins = len(proteins)
#encode each amino acid with an integer
	aa_s = []	
	for protein in proteins:
		for aa in protein:
			aa_s.append(aa)
	int_encoder = LabelEncoder()
	proteins_int_encoded = int_encoder.fit_transform(aa_s)
#one hot encode the amino acids
	proteins_int_encoded = proteins_int_encoded.reshape(n_proteins,9)
	bin_encoder = OneHotEncoder()
	proteins_bin_encoded = bin_encoder.fit_transform(proteins_int_encoded)
    	return proteins_bin_encoded

#recover the encoded peptides
def recover(bin_encoder,int_encoder,proteins_bin_encoded, n_proteins):
	int_encoded = np.array([bin_encoder.active_features_[col] for col in proteins_bin_encoded.sorted_indices().indices]).reshape(n_proteins,9) - bin_encoder.feature_indices_[:-1]
	recovered=int_encoder.inverse_transform([recovered])[0]
    	return recovered

if __name__ == "__main__":
	parse("project_training.txt")
