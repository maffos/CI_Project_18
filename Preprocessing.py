from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from scipy import sparse
import Encodings
import numpy as np

#returns a 2d list with shape(n_peptides,9_aas)
def parse(filename, txtfile):
	proteins=[]
	if txtfile:    	
		labels = []
#read the peptides from the file	
	with open(filename,"r") as raw_data:		
		raw_data.readline()
		if not txtfile:
			for line in raw_data:
				proteins.append(line.split(",")[0])   
		else:		
			for line in raw_data:
				proteins.append(line.split()[0])        		
				labels.append(line.split()[2])
	if txtfile:	
		return proteins,labels
	else:
		return proteins

def parse_regression(filename):
	proteins=[]
	labels = []
#read the peptides from the file	
	with open(filename,"r") as raw_data:		
		raw_data.readline()
		for line in raw_data:
			proteins.append(line.split()[0])        		
			labels.append(float(line.split()[1]))	
	return proteins,labels

#receives a 2d list of shape(n_peptides,9_aas) and returns the encoder together with an integer representation of the input
def int_encode_train(proteins):
#each amino acid is a new object in the list aa_s	
	aa_s = []	
	for protein in proteins:
		for aa in protein:
			aa_s.append(aa)
#encode the amino acids with an integer	
	int_encoder = LabelEncoder()
	int_encoded = int_encoder.fit_transform(aa_s)
	return int_encoder,int_encoded

#receives a 2d list of shape(n_peptides,9_aas) and returns an integer representation of the input
def int_encode_test(proteins,int_encoder):
#each amino acid is a new object in the list aa_s	
	aa_s = []	
	for protein in proteins:
		for aa in protein:
			aa_s.append(aa)
#encode the amino acids with an integer	
	int_encoded = int_encoder.transform(aa_s)
	return int_encoded

#receives a 2d list of shape(n_proteins,9_aas) with int encoded proteins and returns the encoder together with the sparse encoded representation
def sparse_encode_train(proteins):
#one hot encode the amino acids
	bin_encoder = OneHotEncoder()
	bin_encoded = bin_encoder.fit_transform(proteins)
    	return bin_encoder,bin_encoded

#receives a 2d list of shape(n_proteins,9_aas) with int encoded proteins and returns the sparse encoded representation
def sparse_encode_test(proteins,bin_encoder):
#one hot encode the amino acids
	bin_encoded = bin_encoder.transform(proteins)
    	return bin_encoded

# encodes amino acids with a 6-character string representing the physicochemical properties
def encode_physicochemical_train(proteins):	
#aa_s doesn't save the peptides as a single string. Every letter is an entry	
	proteins_encoded = []
	for protein in proteins:
		aa_s = []	
		for aa in protein:
			aa_s.append(Encodings.aa_to_physicochemical[aa])
		proteins_encoded.append(np.concatenate(aa_s))
	print len(proteins_encoded[0])
	#onehot_encoder = OneHotEncoder(categorical_features=[5])
	#proteins_encoded = onehot_encoder.fit_transform(proteins_encoded)
	return proteins_encoded

def encode_physicochemical_test(proteins,onehot_encoder):	
#aa_s doesn't save the peptides as a single string. Every letter is an entry	
	proteins_encoded = []
	for protein in proteins:
		aa_s = []	
		for aa in protein:
			aa_s.append(Encodings.aa_to_physicochemical[aa])
		proteins_encoded.append(np.concatenate(aa_s))
	proteins_encoded = onehot_encoder.transform(proteins_encoded)
	return proteins_encoded,onehot_encoder

#recover the encoded peptides
def recover_from_sparse(bin_encoder,int_encoder,proteins_bin_encoded, n_proteins):
	int_encoded = np.array([bin_encoder.active_features_[col] for col in proteins_bin_encoded.sorted_indices().indices]).reshape(n_proteins,9) - bin_encoder.feature_indices_[:-1]
	recovered=int_encoder.inverse_transform([int_encoded])[0]
    	return recovered



#extract sparse features from the raw data and return every encoding and processing model that was used
def feature_extraction_sparse_train(filename):
	features,labels = parse(filename,True)
	n = len(features)
	int_encoder,int_encoded = int_encode_train(features)
	int_encoded = int_encoded.reshape(n,9)
	sparse_encoder,features = sparse_encode_train(int_encoded)
	return features,labels,sparse_encoder,int_encoder

#extract sparse features from the raw data and return every encoding and processing model that was used
def feature_extraction_regression_train(filename):
	features,labels = parse_regression(filename)
	n = len(features)
	int_encoder,int_encoded = int_encode_train(features)
	int_encoded = int_encoded.reshape(n,9)
	sparse_encoder,features = sparse_encode_train(int_encoded)
	return features,labels, sparse_encoder,int_encoder

#extract sparse features from the raw data
def feature_extraction_sparse_test(filename,sparse_encoder,int_encoder):
	features = parse(filename,False)
	n = len(features)
	int_encoded = int_encode_test(features,int_encoder)
	int_encoded = int_encoded.reshape(n,9)
	features = sparse_encode_test(int_encoded,sparse_encoder)
	return features

def feature_extraction_physicochemical_train(filename):	
	proteins,labels = parse(filename,True)
	proteins_encoded,onehot_encoder = encode_physicochemical_train(proteins)
	return proteins_encoded,labels,onehot_encoder

def feature_extraction_physicochemical_test(filename, onehot_encoder):
	proteins = parse(filename, False)
	proteins_encoded = encode_physicochemical_test(proteins,onehot_encoder)
	return proteins_encoded

def feature_extraction_categorical_numerical_train( filename ):
	features, labels = parse( filename, True)
	n = len(features)
	int_encoder, int_encoded = int_encode_train(features)
	int_encoded = int_encoded.reshape(n,9)
	numerical_encoded = encode_physicochemical_train(features)
	features = np.concatenate((int_encoded,numerical_encoded),axis=1)
	onehot_encoder = OneHotEncoder(categorical_features = np.arange(9))
	features = onehot_encoder.fit_transform(features)
	return features, labels, onehot_encoder,int_encoder

def feature_extraction_categorical_numerical_test( filename,onehot_encoder,int_encoder ):
	features = parse( filename, False)
	n = len(features)
	int_encoded = int_encode_test(features, int_encoder)
	int_encoded = int_encoded.reshape(n,9)
	numerical_encoded = encode_physicochemical_train(features)
	features = np.concatenate((int_encoded,numerical_encoded),axis=1)
	features = sparse_encode_test(features, onehot_encoder)
	return features

if __name__ == "__main__":
	feature_extraction_categorical_numerical_train("project_training.txt")


	
