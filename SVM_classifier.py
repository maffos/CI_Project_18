import Preprocessing
import Encodings
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

#fits an svm to a sparse representation of the features. The regularization constant C and the kernel are determined by grid search with 10 fold cross validation
def train_sparse( filename ):
    	svm = SVC()
    	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	param_grid={'kernel':['rbf','linear','poly'], 'degree': [2,3,4], 'C': [0.1,0.5,0.9,1,2]}
	best_svm = GridSearchCV(svm, param_grid,cv=10)     	
	best_svm.fit(features,labels)
    	return best_svm,sparse_encoder,int_encoder

#traines an svm modell by applying dimensionality reduction to the features first. Hyperparameters, e.g. dimensionality of features, regularization constant C and the kernel being used are determined by Grid search, with 10-fold CV
def train_dimension_reduction(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', SVC())])
	param_grid={'reduce_dim__n_components': [70,75,85,100,120], 'classification__kernel':['rbf','linear','poly'], 'classification__degree': [2,3,4],'classification__C':[0.5,0.9,1,2]}
	svm = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	svm.fit(features,labels)
	return svm,sparse_encoder,int_encoder

def train_physicochemical( filename ):
	svm = SVC()
	features,labels,onehot_encoder = Preprocessing.feature_extraction_physicochemical_train( filename)
	param_grid={'kernel':['rbf','linear','poly'], 'degree': [2,3,4], 'C': [0.5,0.9,1,2]}
	best_svm = GridSearchCV(svm, param_grid,cv=10)     	
	best_svm.fit(features,labels)
	return best_svm,onehot_encoder

def train_categorical_numerical( filename ):
	svm = SVC()
	features,labels, onehot_encoder, int_encoder = Preprocessing.feature_extraction_categorical_numerical_train("project_training.txt")
	param_grid={'kernel':['rbf','linear','poly'], 'degree': [2,3,4], 'C': [0.5,0.9,1,2]}
	best_svm = GridSearchCV(svm, param_grid,cv=10)     		
	best_svm.fit(features,labels)
	return best_svm,onehot_encoder,int_encoder

def train_categorical_numerical_dimension_reduction(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', SVC())])
	param_grid={'reduce_dim__n_components': [70,85,100,120,140,140], 'classification__C':[0.1,0.5,1,2], 'classification__kernel':['linear','poly','rbf'], 'classification__degree':[2,3,4]}
	svm = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,onehot_encoder,int_encoder = Preprocessing.feature_extraction_categorical_numerical_train(filename)
	svm.fit(features,labels)
	return svm,onehot_encoder,int_encoder

def validate_model(testfile,svm,sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(testfile,sparse_encoder,int_encoder)
	prediction = svm.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(testfile, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index]+"\n"))

def validate_physicochemical(testfile, svm, onehot_encoder):
	features = Preprocessing.feature_extraction_physicochemical_test(testfile, onehot_encoder)
	prediction = svm.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(testfile, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index]+"\n"))


def validate_categorical_numerical(testfile, svm, onehot_encoder, int_encoder):
	features = Preprocessing.feature_extraction_categorical_numerical_test(testfile, onehot_encoder,int_encoder)
	prediction = svm.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction2.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(testfile, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction2.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index]+"\n"))

if __name__ == "__main__":
	svm,sparse_encoder,int_encoder = train_dimension_reduction("project_training.txt")
	#svm, sparse_encoder,int_encoder = train_sparse("project_training.txt")
	#svm,onehot_encoder = train_physicochemical("project_training.txt")	
	#svm,onehot_encoder,int_encoder = train_categorical_numerical_dimension_reduction("project_training.txt")
	#svm, onehot_encoder, int_encoder = train_categorical_numerical("project_training.txt")
	print "best params"	
	print svm.best_params_
	print "best score"
	print svm.best_score_
	validate_model("project_sample.csv",svm,sparse_encoder,int_encoder)
	#validate_physicochemical("project_sample.csv", svm, onehot_encoder)
	#validate_categorical_numerical("project_sample.csv",svm, onehot_encoder,int_encoder)
