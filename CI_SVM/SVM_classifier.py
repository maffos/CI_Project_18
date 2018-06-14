import Preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

#fits an svm to a sparse representation of the features. The regularization constant C and the kernel are determined by grid search with 10 fold cross validation
def train_sparse( filename ):
    	svm = SVC()
    	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	param_grid={'kernel':['poly','linear','rbf'], 'C': [0.1,0.5,0.9,1,2]}
	best_svm = GridSearchCV(svm, param_grid,cv=10)     	
	best_svm.fit(features,labels)
    	return best_svm,sparse_encoder,int_encoder

def validate_model(testfile,svm,sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(testfile,sparse_encoder,int_encoder)
	print "prediction: " + str(svm.predict(features))

#traines an svm modell by applying dimensionality reduction to the features first. Hyperparameters, e.g. dimensionality of features, regularization constant C and the kernel being used are determined by Grid search, with 10-fold CV
def train_dimension_reduction(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', SVC())])
	param_grid={'reduce_dim__n_components': [70,75,85,100,120], 'classification__kernel':['poly','linear','rbf'], 'classification__C':[0.1,0.5,1,2]}
	svm = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	svm.fit(features,labels)
	return svm,sparse_encoder,int_encoder

if __name__ == "__main__":
	#svm,sparse_encoder,int_encoder = train_dimension_reduction("project_training.txt")
	svm, sparse_encoder,int_encoder = train_sparse("project_training.txt")	
	print "best params"	
	print svm.best_params_
	print "best score"
	print svm.best_score_
	validate_model("test_input.txt",svm,sparse_encoder,int_encoder)
	
