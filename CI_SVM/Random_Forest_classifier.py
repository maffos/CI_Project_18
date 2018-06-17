from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
import numpy as np
import Preprocessing

def train_sparse(filename):
	clf = RandomForestClassifier()
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	param_grid={'max_depth':[2,4,8], 'n_estimators': [10,50,100,150]}
	best_clf = GridSearchCV(clf, param_grid, cv=10)
	best_clf.fit(features,labels)
	return best_clf,sparse_encoder,int_encoder

def train_dimension_reduction(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', RandomForestClassifier())])
	param_grid={'reduce_dim__n_components': [70,75,85,100,120], 'classification__max_depth':[2,4,8], 'classification__n_estimators':[10,50,100,150]}
	clf = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	clf.fit(features,labels)
	return clf,sparse_encoder,int_encoder

def validate_model(filename, clf, sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(testfile,sparse_encoder,int_encoder)
	print "prediction: " + str(clf.predict(features))

if __name__ == "__main__":
	clf, sparse_encoder,int_encoder = train_dimension_reduction("project_training.txt")
	print "best params"	
	print clf.best_params_
	print "best score"
	print clf.best_score_
	validate_model("test_input.txt",clf,sparse_encoder,int_encoder)
