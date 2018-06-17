from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
import numpy as np
import Preprocessing

def train_sparse(filename):
	clf = RandomForestClassifier(max_depth=2,n_estimators = 100)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	clf.fit(features,labels)
	return clf,sparse_encoder,int_encoder

def train_dimension_reduction(filename):
	#pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', RandomForestClassifier())])
	#param_grid={'reduce_dim__n_components': [70,75,85,100,120], 'classification__max_depth':[2,4,8], 'classification__n_estimators':#[10,50,100,150]}
	#clf = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	svd = TruncatedSVD(n_components=75)
	features = svd.fit_transform(features)
	clf = RandomForestClassifier(max_depth=2,n_estimators=100)
	clf.fit(features,labels)
	return clf,svd,sparse_encoder,int_encoder
	
def train_physicochemical( filename ):
	clf = RandomForestClassifier(max_depth=2,n_estimators = 100)
	features,labels = Preprocessing.feature_extraction_physicochemical( filename,True )   	
	clf.fit(features,labels)
	return clf

#def validate_model(filename, clf, svd, sparse_encoder,int_encoder):
#	features = Preprocessing.feature_extraction_sparse_test(filename,sparse_encoder,int_encoder)
#	features = svd.transform(features)
#	print "prediction: " + str(clf.predict(features))

def validate_model(filename, clf, sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(filename,sparse_encoder,int_encoder)
	print "prediction: " + str(clf.predict(features))

def validate_physicochemical(testfile, clf):
	features = Preprocessing.feature_extraction_physicochemical( testfile,False )
	print "Prediction: " + str(clf.predict(features))

if __name__ == "__main__":
	#clf,svd, sparse_encoder,int_encoder = train_dimension_reduction("project_training.txt")
	#clf,sparse_encoder, int_encoder = train_sparse("project_training.txt")
	#print "best params"	
	#print clf.best_params_
	#print "best score"
	#print clf.best_score_
	#validate_model("test_input.txt",clf,sparse_encoder,int_encoder)
	clf = train_physicochemical("project_training.txt")
	validate_physicochemical("test_input.txt",clf)
