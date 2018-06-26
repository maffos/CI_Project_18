from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import Preprocessing

def train_sparse(filename):
	clf = RandomForestClassifier(max_depth=20,n_estimators = 200)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	score = cross_val_score(clf,features,labels, cv=10)
	print np.mean(score)	
	clf.fit(features,labels)
	return clf,sparse_encoder,int_encoder

def train_dimension_reduction(filename):
	#pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', RandomForestClassifier())])
	#param_grid={'reduce_dim__n_components': [70,75,85,100,120], 'classification__max_depth':[2,4,8], 'classification__n_estimators':#[10,50,100,150]}
	#clf = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	svd = TruncatedSVD(n_components=75)
	features = svd.fit_transform(features)
	clf = RandomForestClassifier(max_depth=20,n_estimators = 200)
	score = cross_val_score(clf,features,labels, cv=10)
	print np.mean(score)
	clf.fit(features,labels)
	return clf,svd,sparse_encoder,int_encoder
	
def train_physicochemical( filename ):
	clf = RandomForestClassifier(max_depth=2,n_estimators = 100)
	features,labels = Preprocessing.feature_extraction_physicochemical( filename,True )   	
	clf.fit(features,labels)
	return clf

def train_categorical_numerical( filename ):
	clf = RandomForestClassifier()
	features,labels, onehot_encoder, int_encoder = Preprocessing.feature_extraction_categorical_numerical_train("project_training.txt")
	#score = cross_val_score(clf,features,labels, cv=10)
	#print np.mean(score)	
	param_grid={'max_depth':[15,20,25,40], 'n_estimators':[100,150,200,250]}	
	best_clf = GridSearchCV(clf,param_grid,cv=10)	
	best_clf.fit(features,labels)
	return best_clf,onehot_encoder,int_encoder

def train_categorical_numerical_dimension_reduction(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD()), ('classification', RandomForestClassifier(max_depth=20,n_estimators = 200))])
	param_grid={'reduce_dim__n_components': [70,85,100,120,140,180], 'classification__max_depth': [15,20,25,40], 'classification__n_estimators': [100,150,200,250]}
	clf = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,onehot_encoder,int_encoder = Preprocessing.feature_extraction_categorical_numerical_train(filename)
	clf.fit(features,labels)
	print clf.best_score_
	return clf,onehot_encoder,int_encoder


def validate_dim_reduct(filename, clf, svd, sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(filename,sparse_encoder,int_encoder)
	features = svd.transform(features)
	print "prediction: " + str(clf.predict(features))

def validate_model(filename, clf, sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(filename,sparse_encoder,int_encoder)
	prediction = clf.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(filename, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index]+"\n"))

def validate_physicochemical(testfile, clf):
	features = Preprocessing.feature_extraction_physicochemical( testfile,False )
	print "Prediction: " + str(clf.predict(features))

def validate_categorical_numerical( filename, clf, onehot_encoder, int_encoder ):
	features = Preprocessing.feature_extraction_categorical_numerical_test(filename,onehot_encoder,int_encoder)
	prediction = clf.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(filename, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index]+"\n"))

if __name__ == "__main__":
	#clf,svd, sparse_encoder,int_encoder = train_dimension_reduction("project_training.txt")
	#clf,sparse_encoder, int_encoder = train_sparse("project_training.txt")
	#print "best params"	
	#print clf.best_params_
	#print "best score"
	#print clf.best_score_
	#validate_model("project_sample.csv",clf,sparse_encoder,int_encoder)
	#clf = train_physicochemical("project_training.txt")
	#validate_physicochemical("test_input.txt",clf)
	clf,onehot_encoder, int_encoder = train_categorical_numerical("project_training.txt")
	#clf,onehot_encoder, int_encoder = train_categorical_numerical_dimension_reduction("project_training.txt")
	print clf.best_params_
	print clf.best_score_	
	validate_categorical_numerical("project_sample.csv",clf,onehot_encoder,int_encoder)
