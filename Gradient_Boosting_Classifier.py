from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import Preprocessing

def train_sparse( filename ):
    	clf = GradientBoostingClassifier()
    	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	param_grid={'n_estimators':[180,200,250,300]}
	best_clf = GridSearchCV(svm, param_grid,cv=10)     	
	best_clf.fit(features,labels)
    	return best_clf,sparse_encoder,int_encoder

def train_dimension_reduction(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD(n_components=70)), ('classification', GradientBoostingClassifier())])
	param_grid={'classification__n_estimators':[180,200,250,300]}
	clf = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	clf.fit(features,labels)
	return clf,sparse_encoder,int_encoder

def validate_model(testfile,clf,sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(testfile,sparse_encoder,int_encoder)
	prediction = clf.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction_gdb2.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(testfile, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction_gdb2.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index]+"\n"))


if __name__ == "__main__":
	clf,sparse_encoder,int_encoder = train_dimension_reduction("project_training.txt")
	print "best params"	
	print clf.best_params_
	print "best score"
	print clf.best_score_
	validate_model("project_sample.csv",clf,sparse_encoder,int_encoder)

