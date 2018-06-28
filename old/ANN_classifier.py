import Preprocessing
import RBM
import NeuralNetwork
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np



def train_regressor(filename):
	pipe = Pipeline([('reduce_dim', TruncatedSVD(n_components=70)), ('regression', MLPRegressor(solver='lbfgs'))])
	param_grid={'regression__hidden_layer_sizes':[(230,),(300,)],'regression__alpha':[0.0001,0.1,0.01]}
	mlp = GridSearchCV(pipe, param_grid,cv=10)
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_regression_train(filename)
	mlp.fit(features,labels)
	return mlp,sparse_encoder,int_encoder

def validate_regressor(testfile,mlp,sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(testfile,sparse_encoder,int_encoder)
	prediction_ic50 = mlp.predict(features)
	prediction = []
	for value in prediction_ic50:
		if value <= 511:
			prediction.append(1)
		else:
			prediction.append(0)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction_mlp_regressor4.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(testfile, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction_mlp_regressor4.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index])+"\n")

	
def train_classifier(filename):
	features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train(filename)
	#svd = TruncatedSVD(n_components=70)
	#features = svd.fit_transform(features)
	mlp = NeuralNetwork.MLPClassifierOverride(activation='relu', max_iter=400, solver='lbfgs',hidden_layer_sizes=(150,))
	#param_grid={'classifier__alpha':[0.0001,0.1,0.9]}
	#mlp = GridSearchCV(mlp_init, param_grid,cv=10)
	mlp.fit(features,labels)
	#print mlp.get_params_
	#return mlp,svd,sparse_encoder,int_encoder
	print mlp.score(features,labels)
	return mlp,sparse_encoder,int_encoder

def validate_classifier(testfile,mlp,sparse_encoder,int_encoder):
	features = Preprocessing.feature_extraction_sparse_test(testfile,sparse_encoder,int_encoder)
	#features = svd.transform(features)
	prediction = mlp.predict(features)
	#print "prediction: " + str(svm.predict(features))
	with open("prediction_mlp_classifier2.csv", "w") as out:
		out.write("Id,Prediction1\n")
	with open(testfile, "r") as f:
		f.readline()
		for index,line in enumerate(f):
			peptide = line.split(",")[0]
			with open("prediction_mlp_classifier2.csv", "a") as out:
				out.write(str(peptide) + "," + str(prediction[index])+"\n")


if __name__ == "__main__":
	mlp,sparse_encoder,int_encoder = train_regressor("project_training.txt")
	print mlp.best_params_
	print mlp.best_score_
	#mlp,sparse_encoder,int_encoder = train_classifier("project_training.txt")
	validate_regressor("project_sample.csv",mlp,sparse_encoder,int_encoder)
	#validate_classifier("project_sample.csv",mlp,sparse_encoder,int_encoder)
