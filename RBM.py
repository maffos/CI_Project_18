import numpy as np
import Preprocessing 
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import GridSearchCV

def train(features,n):
	rbm = BernoulliRBM(n_components=n)
	#param_grid={'learning_rate':[0.9,0.1,0.01,0.001]}
	#best_rbm = GridSearchCV(rbm, param_grid,cv=10)     	
	rbm.fit(features)
	print len(rbm.components_),len(rbm.components_[0])
    	return rbm.components_


