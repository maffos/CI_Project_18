from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import is_classifier
import RBM
import Preprocessing
from sklearn.decomposition import TruncatedSVD

class MLPClassifierOverride(MLPClassifier):
	def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8):

        	sup = super(MLPClassifier, self)
        	sup.__init__(hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation, solver=solver, alpha=alpha,
                     batch_size=batch_size, learning_rate=learning_rate,
                     learning_rate_init=learning_rate_init, power_t=power_t,
                     max_iter=max_iter, loss='log_loss', shuffle=shuffle,
                     random_state=random_state, tol=tol, verbose=verbose,
                     warm_start=warm_start, momentum=momentum,
                     nesterovs_momentum=nesterovs_momentum,
                     early_stopping=early_stopping,
                     validation_fraction=validation_fraction,
		     beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
		self.first_layer_init = True

	def _init_coef(self, fan_in, fan_out):
   		if self.activation == 'logistic':
        		init_bound = np.sqrt(2. / (fan_in + fan_out))
    		elif self.activation in ('identity', 'tanh', 'relu'):
      	  		init_bound = np.sqrt(6. / (fan_in + fan_out))
    		else:
        		raise ValueError("Unknown activation function %s" %
                         self.activation)
    		if self.first_layer_init:
			features,labels,sparse_encoder,int_encoder = Preprocessing.feature_extraction_sparse_train("project_training.txt")
			#svd = TruncatedSVD(n_components=70)
			#features = svd.fit_transform(features)
			weights = RBM.train(features,150)
			weights = weights.reshape(-1,1)
			print fan_in, fan_out	
			coef_init = weights
			self.first_layer_init=False
		else:
			coef_init = self._random_state.uniform(-init_bound, init_bound,
					(fan_in, fan_out))
    		intercept_init = self._random_state.uniform(-init_bound, init_bound,fan_out)

    		return coef_init, intercept_init

	
