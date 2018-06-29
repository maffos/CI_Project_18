from Descriptor import Descriptor
from Encoder import Encoder
from Parser import Parser
from Predictor import Predictor
from Trainer import Trainer
import sys
import time

if __name__ == "__main__":

    enc = Encoder()
    pars = Parser()

    print('|CI|TU2018|MHC Class I Prediction|' + '\n')
    print('This application guides you through the necessary steps for '
          'epitope prediction. You can confirm your input with "enter".' + '\n')
    time.sleep(0.5)

    print('I. Choose training data')
    # Specifies absolute path to file containing training samples <
    input_path = input('[Please specify path to training data]: ')
    proteins, labels = pars.parse_train(input_path)
    print('>Parsing completed' + '\n')
    time.sleep(0.5)

    print('II. Mode selection: ')
    print('- "pre"; svm model with optimized parameters for project training data.')
    print('- "cvg"; svm model with extensive cross validated GridSearch for optimal parameters. '
          'Please take a look at README first.')
    model_selection = input('mode: ')
    time.sleep(0.5)

    # For the project training data optimized parameters.
    if model_selection == "pre":

        print('>Using preoptimized parameters')

        profeat_annotation = ['vv', 'sa']
        modlamp_annotation = ['bi', 'ar']

        dim_red = False
        dimension = 0

        scoring = 'roc_auc'

        params = {'kernel': ['poly'], 'C': [10], 'gamma': ['auto'], 'degree': [3], 'coef0': [1]}

    # Runs a CVGridSearch with user specified parameters for
    # feature annotation, scoring and dimensionality reduction.
    elif model_selection == 'cvg':

        print('>Define parameters for CVGridSearch')

        profeat_annotation = eval(
            input('[Please specify PROFEAT parameters in Python list format. "None" to disable.] '))

        modlamp_annotation = eval(
            input('[Please specify MODLAMP parameters in Python list format. "None" to disable.] '))

        scoring = eval(input('[Please specify SCORING parameter in Python String format] '))

        dimension = eval(input('[Please specify DIMENSION parameter in Python Integer format. "0" to disable.] '))

        if dimension > 0:

            dim_red = True

        else:

            dim_red = False

        # Fixed set for parameters to search. Change if you want to use another range of parameters. <
        params = {'kernel': ['rbf', 'linear', 'poly'],
                  'C': [11e-3, 1e-2, 0.1, 0.5, 1, 1.5, 5, 10, 1e1, 1e2, 1e3],
                  'gamma': ['auto'],
                  'degree': [2, 3],
                  'coef0': [1, 2, 3]}

    else:
        sys.exit('No supported mode was selected.')

    # Instantiates Descriptor object with specified feature annotation.
    desc = Descriptor(profeat=profeat_annotation, modlamp=modlamp_annotation)
    prd = Predictor(desc)

    # Encodes the input data.
    enc_proteins = desc.encode_proteins(proteins)

    print('>Training stared')

    # Trains a model with specified parameters.
    trn = Trainer(scoring=scoring, model_params=params, dim_reduction=dim_red, dim_reduction_n=dimension)

    model = trn.train(enc_proteins, labels)

    trn.cross_validate_svm(model, enc_proteins, labels)

    print('III. Sample prediction')
    sample_path = input('[Please specify path to sample data]: ')
    prd.predict(model, sample_path)

    output_path = input('[Please specify path to save prediction]: ')
    prd.write_prediction(open(output_path, 'w'))
