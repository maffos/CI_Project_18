|CI|TU2018|MHC Class I Prediction|SVM MODEL|
---------------------------------------------------

# SVM Model

## Installation

Please assure to have a distribution of Python 3.3 or higher installed. As most of our methods use
[SciKit-learn](http://scikit-learn.org/stable/), you have to install the library.
To install SciKit-learn you can follow the instructions [here](http://scikit-learn.org/stable/install.html).

This application doesn't need any further installation. You can simply use any Console or IDE and navigate to
the directory where you unpacked the archive. You can start the application with

```
python CI_SVM/Run.py
```

## How to use

This application will guid you through several steps to create a prediction file.

1. Specify a path (absolute or relative) to the file containing the training data you want to use. Please be sure
to supply an existing file. In addition this Application supports .csv and .tsv formats. It will also recognize
for example .txt files with a tab separated formatting.

2. Specify a mode to run. 'pre' (precomputed) mode will use the parameters we have obtained of training our model to
the projects training data. 'cvg' (cross validated grid-search) requires you to specify additional parameters (You find
detailed information at the Parameter section). For our model we are using:

    1. 'kernel': ['rbf', 'linear', 'poly']
    
    2. 'C': [11e-3, 1e-2, 0.1, 0.5, 1, 1.5, 5, 10, 10e1, 10e2, 10e3]
    
    3. 'gamma': ['auto'], this calculates gamma as 1/(#samples)
    
    4. 'degree' : [2,3,4]
    
    5. 'coef0' : [1,2,3]
    
   You can also change those parameters in the source code. You should consider this especially when the resulting
   parameters are on a margin. For example C=1e3, maybe you will perform better with even higher values for C.
   You will find additional information about usable parameters [here.](http://scikit-learn.org/stable/modules/svm.html#svm-classification)

3. The application will now start training our svm model on the training data provided. In addition a cross
validation is started to evaluate the models performance. When the training is completed you will see a list of best
evaluated parameters and a mean/standard deviation value of the cross validation. Default scoring is ROC-AUC.

4. You can now use the trained model to predict sample data. For this specify a path (absolute or relative) to the
file containing the sample data you want to predict. After the prediction completed you can specify a path
(absolute or relative) to a directory you want to write the prediction to. You can pass an existing file
(the prediction values will be written to this file) or enter a valid path with an non-existing file name to generate
a new file. For generating new files you can choose between .csv and .tsv format.

## Parameter

If you are using 'cvg' mode you are asked to specify parameters. Please ensure to use only required input formats:

* For PROFEAT parameters you can choose of ['hy','vv','pl','pa','ch','sa']. These are categorical physicochemical
amino acid annotations. For more information please please take a look at our report or the original
[publication](https://www.researchgate.net/publication/6940836). The abbreviations stand for 

    1. 'hy' -> Hydrophobic
    
    2. 'vv' -> VanDerWaal-Volume
    
    3. 'pl' -> Polarity
    
    4. 'pa' -> Polarizability
    
    5. 'ch' ->  Charge
    
    6. 'sa' -> Solvent-Accessibility 

* For [ModlAMP](https://modlamp.org/) parameters you can choose of ['ar','bi','hy','al','mw']. These are numerical
descriptor values for physicochemical properties of amino acids calculated with the ModlAMP Python library. For more
information please please take a look at our report or the original
[publication](https://academic.oup.com/bioinformatics/article-abstract/33/17/2753/3796392).
The abbreviations stand for

    1. 'ar' -> Aromatic
    
    2. 'bi' -> Boman-Index
    
    3. 'hy' -> Hydrophobic
    
    4. 'al' -> Aliphatic-Index
    
    5. 'mw' -> Molecular-Weight

    

* For SCORING parameter you can choose of several scoring metrics supported by SciKit-learn. We used 'roc_auc' scoring
in our model. You get more information [here](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

* For DIMENSION parameter you can choose an positive integer value. Please be sure to use a value between 1 and the
dimension of your feature matrix - 1. For example only using peptide sequences without feature annotation the
feature matrix has a dimension of 180. Choosing not too low values will result in accelerated calculations but may
decrease the performance of the fitted model.

## Information

If you don't know which directory to use, you can navigate with

```
./Sources/#valid_file_name
```

to a local source folder of the application to write predictions. There you can also find the project_training.txt file containing the
training data used in the project. For a easier handling you can also place your sample files in this directory (this
might be easier then using absolute paths).

## Issues

* Currently you can only predict one sample. After that the application will be closed.






