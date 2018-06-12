
# coding: utf-8

# In[84]:


'''import libraries'''
import numpy as np
import scipy as sc
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[134]:


'''tetsfile'''
file = "project_training.txt"

'''encoding for amino-acids'''
aa_to_binary_dict = {
    "A":0b00000,
    "R":0b00001,
    "N":0b00010,
    "D":0b00100,
    "C":0b01000,
    "Q":0b10000,
    "E":0b00011,
    "G":0b00101,
    "H":0b01001,
    "I":0b10001,
    "L":0b00111,
    "K":0b01011,
    "M":0b10011,
    "F":0b10101,
    "P":0b11001,
    "S":0b11010,
    "T":0b11100,
    "W":0b01111,
    "Y":0b10111,
    "V":0b11011
}


# In[170]:


def train_svm( file ):
    new_svm = svm.SVC()
    '''fit parameters for gridsearch'''
    cv_params = { 'kernel':['rbf'], 'C':[0.1,1,2] }
    folds = 10
    cv = GridSearchCV( new_svm, cv_params, cv = folds )
    
    data_set = parse( file )
    
    proteins = data_set[0]
    labels = data_set[1]
    
    best_params = cv.fit( proteins, labels ).best_params_
    best_svm = svm.SVC( C = best_params['C'], kernel = best_params['kernel'] )
    
    return best_svm.fit( proteins, labels )


# In[161]:


def parse( file ):
    raw_data = open( file, "r" ).readlines()[1:]
    
    proteins = []
    proteins_binary = []
    labels = []
    
    for line in raw_data:
        proteins.append( line.split()[0] )
        labels.append( line.split()[2] )
        
    for protein in proteins:
        proteins_binary.append( encode( protein ) )
        
    return [ proteins_binary, labels ]


# In[162]:


def encode( protein_aa ):
    single_aas = list( protein_aa )
    
    protein_binary = []
    
    for aa in single_aas:
        protein_binary.append( aa_to_binary( aa ) )
        
    return protein_binary


# In[163]:


def aa_to_binary( aa ):
    return aa_to_binary_dict[aa]


# In[171]:


#TEST_SPACE#

#test_svm = train_svm( file )

