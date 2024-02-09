#################################################
### José Javier Gutierrez Gil
### jogugil@gmail.com
#################################################


import os
import random
import warnings
import pandas as pd
import time

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import mne
import yasa
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
import numpy as np
from scipy.signal import resample
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from base.ExractFeatures import ExractFeatures 
from base.Config import Config
from base.Polysom import Polysom 

class Polysom_Data ():
    iPath          = None
    oPath          = None
    conf_data      = None
    lSubject       = []    # Lista de objetos Polysom
     
    def __init__ (self, iPath = './data', oPath = 'data/ISRUC'):
        conf_d = None
        print ('iPath:',iPath)
        if not os.path.exists(iPath) or not os.path.isdir(iPath):
            raise ValueError("Oops! The input Data Path does not exist or is not a directory...")
        self.iPath = iPath
        if not os.path.exists(oPath):
            os.makedirs(oPath)
        elif os.listdir(oPath):
            warnings.warn("Oops! The output Data Path is not empty...")
        self.oPath = oPath
	#conf_d = Config ('base/config_psd.yalm')

        self.conf_data = Config ('config_psd.yalm').config  # Config app  
        #print (self.conf_data)
        self.load ()
    ###
    def getiPath(self):
        return self.iPath

    def getoPath(self):
        return self.oPath
 
    ####
    def load (self):
        for subject in os.listdir(self.getiPath ()):
            try:
                d = os.path.join(self.getiPath (), subject)
                print ('d:',d)
                if os.path.isdir(d) and int (subject):
                    print (' Polysom (d, subject, self.conf_data)')
                    poly = Polysom (d, subject, self.conf_data)
                    self.lSubject.append (poly)
                
                    #if (DEBUG): 
                        #poly.plot_serials (fs = 200)
            except Exception as e:
                print(f"{str(e)}")
                continue
    def getSubject (self, subject):
        sbjct = None
        try:
            sbjct = self.lSubject [subject]
        except Exception as e:
            print(f"{str(e)}")
            
        return sbjct
    def getlSubjects (self):
        return self.lSubject 
    
    import random

    def split_data (self,  train_ratio, val_ratio, test_ratio):
        
        # Asegúrate de que las proporciones sumen 1.0
        assert train_ratio + val_ratio + test_ratio == 1.0
        
        data = self.lSubject
        total_samples = len (data)

        # Calcula la cantidad de datos para cada conjunto
        train_size = int (total_samples * train_ratio)
        val_size   = int (total_samples * val_ratio)
        test_size  = int (total_samples * test_ratio)

        # Mezcla los datos para evitar sesgos
        random.shuffle(data)

        # Divide los datos en conjuntos
        train_data = data [:train_size]
        val_data   = data [train_size:train_size + val_size]
        test_data  = data [train_size + val_size:]

        return train_data, val_data, test_data
    
    def calc_roc_acu (self, classifier, X_dt, y_dt, n_classes = 5):
        rf = []
        rf.append(classifier)
        y_score_train = classifier.predict_proba(X_dt)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc_train = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_dt[:, i], y_score_train[:, i])
            roc_auc_train[i] = auc(fpr[i], tpr[i])

        return roc_auc_train   
    
    
    def calc_result_roc_auc (self, classifier, X_train, y_train, n_ytrain):
        results   = {}
        n_classes = len (y_train[0][0])

        for i, (subject_data, y_subject) in enumerate (zip (X_train, y_train)):
            results[i] = {}
            y_score_subject = classifier.predict_proba (subject_data)
            for j in range(n_classes):
                fpr, tpr, thresholds = roc_curve (y_subject [:, j], y_score_subject[:, j])
                results[i][j] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds,
                        'roc_auc_train': auc(fpr, tpr)
                    }

        return results   
       
    # Función para calcular e imprimir la matriz de confusión
    def calculate_confusion_matrix (self, classifier, X, y):
        y_pred = classifier.predict (X)
        cm     = confusion_matrix (y, y_pred)
        return cm
    def calculate_multilabel_confusion_matrix(self, classifier, X, y):
        y_pred = classifier.predict(X)

        # Assuming y is in a multilabel format
        ml_cm = multilabel_confusion_matrix (y, y_pred)

        return ml_cm   
       
    def calculate_metrics (self, classifier, X, y):
        y_pred    = classifier.predict (X)
        cm        = confusion_matrix (y, y_pred)
        accuracy  = accuracy_score (y, y_pred)
        precision = precision_score (y, y_pred, average='weighted')
        recall    = recall_score (y, y_pred, average='weighted')
        f1        = f1_score (y, y_pred, average='weighted')
        return cm, accuracy, precision, recall, f1