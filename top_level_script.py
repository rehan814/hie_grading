"""
This is the top level script for the project where most of the variables are initialized
 Try to initialize all the variables used in the subsequent functions in this file
"""
import numpy,os
from trainUBM import train_UBM
from DefineDirectories import define_directories
from trainSVM import train_SVM

class Paramertes:
    """
    Parameters values
    """
    num_of_points_per_file=''
    total_lim=''
    pca_threshold=''
    window=''
    shift=''
    gauss=''
    kernel=''
    seq_len=''
    model_sel = ''
    num_of_classes=''


whichfeatures  = numpy.arange(1,56,1)  # mask for features to use

ParametersUBM = Paramertes()
ParametersUBM.num_of_points_per_file = 0
ParametersUBM.total_lim = 90000
ParametersUBM.gauss = 4
ParametersUBM.window = 8
ParametersUBM.shift = 4
ParametersUBM.pca_threshold = 0.98
ParametersUBM.model_sel=0

ParametersSVM=Paramertes()
ParametersSVM.num_of_points_per_file = 0
ParametersSVM.total_lim = 90000
ParametersSVM.gauss= 4
ParametersSVM.window = 8
ParametersSVM.shift = 4
ParametersSVM.kernel = 'KL'
ParametersSVM.seq_len =20
ParametersSVM.num_of_classes = 4
ParametersSVM.equal_number_of_sequences=0
ParametersSVM.svm_classifier_type = 'BI_1vs1' # 1vs1 , 1vsRest, CS
Flag_TrainSVM = 1

expID  = ('GMM_UBM') # experiment ID

(datadir, CurrentDir, targetlist, total_lim, modeldir_UBM, modeldir_SVM, SaveResultFileName_UBM,
SaveResultFileName_SVM,labeldir) = define_directories(expID, ParametersSVM,ParametersUBM,Flag_TrainSVM)

train_UBM(targetlist, modeldir_UBM, datadir, labeldir,  whichfeatures, expID,  total_lim, ParametersUBM)

train_SVM(targetlist, modeldir_UBM, modeldir_SVM, datadir, labeldir,  whichfeatures, expID,  total_lim, ParametersSVM)
