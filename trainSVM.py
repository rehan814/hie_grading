"""
This function creates the UBM models
"""
import os,numpy as np
from ReadData import read_data, train_data
from TrainSVM_Models import *
from MakeSequences import make_sequences

# from DefineDirectories import define_directories as dirs
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import mixture
import pickle


class model_ubm:
    norm_template=[]
    pca_param=[]
    GMM_param=[]

def train_SVM(targetlist, modeldir_UBM, modeldir_SVM, datadir, labeldir, whichfeatures, expID,  total_lim, ParametersSVM):
    """

    :return:
    """
    # Read the list of files whose data need to be read
    fd = open(targetlist)
    names=fd.read()
    names=names.split('\n')
    names=np.asarray(names)

    # Extract the file names
    useind=[]
    for lv in names:
        useind.append(int(lv[4:6]))
    useind=np.asarray(useind)
    patients_actual=np.unique(useind)

    # Create the parent directory where all the UBM models will be saved
    if not os.path.exists(modeldir_UBM):
        os.makedirs(modeldir_UBM)

    # for pat in range(0,patients_actual.__len__(),1):
    for pat in range(4,10,1):

        modeldir_patient_svm = os.path.join(modeldir_SVM, str('patient_' + str(pat+1)))
        if os.path.exists(modeldir_patient_svm):
            print('Directory for this file already exist')
        else:
            os.makedirs(modeldir_patient_svm)
            print('Creating SVM model for Patient', patients_actual[pat])

        # Read data of all the files except for the paitent being trained for
        train_ind = np.where(useind != patients_actual[pat])[0]
        train_ind_files_names = names[train_ind]
        # tr = read_data(train_ind, train_ind_files_names, whichfeatures, datadir, labeldir, total_lim, ParametersSVM)

        # # Load the UBM model file for this patient
        modeldir_patient_ubm = os.path.join(modeldir_UBM, str('patient_' + str(pat+1) + '_GMM_model_UBM.pickle'))
        fid = open(modeldir_patient_ubm,'rb')
        model_ubm = pickle.load(fid)
        fid.close()
        #
        # # Apply Normalization template used earlier to create the UBM model
        # tr.data_g1 = model_ubm.norm_template.transform(np.transpose(tr.data_g1))
        # tr.data_g2 = model_ubm.norm_template.transform(np.transpose(tr.data_g2))
        # tr.data_g3 = model_ubm.norm_template.transform(np.transpose(tr.data_g3))
        # tr.data_g4 = model_ubm.norm_template.transform(np.transpose(tr.data_g4))
        #
        # # Make sequences of the data
        # td_seq = train_data()
        # seq_indx = train_data()
        # td_seq.data_g1 = make_sequences(np.transpose(tr.data_g1), ParametersSVM.seq_len, 0)
        # td_seq.data_g2 = make_sequences(np.transpose(tr.data_g2), ParametersSVM.seq_len, 0)
        # td_seq.data_g3 = make_sequences(np.transpose(tr.data_g3), ParametersSVM.seq_len, 0)
        # td_seq.data_g4 = make_sequences(np.transpose(tr.data_g4), ParametersSVM.seq_len, 0)

        # fid = open('seq_data.dat','wb')
        # pickle.dump(td_seq, fid)
        # fid.close()

        fid = open('seq_data.dat','rb')
        td_seq = pickle.load(fid)
        fid.close()

        # Train the SVMs
        if ParametersSVM.svm_classifier_type is '1vs1':
            trainSVM_models_1vs1(model_ubm, td_seq, modeldir_patient_svm, ParametersSVM)
        elif ParametersSVM.svm_classifier_type is '1vsRest':
            trainSVM_models_1vsRest(model_ubm, td_seq, modeldir_patient_svm, ParametersSVM)
        elif ParametersSVM.svm_classifier_type is 'CS':
            trainSVM_models_crammer_singer(model_ubm, td_seq, modeldir_patient_svm, ParametersSVM)
        elif ParametersSVM.svm_classifier_type is 'BI_1vs1':
            trainSVM_models_builtin_1vs1(model_ubm, td_seq, modeldir_patient_svm, ParametersSVM)
        a=1





