"""
This function creates the UBM models
"""

from ReadData import *
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import mixture, cross_validation
import pickle
from Model_Sel_GMM import model_sel_gmm

class model_ubm:
    norm_template=[]
    pca_param=[]
    GMM_param=[]

def train_UBM(targetlist, modeldir_UBM, datadir, labeldir, whichfeatures, expID,  total_lim, ParametersUBM):
    """

    :return:
    """

    # Load file names in targetlist.
    fd = open(targetlist)
    names=fd.read()
    names=names.split('\n')
    names=np.asarray(names)

    useind=[]
    for lv in names:
        useind.append(int(lv[4:6]))
    useind=np.asarray(useind)
    patients_actual=np.unique(useind)

    # Create directory for UBM models
    if not os.path.exists(modeldir_UBM):
        os.makedirs(modeldir_UBM)
    model_ubm_i = model_ubm()


    for pat in range(4,5,1):

        # Check if the UBM model for this file already exists
        modeldir_patient = os.path.join(modeldir_UBM, str('patient_' + str(pat+1) + '_GMM_model_UBM.pickle'))
        if os.path.isfile(modeldir_patient):
            print ('Model file for Patient',pat+1,'already exist')
            continue
        else:
            print('Creating UBM model for Patient', patients_actual[pat])


        # Select the training files for this iteration
        # Read data of all the files except for the paitent being trained for
        train_ind = np.where(useind != patients_actual[pat])[0]
        train_ind_files_names = names[train_ind]

        # Read Data from the files selected for training
        tr = read_data(train_ind, train_ind_files_names, whichfeatures, datadir, labeldir, total_lim, ParametersUBM)


        # Apply Normalization
        fulltrset = np.concatenate([tr.data_g1, tr.data_g2, tr.data_g3, tr.data_g4], axis=1)
        fulltrset_norm = preprocessing.scale(fulltrset, axis=1)
        model_ubm_i.norm_template = preprocessing.StandardScaler().fit(np.transpose(fulltrset))
        fulltrset_norm=np.transpose(fulltrset_norm)


        if ParametersUBM.model_sel==1:
            labels= np.concatenate((np.ones((np.size(tr.data_g1,1),1)), 2*np.ones((np.size(tr.data_g2,1),1)), 3*np.ones((np.size(tr.data_g3,1),1)), 4*np.ones((np.size(tr.data_g4,1),1))))
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(fulltrset_norm, labels, test_size=0.7, random_state=1)
            del X_test, y_test, y_train
            pca_values=[0.98, 0.99, 1]
            gmm_values=[4, 8, 16, 32]
            pca_threshold, gmm_components =model_sel_gmm(X_train, pca_values, gmm_values)

        else:
            pca_threshold = ParametersUBM.pca_threshold
            gmm_components = ParametersUBM.gauss


        # Apply PCA
        pca = PCA(n_components = pca_threshold)
        model_ubm_i.pca_param = pca.fit(fulltrset_norm)
        fulltrset_norm_pca = pca.transform(fulltrset_norm)


        # Get GMM model
        g = mixture.GMM(n_components= gmm_components)
        model_ubm_i.GMM_param = g.fit(fulltrset_norm_pca)


        fid = open(modeldir_patient,'wb')
        pickle.dump(model_ubm_i,fid)
        fid.close()



