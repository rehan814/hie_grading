import numpy as np
from MAP_GMM import map_gmm
from RemoveNaNs import remove_NaNs


def make_super_vector(seq, data, rel_factor, num_of_seq, model_ubm, ParametersSVM, data_shuffle=0, test_flag=0):
    """

    :type ParametersSVM: Parameters of SVM model
    """
    if test_flag == 0:
        if data_shuffle==0:
            super_vector = np.zeros((model_ubm.GMM_param.n_components * np.size(model_ubm.GMM_param.means_[0]), num_of_seq))
            for k in range(0, num_of_seq, 1):        # sequence from the grade 1
                # Find the sequence 1 from the sequence vector
                idx1 = np.where(seq == k)
                data1 = data[:, idx1[0]]

                # Make the model of the data adopted from the UBM
                adapted_model = prep_and_map(data1, model_ubm, rel_factor)
                super_vector[:, k] = make_gmm_supervector(adapted_model, ParametersSVM)
        else:
            np.random

        return super_vector

    else:
        super_vector = np.zeros((model_ubm.GMM_param.n_components * np.size(model_ubm.GMM_param.means_[0]), num_of_seq))
        DataLabel = np.zeros((num_of_seq,1))
        for k in range(0, num_of_seq, 1):        # sequence from the grade 1
            # Find the sequence 1 from the sequence vector
            idx1 = np.where(seq == k)
            data1 = data[:, idx1[0]]
            data1 = remove_NaNs(data1)[0]

            if np.size(data1,1)<ParametersSVM.limit:
                continue
            else:
                # Make the model of the data adopted from the UBM
                adapted_model = prep_and_map(data1, model_ubm, rel_factor)
                super_vector[:, k] = make_gmm_supervector(adapted_model, ParametersSVM)
                DataLabel[k] = 1
        return super_vector, DataLabel


def prep_and_map(data, model, rel_factor):
    # Determining the eigen values and vectors
    data_trans = model.pca_param.transform(np.transpose(data))  # apply PCA decomposition

    # Using the MAP function
    adapted_model = map_gmm(model, data_trans, rel_factor)
    return adapted_model


def make_gmm_supervector(model, ParametersSVM):
    """

    :type ParametersSVM: object
    """
    means_act = []
    if ParametersSVM.kernel == 'KL':
        for mix_indx in range(0, model.n_components, 1):
            sec_vec = np.asarray(np.tile(model.weights_[mix_indx], (np.size(model.covars_, 1), 1)) /
                                 np.asmatrix(np.sqrt(model.covars_[1])).T)
            means_act = np.append(means_act, np.multiply(sec_vec, np.asmatrix(model.means_[mix_indx]).T))
    else:
        for mix_indx in range(0, model.n_components, 1):
            means_act = np.append(means_act, (model.means_[mix_indx]).T)

    return means_act
