import numpy as np



def map_gmm(ubm_model, data, rel_factor):


    num_feats = np.size(data,1)
    num_mixes = ubm_model.GMM_param.n_components
    num_datapoints = np.size(data,0)

    gaussian_prob = ubm_model.GMM_param.predict_proba(data)

    for data_indx in range(0,np.size(gaussian_prob, 0), 1):
        gaussian_prob[data_indx,:] = gaussian_prob[data_indx,:] * ubm_model.GMM_param.weights_

    sum_prob = np.sum(gaussian_prob,1)
    # sum_prob[np.where(sum_prob==Inf)] = 0
    Pr = np.divide(np.transpose(gaussian_prob), (np.ones([ubm_model.GMM_param.n_components,1])*sum_prob))
    ni = np.sum(Pr, 1)
    alpha = np.divide(ni, (ni + rel_factor))

    data = np.transpose(data)
    adapted_means= np.zeros((num_mixes,num_feats))
    for mu in range(0, ubm_model.GMM_param.n_components,1):
        x_sum = 0
        for t in range(0, num_datapoints, 1):
            x_sum = x_sum + (Pr[mu,t] * data[:,t])

        if not np.isnan(ni[mu]):
            if not ni[mu] == 0:
                x_sum = np.divide(x_sum, ni[mu])

        adapted_means[mu, :] = (alpha[mu] * x_sum) + ((1-alpha[mu]) * ubm_model.GMM_param.means_[mu])

    adapted_model=ubm_model.GMM_param
    adapted_model.means_=adapted_means
    return adapted_model


