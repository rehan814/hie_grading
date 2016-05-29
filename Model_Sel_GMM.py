import numpy as np
from sklearn.decomposition import PCA
from sklearn import mixture


def model_sel_gmm(data, pca_values, gmm_values):

    bic = np.zeros((np.size(pca_values), np.size(gmm_values)))
    i = 0
    for pca_iter in pca_values:
         # Apply PCA
        pca = PCA(n_components = pca_iter)
        pca.fit(data)
        data_pca = pca.transform(data)
        j = 0
        for gmm_iter in gmm_values:
            # Fit GMM model
            gmm = mixture.GMM(n_components=gmm_iter)
            gmm.fit(data_pca)
            bic[i,j] = gmm.bic(data_pca)

            j=j+1
        i=i+1

    # select the values that gives the minimum values.
    min_values= (np.where(bic == bic.min()))
    if min_values[0]>1:
        best_pca_value = pca_values[min_values[0][0]]
        best_gmm_value = gmm_values[min_values[1][0]]
    else:
        best_pca_value = pca_values[min_values[0]]
        best_gmm_value = gmm_values[min_values[1]]

    return best_pca_value, best_gmm_value
