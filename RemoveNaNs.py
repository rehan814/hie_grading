"""
Has a function that removes NaNs from the input data
"""
import numpy as np

def remove_NaNs(data,Labels=1, replace_zeros=0):
    """
    :param data: data should be in the form data= (features, observations)
    :param Labels: optional

    :return:
    data: with nans removed
    Labels with nan columns of data removed
    """

    col=[]
    for lv in range(0,np.size(data,1),1):
        if np.sum(np.isnan(data[:,lv]))>0:
            col=np.append(col,lv)
    if replace_zeros==1:
        data[:,np.asarray(col,dtype=int)]=0
    else:
        data=np.delete(data,col,axis=1)

    if np.any(Labels) == False:
        Labels=[]
    else:
        if replace_zeros==1:
            Labels[np.asarray(col,dtype=int)]=0
        else:
            Labels=np.delete(Labels,col)

    return data, Labels