import  numpy as np
from os import path
from MakeSuperVector import make_super_vector
from RemoveNaNs import remove_NaNs
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import StratifiedKFold
import pickle
from sklearn import svm

def trainSVM_models_1vs1(model_ubm, tr_data, modeldir_patient_svm, ParametersSVM):
    """
    This function will produce the 1vs1 multiclass SVM models
    It will first create the supervectors from the data and then feed them to make_svm_model for selecting c parameter
    and producing a SVM model with best parameter

    :param model_ubm: UBM model to adapt the sequences.
    :param tr_data: training data
    :param modeldir_patient_svm: model directory to save the trained SVM model
    :param ParametersSVM: parameters to be used for selecting relevance factor, and supervector creation.
    :return: nothing
    """

    for i in range(1,5,1):
        for j in range(i+1,5,1):
            data1 = getattr(tr_data, str('data_g' + str(i)))
            data2 = getattr(tr_data, str('data_g' + str(j)))
            r_final = 5

            # Select what ratio of each class should be present for training dataset.
            # Make supervectors
            if ParametersSVM.equal_number_of_sequences == 1:
                if max(data1[0]) < max(data2[0]):
                    num_of_seq = int(max(data1[0]))
                else:
                    num_of_seq = int(max(data2[0]))
                super_vector1 = make_super_vector( data1[0], data1[1], r_final, num_of_seq, model_ubm, ParametersSVM)
                super_vector2 = make_super_vector( data2[0], data2[1], r_final, num_of_seq, model_ubm, ParametersSVM)
            else:
                num_of_seq = int(max(data1[0]))
                super_vector1 = make_super_vector( data1[0], data1[1], r_final, num_of_seq, model_ubm, ParametersSVM)
                num_of_seq = int(max(data2[0]))
                super_vector2 = make_super_vector( data2[0], data2[1], r_final, num_of_seq, model_ubm, ParametersSVM)


            fulltrset_sv = np.concatenate([super_vector1, super_vector2], axis=1)
            labels = np.concatenate([np.ones(np.size(super_vector1, 1)), np.zeros(np.size(super_vector2, 1))])
            del super_vector1, super_vector2

            remove_NaNs(fulltrset_sv, labels)

            print('Building SVM models for grade', i, 'vs', j)
            save_file_name_svm = path.join(modeldir_patient_svm, ('SVM_model_' + str(i) + '_' + str(j)))
            model_svm = make_svm_model(fulltrset_sv, labels)
            model_svm.rel_factor=r_final

            fid = open(save_file_name_svm,'wb')
            pickle.dump(model_svm, fid)
            fid.close()


def trainSVM_models_1vsRest(model_ubm, tr_data, modeldir_patient_svm, ParametersSVM):
    """
    This function will produce the 1vsRest SVM models. It will first create the supervectors from the data and
     then feed them to make_svm_model for selecting c parameter and producing a SVM model with best parameter

    :param model_ubm: UBM model to adapt the sequences.
    :param tr_data: training data
    :param modeldir_patient_svm: model directory to save the trained SVM model
    :param ParametersSVM: parameters to be used for selecting relevance factor, and supervector creation.
    :return: nothing
    """
    targets = np.arange(1,ParametersSVM.num_of_classes+1,1)

    for i in range(1,5,1):

            data1 = getattr(tr_data, str('data_g' + str(i)))

            # Concatenate the data of all classes other then the class(i)
            classes_rest = np.where(targets != i)[0]
            for lv in range(0,np.size(classes_rest),1):
                data2_temp=getattr(tr_data, str('data_g' + str(targets[classes_rest[lv]])))
                if lv == 0:
                    data2=np.asarray(data2_temp)
                else:
                    data2[0] = np.concatenate([data2[0], data2_temp[0]+np.max(data2[0])], axis=0)
                    data2[1] = np.concatenate([data2[1], data2_temp[1]], axis=1)

            del data2_temp

            r_final = 5

            # Select what ratio of each class should be present for training dataset.
            # Make supervectors
            if ParametersSVM.equal_number_of_sequences == 1:
                if max(data1[0]) < max(data2[0]):
                    num_of_seq = int(max(data1[0]))
                else:
                    num_of_seq = int(max(data2[0]))
                data_shuffle=1
                super_vector1 = make_super_vector(data1[0], data1[1], r_final, num_of_seq, model_ubm, ParametersSVM, data_shuffle, test_flag=0,)
                super_vector2 = make_super_vector(data2[0], data2[1], r_final, num_of_seq, model_ubm, ParametersSVM)
            else:
                num_of_seq = int(max(data1[0]))
                super_vector1 = make_super_vector(data1[0], data1[1], r_final, num_of_seq, model_ubm, ParametersSVM)
                num_of_seq = int(max(data2[0]))
                super_vector2 = make_super_vector(data2[0], data2[1], r_final, num_of_seq, model_ubm, ParametersSVM)


            fulltrset_sv = np.concatenate([super_vector1, super_vector2], axis=1)
            labels = np.concatenate([np.ones(np.size(super_vector1, 1)), np.zeros(np.size(super_vector2, 1))])
            del super_vector1, super_vector2

            remove_NaNs(fulltrset_sv, labels)

            print('Building SVM models for grade', i, 'vs_rest')
            save_file_name_svm = path.join(modeldir_patient_svm, ('SVM_model_' + str(i) + '_rest'))
            model_svm = make_svm_model_1vsRest(fulltrset_sv, labels)
            model_svm.rel_factor = r_final

            fid = open(save_file_name_svm,'wb')
            pickle.dump(model_svm, fid)
            fid.close()


def trainSVM_models_crammer_singer(model_ubm, tr_data, modeldir_patient_svm, ParametersSVM):
    """
    This function will train a multiclass SVM based on crammers_singer implementation.
    It will give all the data from all the grades and then finally train a SVM model.

    :param model_ubm: UBM model to adapt the sequences.
    :param tr_data: training data
    :param modeldir_patient_svm: model directory to save the trained SVM model
    :param ParametersSVM: parameters to be used for selecting relevance factor, and supervector creation.
    :return: nothing
    """

    targets = np.arange(1,ParametersSVM.num_of_classes+1,1)
    # Concatenate the data of all classes
    for lv in range(0,ParametersSVM.num_of_classes,1):
        data_temp=getattr(tr_data, str('data_g' + str(targets[lv])))
        if lv == 0:
            data=np.asarray(data_temp)
        else:
            data[0] = np.concatenate([data[0], data_temp[0]+np.max(data[0])], axis=0)
            data[1] = np.concatenate([data[1], data_temp[1]], axis=1)

    num_of_seq = int(max(data[0]))
    r_final=5
    fulltrset_sv =  make_super_vector(data[0], data[1], r_final, num_of_seq, model_ubm, ParametersSVM)
    labels = np.concatenate([np.ones((int(max(tr_data.data_g1[0])))), 2*np.ones((int(max(tr_data.data_g2[0])))),
                             3*np.ones((int(max(tr_data.data_g3[0])))), 4*np.ones((int(max(tr_data.data_g4[0]))))])


    # Do k-fold internal cross validation to select best C parameter
    parameters = [{'C': [1, 10, 100, 1000]}]
    kf = StratifiedKFold(labels, n_folds=3)

    gscv = GridSearchCV(LinearSVC(multi_class='crammer_singer'), parameters, cv=kf)
    model_svm = gscv.fit(fulltrset_sv.T, labels)
    print('best score =', gscv.best_score_ , 'with C ', gscv.best_params_)
    save_file_name_svm = path.join(modeldir_patient_svm, ('SVM_model_CS'))
    model_svm.rel_factor=r_final

    fid = open(save_file_name_svm,'wb')
    pickle.dump(model_svm, fid)
    fid.close()


def make_svm_model(fulltrset_sv, labels):
    """
    This function will do a stratified k fold cross validation to select best C parameter and then return a trained SVM
    with that best parameter.

    :param fulltrset_sv: the data to be used for training
    :param labels: labels of the data
    :return: gscv: trained svm model
    """
    parameters = [{'C': [1, 10, 100, 1000]}]
    kf = StratifiedKFold(labels, n_folds=2)

    gscv = GridSearchCV(SVC(kernel='linear', probability=True), parameters, cv=kf)
    gscv.fit(fulltrset_sv.T, labels)
    print('best score =', gscv.best_score_ , 'with C ', gscv.best_params_)

    return gscv


def make_svm_model_1vsRest(fulltrset_sv, labels):
    """
    This function is different from make_svm_model
    it will do two iterations of stratified k fold cross validation rather than 3.
    There reasons are as follows,
    we want to use SVC svm because it provides the probabilities however it doesn't provide 1vsRest
    multiclass classifiaction. Therefore we have to cross validate the parameters ourself.

    Because it is 1vsRest classification, it is necessary to have representation of all classes for the negative/rest
    class therefore, three strategies are used here.
    1- using stratified k fold so that both positive and negative class have similar representation to whole
        training set in each fold
    2- use shuffling, so that the negative class has the samples from all grades (rest class) in each fold.
    3- lastly two iterations of k-fold are used and the results are then averaged to select the best parameter.


    :param fulltrset_sv: the data to be used for training
    :param labels: labels of the data
    :return: gscv: trained svm model
    """

    # List of parameters to perform cross validation
    parameters = [{'C': [1, 10, 100, 1000]}]

    # 1st iteration of stratified k fold cross validation
    kf = StratifiedKFold(labels, n_folds=2, shuffle=True, random_state=1)
    gscv1 = GridSearchCV(SVC(kernel='linear', probability=True, class_weight = 'balanced'), parameters, cv=kf)
    gscv1.fit(fulltrset_sv.T, labels)


    # 2nd iteration of stratified k fold cross validation
    kf = StratifiedKFold(labels, n_folds=2, shuffle=True, random_state=2)
    gscv2 = GridSearchCV(SVC(kernel='linear', probability=True, class_weight = 'balanced'), parameters, cv=kf)
    gscv2.fit(fulltrset_sv.T, labels)


    # Concatenate all the accuracies and standard deviation of errors from the 2 iterations of k-fold
    grid_mean = np.zeros((np.size(parameters[0]['C'])))
    grid_std = np.zeros((np.size(parameters[0]['C'])))
    for lv in range(0,np.size(parameters[0]['C']),1):
        grid_mean[lv]= np.mean(np.concatenate([gscv1.grid_scores_[lv][2], gscv2.grid_scores_[lv][2]]))
        grid_std[lv] = np.std(np.concatenate([gscv1.grid_scores_[lv][2], gscv2.grid_scores_[lv][2]]))

    grid_mean[0]=grid_mean[3]

    # Select best C
    c = np.asarray(parameters[0]['C'])
    best_c = c[np.where(grid_mean==grid_mean.max())[0]]

    # If there are two best parameters then first try to select with minimum std in error.
    # If still there is a tie then select the minimum c value of the two.
    if best_c.__len__()>1:
        c=np.delete(c,np.where(grid_mean!=grid_mean.max())[0])
        grid_std=np.delete(grid_std,np.where(grid_mean!=grid_mean.max())[0])

        best_c = c[np.where(grid_std==grid_std.min())[0]]
        if best_c.__len__()>1:
            best_c = min(c)

    print('best score =', gscv2.grid_scores_[int(np.where(parameters[0]['C']==best_c)[0])][1], 'with C ', best_c)

    # Train SVM with best parameter selected
    clf = svm.SVC(C=best_c,kernel='linear',probability=True, class_weight = 'balanced')
    clf.fit(fulltrset_sv.T, labels)


    return clf


def trainSVM_models_builtin_1vs1(model_ubm, tr_data, modeldir_patient_svm, ParametersSVM):
    """
    This function will train a multiclass SVM based on crammers_singer implementation.
    It will give all the data from all the grades and then finally train a SVM model.

    :param model_ubm: UBM model to adapt the sequences.
    :param tr_data: training data
    :param modeldir_patient_svm: model directory to save the trained SVM model
    :param ParametersSVM: parameters to be used for selecting relevance factor, and supervector creation.
    :return: nothing
    """

    targets = np.arange(1,ParametersSVM.num_of_classes+1,1)
    # Concatenate the data of all classes
    for lv in range(0,ParametersSVM.num_of_classes,1):
        data_temp=getattr(tr_data, str('data_g' + str(targets[lv])))
        if lv == 0:
            data=np.asarray(data_temp)
        else:
            data[0] = np.concatenate([data[0], data_temp[0]+np.max(data[0])], axis=0)
            data[1] = np.concatenate([data[1], data_temp[1]], axis=1)

    num_of_seq = int(max(data[0]))
    r_final=5
    fulltrset_sv =  make_super_vector(data[0], data[1], r_final, num_of_seq, model_ubm, ParametersSVM)
    labels = np.concatenate([np.ones((int(max(tr_data.data_g1[0])))), 2*np.ones((int(max(tr_data.data_g2[0])))),
                             3*np.ones((int(max(tr_data.data_g3[0])))), 4*np.ones((int(max(tr_data.data_g4[0]))))])


    # Do k-fold internal cross validation to select best C parameter
    parameters = [{'C': [1, 10, 100, 1000]}]
    kf = StratifiedKFold(labels, n_folds=3)

    gscv = GridSearchCV(SVC(probability=True), parameters, cv=kf)
    model_svm = gscv.fit(fulltrset_sv.T, labels)
    print('best score =', gscv.best_score_ , 'with C ', gscv.best_params_)
    save_file_name_svm = path.join(modeldir_patient_svm, ('SVM_model_CS'))
    model_svm.rel_factor=r_final

    fid = open(save_file_name_svm,'wb')
    pickle.dump(model_svm, fid)
    fid.close()