import platform, os

"""
All the initial directories should be defined in this module
"""


def define_directories(expID, ParametersSVM, ParametersUBM, Flag_TrainSVM):
    """
    This function defines the directories.
    :param expID: Experiment ID
    :param ParametersSVM: Parameters to be used by SVM models
    :param ParametersUBM: Parameters to be used by UBM models
    :param Flag_TrainSVM: Flag if SVM training has been performed before. It will prevent the program to use the
                            directories used by UBM function ..
    :return: datadir, labeldir, CurrentDir, targetlist, total_lim, modeldir_UBM, modeldir_SVM, SaveResultFileName_UBM, \
           SaveResultFileName_SVM
    """

    if Flag_TrainSVM == 1:
        datadir = os.path.join('..', '..', 'Feature_Extraction', 'OutputFiles', str('W' + str(ParametersSVM.window)+ '_S'+ str(ParametersSVM.shift)))
    else:
        datadir = os.path.join('..', '..', 'Feature_Extraction', 'OutputFiles', str('W' + str(ParametersUBM.window)+ '_S'+ str(ParametersUBM.shift)))
    labeldir = os.path.join('..', '..', 'Make_Annotations', 'HIE_grades_reviewed_new.xlsx')

    CurrentDir = os.getcwd()
    modeldir = os.path.join('..', 'models\\')
    targetlist = ('..\\..\\Feature_Extraction\\HIE_files_actual.txt')
    total_lim = '_Matlab_Features'  # ending of the feature file name

    modeldir_UBM = str(modeldir + expID +
                       '_' + str(ParametersUBM.num_of_points_per_file) +
                       '_TL' + str(ParametersUBM.total_lim) +
                       '_W' + str(ParametersUBM.window) +
                       '_Sh' + str(ParametersUBM.shift) +
                       '_p' + str(int(ParametersUBM.pca_threshold*100)) +
                       '_g' + str(ParametersUBM.gauss) + '\\')

    modeldir_SVM = str(modeldir_UBM + 'models_SVM\\' +
                               expID + '_' +
                               str(ParametersSVM.num_of_points_per_file) +
                               '_TL' + str(ParametersSVM.total_lim) +
                               '_W' + str(ParametersSVM.window) +
                               '_Sh' + str(ParametersSVM.shift) +
                               '_p' + str(int(ParametersUBM.pca_threshold*100)) +
                               '_g' + str(ParametersSVM.gauss) +
                               '_k' + ParametersSVM.kernel +
                               '_SqLn' + str(ParametersSVM.seq_len) +
                               '_CT_' + ParametersSVM.svm_classifier_type + '\\')

    modeldir_SVM = str(modeldir_UBM + 'models_SVM\\' +
                               expID +
                               '_' + str(ParametersSVM.num_of_points_per_file) +
                               '_TL' + str(ParametersSVM.total_lim) +
                               '_W' + str(ParametersSVM.window) +
                               '_Sh' + str(ParametersSVM.shift) +
                               '_p' + str(int(ParametersUBM.pca_threshold*100)) +
                               '_g' + str(ParametersSVM.gauss) +
                               '_' + ParametersSVM.kernel +
                               '_SqLn' + str(ParametersSVM.seq_len) +
                               '_CT_' + ParametersSVM.svm_classifier_type + '\\')

    SaveResultFileName_UBM = ('..\\results\Results_' +
                              expID + '_' +
                              str(ParametersUBM.num_of_points_per_file) +
                              '_TL' + str(ParametersUBM.total_lim) +
                              '_W' + str(ParametersUBM.window) +
                              '_Sh' + str(ParametersUBM.shift) +
                              '_p' + str(int(ParametersUBM.pca_threshold*100)) +
                              '_g' + str(ParametersUBM.gauss) + '\\')

    SaveResultFileName_SVM = (SaveResultFileName_UBM +
                              expID + '_' +
                              str(ParametersSVM.num_of_points_per_file) +
                              '_TL' + str(ParametersSVM.total_lim) +
                              '_W' + str(ParametersSVM.window) +
                              '_Sh' + str(ParametersSVM.shift) +
                              '_p' + str(int(ParametersUBM.pca_threshold*100)) +
                              '_g' + str(ParametersSVM.gauss) +
                              '_k' + ParametersSVM.kernel +
                              '_SqLn' + str(ParametersSVM.seq_len) +
                              '_CT_' + ParametersSVM.svm_classifier_type + '\\')

    if not os.path.exists(SaveResultFileName_SVM):
        os.makedirs(SaveResultFileName_SVM)

    return datadir, CurrentDir, targetlist, total_lim, modeldir_UBM, modeldir_SVM, SaveResultFileName_UBM, \
           SaveResultFileName_SVM,labeldir
