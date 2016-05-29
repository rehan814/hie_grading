
"""
This is the top level script for the project where most of the variables are initialized
 Try to initialize all the variables used in the subsequent functions in this file
"""
import numpy as np
import os,xlrd, pickle, scipy.io
from DefineDirectories import define_directories
from MakeSequences import make_sequences
from classify_sequence import classify_sequence
from RemoveNaNs import remove_NaNs
from GetGrade import get_grade
from classify_sequence import result_class

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
    seq_len_limit=''


expID  = 'GMM_UBM' # experiment ID
whichfeatures  = np.arange(1,56,1)  # mask for features to use

ParametersUBM = Paramertes()
ParametersUBM.num_of_points_per_file = 0
ParametersUBM.total_lim = 90000
ParametersUBM.gauss = 4
ParametersUBM.window = 8
ParametersUBM.shift = 4
ParametersUBM.pca_threshold = 0.98

ParametersSVM=Paramertes()
ParametersSVM.num_of_points_per_file = 0
ParametersSVM.total_lim = 90000
ParametersSVM.gauss= 4
ParametersSVM.window = 8
ParametersSVM.shift = 4
ParametersSVM.kernel = 'KL'
ParametersSVM.seq_len =20
ParametersSVM.limit=15
ParametersSVM.num_of_classes = 4
ParametersSVM.equal_number_of_sequences=0
ParametersSVM.svm_classifier_type = 'BI_1vs1' # 1vs1 , 1vsRest, cram_sing
ParametersSVM.final_dec = 'prob' # prob, distance

Flag_TrainSVM = 1

(datadir, CurrentDir, targetlist, total_lim, modeldir_UBM, modeldir_SVM, SaveResultFileName_UBM,
SaveResultFileName_SVM,labeldir) = define_directories(expID, ParametersSVM,ParametersUBM,Flag_TrainSVM)

fd = open(targetlist)
names=fd.read()
names=names.split('\n')
names=np.asarray(names)

workbook = xlrd.open_workbook(labeldir)
worksheet = workbook.sheet_by_index(0)
eeg_file_name = worksheet.col_values(0, start_rowx=1, end_rowx=None)
grade_label = np.asarray(worksheet.col_values(1, start_rowx=1, end_rowx=None))

overlap=0
i=1
perChannel=4
overall_grade=np.zeros((1,55))[0]
confidence_level=np.zeros((1,55))[0]
for patient in range(4,5,1):
    if patient==53:
        continue

    SaveFileName_SVM=(SaveResultFileName_SVM + 'Patient_' + str((patient + 1)) + '_o_' + str(overlap) + '.mat')
#         if  0%exist(SaveFileName_SVM,'file')
# %             disp(['Result file for patient ' num2str(patient) ' already exists !!']);
#             d=load(SaveFileName_SVM,'decision','class','votes');
#             class=d.class;
#             votes=d.votes;
#             try
#                 [overallGrade(i,j), confidenceLevel(i,j), secGrade(i,j), distribution(i,:)]=GetGrade(perChannel, class, votes);
#             catch
#                 overallGrade(i,j)=0; confidenceLevel(i,j)=0;
#             end
#             i=i+1;
#             continue;
#         end

    print('Grading patient', names[patient])
    grade_patient = grade_label[patient]

    # %%%%% Assign Directory Paths and Check their existance %%%%%%
    modeldir_patient_ubm = os.path.join(modeldir_UBM, str('patient_' + str(patient + 1) + '_GMM_model_UBM.pickle'))
    modeldir_patient_svm = os.path.join(modeldir_SVM, str('patient_' + str(patient+1)))

    if not os.path.isfile(modeldir_patient_ubm):
        print('UBM Model does not exist')
        break
    elif not os.path.exists(modeldir_patient_svm):
        print('SVM directory does not exist')
        break

    # %%%%%%% Read The Data from the File %%%%%%%%
    data_raw = scipy.io.loadmat(os.path.join(datadir, str(names[patient] + total_lim)))

    feat_vec = data_raw['feat_vec']
    del data_raw
    if len(whichfeatures) == np.size(feat_vec,0):
        data1 = feat_vec
    else:
        data1 = feat_vec[whichfeatures,:,:]
    del feat_vec

    fid = open(modeldir_patient_ubm,'rb')
    model_ubm = pickle.load(fid)
    fid.close()

    result = ''
    grade = ''
    votes = ''

    # class_epochs_channel=np.zeros((tot_num_seq,1))
    decision_channel= {'channel1': result_class, 'channel2': result_class, 'channel3': result_class,
                       'channel4': result_class, 'channel5': result_class, 'channel6': result_class,
                       'channel7': result_class, 'channel8': result_class}

    for channel in range(0,8,1):

        data_channel = data1[:,:,channel]
        # Remove NaNs to apply the normalization template
        [data_channel,labels] = remove_NaNs(data_channel,np.ones((1,np.size(data_channel,1)))[0], replace_zeros=1)
        # Normalization
        data_channel = model_ubm.norm_template.transform(np.transpose(data1[:,:,channel]))
        # Again replace the zeros with NaNs to get proper sequences
        data_channel[np.where(labels==0),:]=np.nan
        # Make Sequences
        seq_data = make_sequences(np.transpose(data_channel), ParametersSVM.seq_len, overlap)

        # Initial channel
        if channel==0:
            class_epochs_channel=np.zeros((max(seq_data[0]),8))
            votes_channel=np.zeros((6,max(seq_data[0]),8))

        # Classify sequences
        string_channel=('channel'+ str(channel+1))
        decision_channel[string_channel] = classify_sequence(seq_data, model_ubm, modeldir_patient_svm, ParametersSVM)

        # Save votes/probability and decisions separately
        # class_epochs_channel[:,channel]=dec[0]
        # votes_channel[:,:,channel]= dec[1]


    # Get the Grade of whole file. Majority voting
    overall_grade[patient], confidence_level[patient]=get_grade(class_epochs_channel,perchannel=0)

a=1
#         try
#         [overallGrade(i,j), confidenceLevel(i,j)]=GetGrade(perChannel, class, votes);
#         catch
#             overallGrade(i,j)=0; confidenceLevel(i,j)=0;
#         end
#
#         decision= overallGrade(i,j);
#
#         save(SaveFileName_SVM,'decision','class','votes');
# %           if decision ==grade_label(patient,2)
# %             disp('correct')
# %         else
# %             disp('wrong')
# %         end
#         i=i+1;
#         clear Data class data  model_UBM seq votes
#     end
#     j=j+1;
# end