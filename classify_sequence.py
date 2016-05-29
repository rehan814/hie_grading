import numpy as np
import pickle,os
from MakeSuperVector import make_super_vector
class result_class:
    """
    final_dec_epoch:  A single column vector of size(number of sequences) decided class for the epoch 1,2,3 or 4
    prob_epoch: (number of grades, number of sequences) probability of each epoch to be from grade 1,2,3 or 4
    dec_epoch: decision of each classifier
                its size will depend on the type of multiclass classifier
                1vsRest= (4, number of sequences)
                1vs1= (6, number of sequences)

    dis_epoch: distance to separating hyperplane of each classifier
                its size will depend on the type of multiclass classifier
                1vsRest= (4, number of sequences)
                1vs1= (6, number of sequences)

    """
    def __init__(self,classes, datasize):
        self.final_dec_epoch=np.zeros((classes))
        self.prob_epoch=np.zeros((classes,datasize))
        self.dec_epoch=np.zeros((classes,datasize))
        self.dis_epoch=np.zeros((classes,datasize))

def classify_sequence(data, model_ubm, modeldir_patient_svm, ParametersSVM):

    if ParametersSVM.svm_classifier_type is '1vs1':
        result_data = classify_1vs1(data, model_ubm, modeldir_patient_svm, ParametersSVM)
    elif ParametersSVM.svm_classifier_type is '1vsRest':
        result_data = classify_1vsRest(data, model_ubm, modeldir_patient_svm, ParametersSVM)
    elif ParametersSVM.svm_classifier_type is 'CS':
        result_data = classify_crammer_singer(data, model_ubm, modeldir_patient_svm, ParametersSVM)
    elif ParametersSVM.svm_classifier_type is 'BI_1vs1':
        result_data=classify_builtin_1vs1(data, model_ubm, modeldir_patient_svm, ParametersSVM)
    return result_data



def classify_1vs1(data, model_ubm, modeldir_patient_svm, ParametersSVM):
    prob_epoch = np.zeros((6,int(max(data[0]))))
    dec_class = np.zeros((6,int(max(data[0]))))
    a=0
    for i in range(1,5,1):
        for j in range(i+1,5,1):

            load_file_name_svm = os.path.join(modeldir_patient_svm, ('SVM_model_' + str(i) + '_' + str(j)))
            fid = open(load_file_name_svm,'rb')
            model_svm = pickle.load(fid)
            fid.close()

            super_vectors = make_super_vector(data[0], data[1], model_svm.rel_factor, int(max(data[0])), model_ubm, ParametersSVM, test_flag=1)
            Dzero=(np.where(super_vectors[1]==0))
            dec_class[a,:] = model_svm.predict(super_vectors[0].T)
            prob_epoch[a,:] = model_svm.predict_proba(super_vectors[0].T)[:,1]

            prob_epoch[a, Dzero] = np.nan
            dec_class[a, Dzero] = 0
            a=a+1



    if ParametersSVM.prob_based_dec==1:
        final_class_epoch = max(prob_epoch)
    else:
        final_class_epoch = get_class(dec_class)

    return final_class_epoch, dec_class, prob_epoch





def classify_1vsRest(data, model_ubm, modeldir_patient_svm, ParametersSVM):

    result=result_class(4,int(max(data[0])))

    a=0
    for i in range(1,5,1):
        load_file_name_svm = os.path.join(modeldir_patient_svm, ('SVM_model_' + str(i) + '_rest'))
        fid = open(load_file_name_svm,'rb')
        model_svm = pickle.load(fid)
        fid.close()

        super_vectors = make_super_vector(data[0], data[1], model_svm.rel_factor, int(max(data[0])), model_ubm, ParametersSVM, test_flag=1)
        Dzero = np.where(super_vectors[1]==0)[0]

        result.dec_epoch[a,:] = model_svm.predict(super_vectors[0].T)
        result.dis_epoch[a,:] = model_svm.decision_function(super_vectors[0].T)
        result.prob_epoch[a,:] = model_svm.predict_proba(super_vectors[0].T)[:,1]

        result.dec_epoch[a,np.where(result.dec_epoch==0)]=5
        result.dec_epoch[a,np.where(result.dec_epoch==1)]=i

        result.prob_epoch[a, Dzero] = np.nan
        result.dec_epoch[a, Dzero] = 0
        result.dis_epoch[a, Dzero] = np.nan

        a=a+1


    if ParametersSVM.final_dec is 'prob':
        result.final_dec_epoch = np.argmax(result.prob_epoch, axis=0)+1
        result.final_dec_epoch[Dzero]=0
    elif ParametersSVM.final_dec is 'distance':
        result.final_dec_epoch = np.argmin(result.dis_epoch, axis=0)+1
        result.final_dec_epoch[Dzero]=0

    return result




def classify_crammer_singer(data, model_ubm, modeldir_patient_svm, ParametersSVM):

    load_file_name_svm = os.path.join(modeldir_patient_svm, ('SVM_model_CS'))
    fid = open(load_file_name_svm,'rb')
    model_svm = pickle.load(fid)
    fid.close()

    super_vectors = make_super_vector(data[0], data[1], model_svm.rel_factor, int(max(data[0])), model_ubm, ParametersSVM, test_flag=1)
    final_class_epoch = model_svm.predict(super_vectors[0].T)
    dec_dist = model_svm.decision_function(super_vectors[0].T)
    Dzero=(np.where(super_vectors[1]==0))

    final_class_epoch[Dzero]=0
    prob_epoch=0

    return final_class_epoch, dec_dist, prob_epoch

def classify_builtin_1vs1(data, model_ubm, modeldir_patient_svm, ParametersSVM):
    result=result_class(4,int(max(data[0])))
    load_file_name_svm = os.path.join(modeldir_patient_svm, ('SVM_model_CS'))
    fid = open(load_file_name_svm,'rb')
    model_svm = pickle.load(fid)
    fid.close()

    super_vectors = make_super_vector(data[0], data[1], model_svm.rel_factor, int(max(data[0])), model_ubm, ParametersSVM, test_flag=1)
    result.final_dec_epoch = model_svm.predict(super_vectors[0].T)
    result.dis_epoch = model_svm.decision_function(super_vectors[0].T)
    Dzero=(np.where(super_vectors[1]==0))[0]

    result.final_dec_epoch[Dzero]=0
    prob_epoch=0

    return result

def get_class(votes1):
    # Calculate the class
    class_epoch=np.zeros((np.size(votes1,1)))
    for m in range(0,np.size(votes1,1)):
        votes=votes1[:,m]
        if sum(votes)==0:
            class_epoch[m]=0
            continue

        b=np.asarray([1,2,3,4])
        a=np.bincount(votes,minlength=5)
        a=a[1:]

        class_1 = b[np.where(a==np.max(a))[0]]
        if np.size(class_1,0)>1:
            v1=np.zeros((3,4))
            a=0
            for i in range(0,4,1):
                for j in range(i+1,4,1):
                    v1[i,j]=votes[a]
                    a=a+1

            if np.size(class_1,0)==2:
                class_1 = v1[class_1[0]-1,class_1[1]-1]

            elif np.size(class_1,0)==3:
                cTemp=np.asarray([v1[class_1[0]-1,class_1[1]-1] , v1[class_1[0]-1,class_1[2]-1] , v1[class_1[1]-1,class_1[2]-1]],dtype=int)
                a=np.bincount(cTemp,minlength=5)
                a=a[1:]
                cTemp2=b[np.where(a==np.max(a))[0]]
                class_1=v1[cTemp2[0],cTemp2[1]]

            elif np.size(class_1,0)==4:
                class_1=0
        else:
            class_epoch[m]=class_1

    return class_epoch