import numpy as np

def get_grade(class_epochs, perchannel=0):

    if perchannel==0:
        class_epochs_1=np.asarray(class_epochs)
        class_epochs_1.flatten()
        c=np.asarray([0,0,0,0])
        for grade in range (0,4,1):
            c[grade]= np.size(np.where(class_epochs==grade+1)[0])
        overall_grade= np.where(c==max(c))[0]
        confidence_level=(c[overall_grade]/sum(c))*100
        return (overall_grade+1, confidence_level)

    elif perchannel==4:
        class_epochs_1=np.asarray(class_epochs)
        class_epochs_1.flatten()
        c=np.asarray([0,0,0,0])
        for grade in range (0,4,1):
            c[grade]= np.size(np.where(class_epochs==grade+1)[0])

        np.sort(c)
        overall_grade= np.where(c==max(c))[0]
        confidence_level=(c[overall_grade]/sum(c))*100
