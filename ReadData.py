"""
Reads data from all the specified files
"""
import xlrd, os, scipy.io
import numpy as np
from RemoveNaNs import remove_NaNs


class train_data:
    data_g1 = []
    data_g2 = []
    data_g3 = []
    data_g4 = []

    def __init__(self):
        self.data_g1 = np.empty((55,1),float)
        self.data_g2 = np.empty((55,1),float)
        self.data_g3 = np.empty((55,1),float)
        self.data_g4 = np.empty((55,1),float)
    #     data_g1 = np.asarray(np.zeros((55,1)))
    #     data_g2 = np.asarray(np.zeros((55,1)))
    #     data_g3 = np.asarray(np.zeros((55,1)))
    #     data_g4 = np.asarray(np.zeros((55,1)))

    # data_g1 = np.empty((55,1),float)
    # data_g2 = np.empty((55,1),float)
    # data_g3 = np.empty((55,1),float)
    # data_g4 = np.empty((55,1),float)


def read_data(train_ind, train_ind_files_names, whichfeatures, datadir, labeldir, total_lim, numsel):
    """

    :param targetlist:
    :param whichfeatures:
    :param patients:
    :param datadir:
    :param labeldir:
    :param total_lim:
    :param numsel:
    :return:
    """


    workbook = xlrd.open_workbook(labeldir)
    worksheet = workbook.sheet_by_index(0)
    eeg_file_name = worksheet.col_values(0, start_rowx=1, end_rowx=None)
    grade_label = np.asarray(worksheet.col_values(1, start_rowx=1, end_rowx=None))

    g1_limit_per_file = int(np.floor(numsel.total_lim / np.size(np.where(grade_label == 1),1)))
    g2_limit_per_file = int(np.floor(numsel.total_lim / np.size(np.where(grade_label == 2),1)))
    g3_limit_per_file = int(np.floor(numsel.total_lim / np.size(np.where(grade_label == 3),1)))
    g4_limit_per_file = int(np.floor(numsel.total_lim / np.size(np.where(grade_label == 4),1)))
    td=train_data()

    for pat_lv in range(0,train_ind_files_names.__len__(),1):

        try:
            data_raw = scipy.io.loadmat(os.path.join(datadir, str(train_ind_files_names[pat_lv] + total_lim)))
        except:
            print('File error: not opening')

        feat_vec = data_raw['feat_vec']
        del data_raw

        if len(whichfeatures) == np.size(feat_vec,0):
            data1 = feat_vec
        else:
            data1 = feat_vec[whichfeatures,:,:]

        del feat_vec


        for channel in range(0,8,1):
            if channel == 0:
                data = data1[:,:,channel]
            else:
                data = np.concatenate((data,data1[:,:,channel]),axis=1)
        data = remove_NaNs(data)[0]

        for lv in range(0,eeg_file_name.__len__(),1):
            if (train_ind_files_names[pat_lv] == eeg_file_name[lv]):
                grade_file=grade_label[lv]
                break


        if (grade_file==1):
            if np.size(td.data_g1,1) > numsel.total_lim:
                continue
            else:
                if (np.size(data,1) > g1_limit_per_file):
                    #td.data_g1 = np.concatenate([td.data_g1, data[:,range(0,g1_limit_per_file,1)]], axis=1)
                    td.data_g1 = np.append(td.data_g1, data[:,range(0,g1_limit_per_file,1)], axis=1)
                else:
                    td.data_g1 = np.append(td.data_g1, data, axis=1)
                    # td.data_g1 = np.concatenate([td.data_g1, data], axis=1)

        elif (grade_file ==2):
            if np.size(td.data_g2,1) > numsel.total_lim:
                continue
            else:
                if (np.size(data,1) > g2_limit_per_file):
                    td.data_g2 = np.concatenate([td.data_g2, data[:,range(0,g2_limit_per_file,1)]], axis=1)
                else:
                    td.data_g2 = np.concatenate([td.data_g2, data], axis=1)



        elif (grade_file==3):
            if np.size(td.data_g3,1) > numsel.total_lim:
                continue
            else:
                if (np.size(data,1) > g3_limit_per_file):
                    td.data_g3 = np.concatenate([td.data_g3, data[:,(range(0,g3_limit_per_file,1))]], axis=1)
                else:
                    td.data_g3 = np.concatenate([td.data_g3, data], axis=1)


        elif (grade_file==4):
            if np.size(td.data_g4,1) > numsel.total_lim:
                continue
            else:
                if (np.size(data,1) > g4_limit_per_file):
                    td.data_g4 = np.concatenate([td.data_g4, data[:,(range(0,g4_limit_per_file,1))]], axis=1)
                else:
                    td.data_g4 = np.concatenate([td.data_g4, data], axis=1)
    td.data_g1 = np.delete(td.data_g1,0 ,axis=1)
    td.data_g2 = np.delete(td.data_g2,0 ,axis=1)
    td.data_g3 = np.delete(td.data_g3,0 ,axis=1)
    td.data_g4 = np.delete(td.data_g4,0 ,axis=1)
    return td