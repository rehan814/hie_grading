import numpy as np


def make_sequences(data, seq_len, overlap):


    seq_indx = []
    tot_num_seq = int(np.floor(np.size(data, 1) / np.floor(seq_len - overlap)))
    ind = 0
    t = seq_len - overlap
    newdata = np.empty((55, 1))

    # if data is less than the length of "seq_len" then just make one sequence
    # and get out.
    if tot_num_seq == 0:
        seq_indx = np.concatenate([seq_indx,  np.ones(np.size(data, 1))], axis=0)
        newdata = data
        print('Warning total number of sequences are less than 1')

    elif tot_num_seq == 1:
            seq_indx = np.concatenate([seq_indx, np.ones(seq_len)])
            newdata = np.concatenate([newdata, data[:, ind: ind + seq_len]])
    else:
        for lv in range(0, tot_num_seq-1, 1):
            seq_indx = np.concatenate([seq_indx, np.ones(seq_len)*lv])
            newdata = np.concatenate([newdata, data[:, ind: ind + seq_len]], axis=1)
            ind = ind + t
            if ind + (seq_len - 1) > np.size(data, 1):
                break        

        newdata = np.delete(newdata, 0, axis=1)  # Remove the first initial vector force-addded at start

    if np.size(newdata, 1) < np.size(data, 1):
        seq_indx = np.concatenate([seq_indx, np.ones((np.size(data, 1) - np.size(newdata, 1))) * (seq_indx[-1] + 1)])
        newdata = np.concatenate([newdata, data[:, np.size(newdata, 1):]], axis=1)
    
    return seq_indx, newdata
