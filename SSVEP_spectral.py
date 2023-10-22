import numpy as np

def psd(eeg_ex,eeg_tr,eeg_danet,sfreq,class_i,i_sub):
    # class array (table)
    if class_i == 11:
        np.save('./PSD/testing_eeg_ex', eeg_ex)
        np.save('./PSD/testing_eeg_tr', eeg_tr)
        np.save('./PSD/testing_eeg_danet', eeg_danet)