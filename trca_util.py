# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.linalg as linalg
from data_preprocessing import bandpass, theshapeof

def filterdata_tv(existing_train_old, existing_valid_old, target_old, classes, ex_train_y, ex_valid_y, tr_y, fb_i, sfreq, filterbank):
    cat_flag = True
    for class_i in classes:
        # print(f"class_number:{class_i}, subject_number:{i_sub+1}")
        # Select data with a specific label
        eeg_ex_train = existing_train_old[..., ex_train_y == class_i]
        eeg_ex_valid = existing_valid_old[..., ex_valid_y == class_i]
        eeg_tr = target_old[...,tr_y == class_i]
        # Filter the signal with fb_i
        eeg_ex_train = bandpass(eeg_ex_train, sfreq,
                            Wp=filterbank[fb_i][0],
                            Ws=filterbank[fb_i][1])

        eeg_ex_valid = bandpass(eeg_ex_valid, sfreq,
                            Wp=filterbank[fb_i][0],
                            Ws=filterbank[fb_i][1])

        eeg_tr = bandpass(eeg_tr,sfreq,
                            Wp=filterbank[fb_i][0],
                            Ws=filterbank[fb_i][1])
        # "mean" target value
        eeg_tr_mean = np.mean(eeg_tr, -1)

        # reshape
        sample,channel,train_trail = eeg_ex_train.shape
        valid_trail = eeg_ex_valid.shape[2]
        eeg_ex_train = torch.Tensor(eeg_ex_train.transpose(2,1,0).copy()).unsqueeze(1)
        eeg_ex_valid = torch.Tensor(eeg_ex_valid.transpose(2,1,0).copy()).unsqueeze(1)
        eeg_tr = torch.Tensor(eeg_tr.transpose(2,1,0).copy()).unsqueeze(1)
        eeg_tr_temp = torch.Tensor(eeg_tr_mean.transpose(1,0).copy()).unsqueeze(0).unsqueeze(1)  
        # average eeg_tr to  map eeg_ex
        eeg_tr_mean_for_train = torch.cat((eeg_tr_temp,eeg_tr_temp))
        if train_trail > 2:
            for i in range(train_trail-2):
                eeg_tr_mean_for_train =  torch.cat((eeg_tr_mean_for_train,eeg_tr_temp))
        
        eeg_tr_mean_for_valid = torch.cat((eeg_tr_temp,eeg_tr_temp))
        if valid_trail > 2:
            for i in range(valid_trail-2):
                eeg_tr_mean_for_valid =  torch.cat((eeg_tr_mean_for_valid,eeg_tr_temp))
        
        if cat_flag == True:
            train_x = eeg_ex_train
            train_y = eeg_tr_mean_for_train
            valid_x = eeg_ex_valid
            valid_y = eeg_tr_mean_for_valid
            cat_flag = False
        else:
            train_x = torch.cat((train_x,eeg_ex_train))
            train_y = torch.cat((train_y,eeg_tr_mean_for_train))
            valid_x = torch.cat((valid_x,eeg_ex_valid))
            valid_y = torch.cat((valid_y,eeg_tr_mean_for_valid))
    return train_x, train_y, valid_x, valid_y

def filterdata_t(existing,target,classes,class_i,ex_y,tr_y,fb_i,sfreq,filterbank):
    # Select data with a specific label
    eeg_ex = existing[..., ex_y == class_i]
    eeg_tr = target[...,tr_y == class_i]
    # Filter the signal with fb_i
    eeg_ex = bandpass(eeg_ex, sfreq,
                        Wp=filterbank[fb_i][0],
                        Ws=filterbank[fb_i][1])
    eeg_tr = bandpass(eeg_tr,sfreq,
                        Wp=filterbank[fb_i][0],
                        Ws=filterbank[fb_i][1])
    
    eeg_tr_mean = np.mean(eeg_tr, -1)

    # reshape
    sample,channel,trail = eeg_ex.shape
    eeg_ex = torch.Tensor(eeg_ex.transpose(2,1,0).copy()).unsqueeze(1)
    eeg_tr = torch.Tensor(eeg_tr.transpose(2,1,0).copy()).unsqueeze(1)
    eeg_tr_temp = torch.Tensor(eeg_tr_mean.transpose(1,0).copy()).unsqueeze(0).unsqueeze(1)
                
    # average eeg_tr to  map eeg_ex
    eeg_tr_mean = torch.cat((eeg_tr_temp,eeg_tr_temp))

    if trail > 2:
        for i in range(trail-2):
            eeg_tr_mean =  torch.cat((eeg_tr_mean,eeg_tr_temp))
    return eeg_ex, eeg_tr_mean

def trca(X):
    n_samples, n_chans, n_trials = theshapeof(X)

    # 1. Compute empirical covariance of all data (to be bounded)
    # -------------------------------------------------------------------------
    # Concatenate all the trials to have all the data as a sequence
    
    UX = np.zeros((n_chans, n_samples * n_trials))
    for trial in range(n_trials):
        UX[:, trial * n_samples:(trial + 1) * n_samples] = X[..., trial].T

    # Mean centering
    UX -= np.mean(UX, 1)[:, None]
    #start = time.time()
    # Covariance
    Q = UX @ UX.T
    #end = time.time()
    #print('UX shape:', UX.shape)
    #print('Covariance:',end - start)
    # 2. Compute average empirical covariance between all pairs of trials
    # -------------------------------------------------------------------------
    S = np.zeros((n_chans, n_chans))
    #start = time.time()
    for trial_i in range(n_trials - 1):
        x1 = np.squeeze(X[..., trial_i])

        # Mean centering for the selected trial
        x1 -= np.mean(x1, 0)

        # Select a second trial that is different
        for trial_j in range(trial_i + 1, n_trials):
            x2 = np.squeeze(X[..., trial_j])

            # Mean centering for the selected trial
            x2 -= np.mean(x2, 0)

            # Compute empirical covariance between the two selected trials and
            # sum it
            S = S + x1.T @ x2 + x2.T @ x1
    #end = time.time()
    #print('Compute empirical covariance:',end - start)
    # 3. Compute eigenvalues and vectors
    # -------------------------------------------------------------------------
    #start = time.time()
    lambdas, W = linalg.eig(S, Q, left=True, right=False)
    #end = time.time()
    #print('linalg:',end - start)
    # Select the eigenvector corresponding to the biggest eigenvalue
    W_best = W[:, np.argmax(lambdas)]

    return W_best

def fast_trca(X):
    n_samples, n_chans, n_trials = theshapeof(X)

    # 1. Compute empirical covariance of all data (to be bounded)
    # -------------------------------------------------------------------------
    # Concatenate all the trials to have all the data as a sequence
    UX = np.zeros((n_chans, n_samples * n_trials))
    for trial in range(n_trials):
        UX[:, trial * n_samples:(trial + 1) * n_samples] = X[..., trial].T
    # Mean centering
    #UX -= np.mean(UX, 1)[:, None]
    # Covariance
    Q = UX @ UX.T
    #print('Q shape:', Q.shape)
    # 2. Compute average empirical covariance between all pairs of trials
    # -------------------------------------------------------------------------
    SX = np.sum(X, 2)
    S = SX.T @ SX
    #print('S shape:', S.shape)
    #end = time.time()
    #print('Compute empirical covariance:',end - start)
    # 3. Compute eigenvalues and vectors
    # -------------------------------------------------------------------------
    lambdas, W = linalg.eig(S-Q, Q, left=True, right=False)
    # Select the eigenvector corresponding to the biggest eigenvalue
    W_best = W[:, np.argmax(lambdas)]

    return W_best

def get_sptialANDtemplate(target, eeg_trans, tr_y, filterbank, fb_i, class_i, sfreq, method):
    # For w/ adapation method
    # Select data with a specific label
    eeg_tr = target[...,tr_y == class_i]
    # Filter the signal with fb_i
    eeg_tr = bandpass(eeg_tr,sfreq,
                        Wp=filterbank[fb_i][0],
                        Ws=filterbank[fb_i][1])
    eeg_tr = torch.Tensor(eeg_tr.transpose(2,1,0).copy()).unsqueeze(1)
    eeg_tr = torch.cat((eeg_tr,eeg_trans[class_i]))
    # reshape
    eeg_tr = torch.permute(eeg_tr.squeeze(),(2,1,0)).numpy()
    # Compute mean of the signal across trials
    if (eeg_tr.ndim == 3):  
        eeg_template = np.mean(eeg_tr, -1)
    else:
        eeg_template = eeg_tr

    # Find the spatial filter for the corresponding filtered signal and label
    if method == 'original':
        w_best = trca(eeg_tr)
    elif method == 'fast':
        w_best = fast_trca(eeg_tr)
    else:
        raise ValueError('Invalid `method` option.')
    return eeg_template, w_best