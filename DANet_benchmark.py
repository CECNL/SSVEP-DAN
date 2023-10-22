#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
from data_preprocessing import round_half_up, train_valid_for_benchmark
from trca_main import TRCA_main
from trca_ablation import TRCA_ab
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--gpu', type=str, default='0', help='number of supplementary_sub')
    parser.add_argument('--repeat', type=int, default=10, help='repeat times')
    parser.add_argument('--supp', type=int, default=34, help='number of supplementary_sub')
    parser.add_argument('--tps', type=int, default=2, help='number of calibration trials per stimulus')
    parser.add_argument('--method', type=str, default='yyy', choices=['trca','Concat','LST',
                                                                           'DANet'])
    parser.add_argument('--ablation', type=str, default='xxx', choices=['origin','woStimulusInd','wo1','wo2',
                                                                        'woACT','wTemporal'])
    parser.add_argument('--file_path', type=str, default='Benchmark_diff_ntps/2tps', help='result save path')
    parser.add_argument('--model_path', type=str, default='Benchmark_diff_ntps/2tps', help='model save path')
    opt = parser.parse_args()
    return opt

opt = parse_option()
print(opt.model_path)
print(opt.supp)
t = time.time()
n_sub = 35 # number of subject. (total)
supplementary_sub = opt.supp # 正常為 34 (n_sub - 1)
os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
n_tps =  opt.tps # number of calibration trails per stimulus
path = 'Benchmark/S'

dur_gaze = 1.5  # data length for target identification [s]
delay = 0.15  # visual latency being considered in the analysis [s]
n_bands = 5  # number of sub-bands in filter bank analysis
is_ensemble = True  # True = ensemble TRCA method; False = TRCA method
sfreq = 250  # sampling rate [Hz]
dur_shift = 0.5  # duration for gaze shifting [s]
list_freqs = np.array(
    [[x + 8.0 for x in range(8)],
     [x + 8.2 for x in range(8)],
     [x + 8.4 for x in range(8)],
     [x + 8.6 for x in range(8)],
     [x + 8.8 for x in range(8)]]).T  # list of stimulus frequencies等同於 Freq_phase那個檔案
n_targets = list_freqs.size  # The number of stimuli

# Useful variables (no need to modify)
dur_gaze_s = round_half_up(dur_gaze * sfreq)  # data length [samples]
delay_s = round_half_up(delay * sfreq)  # visual latency [samples]
dur_sel_s = dur_gaze + dur_shift  # selection time [s]
# filterbank與boosting那篇的數量一致

filterbank = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)] [[fb_i][0],[fb_i][1]]
              [(14, 90), (10, 100)],
              [(22, 90), (16, 100)],
              [(30, 90), (24, 100)],
              [(38, 90), (32, 100)]]
# channel selection
## POz(56), PO3(55), PO4(57), PO7(53), PO8(59), Oz(62), O1(61), O2(63)，但要放在array裡面，所以-1
## POz(56), PO3(55), PO4(57), PO5(54), PO6(58), Oz(62), O1(61), O2(63)，但要放在array裡面，所以-1
channel_num = [53,54,55,56,57,60,61,62]

Performance_record = np.zeros((int(n_sub), opt.repeat))
#Time_record = np.zeros((int(n_sub), 2, opt.repeat))

# In[5]:
##### main ######
for r in range(opt.repeat):#opt.repeat
    for i in range(0,n_sub): # n_sub3
        start = time.time()
        existing_train, existing_valid, target = train_valid_for_benchmark(i ,n_sub, supplementary_sub, 0.8, path, channel_num, 6)
        n_chans, n_samples, n_trials, ex_blocks = existing_train.shape
        existing_valid_blocks = existing_valid.shape[3]
        tr_blocks = target.shape[3]

        ## Convert dummy Matlab format to (sample, channels, trials) and construct
        ## vector of labels
        existing_train = np.reshape(existing_train.transpose([1, 0, 3, 2]),
                            (n_samples, n_chans, n_trials * ex_blocks))
        
        existing_valid = np.reshape(existing_valid.transpose([1, 0, 3, 2]),
                            (n_samples, n_chans, n_trials * existing_valid_blocks))

        target = np.reshape(target.transpose([1, 0, 3, 2]),
                            (n_samples, n_chans, n_trials * tr_blocks))

        ex_training_labels = np.array([x for x in range(n_targets)] * ex_blocks)
        ex_valid_label = np.array([x for x in range(n_targets)] * existing_valid_blocks)
        tr_labels = np.array([x for x in range(n_targets)] * tr_blocks)
        
        crop_data = np.arange(delay_s, delay_s + dur_gaze_s)
        existing_train = existing_train[crop_data]
        existing_valid = existing_valid[crop_data]
        target = target[crop_data]
        train_sub = int(supplementary_sub * 0.8)
        valid_sub = supplementary_sub - train_sub
        '''for visulization
        ## zero mean
        for j in range(n_trials * ex_blocks):
            existing_train[:,:,j] -= existing_train[:,:,j].mean(0)[None,:]
        for j in range(n_trials * existing_valid_blocks):
            existing_valid[:,:,j] -= existing_valid[:,:,j].mean(0)[None,:]
        for j in range(n_trials * tr_blocks):
            target[:,:,j] -= target[:,:,j].mean(0)[None,:]
        '''
        ########## TRCA classification #############
        print(f"Experiment Imformation: existing domain: other subject,target domain:{i+1}, number of calibration trails per stimulus:{n_tps}\n")
        print('Results of the ensemble TRCA-based method:\n')

        # select the first n_tps blocks as training,the final block as testing
        target_tmp = target[...,:(n_tps)*n_trials]
        tr_labels_train = tr_labels[:(n_tps)*n_trials]
        x_testing = target[...,4*n_trials:]
        y_testing = tr_labels[4* n_trials:]

        # output experiment imformation
        print('existing_training domain:',existing_train.shape)
        print('existing_training labels:',ex_training_labels.shape)
        print('existing_valid domain:',existing_valid.shape)
        print('existing_valid labels:',ex_valid_label.shape)
        print('target domain:',target_tmp.shape)
        print('target labels:',tr_labels_train.shape)
        print('testing data(x_target domain):',x_testing.shape)
        print('testing data(y_target domain):',y_testing.shape)

        ######## fit_with_transfer ##########
        # trca
        trca = TRCA_main(sfreq, filterbank, is_ensemble)
        # Construction of the spatial filter and the reference signals
        print('start to fit..')
        if opt.method == 'trca':
            trca.fit_baseline(target_tmp, tr_labels_train)
        elif opt.method == 'Concat':
            eeg_x = np.concatenate((existing_train,existing_valid),2)
            eeg_y = np.concatenate((ex_training_labels, ex_valid_label))
            trca.fit_without_transfer(eeg_x,target_tmp,eeg_y,tr_labels_train)
        elif opt.method == 'LST':
            eeg_x = np.concatenate((existing_train,existing_valid),2)
            eeg_y = np.concatenate((ex_training_labels, ex_valid_label))
            trca.fit_Wlst_mean(eeg_x,target_tmp,eeg_y,tr_labels_train)
        elif opt.method == 'DANet':
            _ = trca.fit_WDANet(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
            #trca.TSNE_visulaize(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path, 100, 100)
        
        trca_ab = TRCA_ab(sfreq, filterbank, is_ensemble)
        if opt.ablation == 'origin':
            _ = trca.fit_WDANet(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'woStimulusInd':
            trca_ab.fit_WDANet_wo_cross(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'wo1':
            eeg_x = np.concatenate((existing_train,existing_valid),2)
            eeg_y = np.concatenate((ex_training_labels, ex_valid_label))
            trca_ab.fit_WDANet_wo_phase1(eeg_x, target_tmp, eeg_y, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'wo2':
            trca_ab.fit_WDANet_wo_phase2(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'woACT':
            trca_ab.fit_WDANet_linear(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'wTemporal':
            trca_ab.fit_WDANet_temporal(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 6, supplementary_sub, i, opt.model_path)
        # Test stage
        print('start predicting...')
        if opt.ablation == 'xxx':
            estimated = trca.predict(x_testing,0)
        else:
            estimated = trca_ab.predict(x_testing,0)
        # Evaluation of the performance for this fold (accuracy and ITR)
        is_correct = estimated == y_testing
        acc = np.mean(is_correct) * 100
        print('acc:',acc)
        Performance_record[i,r] = acc
        print('acc_clg:',Performance_record[:,r])
        if opt.ablation == 'xxx':
            np.save(os.path.join(opt.file_path, opt.method), Performance_record)
        else:
            np.save(os.path.join(opt.file_path, opt.ablation), Performance_record)
        print(f"\nElapsed time: {time.time()-t:.1f} seconds")

print(f"\nElapsed time: {time.time()-t:.1f} seconds")
if opt.ablation == 'xxx':
    np.save(os.path.join(opt.file_path, opt.method), Performance_record)
else:
    np.save(os.path.join(opt.file_path, opt.ablation), Performance_record)