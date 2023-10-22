import os
import time
import numpy as np
import random
from data_preprocessing import round_half_up, train_valid_for_wearable
from trca_main import TRCA_main
from trca_ablation import TRCA_ab
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--gpu', type=str, default='0', help='number of supplementary_sub')
    parser.add_argument('--repeat', type=int, default=10, help='repeat times')
    parser.add_argument('--supp', type=int, default=101, help='number of supplementary_sub')
    parser.add_argument('--tps', type=int, default=2, help='number of calibration trials per stimulus')
    parser.add_argument('--device', type=str, default='dryTOdry', choices=['dryTOdry','dryTOwet','wetTOdry',
                                                                           'wetTOwet'])
    parser.add_argument('--method', type=str, default='yyy', choices=['trca','Concat','LST',
                                                                           'DANet'])
    parser.add_argument('--ablation', type=str, default='xxx', choices=['origin','woStimulusInd','wo1','wo2',
                                                                        'woACT','wTemporal'])
    parser.add_argument('--file_path', type=str, default='Wearable_diff_ntps/dryTOdry/2tps/DANet', help='model save path')
    parser.add_argument('--model_path', type=str, default='Wearable_diff_ntps/dryTOdry2tps', help='model save path')
    opt = parser.parse_args()
    return opt

opt = parse_option()
print(opt.model_path)
print(opt.supp)
os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu

t = time.time()
n_sub = 102 # number of subject.  # 測試建議用8，實際要用35
n_tps = opt.tps # number of calibration trails per stimulus
supplementary_sub = opt.supp # 正常為 101 (n_sub - 1)
path = 'Wearable/'

dur_gaze = 2  # data length for target identification [s]
delay = 0.64 # viual latency being considered in the analysis [s]
n_bands = 3  # number of sub-bands in filter bank analysis
is_ensemble = True  # True = ensemble TRCA method; False = TRCA method
sfreq = 250  # sampling rate [Hz]
dur_shift = 0.5  # duration for gaze shifting [s]
notch_freq = 50 #removal line noise
quality_factor = 35 # for iirnotch function
list_freqs = np.array(
                    [[x + 9.25 for x in range(6)],
                      [x + 9.75 for x in range(6)]]).T
n_targets = list_freqs.size  # The number of stimuli
# Useful variables (no need to modify)
dur_gaze_s = round_half_up(dur_gaze * sfreq)  # data length [samples]
delay_s = round_half_up(delay * sfreq)  # visual latency [samples]
dur_sel_s = dur_gaze + dur_shift  # selection time [s]
# filterbank與boosting那篇的數量一致
'''
filterbank = [[(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
              [(14, 90), (10, 100)],
              [(22, 90), (16, 100)],
              [(30, 90), (24, 100)],
              [(38, 90), (32, 100)]]
'''

filterbank = [[(7.25, 90), (5.25, 100)],  # passband, stopband freqs [(Wp), (Ws)]
              [(13.25, 90), (9.25, 100)],
              [(19.25, 90), (13.25, 100)]]
#filterbank = [[(7.25, 90), (5.25, 100)]]

Performance_record = np.zeros((int(n_sub), opt.repeat))

if opt.device == 'dryTOdry':
    dev_ex = 0
    dev_tr = 0
    print("Cross subject: From dry to dry")
elif opt.device == 'dryTOwet':
    dev_ex = 0
    dev_tr = 1
    print("Cross device: From dry to wet")
elif opt.device == 'wetTOdry':
    dev_ex = 1
    dev_tr = 0
    print("Cross device: From wet to dry")
else:
    dev_ex = 1
    dev_tr = 1
    print("Cross subject: From wet to wet")

for r in range(opt.repeat):#opt.repeat
    # existing data and target data
    for i in range(0,n_sub):  # n_sub
        existing_train, existing_valid, target = train_valid_for_wearable(i ,n_sub, supplementary_sub,0.8, path, dev_ex,dev_tr, 10)
    
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
        ''' for visulization
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
        x_testing = target[...,6*n_trials:]
        y_testing = tr_labels[6* n_trials:]

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
        train_sub = int(supplementary_sub * 0.8)
        valid_sub = supplementary_sub - train_sub
        # Construction of the spatial filter and the reference signals
        print('start to fit...')

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
            _ = trca.fit_WDANet(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
            #trca.TSNE_visulaize(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path, dev_ex, dev_tr)
        
        trca_ab = TRCA_ab(sfreq, filterbank, is_ensemble)
        if opt.ablation == 'origin':
            _ = trca.fit_WDANet(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'woStimulusInd':
            trca_ab.fit_WDANet_wo_cross(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'wo1':
            eeg_x = np.concatenate((existing_train,existing_valid),2)
            eeg_y = np.concatenate((ex_training_labels, ex_valid_label))
            trca_ab.fit_WDANet_wo_phase1(eeg_x, target_tmp, eeg_y, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'wo2':
            trca_ab.fit_WDANet_wo_phase2(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'woACT':
            trca_ab.fit_WDANet_linear(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
        elif opt.ablation == 'wTemporal':
            trca_ab.fit_WDANet_temporal(existing_train, existing_valid, target_tmp, ex_training_labels, ex_valid_label, tr_labels_train, 10, supplementary_sub, i, opt.model_path)
        # Test stage
        print('start predicting...')
        if opt.ablation == 'xxx':
            estimated = trca.predict(x_testing,dev_tr)
        else:
            estimated = trca_ab.predict(x_testing,dev_tr)
        # Evaluation of the performance for this fold (accuracy and ITR)
        is_correct = estimated == y_testing
        print('predict:',estimated)
        acc = np.mean(is_correct) * 100
        print('acc:',acc)
        Performance_record[i,r] = acc
        print('acc_clg:',Performance_record[:,r])
        if opt.ablation == 'xxx':
            np.save(os.path.join(opt.file_path, opt.method), Performance_record)
        else:
            np.save(os.path.join(opt.file_path, opt.ablation), Performance_record)
        print(f"\nElapsed time: {time.time()-t:.1f} seconds")
# average accuracy
print(f"\nElapsed time: {time.time()-t:.1f} seconds")
if opt.ablation == 'xxx':
    np.save(os.path.join(opt.file_path, opt.method), Performance_record)
else:
    np.save(os.path.join(opt.file_path, opt.ablation), Performance_record)
print('wearable')