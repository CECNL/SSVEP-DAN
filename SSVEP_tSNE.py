import numpy as np
from sklearn.manifold import TSNE

def band_tSNE_performance(eeg_ex, eeg_tr, eeg_danet, i_sub, method, device_ex, device_tr):
    # example:(trials,channel,smaple)
    ex_trials, channel, sample = eeg_ex.shape
    print('ex_trials:',ex_trials)
    tr_trials = eeg_tr.shape[0]
    print('tr_trials:',tr_trials)
    danet_trials = eeg_danet.shape[0]
    print('danet_trials:',danet_trials)
    eeg_band = np.concatenate((eeg_ex,eeg_tr),0)
    eeg_band = np.concatenate((eeg_band,eeg_danet),0)
    print(eeg_band.shape)
    
    # data process
    if method == 'cov':
        cov = np.random.rand(eeg_band.shape[0],channel*channel)
        for i in range(eeg_band.shape[0]):
            cov[i] = np.cov(eeg_band[i]).flatten()
        eeg_band = cov # (trials,channel*channel)
    elif method == 'multi_channel':
        multi_ssvep = eeg_band#np.random.rand(eeg_band.shape[0],channel, sample)
        eeg_band = multi_ssvep.reshape(eeg_band.shape[0],-1) # (trials,channel,smaple)-->(trials,channel*sample)
    elif method == 'avg_channel':
        avg_ssvep = np.random.rand(eeg_band.shape[0],1, sample)
        for i in range(eeg_band.shape[0]):
            avg_ssvep[i] = np.mean(eeg_band[i],0)
        eeg_band = avg_ssvep.reshape(eeg_band.shape[0],-1) # (trials,1,smaple)-->(trials,1*sample)
    else:
        raise ValueError('Invalid `method` option.')

    if device_ex == 100 and device_tr == 100:
        file_path = 'Testing/TSNE/DANet_benchmarksub'+str(i_sub)+'_'+method+'.npy'
    else:
        file_path = 'Testing/TSNE/DANet_wetTOdrysub'+str(i_sub)+'_'+method+'.npy'
    X_embedded = TSNE(perplexity=30, n_components=3,init='pca', n_iter=3000).fit_transform(eeg_band) #24000
    np.save(file_path, np.array(X_embedded))
    print('shape of embedded:',X_embedded.shape) # (total_trail,3)