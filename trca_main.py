# -*- coding: utf-8 -*-
import numpy as np
import torch

from DANet_model import DANet
from trainANDdataloader import DANet_main
from trca_util import filterdata_tv, filterdata_t, trca, fast_trca, get_sptialANDtemplate
from data_preprocessing import bandpass, schaefer_strimmer_cov, theshapeof
from SSVEP_tSNE import band_tSNE_performance
from SSVEP_spectral import psd

class TRCA_main:
    def __init__(self, sfreq, filterbank, ensemble=True, method='original',
                 estimator='scm'):
        self.sfreq = sfreq
        self.ensemble = ensemble
        self.filterbank = filterbank
        self.n_bands = len(self.filterbank)
        self.coef_ = None
        self.method = method
        if estimator == 'schaefer':
            self.estimator = schaefer_strimmer_cov
        else:
            self.estimator = estimator
            
    def fit_baseline(self,X,y):
        n_samples, n_chans, _ = theshapeof(X)
        classes = np.unique(y)

        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))

        W = np.zeros((self.n_bands, len(classes), n_chans))

        for class_i in classes:
            # Select data with a specific label
            eeg_tmp = X[..., y == class_i]
            for fb_i in range(self.n_bands):
                # Filter the signal with fb_i
                eeg_tmp = bandpass(eeg_tmp, self.sfreq,
                                   Wp=self.filterbank[fb_i][0],
                                   Ws=self.filterbank[fb_i][1])
                if (eeg_tmp.ndim == 3):
                    # Compute mean of the signal across trials
                    trains[class_i, fb_i] = np.mean(eeg_tmp, -1)
                else:
                    trains[class_i, fb_i] = eeg_tmp
                # Find the spatial filter for the corresponding filtered signal
                # and label
                if self.method == 'original':
                    w_best = trca(eeg_tmp)
                elif self.method == 'fast':
                    w_best = fast_trca(eeg_tmp)
                else:
                    raise ValueError('Invalid `method` option.')

                W[fb_i, class_i, :] = w_best  # Store the spatial filter

        self.trains = trains
        self.coef_ = W
        self.classes = classes
        return self

    def fit_without_transfer(self,existing,target,ex_y,tr_y):
        n_samples, n_chans, _ = theshapeof(existing)
        classes = np.unique(ex_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        W = np.zeros((self.n_bands, len(classes), n_chans))

        for fb_i in range(self.n_bands):
            # Filter the signal with fb_i
            eeg_ex = bandpass(existing, self.sfreq,
                                Wp=self.filterbank[fb_i][0],
                                Ws=self.filterbank[fb_i][1])
            
            eeg_tr = bandpass(target, self.sfreq,
                                Wp=self.filterbank[fb_i][0],
                                Ws=self.filterbank[fb_i][1])
            for class_i in classes:
                # Select data with a specific label
                eeg_ex_class = eeg_ex[..., ex_y == class_i]
                eeg_tr = target[...,tr_y == class_i]
                eeg_trans = np.concatenate((eeg_ex_class,eeg_tr),2)
                eeg_template, w_best = get_sptialANDtemplate(target, eeg_trans, tr_y, self.filterbank, fb_i, class_i, self.sfreq, self.method)
                trains[class_i, fb_i] = eeg_template # Store the template
                W[fb_i, class_i, :] = w_best  # Store the spatial filter
        self.trains = trains
        self.coef_ = W
        self.classes = classes
        return

    def fit_Wlst_mean(self,existing,target,ex_y,tr_y):
        n_samples, n_chans, _ = theshapeof(existing)
        classes = np.unique(ex_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        W = np.zeros((self.n_bands, len(classes), n_chans))

        for fb_i in range(self.n_bands):
            # Filter the signal with fb_i
            eeg_ex_tmp = bandpass(existing, self.sfreq,
                                Wp=self.filterbank[fb_i][0],
                                Ws=self.filterbank[fb_i][1])
            eeg_tr_tmp = bandpass(target,self.sfreq,
                                Wp=self.filterbank[fb_i][0],
                                Ws=self.filterbank[fb_i][1])
            for class_i in classes:
                # Select data with a specific label
                eeg_ex = eeg_ex_tmp[..., ex_y == class_i]
                eeg_tr = eeg_tr_tmp[...,tr_y == class_i]
                eeg_tr_mean = np.mean(eeg_tr, -1)
                # reshape (375,64,2) -->  (2,64,375)               
                sample,channel,trail = eeg_ex.shape
                eeg_ex = eeg_ex.transpose(2,1,0)
                eeg_tr_mean = eeg_tr_mean.transpose(1,0) # (64,375)
                # multivariate least-squares regression for each single-trail
                for i in range(trail):
                    # (2,8,375) --> single-trial (8,375)
                    x_bar = eeg_ex[i].squeeze() # x'   (x = Px')
                    x_bar = np.concatenate((np.ones((1,x_bar.shape[1])),x_bar),0) #(9,375)
                    x = eeg_tr_mean # x
                    P = np.matmul(np.matmul(x,x_bar.T),np.linalg.inv(np.matmul(x_bar,x_bar.T))) # (8,9)*(9,9)=(8,9)
                    new_target = np.matmul(P,x_bar) # (8,375)
                    # reconstruct new_target domain
                    new_target = new_target[:,:,np.newaxis] # new_target.reshape(channel,sample,1)
                    new_target = new_target.transpose(1,0,2)
                    if i == 0:
                        eeg_tmp = new_target
                    else:
                        eeg_tmp = np.concatenate((eeg_tmp,new_target),2)

                eeg_trans = np.concatenate((eeg_tmp,eeg_tr),2)
                eeg_template, w_best = get_sptialANDtemplate(target, eeg_trans, tr_y, self.filterbank, fb_i, class_i, self.sfreq, self.method)
                trains[class_i, fb_i] = eeg_template # Store the template
                W[fb_i, class_i, :] = w_best  # Store the spatial filter

        self.trains = trains
        self.coef_ = W
        self.classes = classes
        return self
    
    def fit_WDANet(self, existing_train, existing_valid, target, ex_train_y, ex_valid_y, tr_y, n_tps, train_sub, i_sub, path):
        n_samples, n_chans, _ = theshapeof(existing_train)
        classes = np.unique(ex_train_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        W = np.zeros((self.n_bands, len(classes), n_chans))
        model_path = path + '.pt' # './CD_all_4trials_5.pt'
        transfer_fine = torch.zeros(len(classes),train_sub*n_tps ,1,n_chans,n_samples)
        for fb_i in range(self.n_bands):
            ########## cat all class existing data ######### 
            train_x, train_y, valid_x, valid_y = filterdata_tv(existing_train, existing_valid, target, classes, ex_train_y, ex_valid_y, tr_y, fb_i, self.sfreq, self.filterbank)
            ########## train my network #########
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            danet = DANet_main()
            dan = DANet().to(dev)
            training_dataloader = danet.train_dataloader(train_x,train_y,256) # n_tps*train_sub // bench: 64, wearable: 256
            valid_dataloader = danet.train_dataloader(valid_x,valid_y,256) # n_tps*valid_sub // bench: 64, wearable: 256
            train_loss_arr, valid_loss_arr = danet.train_network(dan,training_dataloader,valid_dataloader,model_path,500)
            # plot_learning_curves(train_loss_arr, valid_loss_arr, 1000, i_sub, fb_i)
            eeg_x = np.concatenate((existing_train,existing_valid),2)
            eeg_y = np.concatenate((ex_train_y, ex_valid_y))
            for i in range(train_sub):
                left = i * n_tps
                right = left + n_tps
                for class_i in classes:
                    eeg_ex,eeg_tr_mean = filterdata_t(eeg_x[:,:,i*len(classes)*n_tps:(i+1)*len(classes)*n_tps],target,classes,class_i,eeg_y[i*len(classes)*n_tps:(i+1)*len(classes)*n_tps],tr_y,fb_i,self.sfreq,self.filterbank)
                    if class_i == 0:
                        eeg_ex_tmp = eeg_ex
                        eeg_tr_mean_tmp = eeg_tr_mean
                    else:
                        eeg_ex_tmp = np.concatenate((eeg_ex_tmp,eeg_ex))
                        eeg_tr_mean_tmp = np.concatenate((eeg_tr_mean_tmp,eeg_tr_mean))
                eeg_ex_tmp = torch.Tensor(eeg_ex_tmp.copy())
                eeg_tr_mean_tmp = torch.Tensor(eeg_tr_mean_tmp.copy())
                fine_dataloader = danet.train_dataloader(eeg_ex_tmp,eeg_tr_mean_tmp,64)
                transfer_dataloader = danet.trans_dataloader(eeg_ex_tmp,eeg_tr_mean_tmp,64)
                dan.load_state_dict(torch.load(model_path))
                danet.finetune_network(dan,fine_dataloader,150)
                transfer_data = danet.transfer_domain(dan,transfer_dataloader,torch.zeros((len(classes)*n_tps, 1, n_chans,n_samples))) # transfer data is similar to target domain
                transfer_data = transfer_data.cpu()
                for class_i in classes:
                    transfer_fine[class_i,left:right] = transfer_data[class_i*n_tps : class_i*n_tps + n_tps]

            for class_i in classes:
                eeg_template, w_best = get_sptialANDtemplate(target, transfer_fine, tr_y, self.filterbank, fb_i, class_i, self.sfreq, self.method)
                trains[class_i, fb_i] = eeg_template # Store the template
                W[fb_i, class_i, :] = w_best  # Store the spatial filter
                
                
                eeg_tr = target[...,tr_y == class_i]
                eeg_ex = existing_train[...,ex_train_y == class_i]
                # Filter the signal with fb_i
                eeg_tr = bandpass(eeg_tr,self.sfreq,
                                   Wp=self.filterbank[fb_i][0],
                                   Ws=self.filterbank[fb_i][1])
                eeg_ex = bandpass(eeg_ex, self.sfreq,
                                   Wp=self.filterbank[fb_i][0],
                                   Ws=self.filterbank[fb_i][1])
                eeg_tr_for_psd = eeg_tr
                eeg_ex_for_psd = eeg_ex
                eeg_tr = torch.Tensor(eeg_tr.transpose(2,1,0).copy()).unsqueeze(1)
                eeg_tmp = torch.cat((eeg_tr,transfer_fine[class_i]))
                # reshape
                eeg_tmp = torch.permute(eeg_tmp.squeeze(),(2,1,0)).numpy()
                eeg_tmp_for_psd = eeg_tmp
                # plot_power_spectra
                #psd(eeg_ex_for_psd,eeg_tr_for_psd,eeg_tmp_for_psd,self.sfreq,class_i,i_sub)
                if fb_i == 0 and class_i == 11:
                    np.save('./Testing/PSD/testing_eeg_ex_11', eeg_ex_for_psd)
                    np.save('./Testing/PSD/testing_eeg_tr_11', eeg_tr_for_psd)
                    np.save('./Testing/PSD/testing_eeg_danet_11', eeg_tmp_for_psd)
                
        self.trains = trains
        self.coef_ = W
        self.classes = classes
        return self
    
    def TSNE_visulaize(self, existing_train, existing_valid, target, ex_train_y, ex_valid_y, tr_y, n_tps, train_sub, i_sub, path, dev_ex, dev_tr):
        n_samples, n_chans, _ = theshapeof(existing_train)
        classes = np.unique(ex_train_y)
        model_path = path + '.pt' # './CD_all_4trials_5.pt'
        # train_sub = train_sub - 1 # existing domain for training
        transfer_fine = torch.zeros(len(classes),20*n_tps ,1,n_chans,n_samples)
        for fb_i in range(1): # self.n_bands
            ########## cat all class existing data ######### 
            train_x, train_y, valid_x, valid_y = filterdata_tv(existing_train, existing_valid, target, classes, ex_train_y, ex_valid_y, tr_y, fb_i, self.sfreq, self.filterbank)
            ########## train my network #########
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            danet = DANet_main()
            dan = DANet().to(dev)
            training_dataloader = danet.train_dataloader(train_x,train_y,256) # n_tps*train_sub // bench: 64, wearable: 256
            valid_dataloader = danet.train_dataloader(valid_x,valid_y,256) # n_tps*valid_sub // bench: 64, wearable: 256
            train_loss_arr, valid_loss_arr = danet.train_network(dan,training_dataloader,valid_dataloader,model_path,500)
            # plot_learning_curves(train_loss_arr, valid_loss_arr, 1000, i_sub, fb_i)
            random_sub_select = np.random.choice(train_sub,20, replace=False)
            eeg_x = np.concatenate((existing_train,existing_valid),2)
            eeg_y = np.concatenate((ex_train_y, ex_valid_y))
            j = 0
            for i in random_sub_select:
                left = i * n_tps
                right = left + n_tps
                for class_i in classes:
                    eeg_ex,eeg_tr_mean = filterdata_t(eeg_x[:,:,i*len(classes)*n_tps:(i+1)*len(classes)*n_tps],target,classes,class_i,eeg_y[i*len(classes)*n_tps:(i+1)*len(classes)*n_tps],tr_y,fb_i,self.sfreq,self.filterbank)
                    if class_i == 0:
                        eeg_ex_tmp = eeg_ex
                        eeg_tr_mean_tmp = eeg_tr_mean
                        eeg_ex_class1 = eeg_ex
                    else:
                        eeg_ex_tmp = torch.cat((eeg_ex_tmp,eeg_ex))
                        eeg_tr_mean_tmp = torch.cat((eeg_tr_mean_tmp,eeg_tr_mean))
                    if class_i == 1:
                        eeg_ex_class2 = eeg_ex
                fine_dataloader = danet.train_dataloader(eeg_ex_tmp,eeg_tr_mean_tmp,64)
                transfer_dataloader = danet.trans_dataloader(eeg_ex_tmp,eeg_tr_mean_tmp,64)
                dan.load_state_dict(torch.load(model_path))
                danet.finetune_network(dan,fine_dataloader,150)
                transfer_data = danet.transfer_domain(dan,transfer_dataloader,torch.zeros((len(classes)*n_tps, 1, n_chans,n_samples))) # transfer data is similar to target domain
                transfer_data = transfer_data.cpu()
                if j == 0:
                    eeg_ex_class1_tmp = eeg_ex_class1
                    eeg_ex_class2_tmp = eeg_ex_class2
                    eeg_danet_class1_tmp = transfer_data[0*n_tps : 0*n_tps + n_tps]
                    eeg_danet_class2_tmp = transfer_data[1*n_tps : 1*n_tps + n_tps]
                else:
                    eeg_ex_class1_tmp = torch.cat((eeg_ex_class1_tmp,eeg_ex_class1))
                    eeg_ex_class2_tmp = torch.cat((eeg_ex_class2_tmp,eeg_ex_class2))
                    eeg_danet_class1_tmp = torch.cat((eeg_danet_class1_tmp, transfer_data[0*n_tps : 0*n_tps + n_tps]))
                    eeg_danet_class2_tmp = torch.cat((eeg_danet_class2_tmp, transfer_data[1*n_tps : 1*n_tps + n_tps]))
                j = j + 1
            eeg_ex_domain = torch.cat((eeg_ex_class1_tmp,eeg_ex_class2_tmp))
            eeg_ex_domain = eeg_ex_domain.squeeze().numpy()
            transfer_fine_tmp = torch.cat((eeg_danet_class1_tmp,eeg_danet_class2_tmp)).squeeze().numpy()
            for class_i in classes:
                # print(f"fb_i {fb_i} class_i:{class_i}")
                # Select data with a specific label
                eeg_tr = target[...,tr_y == class_i]
                
                # Filter the signal with fb_i
                eeg_tr = bandpass(eeg_tr,self.sfreq,
                                   Wp=self.filterbank[fb_i][0],
                                   Ws=self.filterbank[fb_i][1])
                eeg_tr = np.transpose(eeg_tr,(2,1,0))
                if class_i == 0:
                    eeg_tr_tmp = eeg_tr
                elif class_i == 1:
                    eeg_tr_tmp = np.concatenate((eeg_tr_tmp,eeg_tr))    
            # eeg_ex, eeg_tr, eeg_danet,i_sub, dev_ex, dev_tr, n_bands, class1, class2
            band_tSNE_performance(eeg_ex_domain, eeg_tr_tmp, transfer_fine_tmp, i_sub, 'avg_channel', dev_ex, dev_tr)
            #existing_train, existing_valid, target = filterbank_data(existing_train, existing_valid, target, classes, ex_train_y, ex_valid_y, tr_y, fb_i, self.sfreq, self.filterbank)
        return self
    
    def predict(self, X, device):
        if self.coef_ is None:
            raise RuntimeError('TRCA is not fitted')

        # Alpha coefficients for the fusion of filterbank analysis
        if device == 0: # dry
            fb_coefs = [(x + 1)**(-1.25) + 0.25 for x in range(self.n_bands)]
        else: # wet
            fb_coefs = [(x + 1)**(-1.75) + 0.5 for x in range(self.n_bands)]
        _, _, n_trials = theshapeof(X)

        r = np.zeros((self.n_bands, len(self.classes)))
        pred = np.zeros((n_trials), 'int')  # To store predictions

        for trial in range(n_trials):
            test_tmp = X[..., trial]  # pick a trial to be analysed
            for fb_i in range(self.n_bands):
                
                # filterbank on testdata
                testdata = bandpass(test_tmp, self.sfreq,
                                    Wp=self.filterbank[fb_i][0],
                                    Ws=self.filterbank[fb_i][1])

                for class_i in self.classes:
                    # Retrieve reference signal for class i
                    # (shape: n_chans, n_samples)
                    traindata = np.squeeze(self.trains[int(class_i), fb_i])
                    if self.ensemble:
                        # shape = (n_chans, n_classes)
                        w = np.squeeze(self.coef_[fb_i]).T
                    else:
                        # shape = (n_chans)
                        w = np.squeeze(self.coef_[fb_i, int(class_i)])

                    # Compute 2D correlation of spatially filtered test data
                    # with ref
                    r_tmp = np.corrcoef((testdata @ w).flatten(),
                                        (traindata @ w).flatten())
                    r[fb_i, int(class_i)] = r_tmp[0, 1]
            rho = np.dot(fb_coefs, r)  # fusion for the filterbank analysis
            tau = np.argmax(rho)  # retrieving index of the max
            pred[trial] = int(tau)

        return pred