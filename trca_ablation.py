import numpy as np
import torch

from DANet_model import DANet, DANet_linear, DANet_temporoal
from trainANDdataloader import DANet_main
from trca_util import filterdata_tv, filterdata_t, trca, fast_trca, get_sptialANDtemplate
from data_preprocessing import bandpass, schaefer_strimmer_cov, theshapeof

class TRCA_ab:
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

    def fit_WDANet_wo_phase1(self, existing_train, target, ex_train_y, tr_y, n_tps, train_sub, i_sub, path):
        n_samples, n_chans, _ = theshapeof(existing_train)
        classes = np.unique(ex_train_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        W = np.zeros((self.n_bands, len(classes), n_chans))
        model_path = path + '.pt' # './CD_all_4trials_5.pt'
        transfer_fine = torch.zeros(len(classes),train_sub*n_tps ,1,n_chans,n_samples)

        for fb_i in range(self.n_bands):
            ########## train my network #########
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            danet = DANet_main()
            dan = DANet().to(dev)

            for i in range(train_sub):
                left = i * n_tps
                right = left + n_tps
                for class_i in classes:
                    eeg_ex,eeg_tr_mean = filterdata_t(existing_train[:,:,i*len(classes)*n_tps:(i+1)*len(classes)*n_tps],target,classes,class_i,ex_train_y[i*len(classes)*n_tps:(i+1)*len(classes)*n_tps],tr_y,fb_i,self.sfreq,self.filterbank)
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
                danet.finetune_network(dan,fine_dataloader,150)
                transfer_data = danet.transfer_domain(dan,transfer_dataloader,torch.zeros((len(classes)*n_tps, 1, n_chans,n_samples))) # transfer data is similar to target domain
                transfer_data = transfer_data.cpu()
                for class_i in classes:
                    transfer_fine[class_i,left:right] = transfer_data[class_i*n_tps : class_i*n_tps + n_tps]

            for class_i in classes:
                eeg_template, w_best = get_sptialANDtemplate(target, transfer_fine, tr_y, self.filterbank, fb_i, class_i, self.sfreq, self.method)
                trains[class_i, fb_i] = eeg_template # Store the template
                W[fb_i, class_i, :] = w_best  # Store the spatial filter
        self.trains = trains
        self.coef_ = W
        self.classes = classes

        return self

    def fit_WDANet_wo_phase2(self, existing_train, existing_valid, target, ex_train_y, ex_valid_y, tr_y, n_tps, train_sub, i_sub, path):
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
            training_dataloader = danet.train_dataloader(train_x,train_y,256)
            valid_dataloader = danet.train_dataloader(valid_x,valid_y,256)
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
                        eeg_ex_tmp = torch.cat((eeg_ex_tmp,eeg_ex))
                        eeg_tr_mean_tmp = torch.cat((eeg_tr_mean_tmp,eeg_tr_mean))
                transfer_dataloader = danet.trans_dataloader(eeg_ex_tmp,eeg_tr_mean_tmp,n_tps*len(classes))
                dan.load_state_dict(torch.load(model_path))
                transfer_data = danet.transfer_domain(dan,transfer_dataloader,torch.zeros((len(classes)*n_tps, 1, n_chans,n_samples))) # transfer data is similar to target domain
                transfer_data = transfer_data.cpu()
                for class_i in classes:
                    transfer_fine[class_i,left:right] = transfer_data[class_i*n_tps : class_i*n_tps + n_tps]

            for class_i in classes:
                eeg_template, w_best = get_sptialANDtemplate(target, transfer_fine, tr_y, self.filterbank, fb_i, class_i, self.sfreq, self.method)
                trains[class_i, fb_i] = eeg_template # Store the template
                W[fb_i, class_i, :] = w_best  # Store the spatial filter
                
        self.trains = trains
        self.coef_ = W
        self.classes = classes

        return self
        
    def fit_WDANet_wo_cross(self, existing_train, existing_valid, target, ex_train_y, ex_valid_y, tr_y, n_tps, train_sub, i_sub, path):
        n_samples, n_chans, _ = theshapeof(existing_train)
        classes = np.unique(ex_train_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        model_path = path + '.pt' # './CD_all_4trials_5.pt'
        W = np.zeros((self.n_bands, len(classes), n_chans))

        for class_i in classes:
            print(f"class_number:{class_i}, subject_number:{i_sub+1}")
            # Select data with a specific label
            eeg_train = existing_train[..., ex_train_y == class_i]
            eeg_valid = existing_valid[..., ex_valid_y == class_i]
            eeg_tr = target[...,tr_y == class_i]
            
            for fb_i in range(self.n_bands):
                # Filter the signal with fb_i
                eeg_train = bandpass(eeg_train, self.sfreq,
                                    Wp=self.filterbank[fb_i][0],
                                    Ws=self.filterbank[fb_i][1])
                eeg_valid = bandpass(eeg_valid,self.sfreq,
                                    Wp=self.filterbank[fb_i][0],
                                    Ws=self.filterbank[fb_i][1])
                eeg_tr = bandpass(eeg_tr,self.sfreq,
                                    Wp=self.filterbank[fb_i][0],
                                    Ws=self.filterbank[fb_i][1])

                eeg_train_torch = torch.Tensor(eeg_train.copy())
                eeg_valid_torch = torch.Tensor(eeg_valid.copy())
                eeg_tr_torch = torch.Tensor(eeg_tr.copy())
                
                # "mean" target value 
                eeg_tr_mean = torch.mean(eeg_valid_torch, -1)

                # reshape
                #sample,channel,train_trail = eeg_train.shape
                train_trial = eeg_train.shape[2]
                valid_trial = eeg_valid_torch.shape[2]
                eeg_train_tmp = eeg_train_torch.permute(2,1,0).unsqueeze(1)
                eeg_valid_tmp = eeg_valid_torch.permute(2,1,0).unsqueeze(1)
                eeg_tr_origin_tmp = eeg_tr_torch.permute(2,1,0).unsqueeze(1)
                eeg_tr_temp = eeg_tr_mean.permute(1,0).unsqueeze(0).unsqueeze(1)
                
                # average eeg_tr to  map eeg_train
                eeg_tr_mean_for_train = torch.cat((eeg_tr_temp,eeg_tr_temp))
                
                if train_trial > 2:
                    for i in range(train_trial-2):
                        eeg_tr_mean_for_train =  torch.cat((eeg_tr_mean_for_train,eeg_tr_temp))
                
                # average eeg_tr to  map eeg_valid
                eeg_tr_mean_for_valid = torch.cat((eeg_tr_temp,eeg_tr_temp))
                
                if valid_trial > 2:
                    for i in range(valid_trial-2):
                        eeg_tr_mean_for_valid =  torch.cat((eeg_tr_mean_for_valid,eeg_tr_temp))
                
                # cleeg
                danet = DANet_main()
                dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                dan = DANet().to(dev)
                training_dataloader = danet.train_dataloader(eeg_train_tmp,eeg_tr_mean_for_train,256)
                valid_dataloader = danet.train_dataloader(eeg_valid_tmp,eeg_tr_mean_for_valid,256)
                train_loss_arr, valid_loss_arr = danet.train_network(dan,training_dataloader,valid_dataloader,model_path,500)
                
                transfer_fine = torch.zeros(train_sub*n_tps ,1,n_chans,n_samples)
                eeg_x = torch.cat((eeg_train_tmp,eeg_valid_tmp))
                eeg_y = torch.cat((eeg_tr_mean_for_train, eeg_tr_mean_for_valid))
                for i in range(train_sub):
                    left = i * n_tps
                    right = left + n_tps
                    fine_dataloader = danet.train_dataloader(eeg_x[left:right],eeg_y[left:right],64)
                    transfer_dataloader = danet.trans_dataloader(eeg_x[left:right],eeg_y[left:right],64)
                    dan.load_state_dict(torch.load(model_path))
                    danet.finetune_network(dan,fine_dataloader,150)
                    transfer_data = danet.transfer_domain(dan,transfer_dataloader,torch.zeros((n_tps, 1, n_chans,n_samples))) # transfer data is similar to target domain
                    transfer_data = transfer_data.cpu()
                    transfer_fine[left:right] = transfer_data

                # cat
                transfer_data = transfer_fine.cpu()
                eeg_tmp = torch.cat((eeg_tr_origin_tmp,transfer_fine))
                # reshape
                eeg_tmp = eeg_tmp.squeeze().permute(2,1,0).numpy()

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
    
    def fit_WDANet_linear(self, existing_train, existing_valid, target, ex_train_y, ex_valid_y, tr_y, n_tps, train_sub, i_sub, path):
        n_samples, n_chans, _ = theshapeof(existing_train)
        classes = np.unique(ex_train_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        W = np.zeros((self.n_bands, len(classes), n_chans))
        model_path = path + '_linear.pt' # './CD_all_4trials_5.pt'
        transfer_fine = torch.zeros(len(classes),train_sub*n_tps ,1,n_chans,n_samples)
        for fb_i in range(self.n_bands):
            ########## cat all class existing data ######### 
            train_x, train_y, valid_x, valid_y = filterdata_tv(existing_train, existing_valid, target, classes, ex_train_y, ex_valid_y, tr_y, fb_i, self.sfreq, self.filterbank)
            ########## train my network #########
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            danet = DANet_main()
            dan = DANet_linear().to(dev)
            training_dataloader = danet.train_dataloader(train_x,train_y,256)
            valid_dataloader = danet.train_dataloader(valid_x,valid_y,256)
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
        self.trains = trains
        self.coef_ = W
        self.classes = classes
        return self
    
    def fit_WDANet_temporal(self, existing_train, existing_valid, target, ex_train_y, ex_valid_y, tr_y, n_tps, train_sub, i_sub, path):
        n_samples, n_chans, _ = theshapeof(existing_train)
        classes = np.unique(ex_train_y)
        trains = np.zeros((len(classes), self.n_bands, n_samples, n_chans))
        W = np.zeros((self.n_bands, len(classes), n_chans))
        model_path = path + '_temporoal.pt' # './CD_all_4trials_5.pt'
        transfer_fine = torch.zeros(len(classes),train_sub*n_tps ,1,n_chans,n_samples)
        for fb_i in range(self.n_bands):
            ########## cat all class existing data ######### 
            train_x, train_y, valid_x, valid_y = filterdata_tv(existing_train, existing_valid, target, classes, ex_train_y, ex_valid_y, tr_y, fb_i, self.sfreq, self.filterbank)
            ########## train my network #########
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            danet = DANet_main()
            dan = DANet_temporoal().to(dev)
            training_dataloader = danet.train_dataloader(train_x,train_y,256)
            valid_dataloader = danet.train_dataloader(valid_x,valid_y,256)
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
        self.trains = trains
        self.coef_ = W
        self.classes = classes
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