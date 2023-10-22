import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

class DANet_main():
    def __init__(self):
        self.nothing = None

    def train_dataloader(self,existing,target,batch = 68):
        # run dataset on GPU
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            num_worker = 2
            x_mapping = existing.to(dev)
            y_mapping = target.to(dev)
        else:
            dev = torch.device("cpu")
            num_worker = 2
            x_mapping = existing.to(dev)
            y_mapping = target.to(dev)
        
        # create train and test dataset
        mapping_dataset = Data.TensorDataset(x_mapping,y_mapping)
        
        # DataLoader
        mapping_dl = Data.DataLoader(
                    dataset = mapping_dataset,
                    batch_size = batch,  # 68
                    shuffle = True
                    )

        return mapping_dl

    def trans_dataloader(self,existing,target,batch = 68):
        # run dataset on GPU
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            num_worker = 2
            x_mapping = existing.to(dev)
            y_mapping = target.to(dev)
        else:
            dev = torch.device("cpu")
            num_worker = 2
            x_mapping = existing.to(dev)
            y_mapping = target.to(dev)
        
        # create train and test dataset
        mapping_dataset = Data.TensorDataset(x_mapping,y_mapping)
        
        # DataLoader
        mapping_dl = Data.DataLoader(
                    dataset = mapping_dataset,
                    batch_size = batch,  # mean:32,Nmean:64
                    shuffle = False
                    )

        return mapping_dl
    
    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def model_to_dev(self,net):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.clg = net().to(dev)
        return self.clg
    
    def save_model(self,PATH_model,PATH_opt):
        torch.save(self.clg.state_dict(),PATH_model)
       
    def laod_model(self,PATH_model,PATH_opt):
        self.clg.load_state_dict(torch.load(PATH_model))
        
    def train_network(self,net, train_dl, valid_dl, model_path, epochs=500, loss=0):  #1000
        criterion = nn.MSELoss()
        self.optimizer = optim.Adam(net.parameters(), lr=5*1e-4) #weight_decay=0.0001
        min_loss = 100000
        min_epoch = 0
        train_loss_arr = torch.zeros(epochs)
        valid_loss_arr = torch.zeros(epochs)
        for epoch in range(epochs):
            net.train()
            for x,y in train_dl:
                pred = net(x)
                loss = criterion(pred,y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            net.eval()
            for x,y in valid_dl:
                with torch.no_grad():
                    pred = net(x)
                    valid_loss = criterion(pred,y)
                    if valid_loss < min_loss:
                        min_loss = valid_loss
                        min_epoch = epoch
                        torch.save(net.state_dict(), model_path)
            train_loss_arr[epoch] = loss.item()
            valid_loss_arr[epoch] = valid_loss.item()
            if (epoch + 1)%20 == 0:
                # print(f"epoch {epoch+1} loss:{loss.item()}")
                print("\r",f"epoch {epoch+1} train_loss:{loss} valid_loss:{valid_loss}",end="",flush=True)
        print("min_epoch:",min_epoch)
        return train_loss_arr, valid_loss_arr

    def finetune_network(self,net, train_dl,epochs=500,loss=0):  #1000
        criterion = nn.MSELoss()
        self.optimizer = optim.Adam(net.parameters(), lr=5*1e-4) #weight_decay=0.0001
        net.train()
        for epoch in range(epochs):
            for x,y in train_dl:
                pred = net(x)
                loss = criterion(pred,y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1)%20 == 0:
                # print(f"epoch {epoch+1} loss:{loss.item()}")
                print("\r",f"epoch {epoch+1} loss:{loss}",end="",flush=True)

    def transfer_domain(self,net,dataloader,final_dec):
        flag = True
        net.eval()
        for x,y in dataloader: #existing,target
            with torch.no_grad():
                dec = net(x)
                if flag == True:
                    # global final_dec
                    final_dec = dec.detach()
                    flag = False
                else:
                    final_dec = torch.cat((final_dec,dec.detach()))
        return final_dec