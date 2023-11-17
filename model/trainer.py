import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from model import *
from learning_utils import *
import matplotlib.pyplot as plt
import pickle
import pandas as pd
class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.model = GBN_model(args)

        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)
                          
        
    def train(self, train_data_loader,test_loader):
        for epoch in range(self.epochs):

            for t in range(self.layer_num - 1):  #saw
                self.model.decoder[t + 1].rho = self.model.decoder[t].alphas

            self.model.to(self.args.device)

            loss_t = [0] * (self.layer_num + 1)
            likelihood_t = [0] * (self.layer_num + 1)
            num_data = len(train_data_loader)

            for i, (train_data, _) in enumerate(train_data_loader):
                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                train_data = torch.tensor(train_data, dtype=torch.float).to(self.args.device)


                re_x, theta, loss_list, likelihood, topic_gene , topic_embedding, gene_embedding = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                    elif t < self.layer_num:
                        (1  * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                      #  ((epoch+0.0/self.epochs)  * loss_list[t]).backward(retain_graph=True)
                        (self.args.rate*loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                re_x, theta, loss_list, likelihood, topic_gene , topic_embedding, gene_embedding = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (1 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                    elif t < self.layer_num:
                        (1  * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                        (self.args.rate * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()


            if epoch % 1 == 0:
                    print('epoch {}|{}, layer {}|{}, loss: {}'.format(epoch, self.epochs, t,self.layer_num,sum(loss_t),))

            if  epoch == self.epochs-1 or epoch % 100 == 0:# :#
                torch.save(self.model, self.save_path+"d-scIGM_model_"+self.args.dataname.split('.')[0])
                test_likelihood, test_ppl = self.test(train_data_loader,test_loader)


    def test(self, data_loader,test_loader):
        model = torch.load(self.save_path+"d-scIGM_model_"+self.args.dataname.split('.')[0])
        model.eval()
        likelihood_t = 0
        num_data = len(test_loader)
        ppl_total = 0
        test_theta = None
        for i, (train_data, test_data) in enumerate(test_loader):
            train_data = torch.tensor(train_data, dtype = torch.float).to(self.args.device)
            test_data = torch.tensor(test_data, dtype=torch.float).to(self.args.device)

            re_x, theta, loss_list, likelihood, topic_gene , topic_embedding, gene_embedding = model(train_data)
            temp_theta = np.concatenate((theta[0].T.cpu().detach().numpy(), theta[1].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[2].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[3].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[4].T.cpu().detach().numpy()),axis=1)
            temp_theta = np.concatenate((temp_theta, theta[5].T.cpu().detach().numpy()),axis=1)

            if test_theta is None:
                test_theta = temp_theta
            else:
                test_theta = np.concatenate((test_theta, temp_theta))

        pd.DataFrame(test_theta).to_csv(self.args.output_dir+'d-scIGM_'+ self.args.dataname.split('.')[0]+'_embedding'+'.csv') 
        temp = pd.read_csv(self.args.dataset_dir+self.args.dataname+'',sep=',',index_col=0)
        topic_gene = pd.DataFrame(topic_gene)
        topic_gene.index = temp.columns
        topic_gene.to_csv(self.args.output_dir+'d-scIGM_'+self.args.dataname.split('.')[0]+'_tg.csv') 
        topic_label = []
        for index in range(100):
            topic_label.append("topic_"+str(index+1))
        topic_embedding = pd.DataFrame(topic_embedding)   
        topic_embedding.index = topic_label
        topic_embedding.to_csv(self.args.output_dir+'d-scIGM_'+self.args.dataname.split('.')[0]+'_te.csv') 
        gene_embedding = pd.DataFrame(gene_embedding)     
        gene_embedding.index = temp.columns
        gene_embedding.to_csv(self.args.output_dir+'d-scIGM_'+self.args.dataname.split('.')[0]+'_ge.csv') 
        return likelihood_t, ppl_total

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

