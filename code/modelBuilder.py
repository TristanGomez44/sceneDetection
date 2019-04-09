import sys
from torchvision.models.inception import inception_v3
import cv2
import glob
import torch
from skimage.transform import resize
import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import time

import resnet
import googLeNet
from torch.nn import functional as F

import subprocess
from torch import nn

def buildFeatModel(featModelName,pretrainDataSet):

    if featModelName == "resnet50":

        if pretrainDataSet == "imageNet":
            featModel = resnet.resnet50(pretrained=True)
        elif pretrainDataSet == "places365":
            featModel = resnet.resnet50(pretrained=False,num_classes=365)

            ####### This load code comes from https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py ######

            # load the pre-trained weights
            model_file = '%s_places365.pth.tar' % featModelName
            if not os.access("../models/"+model_file, os.W_OK):
                weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                os.system('wget ' + weight_url)

                os.rename(model_file, "../models/"+model_file)

            checkpoint = torch.load("../models/"+model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            featModel.load_state_dict(state_dict)
        else:
            raise ValueError("Unknown pretrain dataset for model {} : {}".format(featModelName,pretrainDataSet))

    elif featModelName == "googLeNet":

        if pretrainDataSet == "imageNet":
            featModel = googleNet.GoogLeNet(pretrained=True)
        else:
            raise ValueError("Unknown pretrain dataset for model {} : {}".format(featModelName,pretrainDataSet))

    else:
        raise ValueError("Unkown model name :".format(featModelName))

    return featModel

class DiagBlock():

    def __init__(self,cuda,batchSize,feat,pretrainDataSet):

        self.cuda = cuda
        self.batchSize = batchSize
        self.featModelName = feat
        self.featModel = None
        self.pretrainDataSet = pretrainDataSet

    def simMat(self,imagePathList,foldName):

        def preprocc(x):
            x = cv2.imread(x)
            x = resize(x, (299, 299,3), anti_aliasing=True,mode="constant")

            if self.cuda:
                x = torch.tensor(x).cuda()
            else:
                x = torch.tensor(x)

            x = x.permute(2,0,1).float().unsqueeze(0)

            return x

        if not os.path.exists("../results/{}_simMat.csv".format(foldName)):

            if self.featModel is None:

                self.featModel = buildFeatModel(self.featModelName,self.pretrainDataSet)

                if self.cuda:
                    self.featModel = self.featModel.cuda()

                self.featModel.eval()

            print("Computing features")

            #The modulo term increases the total number of batch by one in case
            #One last small batch is required
            nbBatch = len(imagePathList)//self.batchSize + (len(imagePathList) % self.batchSize != 0)
            feat = None
            for i in range(nbBatch):

                indMax = min((i+1)*self.batchSize,len(imagePathList))

                imageBatch = list(map(preprocc,imagePathList[i*self.batchSize:indMax]))

                imageBatch = torch.cat(imageBatch)

                featBatch = self.featModel(imageBatch).detach()

                if feat is None:
                    feat = featBatch
                else:
                    feat = torch.cat([feat,featBatch])

            print("Computing the similarity matrix")

            featCol = feat.unsqueeze(1)
            featRow = feat.unsqueeze(0)

            featCol = featCol.expand(feat.size(0),feat.size(0),feat.size(1))
            featRow = featRow.expand(feat.size(0),feat.size(0),feat.size(1))

            simMatrix = torch.pow(featCol-featRow,2).sum(dim=2)

            simMatrix_np = simMatrix.detach().cpu().numpy()
            np.savetxt("../results/{}_simMat.csv".format(foldName),simMatrix_np)

            plt.imshow(simMatrix_np.astype(int), cmap='gray', interpolation='nearest')
            plt.savefig("../vis/{}_simMat.png".format(foldName))

        else:
            print("Reading similarity matrix")

            simMatrix = torch.tensor(np.genfromtxt("../results/{}_simMat.csv".format(foldName)))

        return simMatrix

    def getPossiblePValues(self,N,K):
        pMax = N*N

        if not os.path.exists("../results/possibP_N{}_K{}.csv".format(N,K)):

            print("Computing possible values of p")

            #Indicates which value of p are possible
            B = torch.zeros((N,K,pMax)).byte()
            lineEnum = torch.arange(N-1,-1,-1).unsqueeze(1).expand(N,pMax)
            colEnum = torch.arange(N*N-1,-1,-1).unsqueeze(0).expand(N,pMax)

            tens1 = torch.arange(N).unsqueeze(1).expand(N,pMax)
            tens2 = torch.pow(torch.arange(pMax),2).unsqueeze(0).expand(N,pMax)

            for n in range(B.size(0)):
                for p in range(B.size(2)):
                    if n*n == p:
                        B[n,0,p] = 1

            for k in range(1,K):

                precMat = B[:,k-1,:]
                print(k)

                for n in range(1,N+1):
                    print("\t",n)

                    for p in range(1,pMax+1):

                        l=1
                        while n-l >= 0 and p-l*l >= 0 and not B[n-1,k,p-1]:
                            B[n-1,k,p-1] = precMat[n-l,p-l*l]
                            l += 1

            possibleValues = torch.nonzero(B)

            np.savetxt("../results/possibP_N{}_K{}.csv".format(N,K),possibleValues.detach().cpu().numpy())

        else:

            possibleValues = torch.tensor(np.genfromtxt("../results/possibP_N{}_K{}.csv".format(N,K)))
            print("Reading possible values of p")

        print(len(possibleValues),"values possible")

        return possibleValues.long()

    def countScenes(self,simMatrix):

        print("Number of scenes : ",end="")

        s = np.linalg.svd(simMatrix,compute_uv=False)
        s = np.log(s[np.where(s > 1)])

        H = np.array([len(s)-1,s[0]-s[len(s)-1]])
        I = np.concatenate((s[:,np.newaxis],np.arange(len(s))[:,np.newaxis]),axis=1)

        K = np.argmin((I*H).sum(axis=1))+1

        print(K)
        return K

    def detectDiagBlock(self,imagePathList,exp_id,model_id):

        foldName = os.path.dirname(imagePathList[0]).split("/")[-2]
        simMatrix = self.simMat(imagePathList,foldName)

        N = int(len(simMatrix))
        pMax = N*N

        print("Number of shots : ",N)

        simMatrix = simMatrix[:N,:N]

        K = self.countScenes(simMatrix.cpu().numpy())

        if self.cuda:
            subprocess.run(["./baseline/build/baseline", "../data/{}/{}_simMat.csv".format(foldName,foldName),"../results/{}/{}_basecuts.csv".format(exp_id,foldName),str(N),str(K),"cuda"])
        else:
            subprocess.run(["./baseline/build/baseline", "../data/{}/{}_simMat.csv".format(foldName,foldName),"../results/{}/{}_basecuts.csv".format(exp_id,foldName),str(N),str(K),"cpu"])

        '''
        C = torch.zeros((N,K,pMax))
        I = torch.zeros((N,K,pMax))
        P = torch.zeros((N,K,pMax))
        G = torch.zeros(N-1)

        C[:,:,:] = float("NaN")
        I[:,:,:] = float("NaN")
        P[:,:,:] = float("NaN")
        G[:] = float("NaN")


        if self.cuda:
            print("Putting tensors on cuda")
            simMatrix = simMatrix.cuda()
            C,I,P,G = C.cuda(),I.cuda(),P.cuda(),G.cuda()

        for k in range(K):
            print("Computing for",k+1,"blocks over",K)
            for n in range(1,N+1):
                for p in range(n-1,n*n):
                        if k == 0:

                            C[n-1,0,p] = simMatrix[n:,n:].sum()/(p+(N-n+1)*(N-n+1)-N)
                            I[n-1,0,p] = N
                            P[n-1,0,p] = (N-n+1)*(N-n+1)
                            if n < 10:
                                print("Init k==0",n-1,0,p)
                        else:

                            G[:] = float("NaN")
                            i=n
                            a = p+(i-n+1)*(i-n+1)

                            while i < N and a < pMax:

                                G[i-1] = simMatrix[n-1:i,n-1:i].sum()/(a+P[i,k-1,a]-N)+C[i,k-1,a]
                                i+=1
                                a = p+(i-n+1)*(i-n+1)

                            G_val = G[n:i]

                            if len(G_val) == 0:
                                minimum,argmin = 0,0
                            else:
                                minimum,argmin = torch.min(G_val,dim=0)

                            C[n-1,k,p] = minimum

                            I[n-1,k,p] = argmin+n-1

                            b = ((I[n-1,k,p]-n+1)*(I[n-1,k,p]-n+1)).long()

                            if p+b < P.size(2):
                                P[n-1,k,p] = b+P[I[n-1,k,p].long(),k-1,p+b]
                            else:
                                print("p+b is too big")

        P_tot = 0
        sceneSplits = [0]
        #print(I[:,1,:]
        for i in range(1,K+1):
            sceneSplits.append(I[sceneSplits[i-1],K-i,P_tot].long().item())
            P_tot += (sceneSplits[i]-sceneSplits[i-1])*(sceneSplits[i]-sceneSplits[i-1])

        np.savetxt("../results/{}/{}_{}.csv".format(exp_id,foldName,model_id),np.array(sceneSplits))
        print(sceneSplits)
        print("Sanity check :",P_tot,P[0,K-1,0].item())
        '''


class CNN_RNN(nn.Module):


    def __init__(self,featModelName,pretrainDataSetFeat,hiddenSize,layerNb,dropout,bidirect,cuda):

        super(CNN_RNN,self).__init__()

        self.featModel = buildFeatModel(featModelName,pretrainDataSetFeat)
        self.featModel.eval()

        #No need to throw an error because one has already been
        #thrown if the model type is unkown
        if featModelName=="resnet50":
            nbFeat = 2048

        self.rnn = nn.LSTM(input_size=nbFeat,hidden_size=hiddenSize,num_layers=layerNb,batch_first=True,dropout=dropout,bidirectional=bidirect)

        self.dense = nn.Linear(hiddenSize*(bidirect+1),2)

    def forward(self,x):

        origBatchSize = x.size(0)
        origSeqLength = x.size(1)

        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        x = x.permute(0,3,1,2)
        x = self.featModel(x)
        x = x.view(origBatchSize,origSeqLength,-1)
        x = x.permute(1,0,2)
        x,_ = self.rnn(x)
        x = self.dense(x)

        return x

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))


def netBuilder(args):
    net = CNN_RNN(args.feat,args.pretrain_dataset,args.hidden_size,args.num_layers,args.dropout,args.bidirect,args.cuda)

    return net
def main():

    imagePathList = np.array(sorted(glob.glob("../data/big_buck_bunny_480p_surround-fix/middleFrames/*"),key=findNumbers),dtype=str)

    diagBlock = DiagBlock(cuda=True,feat="googLeNet")

    diagBlock.detectDiagBlock(imagePathList,"test_exp",1)
if __name__ == "__main__":
    main()
