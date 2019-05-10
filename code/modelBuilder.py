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
import resnetSeg
import googleNet
from torch.nn import functional as F

import subprocess
from torch import nn
import vggish
import vggish_input

from torch.nn import DataParallel

def buildFeatModel(featModelName,pretrainDataSet,layFeatCut=4):

    if featModelName == "resnet50":

        if pretrainDataSet == "imageNet":
            featModel = resnet.resnet50(pretrained=False,layFeatCut=layFeatCut)
            featModel.load_state_dict(torch.load("../models/resnet50_imageNet.pth"))
        elif pretrainDataSet == "places365":
            featModel = resnet.resnet50(pretrained=False,num_classes=365,layFeatCut=layFeatCut)

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
        elif pretrainDataSet == "ADE20K":

            orig_resnet = resnetSeg.resnet50(pretrained=False)
            featModel = resnetSeg.ResnetSeg(orig_resnet)



            featModel.load_state_dict(torch.load("../models/resnet50_ADE20K.pth"))

        else:
            raise ValueError("Unknown pretrain dataset for model {} : {}".format(featModelName,pretrainDataSet))

    elif featModelName == "resnet101":

        if pretrainDataSet == "imageNet":
            featModel = resnet.resnet101(pretrained=False,layFeatCut=layFeatCut)
            featModel.load_state_dict(torch.load("../models/resnet101_imageNet.pth"))
        else:
            raise ValueError("Unknown pretrain dataset for model {} : {}".format(featModelName,pretrainDataSet))

    elif featModelName == "googLeNet":

        if pretrainDataSet == "imageNet":
            featModel = googleNet.googlenet(pretrained=True)
        else:
            raise ValueError("Unknown pretrain dataset for model {} : {}".format(featModelName,pretrainDataSet))

    else:
        raise ValueError("Unkown model name : {}".format(featModelName))

    return featModel

def buildAudioFeatModel(audioFeatModelName):

    if audioFeatModelName == "vggish":
        model = vggish.VGG()
        model.load_state_dict(torch.load("../models/vggish.pth"))
    else:
        raise ValueError("Unkown audio feat model :",audioFeatModelName)

    return model

class DiagBlock():

    def __init__(self,cuda,batchSize,feat,pretrainDataSet,audioFeat):

        self.cuda = cuda
        self.batchSize = batchSize

        self.featModel = None
        self.featModelName = feat

        self.audioFeatModel = None
        self.audioFeatModelName = audioFeat

        self.pretrainDataSet = pretrainDataSet

    def simMat(self,filePathList,foldName,modal="visual"):

        def preprocc_audio(x):
            x = vggish_input.wavfile_to_examples(x)

            if self.cuda:
                x = torch.tensor(x).cuda().unsqueeze(0)
            else:
                x = torch.tensor(x)

            return x.float()

        def preprocc_visual(x):
            x = cv2.imread(x)
            x = resize(x, (299, 299,3), anti_aliasing=True,mode="constant")

            if self.cuda:
                x = torch.tensor(x).cuda()
            else:
                x = torch.tensor(x)

            x = x.permute(2,0,1).float().unsqueeze(0)

            return x

        if modal == "visual":
            preprocc=preprocc_visual
            featModBuilder = buildFeatModel
            kwargs = {'featModelName':self.featModelName,'pretrainDataSet':self.pretrainDataSet}
        elif modal == "audio":
            preprocc=preprocc_audio
            featModBuilder = buildAudioFeatModel
            kwargs = {'audioFeatModelName':self.audioFeatModelName}
        else:
            raise ValueError("Unkown modality : {}".format(modal))

        if not os.path.exists("../results/{}_{}_simMatTEST.csv".format(foldName,modal)):

            featModel = featModBuilder(**kwargs)

            if self.cuda:
                featModel = featModel.cuda()

            featModel.eval()

            print("Computing features")

            #The modulo term increases the total number of batch by one in case
            #One last small batch is required
            nbBatch = len(filePathList)//self.batchSize + (len(filePathList) % self.batchSize != 0)
            feat = None
            for i in range(nbBatch):

                indMax = min((i+1)*self.batchSize,len(filePathList))

                tensorBatch = list(map(preprocc,filePathList[i*self.batchSize:indMax]))

                tensorBatch = torch.cat(tensorBatch)

                featBatch = featModel(tensorBatch).detach()

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
            np.savetxt("../results/{}_{}_simMat.csv".format(foldName,modal),simMatrix_np)

            plt.imshow(simMatrix_np.astype(int), cmap='gray', interpolation='nearest')
            plt.savefig("../vis/{}_{}_simMat.png".format(foldName,modal))

        else:
            print("Reading similarity matrix")

            simMatrix = torch.tensor(np.genfromtxt("../results/{}_{}_simMat.csv".format(foldName,modal))).float()

            if self.cuda:
                simMatrix = simMatrix.cuda()

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

    def detectDiagBlock(self,imagePathList,audioPathList,exp_id,model_id):

        foldName = os.path.dirname(imagePathList[0]).split("/")[-2]
        simMatrix = self.simMat(imagePathList,foldName,modal="visual")

        if not audioPathList is None:
            simMatrix_audio =self.simMat(audioPathList,foldName,modal="audio")

            simMatrix = ((simMatrix/simMatrix.max())+(simMatrix_audio/simMatrix_audio.max()))/2

        np.savetxt("../results/{}_{}_simMat.csv".format(foldName,model_id),simMatrix.detach().cpu().numpy())

        N = int(len(simMatrix))
        pMax = N*N

        print("Number of shots : ",N)

        simMatrix = simMatrix[:N,:N]

        K = self.countScenes(simMatrix.cpu().numpy())

        if self.cuda:
            subprocess.run(["./baseline/build/baseline", "../results/{}_{}_simMat.csv".format(foldName,model_id),"../results/{}/{}_basecuts.csv".format(exp_id,foldName),str(N),str(K),"cuda"])
        else:
            subprocess.run(["./baseline/build/baseline", "../results/{}_{}_simMat.csv".format(foldName,model_id),"../results/{}/{}_basecuts.csv".format(exp_id,foldName),str(N),str(K),"cpu"])

class simpleAttention(nn.Module):
    def __init__(self,nbFeat,frameAttRepSize):

        super(simpleAttention,self).__init__()

        self.linear = nn.Linear(nbFeat,frameAttRepSize)
        self.innerProdWeights = nn.Parameter(torch.randn(frameAttRepSize))

    def forward(self,x):
        x = torch.tanh(self.linear(x))
        attWeights = (x*self.innerProdWeights).sum(dim=-1,keepdim=True)
        return attWeights

class CNN_RNN(nn.Module):

    def __init__(self,featModelName,pretrainDataSetFeat,audioFeatModelName,hiddenSize,layerNb,dropout,bidirect,cuda,layFeatCut,train_visual,framesPerShot,frameAttRepSize,multiGPU):

        super(CNN_RNN,self).__init__()

        self.featModel = buildFeatModel(featModelName,pretrainDataSetFeat,layFeatCut)

        if multiGPU:
            self.featModel = DataParallel(self.featModel,dim=0)

        if audioFeatModelName != "None":
            self.audioFeatModel = buildAudioFeatModel(audioFeatModelName)
            if multiGPU:
                self.audioFeatModel = DataParallel(self.audioFeatModel,dim=0)
        else:
            self.audioFeatModel = None

        self.framesPerShot = framesPerShot

        #No need to throw an error because one has already been
        #thrown if the model type is unkown
        if featModelName=="resnet50" or featModelName=="resnet101":
            nbFeat = 256*2**(layFeatCut-1)
        elif featModelName=="googLeNet":
            nbFeat = 1024

        if not self.audioFeatModel is None:
            nbFeat += 128

        self.frameAtt = simpleAttention(nbFeat,frameAttRepSize)

        self.rnn = nn.LSTM(input_size=nbFeat,hidden_size=hiddenSize,num_layers=layerNb,batch_first=True,dropout=dropout,bidirectional=bidirect)

        self.dense = nn.Linear(hiddenSize*(bidirect+1),1024)

        self.final = nn.Linear(1024,2)

        self.relu = nn.ReLU()
        self.train_visual = train_visual

        self.nb_gpus = torch.cuda.device_count()

    def forward(self,x,audio,h=None,c=None):

        #print("Beg : ",printAlloc())
        #subprocess.call("nvidia-smi >> nvidia.txt", shell=True)
        #print(x.size())
        origBatchSize = x.size(0)
        origSeqLength = x.size(1)//self.framesPerShot

        #print(x.size())

        #print(x.size())
        x = x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        #print(x.size())
        #x = x.permute(0,3,1,2)
        #print(x.size())
        x = self.featModel(x)

        #print("After cnn : ",printAlloc())

        if not self.audioFeatModel is None:
            audio = audio.view(audio.size(0)*audio.size(1),audio.size(2),audio.size(3),audio.size(4))
            audio = self.audioFeatModel(audio)

            x = torch.cat((x,audio),dim=-1)

        attWeights = self.frameAtt(x)


        #print("Forward",h is None)
        #print(x.size(),attWeights.size())

        #Unflattening
        x = x.view(origBatchSize,origSeqLength,self.framesPerShot,-1)
        attWeights = attWeights.view(origBatchSize,origSeqLength,self.framesPerShot,1)
        #print(x.size())
        x = (x*attWeights).sum(dim=-2)/attWeights.sum(dim=-2)

        if not h is None:
            x,(h,c) = self.rnn(x,(h,c))
        else:
            x,(h,c) = self.rnn(x)

        #print("After RNN : ",printAlloc())

        x = self.relu(x)

        x = self.dense(x)
        x = self.relu(x)

        x = self.final(x)

        #print(x.size())

        #print(x.size())
        #x = x.permute(1,0,2)
        #print(x.size())
        #sys.exit(0)
        return x,(h,c)

    def getParams(self):

        featParams = list(self.featModel.parameters())
        params = []

        for p in self.rnn.parameters():
            params.append(p)
        for p in self.dense.parameters():
            params.append(p)

        return (p for p in params)

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

def netBuilder(args):
    net = CNN_RNN(args.feat,args.pretrain_dataset,args.feat_audio,args.hidden_size,args.num_layers,args.dropout,args.bidirect,\
                    args.cuda,args.lay_feat_cut,args.train_visual,args.frames_per_shot,args.frame_att_rep_size,args.multi_gpu)

    return net

def main():


    imagePathList = np.array(sorted(glob.glob("../data/OVSD/Big_buck_bunny/middleFrames/*"),key=findNumbers),dtype=str)
    imagePathList = list(filter(lambda x:x.find(".wav") == -1,imagePathList))

    soundPathList = np.array(sorted(glob.glob("../data/OVSD/Big_buck_bunny/middleFrames/*.wav"),key=findNumbers),dtype=str)

    diagBlock = DiagBlock(cuda=True,batchSize=32,feat="googLeNet",pretrainDataSet="imageNet",audioFeat="vggish")

    diagBlock.detectDiagBlock(imagePathList,soundPathList,"test_exp",1)

if __name__ == "__main__":
    main()
