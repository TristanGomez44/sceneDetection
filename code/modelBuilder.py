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

import processResults

def buildFeatModel(featModelName,pretrainDataSet,layFeatCut=4):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101 or googLeNet
    - pretrainDataSet (str): the dataset on which the architecture should be pretrained. Can be imageNet, places365 or ADE20K depending on the architecture.\
            See code below to check on which pretrain dataset each architecture is available
    - layFeatCut (str): The layer at which to extract the features (ignored if the architecture is not a resnet one.)

    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''

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
    ''' Build an audio feature model pretrained for speech tasks

    Args:
    - audioFeatModelName (str): the name of the architecture. Can only be vggish

    Returns:
    - model (nn.Module): the audio feature extractor

    '''

    if audioFeatModelName == "vggish":
        model = vggish.VGG()
        model.load_state_dict(torch.load("../models/vggish.pth"))
    else:
        raise ValueError("Unkown audio feat model :",audioFeatModelName)

    return model

class FrameAttention(nn.Module):
    ''' A frame attention module that computes a scalar weight for each frame in a batch

    Args:
    - nbFeat (int): the number of feature in the vector representing a frame
    - frameAttRepSize (int): the size of the hidden state of the frame attention

    '''

    def __init__(self,nbFeat,frameAttRepSize):

        super(FrameAttention,self).__init__()

        self.linear = nn.Linear(nbFeat,frameAttRepSize)
        self.innerProdWeights = nn.Parameter(torch.randn(frameAttRepSize))

    def forward(self,x):
        x = torch.tanh(self.linear(x))
        attWeights = (x*self.innerProdWeights).sum(dim=-1,keepdim=True)
        return attWeights



class LSTM_sceneDet(nn.Module):
    ''' A LSTM temporal model

    This models the temporal dependencies between shot representations. It processes the representations one after another.

    Args:
    - nbFeat (int): the number of feature in the vector representing a frame
    - hiddenSize (int): the size of the hidden state
    - layerNb (int): the number of layers. Each layer is one LSTM stacked upon the preceding LSTMs.
    - dropout (float): the dropout amount in the layers of the LSTM, except the last one
    - bidirect (bool): a boolean to indicate if the LSTM is bi-bidirectional or not

    '''

    def __init__(self,nbFeat,hiddenSize,layerNb,dropout,bidirect):

        super(LSTM_sceneDet,self).__init__()

        self.rnn = nn.LSTM(input_size=nbFeat,hidden_size=hiddenSize,num_layers=layerNb,batch_first=True,dropout=dropout,bidirectional=bidirect)
        self.dense = nn.Linear(hiddenSize*(bidirect+1),1024)
        self.final = nn.Linear(1024,1)
        self.relu = nn.ReLU()

    def forward(self,x,h,c):
        ''' The forward pass into the LSTM

        Args:
        - x (torch.tensor): the feature batch
        - h (torch.tensor): the hidden state initial value. Can be set to None to initialize it with zeros
        - c (torch.tensor): the cell state initial value. Can be set to None to initialize it with zeros

        Returns:
        - x (torch.tensor): the scene change score for each shot of each batch
        - h,c (tuple): the hidden state and cell state at the end of processing the sequence

        '''

        if not h is None:
            x,(h,c) = self.rnn(x,(h,c))
        else:
            x,(h,c) = self.rnn(x)
        x = self.relu(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.final(x)

        x = torch.sigmoid(x).squeeze(dim=-1)

        return x,(h,c)

class CNN_sceneDet(nn.Module):
    ''' A CNN temporal model

    This models the temporal dependencies between shot reprensentations. It process all the representations at once.
    This is based on a resnet architecture

    Args:
    - layFeatCut (int): the layer at which the feature from the resnet should be extracted
    - modelType (str): the name of the resnet architecture to use
    - chan (int): the number of channel of the first layer of the architecture. The channel numbers in the following layers are based on this value also
    - pretrained (bool): indicates if the resnet has to be pretrained on imagenet or not.
    - pool (str): the way the feature should be pooled at the end of the processing. Can be 'mean' to simply do an average pooling or 'linear' to use a fully connected layer
    - multiGPU (bool): indicates if the computation should be done on several gpus
    - dilation (int): the dilation of the convolution

    '''

    def __init__(self,layFeatCut,modelType,chan=64,pretrained=True,pool="mean",multiGPU=False,dilation=1,scoreConvWindSize=1,sceneLenCnnPool=0):

        super(CNN_sceneDet,self).__init__()

        if modelType == "resnet50":
            self.cnn = resnet.resnet50(pretrained=False,layFeatCut=layFeatCut,maxPoolKer=(1,3),maxPoolPad=(0,1),stride=(1,2),featMap=True,chan=chan,inChan=3)
            expansion = 4
        elif modelType == "resnet101":
            self.cnn = resnet.resnet101(pretrained=False,layFeatCut=layFeatCut,maxPoolKer=(1,3),maxPoolPad=(0,1),stride=(1,2),featMap=True,chan=chan,inChan=3)
            expansion = 4
        elif modelType == "resnet18":
            self.cnn = resnet.resnet18(pretrained=False,layFeatCut=layFeatCut,maxPoolKer=(1,3),maxPoolPad=(0,1),stride=(1,2),featMap=True,chan=chan,inChan=3)
            expansion = 1
        else:
            raise ValueError("Unkown model type for CNN temporal model : ",modelType)

        if multiGPU:
            self.cnn = DataParallel(self.cnn,dim=0)

        if pretrained:
            if chan != 64:
                raise ValueError("To load the pretrained weights, the CNN temp model should have 64 channels and not {}".format(chan))

            checkpoint = torch.load("../models/{}_imageNet.pth".format(modelType))
            state_dict = {"module."+k: v for k,v in checkpoint.items()}
            self.cnn.load_state_dict(state_dict)

        self.pool = pool
        self.featNb = 2048
        self.hiddSize = 1024
        self.attSize = 1024
        self.layNb = 2
        self.dropout = 0
        self.sceneLenCnnPool = sceneLenCnnPool

        if scoreConvWindSize > 1:
            self.scoreConv = torch.nn.Conv1d(1,1,scoreConvWindSize,padding=scoreConvWindSize//2)
        else:
            self.scoreConv = None

        if self.pool == 'linear':
            self.endLin = nn.Linear(chan*8*self.featNb*expansion,1)
        elif self.pool == 'lstm':

            self.lstm = nn.LSTM(input_size=self.featNb,hidden_size=self.hiddSize,num_layers=self.layNb,batch_first=True,dropout=self.dropout,bidirectional=True)
            self.lin = nn.Linear(2*self.hiddSize,1)

        elif self.pool == "cnn":

            self.cnnPool = nn.Sequential(nn.Linear(self.sceneLenCnnPool*2048//32,self.hiddSize),nn.ReLU(),nn.Linear(self.hiddSize,1),nn.Sigmoid())

    def forward(self,x,gt=None,attList=None):

        x = self.cnn(x)
        x = x.permute(0,2,1,3)

        if self.pool == "mean":
            x = x.mean(dim=-1).mean(dim=-1)

            if self.scoreConv is not None:
                x = self.scoreConv(x.unsqueeze(1)).squeeze(1)

            x = torch.sigmoid(x)
        elif self.pool == "linear":

            origSize = x.size()
            x = x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3))
            x = x.contiguous().view(x.size(0),x.size(1)*x.size(2))
            x = self.endLin(x)
            x = x.view(origSize[0],origSize[1])

            if self.scoreConv is not None:
                x = self.scoreConv(x.unsqueeze(1)).squeeze(1)

            x = torch.sigmoid(x)
        elif self.pool == "lstm":

            x= x.mean(dim=-1)

            batchSceneProp = []
            for i in range(x.size(0)):

                parsedProportion = 0
                scenePropList = []

                while parsedProportion < 1:


                    xToParse = x[i:i+1,int(parsedProportion*x.size(1)):]
                    _,(h,_) = self.lstm(xToParse)

                    h = h.view(self.layNb,2,self.hiddSize)[-1].contiguous().view(2*self.hiddSize).unsqueeze(0)

                    sceneProp = torch.sigmoid(self.lin(h))

                    scenePropList.append(sceneProp.squeeze(1))

                    parsedProportion += sceneProp.data

                batchSceneProp.append(torch.cat(scenePropList,dim=0))

            return batchSceneProp

        elif self.pool == "cnn":

            #print(x.size())
            x = x.mean(dim=2)
            #print(x.size())

            batchSceneProp = []
            for i in range(x.size(0)):

                parsedProportion = 0
                scenePropList = []

                while parsedProportion < 1:

                    res = x.permute(0,2,1)
                    #print(res.size())
                    res = nn.functional.interpolate(res, size=self.sceneLenCnnPool,mode='linear')
                    #print(res.size())
                    res = res.permute(0,2,1)

                    print(res.size())
                    res = res.contiguous().view(res.size(0),res.size(1)*res.size(2))
                    print(res.size())
                    sceneProp = self.cnnPool(res)
                    print(sceneProp.size())
                    scenePropList.append(sceneProp.squeeze(1))

                    parsedProportion += sceneProp.data

                batchSceneProp.append(torch.cat(scenePropList,dim=0))

            return batchSceneProp
        else:
            raise ValueError("Unkown pool mode for CNN temp model {}".format(self.pool))

        return x

class SceneDet(nn.Module):
    ''' This module combines a feature extractor to represent each shot and a temporal model to compute the scene change score for each shot

    Args:
    - temp_model (str): the architecture of the temporal model. Can be 'RNN', 'resnet50' or 'resnet101'. If a resnet is chosen the temporal model will be a CNN
    - featModelName,pretrainDataSetFeat,layFeatCut : the argument to build the visual model (see buildFeatModel())
    - audioFeatModelName (str): the audio architecture (see buildAudioFeatModel()). Should be set to 'None' to not use an audio model
    - hiddenSize,layerNb,dropout,bidirect : the argument to build the LSTM. Ignored is the temporal model is a resnet. (see LSTM_sceneDet module)
    - cuda (bool): whether or not the computation should be done on gpu
    - framesPerShot (int): the number of frame to use to represent a shot
    - frameAttRepSize (int): the size of the frame attention (see FrameAttention module)
    - multiGPU (bool): to run the computation on several gpus. Ignored if cuda is False
    - chanTempMod, pretrTempMod, poolTempMod, dilTempMod: respectively the chan, pretrained, pool and dilTempMod arguments to build the temporal resnet model (see CNN_sceneDet module). Ignored if the temporal \
        model is a resnet.

    '''

    def __init__(self,temp_model,featModelName,pretrainDataSetFeat,audioFeatModelName,hiddenSize,layerNb,dropout,bidirect,cuda,layFeatCut,framesPerShot,frameAttRepSize,multiGPU,\
                        chanTempMod,pretrTempMod,poolTempMod,dilTempMod,scoreConvWindSize,sceneLenCnnPool):

        super(SceneDet,self).__init__()

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
        self.temp_model = temp_model
        #No need to throw an error because one has already been
        #thrown if the model type is unkown
        if featModelName=="resnet50" or featModelName=="resnet101":
            nbFeat = 256*2**(layFeatCut-1)
        elif featModelName=="googLeNet":
            nbFeat = 1024

        if not self.audioFeatModel is None:
            nbFeat += 128

        self.frameAtt = FrameAttention(nbFeat,frameAttRepSize)

        if self.temp_model == "RNN":
            self.tempModel = LSTM_sceneDet(nbFeat,hiddenSize,layerNb,dropout,bidirect)
        elif self.temp_model.find("resnet") != -1:
            self.tempModel = CNN_sceneDet(layFeatCut,self.temp_model,chanTempMod,pretrTempMod,poolTempMod,multiGPU,dilation=dilTempMod,scoreConvWindSize=scoreConvWindSize,sceneLenCnnPool=sceneLenCnnPool)

        self.nb_gpus = torch.cuda.device_count()

    def forward(self,x,audio,h=None,c=None,gt=None):
        ''' The forward pass of the scene change model

        Args:
        - x (torch.tensor): the visual data tensor
        - audio (torch.tensor): the audio data tensor. Can be None if no audio model is used
        - h,c (torch.tensor): the hidden state  and cell initial value. Can be set to None to initialize it with zeros and is ignored when using a resnet as temporal model

        Returns:
        - the scene change score if a resnet is used as temporal model
        - the scene change score along with the final hidden state and cell state if an LSTM is used as temporal model.
        '''

        x = self.computeFeat(x,audio,h,c)

        return self.computeScore(x,h,c,gt)

    def computeFeat(self,x,audio,h=None,c=None):

        origBatchSize = x.size(0)
        origSeqLength = x.size(1)//self.framesPerShot

        x = x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))

        x = self.featModel(x)

        if not self.audioFeatModel is None:
            audio = audio.view(audio.size(0)*audio.size(1),audio.size(2),audio.size(3),audio.size(4))
            audio = self.audioFeatModel(audio)

            x = torch.cat((x,audio),dim=-1)

        attWeights = self.frameAtt(x)

        #Unflattening
        x = x.view(origBatchSize,origSeqLength,self.framesPerShot,-1)
        attWeights = attWeights.view(origBatchSize,origSeqLength,self.framesPerShot,1)

        x = (x*attWeights).sum(dim=-2)/attWeights.sum(dim=-2)
        return x

    def computeScore(self,x,h=None,c=None,gt=None,attList=None):

        if self.temp_model == "RNN":
            return self.tempModel(x,h,c)
        elif self.temp_model.find("resnet") != -1:
            x = x.unsqueeze(1)
            x = x.expand(x.size(0),x.size(1)*3,x.size(2),x.size(3))
            return self.tempModel(x,gt,attList)

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

    net = SceneDet(args.temp_model,args.feat,args.pretrain_dataset,args.feat_audio,args.hidden_size,args.num_layers,args.dropout,args.bidirect,\
                    args.cuda,args.lay_feat_cut,args.frames_per_shot,args.frame_att_rep_size,args.multi_gpu,args.chan_temp_mod,args.pretr_temp_mod,\
                    args.pool_temp_mod,args.dil_temp_mod,args.score_conv_wind_size,args.scene_len_cnn_pool)

    return net

def main():


    imagePathList = np.array(sorted(glob.glob("../data/OVSD/Big_buck_bunny/middleFrames/*"),key=findNumbers),dtype=str)
    imagePathList = list(filter(lambda x:x.find(".wav") == -1,imagePathList))

    soundPathList = np.array(sorted(glob.glob("../data/OVSD/Big_buck_bunny/middleFrames/*.wav"),key=findNumbers),dtype=str)

    diagBlock = DiagBlock(cuda=True,batchSize=32,feat="googLeNet",pretrainDataSet="imageNet",audioFeat="vggish")

    diagBlock.detectDiagBlock(imagePathList,soundPathList,"test_exp",1)

if __name__ == "__main__":
    main()
