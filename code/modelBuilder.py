import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
import resnet
import vggish

import args

def buildFeatModel(featModelName):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''

    featModel = getattr(resnet,featModelName)(pretrained=True)

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

class Discriminator(nn.Module):
    ''' This class defining the discriminator for the adversarial loss term

    This is a three layer MLP outputing the probability that the input representation
    comes from the target domain or not

    Args:
    - inFeat (int): the number of feature outputed by the visual model
    - applyDropout (bool): whether or not to apply dropout on the first two layer of the
                            distriminator

    '''

    def __init__(self,inFeat,applyDropout):

        super(Discriminator,self).__init__()

        self.lin1 = nn.Linear(inFeat,512)
        self.lin2 = nn.Linear(512,128)
        self.lin3 = nn.Linear(128,1)

        self.relu = nn.ReLU()
        self.sigmo = nn.Sigmoid()

        self.applyDropout = applyDropout

        if self.applyDropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        if self.applyDropout:
            x = self.dropout(self.relu(self.lin1(x)))
            x = self.dropout(self.relu(self.lin2(x)))
            return self.sigmo(self.lin3(x))
        else:
            x = self.relu(self.lin1(x))
            x = self.relu(self.lin2(x))
            return self.sigmo(self.lin3(x))

class LSTM_sceneDet(nn.Module):
    ''' A LSTM temporal model

    This models the temporal dependencies between shot representations. It processes the representations one after another.

    Args:
    - nbFeat (int): the number of feature in the vector representing a frame
    - hiddenSize (int): the size of the hidden state
    - layerNb (int): the number of layers. Each layer is one LSTM stacked upon the preceding LSTMs.
    - dropout (float): the dropout amount in the layers of the LSTM, except the last one

    '''

    def __init__(self,nbFeat,hiddenSize,layerNb,dropout):

        super(LSTM_sceneDet,self).__init__()

        self.rnn = nn.LSTM(input_size=nbFeat,hidden_size=hiddenSize,num_layers=layerNb,batch_first=True,dropout=dropout,bidirectional=True)
        self.dense = nn.Linear(hiddenSize*2,1024)
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

class ScoreConv(nn.Module):
    ''' This is a module that reads the scene change scores just before they are passed to the sigmoid by
    the temporal model. It apply one or two convolution layers to the signal and uses 1x1 convolution to
    outputs a transformed signal of the same shape as the input signal.

    It can return this transformed signal and can also returns this transformed signal multiplied
    by the input, like an attention layer.

    Args:
    - kerSize (int): the kernel size of the convolution(s)
    - chan (int): the number of channel when using two convolutions
    - biLay (bool): whether or not to apply two convolutional layers instead of one
    - attention (bool): whether or not to multiply the transformed signal by the input before returning it

    '''

    def __init__(self,kerSize,chan,biLay,attention=False):

        super(ScoreConv,self).__init__()

        self.attention = attention

        if biLay:
            self.conv1 = torch.nn.Conv1d(1,chan,kerSize,padding=kerSize//2)
            self.conv2 = torch.nn.Conv1d(chan,1,1)
            self.layers = nn.Sequential(self.conv1,nn.ReLU(),self.conv2)
        else:
            self.layers = torch.nn.Conv1d(1,1,kerSize,padding=kerSize//2)

    def forward(self,x):

        if not self.attention:
            return self.layers(x)
        else:
            weights = self.layers(x)
            return weights*x

class CNN_sceneDet(nn.Module):
    ''' A CNN temporal model

    This models the temporal dependencies between shot reprensentations. It process all the representations at once.
    This is based on a resnet architecture

    Args:
    - modelType (str): the name of the resnet architecture to use
    - chan (int): the number of channel of the first layer of the architecture. The channel numbers in the following layers are based on this value also
    - pretrained (bool): indicates if the resnet has to be pretrained on imagenet or not.
    - pool (str): the way the feature should be pooled at the end of the processing. Can be 'mean' to simply do an average pooling or 'linear' to use a fully connected layer
    - multiGPU (bool): indicates if the computation should be done on several gpus
    - scoreConvWindSize (int): the kernel size of the score convolution (SC) that will be applied (set to 1 to not apply SC at all).
    - scoreConvChan (int): the number of channel of the SC
    - scoreConvBiLay (bool): to have two layers in the SC instead of one
    - scoreConvAtt (bool): to use the SC as an attention layer

    '''

    def __init__(self,modelType,pool="mean",multiGPU=False,scoreConvWindSize=1,\
                scoreConvChan=8,scoreConvBiLay=False,scoreConvAtt=False):

        super(CNN_sceneDet,self).__init__()

        if modelType == "resnet50" or modelType == "resnet101" or modelType == "resnet152":
            expansion = 4
        else:
            expansion = 1

        self.cnn = getattr(resnet,modelType)(pretrained=True)

        if multiGPU:
            self.cnn = DataParallel(self.cnn,dim=0)

        self.pool = pool
        self.featNb = 2048
        self.hiddSize = 1024
        self.attSize = 1024
        self.layNb = 2
        self.dropout = 0

        if scoreConvWindSize > 1:
            self.scoreConv = ScoreConv(scoreConvWindSize,scoreConvChan,scoreConvBiLay,scoreConvAtt)
        else:
            self.scoreConv = None

        if self.pool == 'linear':
            self.endLin = nn.Linear(chan*8*self.featNb*expansion,1)

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
        else:
            raise ValueError("Unkown pool mode for CNN temp model {}".format(self.pool))

        return x

class SceneDet(nn.Module):
    ''' This module combines a feature extractor to represent each shot and a temporal model to compute the scene change score for each shot

    Args:
    - temp_model (str): the architecture of the temporal model. Can be 'RNN', 'resnet50' or 'resnet101'. If a resnet is chosen the temporal model will be a CNN
    - featModelName : the argument to build the visual model (see buildFeatModel())
    - audioFeatModelName (str): the audio architecture (see buildAudioFeatModel()). Should be set to 'None' to not use an audio model
    - hiddenSize,layerNb,dropout : the argument to build the LSTM. Ignored is the temporal model is a resnet. (see LSTM_sceneDet module)
    - cuda (bool): whether or not the computation should be done on gpu
    - multiGPU (bool): to run the computation on several gpus. Ignored if cuda is False
    - chanTempMod, pretrTempMod, poolTempMod : respectively the chan, pretrained, pool and dilTempMod arguments to build the temporal resnet model (see CNN_sceneDet module). Ignored if the temporal \
        model is a resnet.

    '''

    def __init__(self,temp_model,featModelName,audioFeatModelName,hiddenSize,layerNb,dropout,cuda,multiGPU,\
                        poolTempMod,scoreConvWindSize,scoreConvChan,scoreConvBiLay,scoreConvAtt):

        super(SceneDet,self).__init__()

        self.featModel = buildFeatModel(featModelName)

        if multiGPU:
            self.featModel = DataParallel(self.featModel,dim=0)

        if audioFeatModelName != "None":
            self.audioFeatModel = buildAudioFeatModel(audioFeatModelName)
            if multiGPU:
                self.audioFeatModel = DataParallel(self.audioFeatModel,dim=0)
        else:
            self.audioFeatModel = None

        self.temp_model = temp_model
        #No need to throw an error because one has already been
        #thrown if the model type is unkown
        if featModelName=="resnet50" or featModelName=="resnet101" or featModelName=="resnet151":
            self.nbFeat = 256*2**(4-1)
        else:
            self.nbFeat = 64*2**(4-1)

        if not self.audioFeatModel is None:
            self.nbFeat += 128

        if self.temp_model == "RNN":
            self.tempModel = LSTM_sceneDet(self.nbFeat,hiddenSize,layerNb,dropout)
        elif self.temp_model.find("net") != -1:
            self.tempModel = CNN_sceneDet(self.temp_model,poolTempMod,multiGPU,scoreConvWindSize=scoreConvWindSize,\
                                            scoreConvChan=scoreConvChan,scoreConvBiLay=scoreConvBiLay,scoreConvAtt=scoreConvAtt)

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
        self.features = x
        return self.computeScore(x,h,c,gt)

    def computeFeat(self,x,audio,h=None,c=None):

        origBatchSize = x.size(0)
        origSeqLength = x.size(1)

        x = x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))

        #Computing features
        x = self.featModel(x)

        if not self.audioFeatModel is None:
            audio = audio.view(audio.size(0)*audio.size(1),audio.size(2),audio.size(3),audio.size(4))

            #Adding the audio features
            audio = self.audioFeatModel(audio)
            x = torch.cat((x,audio),dim=-1)

        #Unflattening
        x = x.view(origBatchSize,origSeqLength,-1)

        return x

    def computeScore(self,x,h=None,c=None,gt=None,attList=None):

        if self.temp_model == "RNN":
            return self.tempModel(x,h,c)
        elif self.temp_model.find("net") != -1:
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

def netBuilder(args):

    net = SceneDet(args.temp_model,args.feat,args.feat_audio,args.hidden_size,args.num_layers,args.dropout,\
                    args.cuda,args.multi_gpu,args.pool_temp_mod,args.score_conv_wind_size,args.score_conv_chan,args.score_conv_bilay,\
                    args.score_conv_attention)

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='N',
                        help='the net to use to produce feature for each shot')

    argreader.parser.add_argument('--feat_audio', type=str, metavar='N',
                        help='the net to use to produce audio feature for each shot')

    argreader.parser.add_argument('--pool_temp_mod', type=str, metavar='N',
                        help='The pooling used for the CNN temporal model. Can be \'mean\' or \'linear\'')

    argreader.parser.add_argument('--score_conv_wind_size', type=int, metavar='N',
                        help='The size of the 1d convolution to apply on scores if temp model is a CNN. Set to 1 to remove that layer')

    argreader.parser.add_argument('--score_conv_bilay', type=args.str2bool, metavar='N',
                        help='To apply two convolution (the second is a 1x1 conv) on the scores instead of just one layer')

    argreader.parser.add_argument('--score_conv_attention', type=args.str2bool, metavar='N',
                        help='To apply the score convolution(s) as an attention layer.')

    argreader.parser.add_argument('--score_conv_chan', type=int, metavar='N',
                        help='The number of channel of the score convolution layer (used only if --score_conv_bilay is True)')

    argreader.parser.add_argument('--hidden_size', type=int,metavar='HS',
                        help='The size of the hidden layers in the RNN')

    argreader.parser.add_argument('--num_layers', type=int,metavar='NL',
                        help='The number of hidden layers in the RNN')

    argreader.parser.add_argument('--dropout', type=float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--bidirect', type=args.str2bool,metavar='BIDIR',
                        help='If true, the RNN will be bi-bidirectional')

    argreader.parser.add_argument('--temp_model', type=str,metavar='MODE',
                        help="The architecture to use to model the temporal dependencies. Can be \'RNN\' or a resnet architecture : \'resnet18\', \'resnet50\', \'resnet101\', etc.")

    argreader.parser.add_argument('--discr_dropout', type=args.str2bool, metavar='S',
                        help='To apply dropout on the discriminator (only useful when using adversarial loss term)')

    return argreader
