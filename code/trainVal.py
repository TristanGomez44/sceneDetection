import os
import sys

import args
from args import ArgReader
from args import str2bool

import glob

import numpy as np
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from sklearn.metrics import roc_auc_score

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import modelBuilder
import load_data
import metrics
import utils
import lossTerms
import update
import radam

def epochSeqTr(model,optim,log_interval,loader, epoch, args,writer,**kwargs):
    ''' Train a model during one epoch

    Args:
    - model (torch.nn.Module): the model to be trained
    - optim (torch.optim): the optimiser
    - log_interval (int): the number of epochs to wait before printing a log
    - loader (load_data.TrainLoader): the train data loader
    - epoch (int): the current epoch
    - args (Namespace): the namespace containing all the arguments required for training and building the network
    - writer (tensorboardX.SummaryWriter): the writer to use to log metrics evolution to tensorboardX
    - width (int): the width of the triangular window (i.e. the number of steps over which the window is spreading)

    '''

    model.train()

    print("Epoch",epoch," : train")

    metrDict = {"Loss":0,"Coverage":0,"Overflow":0,"True F-score":0,"AuC":0,\
                "IoU":0,"Disc Accuracy":0,"Dist Pos":0,"Dist Neg":0,"DED":0}

    validBatch = 0

    allOut = None
    allGT = None

    for batch_idx,(data,target,vidNames) in enumerate(loader):

        if target.sum() > 0:

            if (batch_idx % log_interval == 0):
                print("\t",batch_idx*len(data)*len(target[0]),"/",len(loader.dataset))

            #Puting tensors on cuda
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            #Computing predictions
            if args.temp_model.find("net") != -1:

                if args.iou_mode:
                    output,iou_output = model(data)
                else:
                    output = model(data)
                    iou_output = None
            else:
                output,_ = model(data)

            #Computing loss

            output = output[:,args.train_step_to_ignore:output.size(1)-args.train_step_to_ignore]
            target = target[:,args.train_step_to_ignore:target.size(1)-args.train_step_to_ignore]

            weights = getWeights(target,args.class_weight)
            loss = args.nll_weight*F.binary_cross_entropy(output, target,weight=weights)

            #Adding loss term
            loss = lossTerms.addDistTerm(loss,args,output,target)
            loss,discMeanAcc = lossTerms.addAdvTerm(loss,args,model.features,model.featModel,kwargs["discrModel"],kwargs["discrIter"],kwargs["discrOptim"])
            loss,distPos,distNeg = lossTerms.addSiamTerm(loss,args,model.features,target)
            loss = -lossTerms.addIoUTerm(loss,args,iou_output,target)

            loss.backward()
            optim.step()
            optim.zero_grad()

            #Metrics
            pred = output.data > 0.5
            cov,overflow,iou = metrics.binaryToMetrics(pred,target)

            metrDict["Coverage"] += cov
            metrDict["Overflow"] += overflow
            metrDict["True F-score"] += 2*cov*(1-overflow)/(cov+1-overflow)
            metrDict["IoU"] += iou
            metrDict["Disc Accuracy"] += discMeanAcc
            metrDict["Dist Pos"] += distPos
            metrDict["Dist Neg"] += distNeg
            metrDict["DED"] += metrics.computeDED(output.data>0.5,target.long())

            if allOut is None:
                allOut = output.data
                allGT = target
            else:
                allOut = torch.cat((allOut,output.data),dim=-1)
                allGT = torch.cat((allGT,target),dim=-1)

            metrDict["Loss"] += loss.detach().data.item()
            validBatch += 1

            if validBatch > 3 and args.debug:
                break

    #If the training set is empty (which we might want to for kust evaluate the model), then allOut and allGT will still be None
    if not allGT is None:
        metrDict["AuC"] = roc_auc_score(allGT.view(-1).cpu().numpy(),allOut.view(-1).cpu().numpy())
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))
        writeSummaries(metrDict,validBatch,writer,epoch,"train",args.model_id,args.exp_id)

def epochSeqVal(model,log_interval,loader, epoch, args,writer,metricEarlyStop,metricLastVal,maximiseMetric):
    '''
    Validate a model. This function computes several metrics and return the best value found until this point.

    Args:
     - model (torch.nn.Module): the model to validate
     - log_interval (int): the number of epochs to wait before printing a log
     - loader (load_data.TrainLoader): the train data loader
     - epoch (int): the current epoch
     - args (Namespace): the namespace containing all the arguments required for training and building the network
     - writer (tensorboardX.SummaryWriter): the writer to use to log metrics evolution to tensorboardX
     - width (int): the width of the triangular window (i.e. the number of steps over which the window is spreading)
     - metricEarlyStop (str): the name of the metric to use for early stopping. Can be any of the metrics computed in the metricDict variable of the writeSummaries function
     - metricLastVal (float): the best value of the metric to use early stopping until now
     - maximiseMetric (bool): If true, the model maximising this metric will be kept.
    '''

    model.eval()

    print("Epoch",epoch," : val")

    metrDict = {"Loss":0,"Coverage":0,"Overflow":0,"True F-score":0,"AuC":0,\
                "IoU":0,"Disc Accuracy":0,"DED":0}

    nbVideos = 0

    outDict = {}
    targDict = {}
    frameIndDict = {}
    precVidName = "None"
    videoBegining = True
    validBatch = 0
    nbVideos = 0

    for batch_idx, (data,target,vidName,frameInds) in enumerate(loader):

        newVideo = (vidName != precVidName) or videoBegining

        if (batch_idx % log_interval == 0):
            print("\t",loader.sumL+1,"/",loader.nbShots)

        if args.cuda:
            data, target,frameInds = data.cuda(), target.cuda(),frameInds.cuda()

        if args.temp_model.find("net") != -1:
            output = model.computeFeat(data).data
            print(vidName)
        else:
            if newVideo:
                output,(h,c) = model(data)
            else:
                output,(h,c) = model(data,h,c)

            output,h,c = output.data,h.data,c.data

        update.updateFrameDict(frameIndDict,frameInds,vidName)

        if newVideo and not videoBegining:
            allOutput,nbVideos = update.updateMetrics(args,model,allOutput,allTarget,precVidName,nbVideos,metrDict,outDict,targDict)
        if newVideo:
            allTarget = target
            allOutput = output
            videoBegining = False
        else:
            allTarget = torch.cat((allTarget,target),dim=1)
            allOutput = torch.cat((allOutput,output),dim=1)

        precVidName = vidName

        if nbVideos > 1 and args.debug:
            break

    if not args.debug:
        allOutput,nbVideos = update.updateMetrics(args,model,allOutput,allTarget,precVidName,nbVideos,metrDict,outDict,targDict)

    for key in outDict.keys():
        fullArr = torch.cat((frameIndDict[key].float(),outDict[key].permute(1,0)),dim=1)
        np.savetxt("../results/{}/{}_epoch{}_{}.csv".format(args.exp_id,args.model_id,epoch,key),fullArr.cpu().detach().numpy())

    writeSummaries(metrDict,validBatch,writer,epoch,"val",args.model_id,args.exp_id,nbVideos=nbVideos)

    metricVal = metrDict[metricEarlyStop]

    if metricLastVal is None:
        betterFound = True
    elif maximiseMetric and metricVal > metricLastVal:
        betterFound = True
    elif (not maximiseMetric) and metricVal < metricLastVal:
        betterFound = True
    else:
        betterFound = False

    if betterFound:
        paths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id, epoch))
        if len(paths) > 1:
            raise ValueError("More than one best model found for exp {} and model {}".format(exp_id,model_id))

        if len(paths) != 0:
            os.remove(paths[0])
        torch.save(model.state_dict(), "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, epoch))
        return metricVal,outDict,targDict

    else:
        return metricLastVal,outDict,targDict

def computeScore(model,allFeats,allTarget,valLTemp,poolTempMod,vidName):

    allOutput = None
    splitSizes = [valLTemp for _ in range(allFeats.size(1)//valLTemp)]

    if allFeats.size(1)%valLTemp > 0:
        splitSizes.append(allFeats.size(1)%valLTemp)

    chunkList = torch.split(allFeats,split_size_or_sections=splitSizes,dim=1)

    sumSize = 0

    for i in range(len(chunkList)):

        output = model.computeScore(chunkList[i])
        if type(output) is tuple:
            output = output[0]

        if allOutput is None:
            allOutput = output
        else:
            allOutput = torch.cat((allOutput,output),dim=1)

        sumSize += len(chunkList[i])

    return allOutput

def getWeights(target,classWeight):
    '''
    Compute the weights of each instance to compensate overrepresentation of the 0s in the dataset. This is done by interpolating between equal weights and perfect balancing
    Args:
     - target (torch.tensor): the target tensor
     - classWeight : a scalar determining at which point the weights will be close from equal weights or from perfect balance.
                    if classWeight = 0, the weights are set equal, if classWeight = 1, the weights are set to compensate exactly the small number of 'one' in the target tensor.
    Returns:
     - weights (torch.tensor): the weight for each instance.
    '''

    oneNb = target.sum().float()
    zeroNb = (target.numel() - oneNb).float()

    unBalWeight = torch.tensor([0.5,0.5])
    balWeight = torch.tensor([oneNb/target.numel(),zeroNb/target.numel()])

    weight = unBalWeight*(1-classWeight)+balWeight*classWeight

    weights = torch.zeros_like(target)

    weights[target==1] = weight[1]
    weights[target==0] = weight[0]

    if target.is_cuda:
        weights = weights.cuda()

    return weights

def writeSummaries(metrDict,batchNb,writer,epoch,mode,model_id,exp_id,nbVideos=None):
    ''' Write the metric computed during an evaluation in a tf writer and in a csv file

    Args:
    - total_loss (float): the loss summed over every batch. It will be divided by the number of batches to obtain the mean loss per example
    - total_cover (float): the coverage summed over every batch. Same as total_loss
    - total_overflow (float): the overflow summed over every batch. Same as total_loss
    - total_auc (float): the area under the ROC curve summed over every batch. Same as total_loss
    - total_iou (float): the IoU summed over every batch. Same as total_loss
    - batchNb (int): the total number of batches during the epoch
    - writer (tensorboardX.SummaryWriter): the writer to use to write the metrics to tensorboardX
    - mode (str): either \'train\' or \'val\' to indicate if the epoch was a training epoch or a validation epoch
    - model_id (str): the id of the model
    - exp_id (str): the experience id
    - nbVideos (int): During validation the metrics are computed over whole videos and not batches, therefore the number of videos should be indicated \
        with this argument during validation

    Returns:
    - metricDict (dict): a dictionnary containing the metrics value

    '''

    sampleNb = batchNb if mode == "train" else nbVideos

    for metric in metrDict.keys():

        if metric == "AuC":
            if mode != "train":
                metrDict["AuC"] /= sampleNb
        else:
            metrDict[metric] /= sampleNb

    metrDict["F-score"] = 2*metrDict["Coverage"]*(1-metrDict["Overflow"])/(metrDict["Coverage"]+1-metrDict["Overflow"])

    for metric in metrDict:
        writer.add_scalars(metric,{model_id+"_"+mode:metrDict[metric]},epoch)

    if not os.path.exists("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode)):
        header = [metric.lower().replace(" ","_") for metric in metrDict.keys()]
    else:
        header = ""

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]),file=text_file)

    return metrDict

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr == "RAdam":
        optimConst = radam.RAdam
        kwargs = {}
    elif optimStr != "AMSGrad":
        optimConst = getattr(torch.optim,optimStr)
        if optimStr == "SGD":
            kwargs= {'momentum': momentum}
        elif optimStr == "Adam":
            kwargs = {}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'amsgrad':True}

    return optimConst,kwargs

def initialize_Net_And_EpochNumber(net,exp_id,model_id,cuda,start_mode,init_path,init_path_visual_temp):
    '''Initialize a network

    If init is None, the network will be left unmodified. Its initial parameters will be saved.

    Args:
        net (CNN_RNN): the net to be initialised

        exp_id (string): the name of the experience
        model_id (int): the id of the network
        cuda (bool): whether to use cuda or not
        start_mode (str): a string indicating the start mode. Can be \'scratch\' or \'fine_tune\'.
        init_path (str): the path to the weight file to use to initialise. Ignored is start_mode is \'scratch\'.
        init_path_visual_temp (str): the path to the weight file to use to initialise visual and temp model. This is \
                                different from --init_path if a temporal CNN is used with a LSTM pooling, in which case \
                                the weight of the LSTM generator will be not initialised with this arg. \
                                Ignored if start_mode is \'scratch\' and if --init_path is not "None".

    Returns: the start epoch number
    '''

    if start_mode == "scratch":
        #Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id,model_id))
        startEpoch = 1
    elif start_mode == "fine_tune":


        if init_path != "None":
            params = torch.load(init_path,map_location=torch.device('cpu') if not cuda else torch.device("cuda0"))

            state_dict = {k.replace("module.cnn.","cnn.module.").replace("scoreConv.weight","scoreConv.layers.weight").replace("scoreConv.bias","scoreConv.layers.bias"): v for k,v in params.items()}

            paramToRemove = []
            for param in state_dict.keys():
                if param.find("frameAtt") != -1:
                    paramToRemove.append(param)
            for param in paramToRemove:
                state_dict.pop(param)

            net.load_state_dict(state_dict)
            startEpoch = utils.findLastNumbers(init_path)
        else:

            params = torch.load(init_path_visual_temp)
            for key in params.keys():

                if cuda:
                    params[key] = params[key].cuda()

                if key in net.state_dict().keys():
                    net.state_dict()[key].data += params[key].data -net.state_dict()[key].data

                startEpoch = utils.findLastNumbers(init_path_visual_temp)

    return startEpoch

def resetAdvIter(kwargsTr):
    if not kwargsTr["discrLoader"] is None:
        kwargsTr["discrIter"] = iter(kwargsTr["discrLoader"])
    else:
        kwargsTr["discrIter"] = None
    return kwargsTr

def evalAllImages(exp_id,model_id,model,testLoader,cuda,log_interval):
    '''
    Pass all the images and/or the sound extracts of a loader in a feature model and save the feature vector in one csv for each image.
    Args:
    - exp_id (str): The experience id
    - model (nn.Module): the model to process the images
    - testLoader (load_data.TestLoader): the image and/or sound loader
    - cuda (bool): True is the computation has to be done on cuda
    - log_interval (int): the number of batches to wait before logging progression
    '''

    for batch_idx, (data, _,vidName,frameInds) in enumerate(testLoader):

        if (batch_idx % log_interval == 0):
            print("\t",testLoader.sumL+1,"/",testLoader.nbShots)

        if not data is None:
            if cuda:
                data = data.cuda()
            data = data[:,:len(frameInds)]
            data = data.view(data.size(0)*data.size(1),data.size(2),data.size(3),data.size(4))

        if not os.path.exists("../results/{}/{}".format(exp_id,vidName)):
            os.makedirs("../results/{}/{}".format(exp_id,vidName))

        feats = model(data)
        for i,feat in enumerate(feats):
            imageName = frameInds[i]
            if not os.path.exists("../results/{}/{}/{}_{}.csv".format(exp_id,vidName,imageName,model_id)):

                np.savetxt("../results/{}/{}/{}_{}.csv".format(exp_id,vidName,imageName,model_id),feat.detach().cpu().numpy())

def addInitArgs(argreader):
    argreader.parser.add_argument('--start_mode', type=str,metavar='SM',
                help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.')
    argreader.parser.add_argument('--init_path', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the network')
    argreader.parser.add_argument('--init_path_visual', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the visual model')
    argreader.parser.add_argument('--init_path_visual_temp', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the visual and the temporal model')

    return argreader
def addLossArgs(argreader):

    argreader.parser.add_argument('--nll_weight', type=float, metavar='NLLWEIGHT',
                        help='The weight of the nll likelihoos term.')

    argreader.parser.add_argument('--class_weight', type=float, metavar='CW',
                        help='Set the importance of balancing according to class instance number in the loss function. 0 makes equal weights and 1 \
                        makes weights proportional to the class instance number of the other class.')

    argreader.parser.add_argument('--dist_weight', type=float,metavar='DW',
                        help="The weight of the distance to scene change term in the loss function")

    argreader.parser.add_argument('--adv_weight', type=float,metavar='DW',
                        help="The weight of the adversarial term in the loss function. This term penalise the model \
                            if the discriminator is able to find if the shots comes from the training dataset or the \
                            auxilliary dataset (see --dataset_adv in load_data.py)")

    argreader.parser.add_argument('--siam_weight', type=float,metavar='DW',
                        help="The weight of the siamese term in the loss function. This term penalise the model \
                            if the features are too close when they belong to different scene and if they are too\
                            far away when they belong to the same scenes.")

    argreader.parser.add_argument('--siam_margin', type=float,metavar='DW',
                        help="The margin of the siamese term. The model is penalized is the distance between features \
                            belonging to the same scene is above this argument.")

    argreader.parser.add_argument('--siam_nb_samples', type=int,metavar='DW',
                        help="The number of feature pairs to build at each batch.")

    argreader.parser.add_argument('--iou_weight', type=float,metavar='FLOAT',
                        help="The weight of the IoU term.")

    return argreader
def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=args.str2FloatList,metavar='LR',
                        help='learning rate (it can be a schedule : --lr 0.01,0.001,0.0001)')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                        help='SGD momentum')
    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                        help='the optimizer to use (default: \'SGD\')')
    return argreader
def addValArgs(argreader):
    argreader.parser.add_argument('--train_step_to_ignore', type=int,metavar='LMAX',
                    help='Number of steps that will be ignored at the begining and at the end of the training sequence for binary cross entropy computation')

    argreader.parser.add_argument('--val_l_temp', type=int,metavar='LMAX',help='Length of sequences for computation of scores when using a CNN temp model.')

    argreader.parser.add_argument('--metric_early_stop', type=str,metavar='METR',
                    help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_metric', type=args.str2bool,metavar='BOOL',
                    help='If true, The chosen metric for chosing the best model will be maximised')

    argreader.parser.add_argument('--compute_val_metrics', type=args.str2bool,metavar='BOOL',
                    help='If false, the metrics will not be computed during validation, but the scores produced by the models will still be saved')

    return argreader

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--comp_feat', action='store_true',help='To compute and write in a file the features of all images in the test set. All the arguments used to \
                                    build the model and the test data loader should be set.')
    argreader.parser.add_argument('--no_train', type=str,nargs=2,help='To use to re-evaluate a model at each epoch after training. At each epoch, the model is not trained but \
                                                                            the weights of the corresponding epoch are loaded and then the model is evaluated.\
                                                                            The values of this argument are the exp_id and the model_id of the model to get the weights from.')

    argreader = addInitArgs(argreader)
    argreader = addLossArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.redirect_out:
        sys.stdout = open("python.out", 'w')

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id,args.model_id))

    writer = SummaryWriter("../results/{}".format(args.exp_id))

    print("Model :",args.model_id,"Experience :",args.exp_id)

    if args.comp_feat:

        testLoader = load_data.TestLoader(args.val_l,args.dataset_test,args.test_part_beg,args.test_part_end,args.img_size,\
                                          args.resize_image,args.exp_id,args.random_frame_val)

        if args.feat != "None":
            featModel = modelBuilder.buildFeatModel(args.feat,args.pretrain_dataset,args.lay_feat_cut)
            if args.cuda:
                featModel = featModel.cuda()
            if args.init_path_visual != "None":
                featModel.load_state_dict(torch.load(args.init_path_visual))
            elif args.init_path != "None":
                model = modelBuilder.netBuilder(args)
                params = torch.load(args.init_path)
                state_dict = {k.replace("module.cnn.","cnn.module."): v for k,v in params.items()}
                model.load_state_dict(state_dict)
                featModel = model.featModel

            featModel.eval()
        else:
            featModel = None

        with torch.no_grad():
            evalAllImages(args.exp_id,args.model_id,featModel,testLoader,args.cuda,args.log_interval)

    else:

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        paramToOpti = []


        trainLoader,trainDataset = load_data.buildSeqTrainLoader(args)

        valLoader = load_data.TestLoader(args.val_l,args.dataset_val,args.val_part_beg,args.val_part_end,\
                                            args.img_size,args.resize_image,\
                                            args.exp_id,args.random_frame_val)

        #Building the net
        net = modelBuilder.netBuilder(args)


        if args.cuda:
            net = net.cuda()


        trainFunc = epochSeqTr
        valFunc = epochSeqVal

        kwargsTr = {'log_interval':args.log_interval,'loader':trainLoader,'args':args,'writer':writer}
        kwargsVal = kwargsTr.copy()

        kwargsVal['loader'] = valLoader
        kwargsVal.update({"metricEarlyStop":args.metric_early_stop,"maximiseMetric":args.maximise_metric})

        if args.adv_weight > 0:
            kwargsTr["discrModel"] = modelBuilder.Discriminator(net.nbFeat,args.discr_dropout)
            kwargsTr["discrModel"] = kwargsTr["discrModel"].cuda() if args.cuda else kwargsTr["discrModel"].cpu()
            kwargsTr["discrLoader"] = load_data.buildFrameTrainLoader(args)
            kwargsTr["discrOptim"] = torch.optim.SGD(kwargsTr["discrModel"].parameters(), lr=args.lr,momentum=args.momentum)
        else:
            kwargsTr["discrModel"],kwargsTr["discrLoader"],kwargsTr["discrOptim"] = None,None,None

        for p in net.parameters():
            paramToOpti.append(p)

        paramToOpti = (p for p in paramToOpti)

        #Getting the contructor and the kwargs for the choosen optimizer
        optimConst,kwargsOpti = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

        startEpoch = initialize_Net_And_EpochNumber(net,args.exp_id,args.model_id,args.cuda,args.start_mode,args.init_path,args.init_path_visual_temp)

        #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
        #the args.lr argument will be a float and not a float list.
        #Converting it to a list with one element makes the rest of processing easier
        if type(args.lr) is float:
            args.lr = [args.lr]

        lrCounter = 0

        metricLastVal = None

        outDictEpochs = {}
        targDictEpochs = {}

        for epoch in range(startEpoch, args.epochs + 1):

            kwargsOpti,kwargsTr,lrCounter = update.updateLR(epoch,args.epochs,args.lr,startEpoch,kwargsOpti,kwargsTr,lrCounter,net,optimConst)

            kwargsTr["epoch"],kwargsVal["epoch"] = epoch,epoch
            kwargsTr["model"],kwargsVal["model"] = net,net

            kwargsTr = resetAdvIter(kwargsTr)

            if not args.no_train:
                trainFunc(**kwargsTr)
            else:
                net.load_state_dict(torch.load("../models/{}/model{}_epoch{}".format(args.no_train[0],args.no_train[1],epoch)))

            kwargsVal["metricLastVal"] = metricLastVal

            #Checking if validation has already been done
            if len(glob.glob("../results/{}/{}_epoch{}_*".format(args.exp_id,args.model_id,epoch))) < len(kwargsVal["loader"].videoPaths):
                with torch.no_grad():
                    metricLastVal,outDict,targDict = valFunc(**kwargsVal)
                outDictEpochs[epoch] = outDict
                targDictEpochs[epoch] = targDict
                update.updateHist(writer,args.model_id,outDictEpochs,targDictEpochs)
            else:
                print("Validation epoch {} already done !".format(epoch))

if __name__ == "__main__":
    main()
