from args import ArgReader
from args import str2bool
import modelBuilder
import glob
import numpy as np
import load_data
import torch
import load_data
import os
import sys

import processResults
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from torch.nn import DataParallel
import gc
import torch.backends.cudnn as cudnn

import subprocess
from sklearn.metrics import roc_auc_score
torch.backends.cudnn.benchmark = True

def softTarget(predBatch,targetBatch,width):
    ''' Convolve the target with a triangular window to make it smoother

    Args :
    - predBatch (torch.tensor): the batch of prediction
    - targetBatch (torch.tensor): the batch of target
    - width (int): the width of the triangular window (i.e. the number of steps over which the window is spreading)

    Returns
    - softTargetBatch (torch.tensor): the batch of soft targets
    '''

    device = predBatch.device
    softTargetBatch = torch.zeros_like(targetBatch)

    for i,target in enumerate(targetBatch):

        inds = torch.arange(len(target)).unsqueeze(1).to(device).float()

        sceneChangeInds = target.nonzero().view(-1).unsqueeze(0).float()

        #A matrix containing the triangular window applied to each one of the target
        softTarg = F.relu(1-(1.0/(width+1))*torch.abs(inds-sceneChangeInds))

        #Agregates the columns of the matrix
        softTarg = softTarg.mean(dim=1)

        #By doing this, the element in the original target equal to 1 stays equal to 1
        softTargetBatch[i] = torch.max(softTarg,target.float())

    return softTargetBatch

def baselineAllVids(dataset,batchSize,visualFeat,audioFeat,pretrainSet,cuda,intervToProcess,audioLen):

    #Extract the middle frame first
    load_data.getMiddleFrames(dataset,audioLen)

    vidPaths = np.array(sorted(glob.glob("../data/{}/*/middleFrames/".format(dataset))),dtype=str)[intervToProcess]

    for vidPath in vidPaths:
        print(vidPath)

        imagePathList = np.array(sorted(glob.glob(vidPath+"/*"),key=modelBuilder.findNumbers),dtype=str)
        imagePathList = list(filter(lambda x:x.find(".wav") == -1,imagePathList))

        if audioFeat != "None":
            audioPathList = np.array(sorted(glob.glob(vidPath+"/*.wav"),key=modelBuilder.findNumbers),dtype=str)
        else:
            audioPathList = None

        diagBlock = modelBuilder.DiagBlock(cuda=cuda,batchSize=batchSize,feat=visualFeat,pretrainDataSet=pretrainSet,audioFeat=audioFeat)

        diagBlock.detectDiagBlock(imagePathList,audioPathList,"test_exp",1)

def epochSiam(model,optim,log_interval,loader, epoch, args,writer,kwargs,mode):

    model.train()

    print("Epoch",epoch," : ",mode)

    total_loss = 0
    total_posDist = 0
    total_negDist = 0
    correct =0
    repList = None

    if not kwargs["audioModel"] is None:
        audioModel = kwargs["audioModel"]

    if kwargs["mining_mode"] == "offline":

        for batch_idx, (_,anch,pos,neg,anchAudio,posAudio,negAudio,_,_,_) in enumerate(loader):

            if (batch_idx % log_interval == 0):
                print("\t",batch_idx+1,"/",loader.batchNb)

            if args.cuda:
                anch,pos,neg = anch.cuda(),pos.cuda(),neg.cuda()

            anchRep = model(anch)
            posRep = model(pos)
            negRep = model(neg)

            if not kwargs["audioModel"] is None:

                if args.cuda:
                    anchAudio,posAudio,negAudio = anchAudio.cuda(),posAudio.cuda(),negAudio.cuda()

                anchAudRep = audioModel(anchAudio)
                posAudRep = audioModel(posAudio)
                negAudRep = audioModel(negAudio)

                anchRep = torch.cat((anchRep,anchAudRep),dim=1)
                posRep = torch.cat((posRep,posAudRep),dim=1)
                negRep = torch.cat((negRep,negAudRep),dim=1)

            #Loss
            p = kwargs["dist_order"]
            margin = kwargs["margin"]
            loss = torch.nn.functional.triplet_margin_loss(anchRep, posRep, negRep,margin=margin,p=p)

            posDist = torch.pow(torch.pow(anchRep-posRep,p).sum(dim=1),1/p)
            negDist = torch.pow(torch.pow(anchRep-negRep,p).sum(dim=1),1/p)

            correct += (posDist < negDist-margin).sum().float()/posDist.size(0)

            total_posDist += posDist.mean()
            total_negDist += negDist.mean()

            if mode == "train":
                loss.backward()
                optim.step()
                optim.zero_grad()
            else:
                loss.backward()
                optim.zero_grad()

            #Metrics
            total_loss += loss

    if mode == "train":
        torch.save(model.state_dict(), "../models/{}/{}_siam_epoch{}".format(args.exp_id,args.model_id, epoch))
        torch.save(audioModel.state_dict(), "../models/{}/{}_siam_audio_epoch{}".format(args.exp_id,args.model_id, epoch))

    writeSummariesSiam(total_loss,correct,total_posDist,total_negDist,batch_idx+1,writer,epoch,mode,args.model_id,args.exp_id)

def compDistMat(feat):

    featCol = feat.unsqueeze(1)
    featRow = feat.unsqueeze(0)

    featCol = featCol.expand(feat.size(0),feat.size(0),feat.size(1))
    featRow = featRow.expand(feat.size(0),feat.size(0),feat.size(1))

    simMatrix = torch.pow(featCol-featRow,2).sum(dim=2)
def epochSeqTr(model,optim,log_interval,loader, epoch, args,writer,width):
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

    total_loss = 0
    total_cover = 0
    total_overflow = 0
    total_auc = 0
    total_iou = 0
    validBatch = 0

    allOut = None
    allGT = None

    for batch_idx, (data, audio,target,vidNames) in enumerate(loader):

        if target.sum() > 0:

            if (batch_idx % log_interval == 0):
                print("\t",loader.sumL+1,"/",loader.nbShots)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                if not audio is None:
                    audio = audio.cuda()

            if args.temp_model.find("resnet") != -1:
                output = model(data,audio)
            else:
                output,_ = model(data,audio)

            #Metrics
            pred = output.data > 0.5
            cov,overflow,iou = processResults.binaryToMetrics(pred,target)

            total_cover += cov
            total_overflow += overflow
            total_iou += iou

            if allOut is None:
                allOut = output.data
                allGT = target
            else:
                allOut = torch.cat((allOut,output.data),dim=-1)
                allGT = torch.cat((allGT,target),dim=-1)

            #Loss
            if args.soft_loss:
                target = softTarget(output,target,width)
                weights = None
            else:
                weights = getWeights(target,args.class_weight)

            loss = F.binary_cross_entropy(output, target,weight=weights)

            loss.backward()
            optim.step()
            optim.zero_grad()

            total_loss += loss.detach().data.item()
            validBatch += 1

            if validBatch > 3 and args.debug:
                break

    total_auc = roc_auc_score(allGT.view(-1).cpu().numpy(),allOut.view(-1).cpu().numpy())

    torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))
    writeSummaries(total_loss,total_cover,total_overflow,total_auc,total_iou,validBatch,writer,epoch,"train",args.model_id,args.exp_id)

def epochSeqVal(model,log_interval,loader, epoch, args,writer,width,metricEarlyStop,metricLastVal,maximiseMetric):
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

    total_loss,total_cover,total_overflow,total_auc,total_iou,nbVideos = 0,0,0,0,0,0

    outDict = {}
    frameIndDict = {}
    currVidName = "None"
    videoBegining = True
    validBatch = 0
    nbVideos = 0

    for batch_idx, (data, audio,target,vidName,frameInds) in enumerate(loader):

        newVideo = (vidName != currVidName) or videoBegining
        currVidName = vidName

        if (batch_idx % log_interval == 0):
            print("\t",loader.sumL+1,"/",loader.nbShots)

        if args.cuda:
            data, target,frameInds = data.cuda(), target.cuda(),frameInds.cuda()
            if not audio is None:
                audio = audio.cuda()
        #print(data.size())

        if args.temp_model.find("resnet") != -1:
            output = model(data,audio).data
        else:
            if newVideo:
                output,(h,c) = model(data,audio)
            else:
                output,(h,c) = model(data,audio,h,c)

            output,h,c = output.data,h.data,c.data

        updateOutDict(outDict,output,frameIndDict,frameInds,vidName)

        #loss.backward()
        #optim.zero_grad()

        if batch_idx > 3 and args.debug:
            break

        #Metrics
        if newVideo and not videoBegining:
            cov,overflow,iou = processResults.binaryToMetrics(allOutput>0.5,allTarget)
            total_cover += cov
            total_overflow += overflow
            total_iou += iou

            if args.soft_loss:
                weights = None
            else:
                weights = getWeights(allTarget,args.class_weight)

            total_auc += roc_auc_score(allTarget.view(-1).cpu().numpy(),allOutput.view(-1).cpu().numpy())

            #Loss
            if args.soft_loss:
                allTarget = softTarget(allOutput,allTarget,width)

            loss = F.binary_cross_entropy(allOutput,allTarget,weight=weights).data.item()
            total_loss += loss

            nbVideos += 1

        if newVideo:
            allTarget = target
            allOutput = output
            videoBegining = False
        else:
            allTarget = torch.cat((allTarget,target),dim=1)
            allOutput = torch.cat((allOutput,output),dim=1)

    for key in outDict.keys():
        fullArr = torch.cat((frameIndDict[key].float(),outDict[key].unsqueeze(1)),dim=1)
        np.savetxt("../results/{}/{}_epoch{}_{}.csv".format(args.exp_id,args.model_id,epoch,key),fullArr.cpu().detach().numpy())

    metricDict = writeSummaries(total_loss,total_cover,total_overflow,total_auc,total_iou,validBatch,writer,epoch,"vali",args.model_id,args.exp_id,nbVideos=nbVideos)

    metricVal = metricDict[metricEarlyStop]

    if metricLastVal is None:
        betterFound = True
    elif maximiseMetric and metricVal > metricLastVal:
        betterFound = True
    elif (not maximiseMetric) and metricVal < metricLastVal:
        betterFound = True
    else:
        betterFound = False


    if betterFound:
        print("Better {} found : {}".format(metricEarlyStop,metricLastVal))
        paths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id, epoch))
        if len(paths) > 1:
            raise ValueError("More than one best model found for exp {} and model {}".format(exp_id,model_id))

        if len(paths) != 0:
            os.remove(paths[0])
        torch.save(model.state_dict(), "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, epoch))
        return metricVal

    else:
        return metricLastVal

def updateOutDict(outDict,output,frameIndDict,frameInds,vidName):
    ''' Store the prediction of a model in a dictionnary with one entry per movie

    Args:
     - outDict (dict): the dictionnary where the scores will be stored
     - output (torch.tensor): the output batch of the model
     - frameIndDict (dict): a dictionnary collecting the index of each frame used
     - vidName (str): the name of the video from which the score are produced

    '''

    if vidName in outDict.keys():
        outDict[vidName] = torch.cat((output[0,:],outDict[vidName]),dim=0)

        reshFrInds = frameInds.view(len(output[0,:]),-1).clone()
        frameIndDict[vidName] = torch.cat((frameIndDict[vidName],reshFrInds),dim=0)

    else:
        outDict[vidName] = output[0,:]
        frameIndDict[vidName] = frameInds.view(len(output[0,:]),-1).clone()

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

def writeSummariesSiam(total_loss,correct,total_posDist,total_negDist,batchNb,writer,epoch,mode,model_id,exp_id):
    '''


    '''


    total_loss /= batchNb
    total_posDist /= batchNb
    total_negDist /= batchNb
    accuracy = correct/batchNb

    writer.add_scalars('Losses',{model_id+"_"+mode:total_loss},epoch)
    writer.add_scalars('Accuracies',{model_id+"_"+mode:accuracy},epoch)
    writer.add_scalars('Pos_dist',{model_id+"_"+mode:total_posDist},epoch)
    writer.add_scalars('Neg_dist',{model_id+"_"+mode:total_negDist},epoch)

    if not os.path.exists("../results/{}/{}_siam_epoch{}_metrics_{}.csv".format(model_id,epoch,mode)):
        header = "epoch,loss,accuracy,posDist,negDist"
    else:
        header = ""

    with open("../results/{}/{}_siam_epoch{}_metrics_{}.csv".format(model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print("{},{},{},{},{}".format(epoch,total_loss,accuracy,total_posDist,total_negDist),file=text_file)

def writeSummaries(total_loss,total_cover,total_overflow,total_auc,total_iou,validBatch,writer,epoch,mode,model_id,exp_id,nbVideos=None):

    sampleNb = validBatch if mode == "train" else nbVideos
    total_loss /= sampleNb
    total_iou /= sampleNb
    total_cover /= sampleNb
    total_overflow /= sampleNb
    f_score = 2*total_cover*(1-total_overflow)/(total_cover+1-total_overflow)

    if mode != "train":
        total_auc /= sampleNb

    metricDict = {'Losse':total_loss,'Coverage':total_cover,'Overflow':total_overflow,'F-score':f_score,'AuC':total_auc,'IoU':total_iou}

    for metric in metricDict:
        writer.add_scalars(metric,{model_id+"_"+mode:metricDict[metric]},epoch)

    if not os.path.exists("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode)):
        header = "epoch,loss,coverage,overflow,f-score,auc,iou"
    else:
        header = ""

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print("{},{},{},{},{},{},{}\n".format(epoch,total_loss,total_cover,total_overflow,f_score,total_auc,total_iou),file=text_file)

    return metricDict

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
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

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)

def initialize_Net_And_EpochNumber(net,exp_id,model_id,cuda,start_mode,init_path):
    '''Initialize a network

    If init is None, the network will be left unmodified. Its initial parameters will be saved.

    Args:
        net (CNN_RNN): the net to be initialised

        exp_id (string): the name of the experience
        model_id (int): the id of the network
        cuda (bool): whether to use cuda or not
        start_mode (str): a string indicating the start mode. Can be \'scratch\' or \'fine_tune\'.
        init_path (str): the path to the weight file of the net to use to initialise. Ignored is start_mode is \'scratch\'.

    Returns: the start epoch number
    '''

    if start_mode == "scratch":
        #Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id,model_id))
        startEpoch = 1
    elif start_mode == "fine_tune":

        params = torch.load(init_path)

        state_dict = {k.replace("module.cnn.","cnn.module."): v for k,v in params.items()}

        net.load_state_dict(state_dict)
        startEpoch = findLastNumbers(init_path)

    return startEpoch

def evalAllImages(exp_id,model_id,model,audioModel,dataset_test,testLoader,cuda,log_interval):

    for batch_idx, (data,audio, _,vidName,frameInds) in enumerate(testLoader):


        print(vidName,frameInds.shape,frameInds)

        if (batch_idx % log_interval == 0):
            print("\t",testLoader.sumL+1,"/",testLoader.nbShots)

        if not data is None:
            if cuda:
                data = data.cuda()
            data = data[:,:len(frameInds)]
            data = data.view(data.size(0)*data.size(1),data.size(2),data.size(3),data.size(4))

        if not audio is None:
            if cuda:
                audio = audio.cuda()
            audio = audio[:,:len(frameInds)]
            audio = audio.view(audio.size(0)*audio.size(1),audio.size(2),audio.size(3),audio.size(4))

        if not os.path.exists("../results/{}/{}".format(exp_id,vidName)):
            os.makedirs("../results/{}/{}".format(exp_id,vidName))

        if (not audioModel is None) and (not model is None):
            audioFeats = audioModel(audio)
            feats = model(data)
            feats = torch.cat((feats,audioFeats),dim=-1)
        elif (not audioModel is None):
            audioFeats = audioModel(audio)
            feats = audioFeats
        elif (not model is None):
            feats = model(data)
        for i,feat in enumerate(feats):
            imageName = frameInds[i]
            if not os.path.exists("../results/{}/{}/{}_{}.csv".format(exp_id,vidName,imageName,model_id)):

                np.savetxt("../results/{}/{}/{}_{}.csv".format(exp_id,vidName,imageName,model_id),feat.detach().cpu().numpy())

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--baseline', type=int,nargs=2, metavar='N',help='To run the baseline on every video of the dataset. The --dataset argument must\
                                                                        also be used. The values of the argument are the index of the first and the last video to process.')

    argreader.parser.add_argument('--comp_feat', action='store_true',help='To compute and write in a file the features of all images in the test set')
    argreader.parser.add_argument('--train_siam', action='store_true',help='To train a siamese network instead of a CNN-RNN')

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

    if args.baseline:
        baselineAllVids(args.dataset_test,args.batch_size,args.feat,args.feat_audio,args.pretrain_dataset,args.cuda,args.baseline,args.audio_len)
    elif args.comp_feat:

        img_size = (args.img_width,args.img_heigth)
        testLoader = load_data.TestLoader(args.val_l,args.dataset_test,args.test_part_beg,args.test_part_end,(args.img_width,args.img_heigth),args.audio_len,args.resize_image,args.frames_per_shot,args.exp_id)

        if args.feat != "None":
            featModel = modelBuilder.buildFeatModel(args.feat,args.pretrain_dataset,args.lay_feat_cut)
            if args.cuda:
                featModel = featModel.cuda()
            if args.init_path_visual != "None":
                featModel.load_state_dict(torch.load(args.init_path_visual))
            elif args.init_path != "None":
                model = modelBuilder.netBuilder(args)
                model.load_state_dict(torch.load(args.init_path))
                featModel = model.featModel

            featModel.eval()
        else:
            featModel = None

        if args.feat_audio != "None":
            audioFeatModel = modelBuilder.buildAudioFeatModel(args.feat_audio)
            if args.cuda:
                audioFeatModel = audioFeatModel.cuda()
            if args.init_path_audio != "None":
                audioFeatModel.load_state_dict(torch.load(args.init_path_audio))
            elif args.init_path != "None":
                model = modelBuilder.netBuilder(args)
                model.load_state_dict(torch.load(args.init_path))
                audioFeatModel = model.audioFeatModel

            audioFeatModel.eval()
        else:
            audioFeatModel = None

        evalAllImages(args.exp_id,args.model_id,featModel,audioFeatModel,args.dataset_test,testLoader,args.cuda,args.log_interval)

    else:

        if args.img_width == -1 or args.img_heigth == -1:
            img_size = None
        else:
            img_size = (args.img_width,args.img_heigth)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        if args.feat_audio != "None":
            audioLen = args.audio_len
        else:
            audioLen = 0
        paramToOpti = []

        if args.train_siam:

            trainLoader = load_data.PairLoader(args.dataset_train,args.batch_size,img_size,args.train_part_beg,args.train_part_end,True,audioLen,args.resize_image)
            valLoader = load_data.PairLoader(args.dataset_val,args.val_batch_size,img_size,args.val_part_beg,args.val_part_end,True,audioLen,args.resize_image)

            net = modelBuilder.buildFeatModel(args.feat,args.pretrain_dataset,args.lay_feat_cut)

            trainFunc = epochSiam
            valFunc = epochSiam
            if args.feat_audio != "None":
                audioNet = modelBuilder.buildAudioFeatModel(args.feat_audio)

                for p in audioNet.parameters():
                    paramToOpti.append(p)

                    model,optim,log_interval,loader, epoch, args,writer,kwargs,mode

            kwargsTr = {'model':model,'optim':optim,'log_interval':log_interval,'loader':loader,'epoch':epoch,'args':args,'writer':writer,'mode':'train',\
                    'margin':args.margin,"dist_order":args.dist_order,"mining_mode":args.mining_mode,"audioModel":audioNet}

            kwargsVal = kwargsTr
            kwargsVal["mode"] = 'val'

            if args.cuda and args.feat_audio != "None":
                audioNet = audioNet.cuda()
        else:

            trainLoader = load_data.TrainLoader(args.batch_size,args.dataset_train,args.train_part_beg,args.train_part_end,args.l_min,args.l_max,\
                                                (args.img_width,args.img_heigth),audioLen,args.resize_image,args.frames_per_shot,args.exp_id)

            valLoader = load_data.TestLoader(args.val_l,args.dataset_val,args.test_part_beg,args.test_part_end,\
                                                (args.img_width,args.img_heigth),audioLen,args.resize_image,args.frames_per_shot,args.exp_id)

            #Building the net
            net = modelBuilder.netBuilder(args)

            trainFunc = epochSeqTr
            valFunc = epochSeqVal

            kwargsTr = {'model':model,'optim':optim,'log_interval':log_interval,'loader':loader,'epoch':epoch,'args':args,'writer':writer,'width':width}
            kwargsVal = kwargsTr

            kwargsVal.update({"metricEarlyStop":metricEarlyStop,"metricLastVal":metricLastVal,"maximiseMetric":maximiseMetric})
            kwargsVal.pop("optim")

        for p in net.parameters():
            paramToOpti.append(p)

        paramToOpti = (p for p in paramToOpti)

        img_size = (args.img_width,args.img_heigth)

        if args.cuda:
            net = net.cuda()

        #Getting the contructor and the kwargs for the choosen optimizer
        optimConst,kwargsOpti = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

        startEpoch = initialize_Net_And_EpochNumber(net,args.exp_id,args.model_id,args.cuda,args.start_mode,args.init_path)

        #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
        #the args.lr argument will be a float and not a float list.
        #Converting it to a list with one element makes the rest of processing easier
        if type(args.lr) is float:
            args.lr = [args.lr]

        if type(args.soft_loss_width) is float:
            args.soft_loss_width = [args.soft_loss_width]

        lrCounter = 0
        widthCounter = 0
        metricLastVal = None

        for epoch in range(startEpoch, args.epochs + 1):

            #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
            #The optimiser have to be rebuilt every time the learning rate is updated
            if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

                kwargsOpti['lr'] = args.lr[lrCounter]
                if args.train_visual:
                    optim = optimConst(net.parameters(), **kwargsOpti)
                else:
                    optim = optimConst(net.getParams(), **kwargsOpti)

                if lrCounter<len(args.lr)-1:
                    lrCounter += 1

            if (epoch-1) % ((args.epochs + 1)//len(args.soft_loss_width)) == 0 or epoch==startEpoch:

                width = args.soft_loss_width[widthCounter]
                if widthCounter<len(args.soft_loss_width)-1:
                    widthCounter += 1

            trainFunc(**kwargsTr)
            metricLastVal = valFunc(**kwargsVal)

            if not args.train_siam:
                kwargsVal["metricLastVal"] = metricLastVal

            kwargsTr["epoch"],kwargsVal["epoch"] = epoch,epoch
            kwargsTr["model"],kwargsVal["model"] = model,model

            #val(net,optim,valLoader,epoch,args,writer,5)

if __name__ == "__main__":
    main()
