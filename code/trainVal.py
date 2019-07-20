from args import ArgReader
from args import str2bool
import args
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
torch.backends.cudnn.enabled = True
import time
from skimage.transform import resize

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sys
from PIL import Image

def sparsiRew(output,wind,thres):

    weight = torch.ones(1,1,wind).to(output.device)
    output = output.unsqueeze(1)
    avg_output = torch.nn.functional.conv1d(output, weight,padding=wind//2)
    avg_output = F.relu(avg_output - thres).sum()

    return avg_output

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

    hmDict = {}

    for batch_idx, (data, audio,target,vidNames) in enumerate(loader):

        if target.sum() > 0:

            if (batch_idx % log_interval == 0):
                print("\t",batch_idx*len(data)*len(target[0]),"/",len(loader.dataset))

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                if not audio[0] is None:
                    audio = audio.cuda()

            if args.temp_model.find("net") != -1:
                if args.pool_temp_mod == "lstm" or args.pool_temp_mod == "cnn":
                    output = model(data,audio,None,None,target)
                else:
                    output = model(data,audio)
            else:
                output,_ = model(data,audio)

            #Loss
            if args.pool_temp_mod == 'lstm' or args.pool_temp_mod=="cnn":
                loss = -processResults.continuousIoU(output, target)
            elif args.soft_loss:
                output = output[:,args.train_step_to_ignore:output.size(1)-args.train_step_to_ignore]
                target = target[:,args.train_step_to_ignore:output.size(1)-args.train_step_to_ignore]

                softTarg = softTarget(output,target,width)
                loss = F.binary_cross_entropy(output, softTarg)
            else:
                output = output[:,args.train_step_to_ignore:output.size(1)-args.train_step_to_ignore]
                target = target[:,args.train_step_to_ignore:target.size(1)-args.train_step_to_ignore]

                weights = getWeights(target,args.class_weight)
                loss = F.binary_cross_entropy(output, target,weight=weights)

            if args.sparsi_weig > 0:
                loss += args.sparsi_weig*sparsiRew(output,args.sparsi_wind,args.sparsi_thres)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if args.pool_temp_mod == "lstm" or args.pool_temp_mod=="cnn":
                output = processResults.scenePropToBinary(output,data.size(1))

            #Metrics
            pred = output.data > 0.5
            cov,overflow,iou = processResults.binaryToMetrics(pred,target)

            total_cover += cov
            total_overflow += overflow
            total_iou += iou

            #Store the f score of each example
            updateHMDict(hmDict,cov,overflow,vidNames)

            if allOut is None:
                allOut = output.data
                allGT = target
            else:
                allOut = torch.cat((allOut,output.data),dim=-1)
                allGT = torch.cat((allGT,target),dim=-1)

            total_loss += loss.detach().data.item()
            validBatch += 1

            if validBatch > 3 and args.debug:
                break

    #If the training set is empty (which we might want to for kust evaluate the model), then allOut and allGT will still be None
    if not allGT is None:
        total_auc = roc_auc_score(allGT.view(-1).cpu().numpy(),allOut.view(-1).cpu().numpy())
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))
        writeSummaries(total_loss,total_cover,total_overflow,total_auc,total_iou,validBatch,writer,epoch,"train",args.model_id,args.exp_id)

    #Compute the mean f-score of the all the videos processed during this training epoch
    agregateHMDict(hmDict)

    return hmDict

def updateHMDict(hmDict,coverage,overflow,vidNames):

    for vidName in vidNames:
        f_score = 2*coverage*(1-overflow)/(coverage+1-overflow)

        if vidName in hmDict.keys():
            hmDict[vidName].append(f_score)
        else:
            hmDict[vidName] = [f_score]

def agregateHMDict(hmDict):

    for vidName in hmDict.keys():
        hmDict[vidName] = np.array(hmDict[vidName]).mean()

def updateMetrics(args,model,allOutput,allTarget,precVidName,width,nbVideos,total_loss,total_cover,total_overflow,total_iou,total_auc,outDict,targDict):
    if args.temp_model.find("net") != -1:
        allOutput = computeScore(model,allOutput,allTarget,args.val_l_temp,args.pool_temp_mod,args.val_l_temp_overlap,precVidName)

    if args.pool_temp_mod == 'lstm' or args.pool_temp_mod == 'cnn'  :
        loss = -processResults.continuousIoU(allOutput, allTarget)
    elif args.soft_loss:
        softAllTarget = softTarget(allOutput,allTarget,width)
        loss = F.binary_cross_entropy(allOutput,softAllTarget).data.item()
    else:
        weights = getWeights(allTarget,args.class_weight)
        loss = F.binary_cross_entropy(allOutput,allTarget,weight=weights).data.item()

    if args.sparsi_weig > 0:
        loss += args.sparsi_weig*sparsiRew(allOutput.data,args.sparsi_wind,args.sparsi_thres)

    total_loss += loss

    if args.pool_temp_mod=="lstm" or args.pool_temp_mod=="cnn":
        allOutput = processResults.scenePropToBinary(allOutput,allTarget.size(1))

    outDict[precVidName] = allOutput
    targDict[precVidName] = allTarget

    cov,overflow,iou = processResults.binaryToMetrics(allOutput>0.5,allTarget)
    total_cover += cov
    total_overflow += overflow
    total_iou += iou

    total_auc += roc_auc_score(allTarget.view(-1).cpu().numpy(),allOutput.view(-1).cpu().numpy())

    nbVideos += 1

    return allOutput,nbVideos,total_loss,total_cover,total_overflow,total_iou,total_auc

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
    targDict = {}
    frameIndDict = {}
    precVidName = "None"
    videoBegining = True
    validBatch = 0
    nbVideos = 0


    for batch_idx, (data, audio,target,vidName,frameInds) in enumerate(loader):

        #print(vidName,target.sum(),target.size())

        newVideo = (vidName != precVidName) or videoBegining

        if (batch_idx % log_interval == 0):
            print("\t",loader.sumL+1,"/",loader.nbShots)

        if args.cuda:
            data, target,frameInds = data.cuda(), target.cuda(),frameInds.cuda()
            if not audio is None:
                audio = audio.cuda()
        #print(data.size())

        if args.temp_model.find("net") != -1:
            output = model.computeFeat(data,audio).data

        else:
            if newVideo:
                output,(h,c) = model(data,audio)
            else:
                output,(h,c) = model(data,audio,h,c)

            output,h,c = output.data,h.data,c.data

        updateFrameDict(frameIndDict,frameInds,vidName)

        if newVideo and not videoBegining:
            #print("targDict",targDict)
            allOutput,nbVideos,total_loss,total_cover,total_overflow,total_iou,total_auc = updateMetrics(args,model,allOutput,allTarget,precVidName,width,nbVideos,\
                                                                                                    total_loss,total_cover,total_overflow,total_iou,total_auc,outDict,targDict)
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
        allOutput,nbVideos,total_loss,total_cover,total_overflow,total_iou,total_auc = updateMetrics(args,model,allOutput,allTarget,precVidName,width,nbVideos,\
                                                                                        total_loss,total_cover,total_overflow,total_iou,total_auc,outDict,targDict)

    for key in outDict.keys():
        fullArr = torch.cat((frameIndDict[key].float(),outDict[key].permute(1,0)),dim=1)
        np.savetxt("../results/{}/{}_epoch{}_{}.csv".format(args.exp_id,args.model_id,epoch,key),fullArr.cpu().detach().numpy())

    metricDict = writeSummaries(total_loss,total_cover,total_overflow,total_auc,total_iou,validBatch,writer,epoch,"val",args.model_id,args.exp_id,nbVideos=nbVideos)

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
        paths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id, epoch))
        if len(paths) > 1:
            raise ValueError("More than one best model found for exp {} and model {}".format(exp_id,model_id))

        if len(paths) != 0:
            os.remove(paths[0])
        torch.save(model.state_dict(), "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, epoch))
        return metricVal,outDict,targDict

    else:
        return metricLastVal,outDict,targDict

def computeScore(model,allFeats,allTarget,valLTemp,poolTempMod,overlap,vidName):

    allOutput = None
    splitSizes = [valLTemp for _ in range(allFeats.size(1)//valLTemp)]

    if allFeats.size(1)%valLTemp > 0:
        splitSizes.append(allFeats.size(1)%valLTemp)

    #chunkList = torch.split(allFeats,split_size_or_sections=splitSizes,dim=1)
    chunkList = splitWithOverlap(allFeats,splitSizes,overlap)

    sumSize = 0

    if poolTempMod == "lstm" or poolTempMod=="cnn":
        attList = []
    else:
        attList = None

    for i in range(len(chunkList)):

        if poolTempMod == "lstm" or poolTempMod=="cnn":
            targets = allTarget[:,sumSize:sumSize+chunkList[i].size(1)]
            output = model.computeScore(chunkList[i],None,None,targets)[0].unsqueeze(0).data[:,overlap:chunkList[i].size(1)-overlap]

            #print(output.size(),len(attList))
        else:
            output = model.computeScore(chunkList[i]).data[:,overlap:chunkList[i].size(1)-overlap]

        if allOutput is None:
            allOutput = output
        else:
            allOutput = torch.cat((allOutput,output),dim=1)

        sumSize += len(chunkList[i])

    return allOutput

def splitWithOverlap(allFeat,splitSizes,overlap):

    chunkList = torch.split(allFeat,split_size_or_sections=splitSizes,dim=1)

    cumSize = torch.cumsum(torch.tensor(splitSizes),dim=0)
    cumSize= torch.cat((torch.tensor([0]),cumSize),dim=0)

    padd = torch.zeros(allFeat.size(0),overlap,allFeat.size(2)).to(allFeat.device)
    allFeat = torch.cat((padd,allFeat,padd),dim=1)

    offset = overlap
    overlappedChunks = []
    for i in range(len(chunkList)):

        overlChunkList = torch.cat((allFeat[:,cumSize[i]:cumSize[i]+overlap],chunkList[i],allFeat[:,cumSize[i+1]+overlap:cumSize[i+1]+2*overlap]),dim=1)
        overlappedChunks.append(overlChunkList)

    return overlappedChunks

def updateFrameDict(frameIndDict,frameInds,vidName):
    ''' Store the prediction of a model in a dictionnary with one entry per movie

    Args:
     - outDict (dict): the dictionnary where the scores will be stored
     - output (torch.tensor): the output batch of the model
     - frameIndDict (dict): a dictionnary collecting the index of each frame used
     - vidName (str): the name of the video from which the score are produced

    '''

    if vidName in frameIndDict.keys():
        reshFrInds = frameInds.view(len(frameInds),-1).clone()
        frameIndDict[vidName] = torch.cat((frameIndDict[vidName],reshFrInds),dim=0)

    else:
        frameIndDict[vidName] = frameInds.view(len(frameInds),-1).clone()

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

def writeSummaries(total_loss,total_cover,total_overflow,total_auc,total_iou,batchNb,writer,epoch,mode,model_id,exp_id,nbVideos=None):
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
            params = torch.load(init_path)

            state_dict = {k.replace("module.cnn.","cnn.module."): v for k,v in params.items()}

            net.load_state_dict(state_dict)
            startEpoch = findLastNumbers(init_path)
        else:

            params = torch.load(init_path_visual_temp)
            for key in params.keys():

                if cuda:
                    params[key] = params[key].cuda()

                net.state_dict()[key].data += params[key].data -net.state_dict()[key].data

                startEpoch = findLastNumbers(init_path_visual_temp)

    return startEpoch

def evalAllImages(exp_id,model_id,model,audioModel,testLoader,cuda,log_interval):
    '''
    Pass all the images and/or the sound extracts of a loader in a feature model and save the feature vector in one csv for each image.
    Args:
    - exp_id (str): The experience id
    - model (nn.Module): the model to process the images
    - audioModel (nn.Module): the model to process the sound extracts
    - testLoader (load_data.TestLoader): the image and/or sound loader
    - cuda (bool): True is the computation has to be done on cuda
    - log_interval (int): the number of batches to wait before logging progression
    '''

    for batch_idx, (data,audio, _,vidName,frameInds) in enumerate(testLoader):

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

def updateHist(writer,model_id,outDictEpochs,targDictEpochs):

    firstEpoch = list(outDictEpochs.keys())[0]

    for i,vidName in enumerate(outDictEpochs[firstEpoch].keys()):

        fig = plt.figure()

        targBounds = np.array(processResults.binaryToSceneBounds(targDictEpochs[firstEpoch][vidName][0]))

        cmap = cm.plasma(np.linspace(0, 1, len(targBounds)))

        width = 0.2*len(outDictEpochs.keys())
        off = width/2

        plt.bar(len(outDictEpochs.keys())+off,targBounds[:,1]+1-targBounds[:,0],width,bottom=targBounds[:,0],color=cmap,edgecolor='black')

        for j,epoch in enumerate(outDictEpochs.keys()):

            predBounds = np.array(processResults.binaryToSceneBounds(outDictEpochs[epoch][vidName][0]))
            cmap = cm.plasma(np.linspace(0, 1, len(predBounds)))

            plt.bar(epoch-1,predBounds[:,1]+1-predBounds[:,0],1,bottom=predBounds[:,0],color=cmap,edgecolor='black')

        writer.add_figure(model_id+"_val"+"_"+vidName,fig,firstEpoch)

def updateHardMin(hmDict,trainDataset,args):

    paths= trainDataset.videoPaths
    names = list(map(lambda x: os.path.basename(os.path.splitext(x)[0]),paths))

    for i in range(len(names)):
        if names[i] in hmDict.keys():
            hmDict[names[i]] = (i,hmDict[names[i]])

    vidIndsScores = list(sorted([hmDict[name] for name in hmDict.keys()],key=lambda x:x[1]))
    vidInds = list(map(lambda x:x[0],vidIndsScores))

    sampler = load_data.Sampler(len(trainDataset.videoPaths),trainDataset.nbShots,args.l_max,vidInds,args.hm_prop)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,batch_size=args.batch_size,sampler=sampler, collate_fn=load_data.collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers)

    return trainLoader

#Init args
def addInitArgs(argreader):
    argreader.parser.add_argument('--start_mode', type=str,metavar='SM',
                help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.')
    argreader.parser.add_argument('--init_path', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the network')
    argreader.parser.add_argument('--init_path_visual', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the visual model')
    argreader.parser.add_argument('--init_path_visual_temp', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the visual and the temporal model')
    argreader.parser.add_argument('--init_path_audio', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the audio model')
    return argreader

#Loss args
def addLossArgs(argreader):
    argreader.parser.add_argument('--class_weight', type=float, metavar='CW',
                        help='Set the importance of balancing according to class instance number in the loss function. 0 makes equal weights and 1 \
                        makes weights proportional to the class instance number of the other class.')
    argreader.parser.add_argument('--sparsi_weig', type=float,metavar='SW',
                        help='Weight of the term rewarding the sparsity in the score prediction')
    argreader.parser.add_argument('--sparsi_wind', type=int,metavar='SW',
                        help='Half-size of the window taken into account for the sparsity term')
    argreader.parser.add_argument('--sparsi_thres', type=float,metavar='ST',
                        help='Threshold above which the sum of scores in the window is considered too big ')
    argreader.parser.add_argument('--soft_loss', type=args.str2bool,metavar='BOOL',
                        help="To use target soften with a triangular kernel.")
    argreader.parser.add_argument('--soft_loss_width', type=args.str2FloatList,metavar='WIDTH',
                        help="The width of the triangular window of the soft loss (in number of shots). Can be a schedule like learning rate")

    return argreader

#Optim args
def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=args.str2FloatList,metavar='LR',
                        help='learning rate (it can be a schedule : --lr 0.01,0.001,0.0001)')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                        help='SGD momentum')
    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                        help='the optimizer to use (default: \'SGD\')')
    return argreader

#Validation arguments
def addValArgs(argreader):
    argreader.parser.add_argument('--train_step_to_ignore', type=int,metavar='LMAX',
                    help='Number of steps that will be ignored at the begining and at the end of the training sequence for binary cross entropy computation')
    argreader.parser.add_argument('--val_l_temp_overlap', type=int,metavar='LMAX',
                    help='Size of the overlap between sequences passed to the CNN temp model')
    argreader.parser.add_argument('--val_l_temp', type=int,metavar='LMAX',help='Length of sequences for computation of scores when using a CNN temp model.')

    argreader.parser.add_argument('--metric_early_stop', type=str,metavar='METR',
                    help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_metric', type=args.str2bool,metavar='BOOL',
                    help='If true, The chosen metric for chosing the best model will be maximised')

    return argreader

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--comp_feat', action='store_true',help='To compute and write in a file the features of all images in the test set. All the arguments used to \
                                    build the model and the test data loader should be set.')
    argreader.parser.add_argument('--no_train', action='store_true',help='To use to re-evaluate a model at each epoch after training. At each epoch, the model is not trained but \
                                                                            the weights of the corresponding epoch are loaded and then the model is evaluated')

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

        img_size = (args.img_width,args.img_heigth)
        testLoader = load_data.TestLoader(args.val_l,args.dataset_test,args.test_part_beg,args.test_part_end,(args.img_width,args.img_heigth),\
                                          args.audio_len,args.resize_image,args.frames_per_shot,args.exp_id,args.random_frame_val)

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

        if args.feat_audio != "None":
            audioFeatModel = modelBuilder.buildAudioFeatModel(args.feat_audio)
            if args.cuda:
                audioFeatModel = audioFeatModel.cuda()
            if args.init_path_audio != "None":
                audioFeatModel.load_state_dict(torch.load(args.init_path_audio))
            elif args.init_path != "None":
                model = modelBuilder.netBuilder(args)
                params = torch.load(args.init_path)
                state_dict = {k.replace("module.cnn.","cnn.module."): v for k,v in params.items()}
                model.load_state_dict(state_dict)
                audioFeatModel = model.audioFeatModel

            audioFeatModel.eval()
        else:
            audioFeatModel = None

        evalAllImages(args.exp_id,args.model_id,featModel,audioFeatModel,testLoader,args.cuda,args.log_interval)

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

        if True:

            train_dataset = load_data.SeqTrDataset(args.dataset_train,args.train_part_beg,args.train_part_end,args.l_min,args.l_max,\
                                                (args.img_width,args.img_heigth),audioLen,args.resize_image,args.frames_per_shot,args.exp_id,args.max_shots)
            sampler = load_data.Sampler(len(train_dataset.videoPaths),train_dataset.nbShots,args.l_max)
            trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,sampler=sampler, collate_fn=load_data.collateSeq, # use custom collate function here
                              pin_memory=False,num_workers=args.num_workers)

            valLoader = load_data.TestLoader(args.val_l,args.dataset_val,args.val_part_beg,args.val_part_end,\
                                                (args.img_width,args.img_heigth),audioLen,args.resize_image,\
                                                args.frames_per_shot,args.exp_id,args.random_frame_val)

            #Building the net
            net = modelBuilder.netBuilder(args)

            trainFunc = epochSeqTr
            valFunc = epochSeqVal

            kwargsTr = {'log_interval':args.log_interval,'loader':trainLoader,'args':args,'writer':writer}
            kwargsVal = kwargsTr.copy()

            kwargsVal['loader'] = valLoader
            kwargsVal.update({"metricEarlyStop":args.metric_early_stop,"maximiseMetric":args.maximise_metric})

        for p in net.parameters():
            paramToOpti.append(p)

        paramToOpti = (p for p in paramToOpti)

        img_size = (args.img_width,args.img_heigth)

        if args.cuda:
            net = net.cuda()

        #Getting the contructor and the kwargs for the choosen optimizer
        optimConst,kwargsOpti = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

        startEpoch = initialize_Net_And_EpochNumber(net,args.exp_id,args.model_id,args.cuda,args.start_mode,args.init_path,args.init_path_visual_temp)

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

        if True:
            outDictEpochs = {}
            targDictEpochs = {}

        for epoch in range(startEpoch, args.epochs + 1):

            #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
            #The optimiser have to be rebuilt every time the learning rate is updated
            if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

                kwargsOpti['lr'] = args.lr[lrCounter]
                if args.train_visual:
                    optim = optimConst(net.parameters(), **kwargsOpti)
                else:
                    optim = optimConst(net.getParams(), **kwargsOpti)

                kwargsTr["optim"] = optim

                if lrCounter<len(args.lr)-1:
                    lrCounter += 1

            if (epoch-1) % ((args.epochs + 1)//len(args.soft_loss_width)) == 0 or epoch==startEpoch:

                width = args.soft_loss_width[widthCounter]
                if widthCounter<len(args.soft_loss_width)-1:
                    widthCounter += 1
                kwargsTr["width"] = width

            kwargsTr["epoch"],kwargsVal["epoch"] = epoch,epoch
            kwargsTr["model"],kwargsVal["model"] = net,net

            if not args.no_train:
                hmDict = trainFunc(**kwargsTr)

                if epoch % args.epochs_hm == 0 and args.epochs_hm != -1:
                    train_loader = updateHardMin(hmDict,train_dataset,args)
                    kwargsTr["loader"] = train_loader

            else:
                net.load_state_dict(torch.load("../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id,epoch)))

            if True:
                kwargsVal["metricLastVal"] = metricLastVal
                kwargsVal["width"] = width
                metricLastVal,outDict,targDict = valFunc(**kwargsVal)

                outDictEpochs[epoch] = outDict
                targDictEpochs[epoch] = targDict
                updateHist(writer,args.model_id,outDictEpochs,targDictEpochs)

if __name__ == "__main__":
    main()
