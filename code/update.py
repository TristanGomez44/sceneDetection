from torch.nn import functional as F
import metrics
import trainVal
import numpy as np
import load_data
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_auc_score
import utils

def updateMetrics(args,model,allOutput,allTarget,precVidName,nbVideos,metrDict,outDict,targDict):

    if args.temp_model.find("net") != -1:
        allOutput = trainVal.computeScore(model,allOutput,allTarget,args.val_l_temp,args.pool_temp_mod,precVidName)

    outDict[precVidName] = allOutput
    targDict[precVidName] = allTarget

    if args.compute_val_metrics:
        weights = trainVal.getWeights(allTarget,args.class_weight)
        loss = F.binary_cross_entropy(allOutput,allTarget,weight=weights).data.item()
        metrDict["Loss"] += loss
        cov,overflow,iou = metrics.binaryToMetrics(allOutput>0.5,allTarget)
        metrDict["Coverage"] += cov
        metrDict["Overflow"] += overflow
        metrDict["True F-score"] += 2*cov*(1-overflow)/(cov+1-overflow)
        metrDict["IoU"] += iou
        metrDict["AuC"] += roc_auc_score(allTarget.view(-1).cpu().numpy(),allOutput.view(-1).cpu().numpy())
        metrDict['DED'] += metrics.computeDED(allOutput.data>0.5,allTarget.long())

    nbVideos += 1

    return allOutput,nbVideos
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
def updateHist(writer,model_id,outDictEpochs,targDictEpochs):

    firstEpoch = list(outDictEpochs.keys())[0]

    for i,vidName in enumerate(outDictEpochs[firstEpoch].keys()):

        fig = plt.figure()

        targBounds = np.array(utils.binaryToSceneBounds(targDictEpochs[firstEpoch][vidName][0]))

        cmap = cm.plasma(np.linspace(0, 1, len(targBounds)))

        width = 0.2*len(outDictEpochs.keys())
        off = width/2

        plt.bar(len(outDictEpochs.keys())+off,targBounds[:,1]+1-targBounds[:,0],width,bottom=targBounds[:,0],color=cmap,edgecolor='black')

        for j,epoch in enumerate(outDictEpochs.keys()):

            predBounds = np.array(utils.binaryToSceneBounds(outDictEpochs[epoch][vidName][0]))
            cmap = cm.plasma(np.linspace(0, 1, len(predBounds)))

            plt.bar(epoch-1,predBounds[:,1]+1-predBounds[:,0],1,bottom=predBounds[:,0],color=cmap,edgecolor='black')

        writer.add_figure(model_id+"_val"+"_"+vidName,fig,firstEpoch)
def updateHardMin(epoch,epochs_hm,hmDict,trainDataset,args,kwargsTr,trainLoader):

    if epoch % epochs_hm == 0 and epochs_hm != -1 and args.hm_prop > 0:

        paths= trainDataset.videoPaths
        names = list(map(lambda x: os.path.basename(os.path.splitext(x)[0]),paths))

        for i in range(len(names)):
            if names[i] in hmDict.keys():
                hmDict[names[i]] = (i,hmDict[names[i]])

        vidIndsScores = list(sorted([hmDict[name] for name in hmDict.keys()],key=lambda x:x[1]))
        vidInds = list(map(lambda x:x[0],vidIndsScores))

        sampler = load_data.Sampler(len(paths),trainDataset.nbShots,args.l_max,vidInds,args.hm_prop)
        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,batch_size=args.batch_size,sampler=sampler, collate_fn=load_data.collateSeq, # use custom collate function here
                          pin_memory=False,num_workers=args.num_workers)

        kwargsTr["loader"] = trainLoader

    return trainLoader,kwargsTr
def updateLR(epoch,maxEpoch,lr,startEpoch,kwargsOpti,kwargsTr,lrCounter,net,optimConst):
    #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
    #The optimiser have to be rebuilt every time the learning rate is updated
    if (epoch-1) % ((maxEpoch + 1)//len(lr)) == 0 or epoch==startEpoch:

        kwargsOpti['lr'] = lr[lrCounter]
        optim = optimConst(net.parameters(), **kwargsOpti)

        kwargsTr["optim"] = optim

        if lrCounter<len(lr)-1:
            lrCounter += 1

    return kwargsOpti,kwargsTr,lrCounter
