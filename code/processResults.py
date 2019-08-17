
from args import ArgReader

import os
import glob

import torch
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import matplotlib.cm as cm
import pims
import cv2
from PIL import Image

import load_data
import modelBuilder

import metrics
import utils

def evalModel_leaveOneOut(exp_id,model_id,model_name,dataset_test,epoch,firstThres,lastThres,lenPond,relativeToFrame):
    '''
    Evaluate a model. It requires the scores for each video to have been computed already with the trainVal.py script. Check readme to
    see how to compute the scores for each video.

    It computes the performance of a model using the default decision threshold (0.5) and the best decision threshold.

    The best threshold is computed for each video by looking for the best threshold on all the other videos. The best threshold is also
    computed for each metric.

    To find the best threshold, a range of threshold are evaluated and the best is selected.

    Args:
    - exp_id (str): the name of the experience
    - model_id (str): the id of the model to evaluate. Eg : "res50_res50_youtLarg"
    - model_name (str): the label of the model. It will be used to identify the model in the result table. Eg. : 'Res50-Res50 (Youtube-large)'
    - dataset_test (str): the dataset to evaluate
    - epoch (int): the epoch at which to evaluate
    - firstThres (float): the lower bound of the threshold range to evaluate
    - lastThres (float): the upper bound of the threshold range to evaluate
    '''


    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,epoch)),key=utils.findNumbers))
    videoNameDict = buildVideoNameDict(dataset_test,0,1,resFilePaths)
    resFilePaths = np.array(list(filter(lambda x:x in videoNameDict.keys(),resFilePaths)))

    thresList = np.arange(firstThres,lastThres,step=(lastThres-firstThres)/10)

    #Store the value of the f-score of for video and for each threshold
    metTun = {}
    metEval = {"IoU":    np.zeros(len(resFilePaths)),\
               "F-score":np.zeros(len(resFilePaths)),\
               "F-score New":np.zeros(len(resFilePaths)),\
               "DED":    np.zeros(len(resFilePaths))}

    metDef = {"IoU":    np.zeros(len(resFilePaths)),\
              "F-score":np.zeros(len(resFilePaths)),\
              "F-score New":np.zeros(len(resFilePaths)),\
              "DED":    np.zeros(len(resFilePaths))}

    for j,path in enumerate(resFilePaths):

        fileName = os.path.basename(os.path.splitext(path)[0])
        videoName = videoNameDict[path]

        metEval["F-score"][j],metDef["F-score"][j] = findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"F-score",lenPond,relativeToFrame)

        if not relativeToFrame:
            metEval["F-score New"][j],metDef["F-score New"][j] = findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"F-score New",lenPond)
            metEval["IoU"][j],metDef["IoU"][j] = findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"IoU",lenPond)
            metEval["DED"][j],metDef["DED"][j] = findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"DED",lenPond)

    printHeader = not os.path.exists("../results/{}_metrics.csv".format(dataset_test))

    with open("../results/{}_metrics.csv".format(dataset_test),"a") as text_file:
        if printHeader:
            print("Model,Threshold tuning,F-score,F-score New,IoU,DED",file=text_file)

        print("\multirow{2}{*}{"+model_name+"}"+"&"+"Yes"+"&"+formatMetr(metEval["F-score"])+"&"+formatMetr(metEval["F-score New"])+"&"+formatMetr(metEval["IoU"])+"&"+formatMetr(metEval["DED"])+"\\\\",file=text_file)
        print("&"+"No"+"&"+formatMetr(metDef["F-score"])+"&"+formatMetr(metDef["F-score New"])+"&"+formatMetr(metDef["IoU"])+"&"+formatMetr(metDef["DED"])+"\\\\",file=text_file)
        print("\hline",file=text_file)

    print("Best F-score : ",str(round(metEval["F-score"].mean(),2)),"Default F-score :",str(round(metDef["F-score"].mean(),2)),\
          "Best IoU :",str(round(metEval["IoU"].mean(),2)),"Default IoU :",str(round(metDef["IoU"].mean(),2)))

def findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,metric,lenPond,relativeToFrame=False):
    _,thres = bestThres(videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,metric,relativeToFrame=relativeToFrame)

    gt = load_data.getGT(dataset_test,videoName,relativeToFrame).astype(int)
    scores = np.genfromtxt(path)[:,1]

    pred = (scores > thres).astype(int)

    if relativeToFrame:
        pred = shotInd_to_frameInd(pred,dataset_test,videoName)

    metr_dict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]),lenPond)

    pred = (scores > 0.5).astype(int)

    if relativeToFrame:
        pred = shotInd_to_frameInd(pred,dataset_test,videoName)

    def_metr_dict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]),lenPond)

    return metr_dict[metric],def_metr_dict[metric]

def formatMetr(metricValuesArr):

    return "$"+str(round(metricValuesArr.mean(),2))+" \pm "+str(round(metricValuesArr.std(),2))+"$"

def bestThres(videoToEvalName,resFilePaths,thresList,dataset,videoNameDict,metTun,metric,relativeToFrame=False):

    optiFunc = np.min if metric == "DED" else np.max
    argOptiFunc = np.argmin if metric == "DED" else np.argmax

    metr_list = np.zeros(len(thresList))

    for i,thres in enumerate(thresList):

        mean = 0
        for j,path in enumerate(resFilePaths):

            if videoNameDict[path] != videoToEvalName:

                key = videoToEvalName+str(thres)+metric
                if not key in metTun.keys():

                    gt = load_data.getGT(dataset,videoNameDict[path],relativeToFrame).astype(int)
                    scores = np.genfromtxt(path)[:,1]
                    pred = (scores > thres).astype(int)

                    if relativeToFrame:
                        pred = shotInd_to_frameInd(pred,dataset,videoNameDict[path])

                    metrDict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))
                    metTun[key] = metrDict[metric]

                mean += metTun[key]

        metr_list[i] = mean/(len(resFilePaths)-1)

    return optiFunc(metr_list),thresList[argOptiFunc(metr_list)]

def shotInd_to_frameInd(pred,dataset,videoName):

    shotsF = np.genfromtxt("../data/"+dataset+"/"+videoName+"/result.csv").astype(int)

    frameInds = shotsF[pred.nonzero()].mean(axis=1).astype(int)

    lastFrameInd = shotsF[-1,1]
    predF = np.zeros(lastFrameInd+1).astype(int)
    predF[frameInds] = 1

    return predF

def tsne(dataset,exp_id,model_id,seed,nb_scenes=10):
    '''
    Plot the representations of the shots of a video in a 2D space using t-sne algorithm. Each point represents a shot,
    its color indicates from which scene it comes from

    Args:
    - dataset (str): the video dataset
    - exp_id (str): the experience name
    - model_id (str): the model name
    - seed (int): the seed to initialise the t-sne algorithm
    - nb_scenes (int): the number of scenes to plot

    '''

    repFile = glob.glob("../results/{}/")

    videoPathList = glob.glob("../data/{}/*.*".format(dataset))
    videoPathList = list(filter(lambda x:x.find(".wav")==-1,videoPathList))

    np.random.seed(seed)

    cmap = cm.rainbow(np.linspace(0, 1, nb_scenes))

    for videoPath in videoPathList:
        videoName = os.path.splitext(os.path.basename(videoPath))[0]
        print(videoName)
        if os.path.exists("../results/{}/{}/".format(exp_id,videoName)):
            print("\tFeature for video {} exist".format(videoName))
            if not os.path.exists("../vis/{}/{}_model{}_tsne.png".format(exp_id,videoName,model_id)):

                print("\tComputing t-sne")
                repFilePaths = sorted(glob.glob("../results/{}/{}/*_{}.csv".format(exp_id,videoName,model_id)))
                frameInd = np.array(list(map(lambda x:int(os.path.basename(x).split("_")[0]),repFilePaths)))

                gt = load_data.getGT(dataset,videoName)

                cmap = cm.rainbow(np.linspace(0, 1, gt.sum()+1))
                reps = np.array(list(map(lambda x:np.genfromtxt(x),repFilePaths)))
                imageRep = 255*(reps-reps.min())/(reps.max()-reps.min())
                imageRep = imageRep.transpose()
                imageRep = resize(imageRep,(300,1000),mode="constant", order=0,anti_aliasing=True)
                imageRep = Image.fromarray(imageRep)
                imageRep = imageRep.convert('RGB')
                imageRep.save("../vis/{}/{}_model{}_catRepr.png".format(exp_id,model_id,videoName))

                #Computing the scene/color index of each frame
                gt = gt[:len(reps)]
                colorInds = np.cumsum(gt)

                #Load the gt with the interval format
                gt_interv = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,videoName)).astype(int)

                repr_tsne = TSNE(n_components=2,init='pca',random_state=1,learning_rate=20).fit_transform(reps)
                plt.figure()
                plt.title("T-SNE view of feature from {}".format(videoName))
                plt.scatter(repr_tsne[:,0],repr_tsne[:,1], zorder=2,color=cmap[colorInds])
                plt.savefig("../vis/{}/{}_model{}_tsne.png".format(exp_id,videoName,model_id))
            else:
                print("\tT-sne already done")
        else:
            print("\tFeature for video {} does not exist".format(videoName))

def plotScore(exp_id,model_id,exp_id_init,model_id_init,dataset_test,plotDist,epoch):

    resFilePaths = sorted(glob.glob("../results/{}/{}_epoch{}*.csv".format(exp_id,model_id,epoch)))

    videoPaths = load_data.findVideos(dataset_test,propStart=0,propEnd=1)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    print("../results/{}/{}_epoch*.csv".format(exp_id,model_id))

    scoresNewScAll,scoresNoScAll,scoAccNewScAll,scoAccNoScAll = np.array([]),np.array([]),np.array([]),np.array([])

    for path in resFilePaths:

        videoName = None
        for candidateVideoName in videoNames:
            if "_"+candidateVideoName.replace("__","_")+".csv" in path:
                videoName = candidateVideoName

        if not videoName is None:
            fileName = os.path.basename(os.path.splitext(path)[0])

            scores = np.genfromtxt(path)[:,1]

            gt = load_data.getGT(dataset_test,videoName)

            fig = plt.figure(figsize=(30,5))
            ax1 = fig.add_subplot(111)

            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(20)

            legHandles = []

            #Plot the ground truth transitions
            ax1.vlines(gt.nonzero(),0,1,linewidths=3,color='gray')

            #Plot the decision threshold
            ax1.hlines([0.5],0,len(scores),linewidths=3,color='red')

            #Plot the scores
            legHandles += ax1.plot(np.arange(len(scores)),scores,color="blue",label="Scene change score")

            plt.xlabel("Time (shot index)")
            plt.ylabel("Probability of scene change")
            plt.tight_layout()
            plt.savefig("../vis/{}/Scores_{}.png".format(exp_id,fileName))
            plt.close()
            if plotDist:
                fig = plt.figure(figsize=(30,5))
                ax1 = fig.add_subplot(111)

                #Plot the distance between the successive features
                legHandles,dists = plotFeat(exp_id,model_id,videoName,len(scores),"orange","Distance between shot trained representations",legHandles,ax1)
                legHandles,_ = plotFeat(exp_id_init,model_id_init,videoName,len(scores),"green","Distance between untrained shot representations",legHandles,ax1)

                legend = fig.legend(handles=legHandles, loc='center right' ,title="")
                fig.gca().add_artist(legend)

                plt.xlabel("Time (shot index)")
                plt.ylabel("Distance")
                plt.tight_layout()
                plt.savefig("../vis/{}/Dists_{}_.png".format(exp_id,fileName))
                plt.close()

                #The distance between the shot features and the preceding shot features
                #is not defined for the first shot, so we remove it
                plotHist(gt[1:],dists,exp_id,fileName,sigName="Distance",sigShortName="dist")

            plotHist(gt,scores,exp_id,fileName,sigName="Score",sigShortName="sco")

            scoAcc = scores[2:]-2*scores[1:-1]+scores[:-2]
            plotHist(gt[1:-1],scoAcc,exp_id,fileName,sigName="Score Acceleration",sigShortName="scoAcc")

            scoresNewSc, scoAccNewSc, scoresNoSc, scoAccNoSc = split_and_plot2Hist(gt[1:-1],scores[2:],scoAcc,exp_id,fileName,sigName1="Score",sigName2="Score Acceleration",sigShortName1="sco",sigShortName2="accSco")

            scoresNewScAll = np.concatenate((scoresNewScAll,scoresNewSc),axis=0)
            scoresNoScAll = np.concatenate((scoresNoScAll,scoresNoSc),axis=0)
            scoAccNewScAll = np.concatenate((scoAccNewScAll,scoAccNewSc),axis=0)
            scoAccNoScAll =np.concatenate((scoAccNoScAll,scoAccNoSc),axis=0)

        else:
            raise ValueError("Unkown video : ",path)

    sig1Max,sig1Min = max(scoresNewScAll.max(),scoresNoScAll.max()),min(scoresNewScAll.min(),scoresNoScAll.min())
    sig2Max,sig2Min = max(scoAccNewScAll.max(),scoAccNoScAll.max()),min(scoAccNewScAll.min(),scoAccNoScAll.min())

    plot2DHist("Scores","Score acceleration","sco","scoAcc",scoresNewScAll,scoAccNewScAll,(sig1Min,sig1Max),(sig2Min,sig2Max),\
                "Scores Score acceleration when scene change",exp_id,"allEp_newSc")

    plot2DHist("Scores","Score acceleration","sco","scoAcc",scoresNoScAll,scoAccNoScAll,(sig1Min,sig1Max),(sig2Min,sig2Max),\
                "Scores Score acceleration when no scene change",exp_id,"allEp_noSc")

def plotHist(gt,signal,exp_id,fileName,sigName,sigShortName):
    plt.figure()

    newSceneInds = torch.tensor(gt.nonzero()[0])
    noNewSceneInds = torch.tensor((1-gt).nonzero()[0])

    signal_newScene = signal[newSceneInds]
    signal_noNewScene = signal[noNewSceneInds]

    sigMax = max(signal_newScene.max(),signal_noNewScene.max())
    sigMin = min(signal_newScene.min(),signal_noNewScene.min())

    plt.ylabel("Density")
    plt.xlabel("Probability of scene change")
    plt.hist(signal_newScene,label="{} when scene change".format(sigName),alpha=0.5,range=(sigMin,sigMax),density=True,bins=30)
    plt.hist(signal_noNewScene,label="{} when no scene change".format(sigName),alpha=0.5,range=(sigMin,sigMax),density=True,bins=30)
    plt.legend()
    plt.savefig("../vis/{}/Hist_{}_{}.png".format(exp_id,fileName,sigShortName))
    plt.tight_layout()
    plt.close()

def plot2DHist(sigName1,sigName2,sigShortName1,sigShortName2,sig1,sig2,sig1Range,sig2Range,label,exp_id,fileName):

    plt.figure()
    plt.xlabel(sigName1)
    plt.ylabel(sigName2)
    plt.hist2d(sig1,sig2,label=label,range=[sig1Range,sig2Range],bins=30)
    plt.legend()
    plt.savefig("../vis/{}/2dHist_{}_{}_{}.png".format(exp_id,fileName,sigShortName1,sigShortName2))
    plt.close()

def split_and_plot2Hist(gt,signal1,signal2,exp_id,fileName,sigName1,sigName2,sigShortName1,sigShortName2):
    plt.figure()

    newSceneInds = torch.tensor(gt.nonzero()[0])
    noNewSceneInds = torch.tensor((1-gt).nonzero()[0])

    signal1_newSc,signal1_noSc = signal1[newSceneInds],signal1[noNewSceneInds]
    signal2_newSc,signal2_noSc = signal2[newSceneInds],signal2[noNewSceneInds]

    sig1Max,sig1Min = max(signal1_newSc.max(),signal1_noSc.max()),min(signal1_newSc.min(),signal1_noSc.min())
    sig2Max,sig2Min = max(signal2_newSc.max(),signal2_noSc.max()),min(signal2_newSc.min(),signal2_noSc.min())


    plot2DHist("Scores","Score acceleration","sco","scoAcc",signal1_newSc,signal2_newSc,(sig1Min,sig1Max),(sig2Min,sig2Max),\
                "{} {} when scene change".format(sigName1,sigName2),exp_id,fileName+"_newSc")
    plot2DHist("Scores","Score acceleration","sco","scoAcc",signal1_noSc,signal2_noSc,(sig1Min,sig1Max),(sig2Min,sig2Max),\
                "{} {} when no scene change".format(sigName1,sigName2),exp_id,fileName+"_noSc")

    return signal1_newSc, signal2_newSc, signal1_noSc, signal2_noSc

def plotFeat(exp_id,model_id,videoName,nbShots,color,label,legHandles,ax1):
    featPaths = sorted(glob.glob("../results/{}/{}/*_{}.csv".format(exp_id,videoName,model_id)),key=utils.findNumbers)

    if len(featPaths) > 0:
        feats = np.array(list(map(lambda x:np.genfromtxt(x),featPaths)))
        dists = np.sqrt(np.power(feats[:-1]-feats[1:],2).sum(axis=1))
        legHandles += ax1.plot(np.arange(nbShots-1)+1,dists,color=color,label=label)

    return legHandles,dists

def buildVideoNameDict(dataset_test,test_part_beg,test_part_end,resFilePaths):

    videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset_test)))))
    videoPaths = list(filter(lambda x:x.find(".xml") == -1,videoPaths))
    videoPaths = list(filter(lambda x:os.path.isfile(x),videoPaths))
    videoPaths = np.array(videoPaths)[int(test_part_beg*len(videoPaths)):int(test_part_end*len(videoPaths))]
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))
    videoNameDict = {}

    for path in resFilePaths:
        for videoName in videoNames:
            if "_"+videoName.replace("__","_")+".csv" in path.replace("__","_"):
                videoNameDict[path] = videoName
        if path not in videoNameDict.keys():
            raise ValueError("The path "+" "+path+" "+"doesnt have a video name")

    return videoNameDict

def evalModel(exp_id,model_id,dataset_test,test_part_beg,test_part_end,firstEpoch,lastEpoch,firstThres,lastThres):

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch*.csv".format(exp_id,model_id)),key=utils.findNumbers))

    if firstEpoch is None:
        firstEpoch = utils.findNumbers(resFilePaths[0][resFilePaths[0].find("epoch")+5:].split("_")[0])

    if lastEpoch is None:
        lastEpoch = utils.findNumbers(resFilePaths[-1][resFilePaths[-1].find("epoch")+5:].split("_")[0])

    #If there's only one epoch to plot, theres only one point to plot which is why the marker has to be visible
    if firstEpoch == lastEpoch:
        mark = "o"
    else:
        mark = ","

    videoNameDict = buildVideoNameDict(dataset_test,test_part_beg,test_part_end,resFilePaths)

    resFilePaths = np.array(list(filter(lambda x:x in videoNameDict.keys(),resFilePaths)))

    #thresList = np.array([0.5,0.6,0.65,0.7,0.75,0.8])
    thresList = np.arange(firstThres,lastThres,step=(lastThres-firstThres)/10)

    iouArr = np.zeros((len(thresList),lastEpoch-firstEpoch+1))
    iouArr_pred,iouArr_gt = iouArr.copy(),iouArr.copy()
    fscoArr = np.zeros((len(thresList),lastEpoch-firstEpoch+1))

    cmap = cm.rainbow(np.linspace(0, 1, len(thresList)))

    fig = plt.figure(1,figsize=(13,8))
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    fig_pred = plt.figure(2,figsize=(13,8))
    ax_pred = fig_pred.add_subplot(111)
    box = ax_pred.get_position()
    ax_pred.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    fig_gt = plt.figure(3,figsize=(13,8))
    ax_gt = fig_gt.add_subplot(111)
    box = ax_gt.get_position()
    ax_gt.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    fig_sco = plt.figure(4,figsize=(13,8))
    ax_fsco = fig_sco.add_subplot(111)
    box = ax_fsco.get_position()
    ax_fsco.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    for i,thres in enumerate(thresList):

        for k in range(lastEpoch-firstEpoch+1):

            resFilePaths = sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,k+firstEpoch)),key=utils.findNumbers)

            iou_mean,iou_pred_mean,iou_gt_mean,f_sco_mean = 0,0,0,0

            for j,path in enumerate(resFilePaths):

                fileName = os.path.basename(os.path.splitext(path)[0])
                videoName = videoNameDict[path]

                gt = load_data.getGT(dataset_test,videoName).astype(int)
                #gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset_test,videoName)).astype(int)
                scores = np.genfromtxt(path)[:,1]

                pred = (scores > thres).astype(int)

                metrDict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))

                iou_mean += metrDict["IoU"]
                iou_pred_mean += metrDict["IoU_pred"]
                iou_gt_mean += metrDict["IoU_gt"]
                f_sco_mean +=  metrDict["F-score"]

            iouArr[i,k] = iou_mean/len(resFilePaths)
            iouArr_pred[i,k] = iou_pred_mean/len(resFilePaths)
            iouArr_gt[i,k] = iou_gt_mean/len(resFilePaths)
            fscoArr[i,k] = f_sco_mean/len(resFilePaths)

        ax.plot(np.arange(firstEpoch,lastEpoch+1),iouArr[i],label=thres,color=cmap[i], marker=mark)
        ax_pred.plot(np.arange(firstEpoch,lastEpoch+1),iouArr_pred[i],label=thres,color=cmap[i],marker=mark)
        ax_gt.plot(np.arange(firstEpoch,lastEpoch+1),iouArr_gt[i],label=thres,color=cmap[i],marker=mark)
        ax_fsco.plot(np.arange(firstEpoch,lastEpoch+1),fscoArr[i],label=thres,color=cmap[i],marker=mark)

    plt.figure(1,figsize=(13,8))
    plt.legend(loc="center right",bbox_to_anchor=(1.5,0.5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(0,1)

    plt.figure(2,figsize=(13,8))
    plt.legend(loc="center right",bbox_to_anchor=(1.5,0.5))
    ax_pred.set_xlabel("Epoch")
    ax_pred.set_ylabel("IoU pred")
    ax_pred.set_ylim(0,1)

    plt.figure(3,figsize=(13,8))
    plt.legend(loc="center right",bbox_to_anchor=(1.5,0.5))
    ax_gt.set_xlabel("Epoch")
    ax_gt.set_ylabel("IoU gt")
    ax_gt.set_ylim(0,1)

    plt.figure(4,figsize=(13,8))
    plt.legend(loc="center right",bbox_to_anchor=(1.5,0.5))
    ax_fsco.set_xlabel("Epoch")
    ax_fsco.set_ylabel("F-score")
    ax_fsco.set_ylim(0,1)

    if not os.path.exists("../vis/{}".format(exp_id)):
        os.makedirs("../vis/{}".format(exp_id))

    fig.savefig("../vis/{}/model{}_iouThres.png".format(exp_id,model_id))
    fig_pred.savefig("../vis/{}/model{}_iouPredThres.png".format(exp_id,model_id))
    fig_gt.savefig("../vis/{}/model{}_iouGTThres.png".format(exp_id,model_id))
    fig_sco.savefig("../vis/{}/model{}_fscoThres.png".format(exp_id,model_id))

    bestInd = np.argmax(iouArr)
    bestThresInd,bestEpochInd = bestInd//iouArr.shape[1],bestInd%iouArr.shape[1]

    print("Best IoU's : ",iouArr[bestThresInd],"epoch",bestEpochInd+firstEpoch,"threshold",round(thresList[bestThresInd],3),end=" ")

    bestInd = np.argmax(fscoArr)
    bestThresInd,bestEpochInd = bestInd//iouArr.shape[1],bestInd%iouArr.shape[1]

    print("Best F-scores : ",fscoArr[bestThresInd],"epoch",bestEpochInd+firstEpoch,"threshold",round(thresList[bestThresInd],2))

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    ########### PLOT TSNE ################
    argreader.parser.add_argument('--tsne',action='store_true',help='To plot t-sne representation of feature extracted. Also plots the representation of a video side by side to make an image. \
                                    The --exp_id, --model_id, --seed and --dataset_test arguments should be set.')

    ########### PLOT SCORE EVOLUTION ALONG VIDEO ##################
    argreader.parser.add_argument('--plot_score',action="store_true",help='To plot the scene change probability of produced by a model for all the videos processed by this model during validation for all epochs.\
                                                                            The --model_id argument must be set, along with the --exp_id, --dataset_test and --epoch_to_plot arguments.')

    argreader.parser.add_argument('--epoch_to_plot',type=int,metavar="N",help='The epoch at which to plot the predictions when using the --plot_score argument')

    argreader.parser.add_argument('--untrained_exp_and_model_id',default=[None,None],type=str,nargs=2,help='To plot the distance between features computed by the network before training when using the --plot_score arg.\
                                                                                        The values are the exp_id and the model_id of the model not trained on scene change detection (i.e. the model with the ImageNet weights)')
    argreader.parser.add_argument('--plot_dist',action="store_true",help='To plot the distance when using the --plot_score argument')

    ########## COMPUTE METRICS AND PUT THEM IN AN LATEX TABLE #############
    argreader.parser.add_argument('--eval_model_leave_one_out',type=float,nargs=3,help='To evaluate a model by tuning its decision threshold on the video on which it is not\
                                    evaluated. The --model_id argument must be set, along with the --model_name, --exp_id and --dataset_test arguments. \
                                    The values of this args are the epoch at which to evaluate , followed by the minimum and maximum decision threshold \
                                    to evaluate. Use the --len_pond to ponderate the overflow and coverage by scene lengths.')

    argreader.parser.add_argument('--model_name',type=str,metavar="NAME",help='The name of the model as will appear in the latex table produced by the --eval_model_leave_one_out argument.')
    argreader.parser.add_argument('--len_pond',action="store_true",help='Use this argument to ponderate the coverage and overflow by the GT scene length when using --eval_model_leave_one_out.')
    argreader.parser.add_argument('--relative_to_frame',action="store_true",help='Use this argument to compute the metrics relative to frame index and not shot index when using --eval_model_leave_one_out.')

    argreader.parser.add_argument('--fine_tuned_thres',action="store_true",help='To automatically fine tune the decision threshold of the model. Only useful for the --bbc_annot_dist arg. \
                                                                                Check the help of this arg.')
    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.tsne:
        tsne(args.dataset_test,args.exp_id,args.model_id,args.seed)
    if args.plot_score:
        plotScore(args.exp_id,args.model_id,args.untrained_exp_and_model_id[0],args.untrained_exp_and_model_id[1],args.dataset_test,args.plot_dist,args.epoch_to_plot)
    if args.eval_model_leave_one_out:
        epoch = int(args.eval_model_leave_one_out[0])
        thresMin = args.eval_model_leave_one_out[1]
        thresMax = args.eval_model_leave_one_out[2]
        evalModel_leaveOneOut(args.exp_id,args.model_id,args.model_name,args.dataset_test,epoch,thresMin,thresMax,args.len_pond,args.relative_to_frame)

if __name__ == "__main__":
    main()
