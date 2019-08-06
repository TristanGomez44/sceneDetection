
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

import metrics
import utils

def resultTables(exp_ids,modelIds,thresList,epochList,dataset):

    videoPaths = load_data.findVideos(dataset,0,1)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    #This csv will contain the means metric
    metrDict = {"F-score":0,"IoU":0,"DED":0}
    csvMeans = "Model,"+",".join([metric for metric in metrDict.keys()])+"\n"

    for i,modelId in enumerate(modelIds):

        metrDict = {"F-score":np.zeros(len(videoNames)),"IoU":np.zeros(len(videoNames)),"DED":np.zeros(len(videoNames))}
        #This csv wil contain the performance for each video
        csvVids = "Video,"+",".join([metricName for metricName in metrDict.keys()])+"\n"

        for j,videoName in enumerate(videoNames):

            target = load_data.getGT(dataset,videoName).astype(int)
            scores = np.genfromtxt("../results/{}/{}_epoch{}_{}.csv".format(exp_ids[i],modelId,epochList[i],videoName))[:,1]
            pred = (scores > thresList[i]).astype(int)

            metrDictVid = metrics.binaryToAllMetrics(torch.tensor(pred).unsqueeze(0),torch.tensor(target).unsqueeze(0))

            metrDict["F-score"][j] = metrDictVid["F-score"]
            metrDict["IoU"][j] = metrDictVid["IoU"]
            metrDict["DED"][j] = metrDictVid["DED"]

            csvVids += videoName+","+",".join([str(f_score),str(iou),str(ded)])+"\n"

        with open("../results/{}/{}_{}_metrics.csv".format(exp_ids[i],modelId,dataset),"w") as text_file:
            print(csvVids,file=text_file)

        for metricName in metrDict.keys():
            metrDict[metricName] = "{} \pm {}".format(metrDict[metricName].mean(),metrDict[metricName].std())

        csvMeans += modelId+","+",".join([metrDict[metricName] for metricName in metrDict.keys()])+"\n"

    with open("../results/{}_metrics.csv".format(dataset),"w") as text_file:
        print(csvMeans,file=text_file)

def evalModel_leaveOneOut(exp_id,model_id,model_name,dataset_test,epoch,firstThres,lastThres,lenPond):

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

        metEval["F-score"][j],metDef["F-score"][j] = findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"F-score",lenPond)
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

def findBestThres_computeMetrics(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,metric,lenPond):
    _,thres = bestThres(videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,metric)

    gt = load_data.getGT(dataset_test,videoName).astype(int)
    scores = np.genfromtxt(path)[:,1]

    pred = (scores > thres).astype(int)
    metr_dict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]),lenPond)

    pred = (scores > 0.5).astype(int)
    def_metr_dict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]),lenPond)

    return metr_dict[metric],def_metr_dict[metric]

def formatMetr(metricValuesArr):

    return "$"+str(round(metricValuesArr.mean(),2))+" \pm "+str(round(metricValuesArr.std(),2))+"$"

def bestThres(videoToEvalName,resFilePaths,thresList,dataset,videoNameDict,metTun,metric,annotatorGT=0):

    optiFunc = np.min if metric == "DED" else np.max
    argOptiFunc = np.argmin if metric == "DED" else np.argmax

    metr_list = np.zeros(len(thresList))

    for i,thres in enumerate(thresList):

        mean = 0
        for j,path in enumerate(resFilePaths):

            if videoNameDict[path] != videoToEvalName:

                key = videoToEvalName+str(thres)+metric
                if not key in metTun.keys():

                    gt = load_data.getGT(dataset,videoNameDict[path],annotatorGT).astype(int)
                    scores = np.genfromtxt(path)[:,1]
                    pred = (scores > thres).astype(int)
                    metrDict = metrics.binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))
                    metTun[key] = metrDict[metric]

                mean += metTun[key]

        metr_list[i] = mean/(len(resFilePaths)-1)

    return optiFunc(metr_list),thresList[argOptiFunc(metr_list)]

def tsne(dataset,exp_id,model_id,seed,framesPerShots,nb_scenes=10):
    '''
    Plot the representations of the shots of a video in a 2D space using t-sne algorithm. Each point represents a shot,
    its color indicates from which scene it comes from

    Args:
    - dataset (str): the video dataset
    - exp_id (str): the experience name
    - model_id (str): the model name
    - seed (int): the seed to initialise the t-sne algorithm
    - framesPerShots (int): the number of frames per shot
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

def bbcAnnotDist(annotFold,modelExpId,modelId,modelEpoch,fine_tuned_thres):

    #The number and names of episodes
    epiNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),sorted(glob.glob(annotFold+"/scenes/annotator_0/*.txt"))))
    epNb = len(epiNames)
    #The number of annotators
    annotNb = len(list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),sorted(glob.glob(annotFold+"/scenes/annotator_*/")))))
    distMat = -np.ones((epNb,annotNb+1,annotNb+1))
    distMat = {"DED":distMat.copy(),"IoU":distMat.copy(),"F-score":distMat.copy()}

    figCount = 0

    thresList = np.arange(10)/10

    #This dict will contain argument only useful for when the model is evaluated
    kwargs = {"resFilePaths" : glob.glob("../results/{}/{}_epoch{}_*.csv".format(modelExpId,modelId,modelEpoch)),\
              "thresList" : thresList,"dataset":"bbc","metTun":{},"fine_tuned_thres":fine_tuned_thres}
    kwargs["videoNameDict"] = buildVideoNameDict("bbc",0,1,kwargs["resFilePaths"])

    for i in range(epNb):
        print("Episode ",i)
        shotNb = np.genfromtxt("../data/bbc/{}/result.csv".format(i)).shape[0]
        kwargs["videoName"] = str(i)

        for j in range(annotNb+1):
            print("\t Annot",j)

            for k in range(annotNb+1):

                distMat["DED"][i,j,k] = computeMetric("DED",annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,j,annotNb,shotNb,kwargs)
                distMat["F-score"][i,j,k] =computeMetric("F-score",annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,j,annotNb,shotNb,kwargs)
                distMat["IoU"][i,j,k] = computeMetric("IoU",annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,j,annotNb,shotNb,kwargs)

        for metric in distMat.keys():
            plotHeatMapWithValues(figCount,distMat[metric][i],"../vis/bbc_annot{}_ep{}_w{}_thresFT={}.png".format(metric,i,modelId,fine_tuned_thres))
            figCount+=1

    for metric in distMat.keys():
        plotHeatMapWithValues(figCount,distMat[metric].mean(axis=0),"../vis/bbc_annot{}_allEp_w{}_thresFT={}.png".format(metric,modelId,fine_tuned_thres))
        figCount+=1

def computeMetric(metric,annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,j,annotNb,shotNb,kwargs):

    kwargs["metric"] = metric

    if metric == "DED":
        kwargs["annotatorGT"] = k
        segmJ = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb,**kwargs)
        kwargs["annotatorGT"] = j
        segmK = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb,**kwargs)
        return metrics.computeDED(torch.tensor(segmJ).unsqueeze(0),torch.tensor(segmK).unsqueeze(0))

    elif metric == "IoU":
        kwargs["annotatorGT"] = k
        segmJ = utils.binaryToSceneBounds(utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb,**kwargs))
        kwargs["annotatorGT"] = j
        segmK = utils.binaryToSceneBounds(utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb,**kwargs))
        return metrics.IoU(np.array(segmJ),np.array(segmK))

    elif metric == "F-score":
        kwargs["annotatorGT"] = k
        segmJ = utils.binaryToSceneBounds(utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb,**kwargs))
        kwargs["annotatorGT"] = j
        segmK = utils.binaryToSceneBounds(utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb,**kwargs))
        over = metrics.overflow(np.array(segmJ),np.array(segmK),lenPond=True)
        cover = metrics.coverage(np.array(segmJ),np.array(segmK),lenPond=True)
        return 2*cover*(1-over)/(cover+1-over)
    else:
        raise ValueError("Unkown metric : ",metric)

def readBothSeg(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,k,annotNb,shotNb):

    if j==annotNb:
        segmK = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb)
        segmJ = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb,segmK.sum())
    elif k==annotNb:
        segmJ = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb)
        segmK = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb,segmJ.sum())
    else:
        segmJ = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb)
        segmK = utils.toBinary(utils.getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb)

    return segmJ,segmK

def plotHeatMapWithValues(figInd,mat,path):
    # Limits for the extent
    x_start = 0.0
    x_end = mat.shape[0]
    y_start = 0
    y_end = mat.shape[1]

    extent = [x_start, x_end, y_start, y_end]

    size = mat.shape[0]

    # The normal figure
    fig = plt.figure(figInd,figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, extent=extent, origin='lower', interpolation='None', cmap='gray',vmin=0,vmax=1)

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = round(mat[y_index, x_index],2)
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color="white" if mat[y_index, x_index] < 0.5 else "black", ha='center', va='center',fontsize='xx-large')

    fig.colorbar(im)
    fig.savefig(path)
    plt.close()

def scoreVis_video(dataset,exp_id,resFilePath,nbScoToPlot=11):
    ''' Plot the scene change score on the image of a video

    Args:
    - dataset (str): the video dataset
    - exp_id (str): the experience name
    - resFilePath (str): the path to a csv file containing the score of each shot of a video. Such a file is produced by using the trainVal.py script with \
                        the --comp_feat argument
    - nbScoToPlot (int): the number of score to plot at the same time on the image. The score plot on the center is the score of the current shot, the scores \
                        plot on the left and on the right correspond respectively to the scores of the prededing and following shots.

    '''

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    resFile = np.genfromtxt(resFilePath)
    frameInds = resFile[:,:-1]
    scores = resFile[:,-1]

    splitResFileName = os.path.splitext(os.path.basename(resFilePath))[0].split("_")

    i=0
    epochFound = False
    while i < len(splitResFileName) and not epochFound:

        if splitResFileName[i].find("epoch") != -1:
            epochFound = True
        else:
            i+=1

    videoName = "_".join(splitResFileName[i+1:])
    #videoName = "_".join(videoName)

    modelId =  "_".join(splitResFileName[:i])

    videoPath = list(filter(lambda x:x.find("wav")==-1,glob.glob("../data/"+dataset+"/"+videoName+"*.*")))[0]
    print(videoPath)
    shotFrames = utils.xmlToArray("../data/{}/{}/result.xml".format(dataset))

    cap = cv2.VideoCapture(videoPath)
    print(cap)
    shotCount = 0

    widProp = 0.6
    heigProp = 0.1
    heigShifProp = 0.1

    i=0
    success=True
    imgWidth,imgHeigth = None,None

    fps = utils.getVideoFPS(videoPath)
    print("Fps",fps)
    videoRes = None
    while success:
        success, imageRaw = cap.read()

        if not videoRes:
            print("../vis/{}/{}_{}_score.mp4".format(exp_id,modelId,videoName))
            videoRes = cv2.VideoWriter("../vis/{}/{}_{}_score.mp4".format(exp_id,modelId,videoName), fourcc, fps, (imageRaw.shape[0],imageRaw.shape[1]))

        if not imgWidth:
            imgWidth = imageRaw.shape[0]
            imgHeigth = imageRaw.shape[1]

        if i > shotFrames[shotCount,1]:
            shotCount += 1

        scoresToPlot = scores[max(shotCount-nbScoToPlot//2,0):min(shotCount+nbScoToPlot//2+1,len(scores))]

        if shotCount<nbScoToPlot//2:
            shift = nbScoToPlot-(len(scoresToPlot)+1)
        elif shotCount>len(scores)-nbScoToPlot//2:
            shift = (len(scoresToPlot)+1)-nbScoToPlot
        else:
            shift = 0

        #Background
        startW = int(imgWidth*(1-widProp)//2)
        endW = imgWidth-startW
        cv2.rectangle(imageRaw, (startW,int(imgHeigth*(1-heigShifProp))-int(imgHeigth*heigProp)), (endW,int(imgHeigth*(1-heigShifProp))), (0,0,0),thickness=-1)

        #Top and bottom lines
        topLineHeig = int(imgHeigth*(1-heigShifProp) - imgHeigth*heigProp)
        cv2.line(imageRaw,(startW,topLineHeig),(endW,topLineHeig),(255,0,0),2)

        botLineHeig = int(imgHeigth*(1-heigShifProp))
        cv2.line(imageRaw,(startW,botLineHeig),(endW,botLineHeig),(255,0,0),2)

        #Focus lines
        shiftFocus = int(imgWidth*widProp//nbScoToPlot)//2
        cv2.line(imageRaw,(imgWidth//2-shiftFocus,topLineHeig),(imgWidth//2-shiftFocus,botLineHeig),(255,0,0),2)
        cv2.line(imageRaw,(imgWidth//2+shiftFocus,topLineHeig),(imgWidth//2+shiftFocus,botLineHeig),(255,0,0),2)

        #Dots
        xList = []
        yList = []
        for j,score in enumerate(scoresToPlot):

            posX = int(imgWidth//2 - imgWidth*widProp//2 + (imgWidth*widProp//nbScoToPlot)*((j+1)+shift))
            posX += int(imgWidth*widProp/(2*nbScoToPlot))
            posY = int(imgHeigth*(1-heigShifProp) - imgHeigth*heigProp*score)

            xList.append(posX)
            yList.append(posY)

            cv2.circle(imageRaw,(posX,posY), 2, (255,255,255), -1)

        #Lines between dots
        for j in range(len(scoresToPlot)-1):
            cv2.line(imageRaw,(xList[j],yList[j]),(xList[j+1],yList[j+1]),(255,255,255),1)

        videoRes.write(imageRaw)

        i+=1

        if shotCount == len(shotFrames):
            success = False

    videoRes.release()

def convScoPlot(weightFile):

    model_id = os.path.basename(weightFile)
    end = model_id.find("_epoch")
    model_id = model_id[5:end]

    exp_id = weightFile.split("/")[-2]

    epoch = int(os.path.basename(weightFile).split("_")[-1][5:])

    paramDict = torch.load(weightFile)

    for key in paramDict.keys():
        if key.find("scoreConv.weight") != -1:
            weight =  paramDict[key]

    inSize = 100
    impAmpl = 1

    inp = torch.zeros(1,1,inSize).to(weight.device)
    inp[0,0,inSize//2] = impAmpl

    out = torch.nn.functional.conv1d(inp, weight,padding=weight.size(-1)//2)[0,0].cpu().detach().numpy()

    plt.figure()
    plt.title("Impulse response of {} score filter".format(model_id))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.plot(out)
    plt.savefig("../vis/{}/model{}_epoch{}_impResp.png".format(exp_id,model_id,epoch))

    fft = np.abs(np.fft.fft(out))[:inSize//2]

    plt.figure()
    plt.title("Fourrier transform of {} score filter".format(model_id))
    plt.xlabel("frequency")
    plt.ylabel("Amplitude")
    plt.ylim(0,max(fft)*1.2)
    plt.plot(fft)
    plt.savefig("../vis/{}/model{}_epoch{}_fourrier.png".format(exp_id,model_id,epoch))

def plotScore(exp_id,model_id,exp_id_init,model_id_init,dataset_test,test_part_beg,test_part_end,plotDist=False):

    resFilePaths = sorted(glob.glob("../results/{}/{}_epoch*.csv".format(exp_id,model_id)))

    videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset_test)))))
    videoPaths = list(filter(lambda x:x.find(".xml") == -1,videoPaths))
    videoPaths = list(filter(lambda x:os.path.isfile(x),videoPaths))
    videoPaths = np.array(videoPaths)[int(test_part_beg*len(videoPaths)):int(test_part_end*len(videoPaths))]
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

            legHandles = []

            #Plot the ground truth transitions
            ax1.vlines(gt.nonzero(),0,1,linewidths=3,color='gray')

            #Plot the decision threshold
            ax1.hlines([0.5],0,len(scores),linewidths=3,color='red')

            #Plot the scores
            legHandles += ax1.plot(np.arange(len(scores)),scores,color="blue",label="Scene change score")

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

    plt.hist(signal_newScene,label="{} when scene change".format(sigName),alpha=0.5,range=(sigMin,sigMax),density=True,bins=30)
    plt.hist(signal_noNewScene,label="{} when no scene change".format(sigName),alpha=0.5,range=(sigMin,sigMax),density=True,bins=30)
    plt.legend()
    plt.savefig("../vis/{}/Hist_{}_{}.png".format(exp_id,fileName,sigShortName))
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

    argreader.parser.add_argument('--frame',action='store_true',help='To compute the metrics at the frame level.')

    argreader.parser.add_argument('--metric',type=str,default="IoU",metavar='METRIC',help='The metric to use. Can only be \'IoU\' for now.')

    argreader.parser.add_argument('--tsne',action='store_true',help='To plot t-sne representation of feature extracted. Also plots the representation of a video side by side to make an image. \
                                    The --exp_id, --model_id, --frames_per_shot, --seed and --dataset_test arguments should be set.')

    argreader.parser.add_argument('--score_vis_video',type=str,help='To plot the scene change score on the video itself. Requires the --dataset_test and --exp_id arguments to be set. The value is a path to a result file.')
    argreader.parser.add_argument('--plot_cat_repr',type=str,help=' The value must a path to a folder containing representations vector as CSV files.')
    argreader.parser.add_argument('--conv_sco_plot',type=str,help='To plot the frequency response of the 1D filter learned by a model filtering its scores. The value is the path to the weight file.')

    argreader.parser.add_argument('--plot_score',action="store_true",help='To plot the scene change probability of produced by a model for all the videos processed by this model during validation for all epochs.\
                                                                            The --model_id argument must be set, along with the --exp_id, --dataset_test, --test_part_beg  and --test_part_end arguments.')
    argreader.parser.add_argument('--untrained_exp_and_model_id',default=[None,None],type=str,nargs=2,help='To plot the distance between features computed by the network before training when using the --plot_score arg.\
                                                                                        The values are the exp_id and the model_id of the model not trained on scene change detection (i.e. the model with the ImageNet weights)')
    argreader.parser.add_argument('--plot_dist',action="store_true",help='To plot the distance when using the --plot_score argument')

    argreader.parser.add_argument('--eval_model',type=float,nargs=4,help='To evaluate a model and plot the mean metrics as a function of the score threshold.\
                                    The --model_id argument must be set, along with the --exp_id, --dataset_test, --test_part_beg  and --test_part_end arguments. \
                                    The values of this args are the epochs at which to start and end the plot, followed by the minimum and maximum decision threshold \
                                    to evaluate.')

    argreader.parser.add_argument('--eval_model_leave_one_out',type=float,nargs=3,help='To evaluate a model by tuning its decision threshold on the video on which it is not\
                                    evaluated. The --model_id argument must be set, along with the --model_name, --exp_id and --dataset_test arguments. \
                                    The values of this args are the epoch at which to evaluate , followed by the minimum and maximum decision threshold \
                                    to evaluate. Use the --len_pond to ponderate the overflow and coverage by scene lengths.')

    argreader.parser.add_argument('--model_name',type=str,metavar="NAME",help='The name of the model as will appear in the latex table produced by the --eval_model_leave_one_out argument.')
    argreader.parser.add_argument('--len_pond',action="store_true",help='Use this argument to ponderate the coverage and overflow by the GT scene length when using --eval_model_leave_one_out.')

    argreader.parser.add_argument('--bbc_annot_dist',type=str,nargs=4,help='To comute the differential edit distance (DED) between annotators of the BBC dataset and one model. \
                                                                            It requires to have already evaluated the model on the bbc database.\
                                                                            The values of the arg are the following :\
                                                                            - is the path to the BBC annnotation folder downloaded.\
                                                                            - the exp_id of the model \
                                                                            - the id of the model \
                                                                            - the epoch at which the model has been evaluated. \
                                                                            The --fine_tuned_thres can also be used to tune the decision threshold of the model for each annotator,\
                                                                            each metric and each video in a leave-one-out manner')

    argreader.parser.add_argument('--fine_tuned_thres',action="store_true",help='To automatically fine tune the decision threshold of the model. Only useful for the --bbc_annot_dist arg. \
                                                                                Check the help of this arg.')

    argreader.parser.add_argument('--results_table',action="store_true",help='To write the metric value for several models. The arguments that must be set are \
                                                                            --exp_ids, --model_ids, --thres_list, --epoch_list and --dataset_test')

    argreader.parser.add_argument('--exp_id_list',type=str,nargs="*",help='The list of model experience ids (useful for the --results_table argument')
    argreader.parser.add_argument('--model_id_list',type=str,nargs="*",help='The list of model ids (useful for the --results_table argument')
    argreader.parser.add_argument('--thres_list',type=float,nargs="*",help='The list of decision threshold (useful for the --results_table argument')
    argreader.parser.add_argument('--epoch_list',type=int,nargs="*",help='The list of epoch at which is model is evaluated (useful for the --results_table argument and for --bbc_annot_dist.')

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.tsne:
        tsne(args.dataset_test,args.exp_id,args.model_id,args.seed,args.frames_per_shot)
    if args.score_vis_video:
        scoreVis_video(args.dataset_test,args.exp_id,args.score_vis_video)
    if args.plot_cat_repr:
        plotCatRepr(args.plot_cat_repr)
    if args.conv_sco_plot:
        convScoPlot(args.conv_sco_plot)
    if args.plot_score:
        plotScore(args.exp_id,args.model_id,args.untrained_exp_and_model_id[0],args.untrained_exp_and_model_id[1],args.dataset_test,args.test_part_beg,args.test_part_end,args.plot_dist)
    if args.eval_model:
        epochStart = int(args.eval_model[0])
        epochEnd = int(args.eval_model[1])
        thresMin = args.eval_model[2]
        thresMax = args.eval_model[3]
        evalModel(args.exp_id,args.model_id,args.dataset_test,args.test_part_beg,args.test_part_end,epochStart,epochEnd,thresMin,thresMax)
    if args.eval_model_leave_one_out:
        epoch = int(args.eval_model_leave_one_out[0])
        thresMin = args.eval_model_leave_one_out[1]
        thresMax = args.eval_model_leave_one_out[2]
        evalModel_leaveOneOut(args.exp_id,args.model_id,args.model_name,args.dataset_test,epoch,thresMin,thresMax,args.len_pond)
    if args.bbc_annot_dist:
        bbcAnnotDist(args.bbc_annot_dist[0],args.bbc_annot_dist[1],args.bbc_annot_dist[2],args.bbc_annot_dist[3],args.fine_tuned_thres)
    if args.results_table:
        resultTables(args.exp_id_list,args.model_id_list,args.thres_list,args.epoch_list,args.dataset_test)

if __name__ == "__main__":
    main()
