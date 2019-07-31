
from args import ArgReader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
import xml.etree.ElementTree as ET
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import pims
import cv2
from PIL import Image
import torch
import subprocess
from skimage.transform import resize
import load_data
import modelBuilder
import scipy as sp

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

            metrDictVid = binaryToAllMetrics(torch.tensor(pred).unsqueeze(0),torch.tensor(target).unsqueeze(0))

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

                gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,videoName)).astype(int)
                cmap = cm.rainbow(np.linspace(0, 1, gt.sum()+1))

                reps = np.array(list(map(lambda x:np.genfromtxt(x),repFilePaths)))
                imageRep = 255*(reps-reps.min())/(reps.max()-reps.min())
                imageRep = imageRep.transpose()
                imageRep = resize(imageRep,(300,1000),mode="constant", order=0,anti_aliasing=True)
                imageRep = Image.fromarray(imageRep)
                imageRep = imageRep.convert('RGB')
                imageRep.save("../vis/{}/{}_model{}_catRepr.png".format(exp_id,model_id,videoName))

                gt = gt[:len(reps)]

                colorInds = np.cumsum(gt)

                #Load the gt with the interval format
                gt_interv = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,videoName)).astype(int)

                #Computing the scene/color index of each frame
                colorInds = ((gt_interv[:,0].reshape(-1,1) <= frameInd.reshape(1,-1))*(frameInd.reshape(1,-1) <= gt_interv[:,1].reshape(-1,1))).nonzero()[0]

                repr_tsne = TSNE(n_components=2,init='pca',random_state=1,learning_rate=20).fit_transform(reps)
                plt.figure()
                plt.title("T-SNE view of feature from {}".format(videoName))
                plt.scatter(repr_tsne[:,0],repr_tsne[:,1], zorder=2,color=cmap[colorInds])
                plt.savefig("../vis/{}/{}_model{}_tsne.png".format(exp_id,videoName,model_id))
            else:
                print("\tT-sne already done")
        else:
            print("\tFeature for video {} does not exist".format(videoName))

def binaryToMetrics(pred,target):
    ''' Computes metrics of a predicted scene segmentation using a gt and a prediction encoded in binary format

    Args:
    - pred (list): the predicted scene segmentation. It is a list indicating for each shot if it is the begining of a new scene or not. A 1 indicates that \
                    the shot is the first shot of a new scene.
    - target (list): the ground truth scene segmentation. Formated the same way as pred.

    '''

    predBounds = []
    targBounds = []

    for i in range(len(pred)):

        pred_bounds = binaryToSceneBounds(pred[i])
        targ_bounds = binaryToSceneBounds(target[i])

        if len(targ_bounds) > 1:
            predBounds.append(pred_bounds)
            targBounds.append(targ_bounds)

    cov_val,overflow_val,iou_val = 0,0,0
    for pred,targ in zip(predBounds,targBounds):

        cov_val += coverage(np.array(targ),np.array(pred))
        overflow_val += overflow(np.array(targ),np.array(pred))
        iou_val += IoU(np.array(targ),np.array(pred))

    cov_val /= len(targBounds)
    overflow_val /= len(targBounds)
    iou_val /= len(targBounds)

    return cov_val,overflow_val,iou_val

def binaryToAllMetrics(predBin,targetBin):
    ''' Computes the IoU of a predicted scene segmentation using a gt and a prediction encoded in binary format

    This computes IoU relative to prediction and to ground truth and also computes the mean of the two. \

    Args:
    - predBin (list): the predicted scene segmentation. It is a list indicating for each shot if it is the begining of a new scene or not. A 1 indicates that \
                    the shot is the first shot of a new scene.
    - targetBin (list): the ground truth scene segmentation. Formated the same way as pred.


    '''

    predBounds = []
    targBounds = []

    for i in range(len(predBin)):

        pred_bounds = binaryToSceneBounds(predBin[i])
        targ_bounds = binaryToSceneBounds(targetBin[i])

        if len(targ_bounds) > 1:
            predBounds.append(pred_bounds)
            targBounds.append(targ_bounds)

    iou,iou_pred,iou_gt,over,cover,ded = 0,0,0,0,0,0
    for i,(pred,targ) in enumerate(zip(predBounds,targBounds)):

        iou_pred += IoU_oneRef(np.array(targ),np.array(pred))
        iou_gt += IoU_oneRef(np.array(pred),np.array(targ))
        iou += iou_pred*0.5+iou_gt*0.5
        over += overflow(np.array(targ),np.array(pred))
        cover += coverage(np.array(targ),np.array(pred))
        ded += computeDED(targetBin[i].unsqueeze(0),predBin[i].unsqueeze(0))

    iou_pred /= len(targBounds)
    iou_gt /= len(targBounds)
    iou /= len(targBounds)
    over /= len(targBounds)
    cover /= len(targBounds)
    ded /= len(targBounds)

    f_score = 2*cover*(1-over)/(cover+1-over)

    return {"IoU":iou,"IoU_pred":iou_pred,"IoU_gt":iou_gt,"F-score":f_score,"DED":ded}

def binaryToSceneBounds(scenesBinary):
    ''' Convert a list indicating for each shot if it is the first shot of a new scene or not \
                into a list of intervals i.e. a scene boundary array relative to shot index
    Args:
    - scenesBinary (list): a list indicating for each shot if it is the begining of a new scene or not. A 1 indicates that \
                    the shot is the first shot of a new scene. A 0 indicates otherwise.

    '''

    sceneBounds = []
    currSceneStart=0

    for i in range(len(scenesBinary)):

        if scenesBinary[i]:
            sceneBounds.append([currSceneStart,i-1])
            currSceneStart = i

    sceneBounds.append([currSceneStart,len(scenesBinary)-1])

    return sceneBounds

def scenePropToBinary(sceneProps,shotNb):
    ''' Convert a list of scene length into the binary format '''

    if type(shotNb) is int:
        shotNb = [shotNb for _ in range(len(sceneProps))]

    #Looping over batches
    binary = torch.zeros((len(sceneProps),shotNb[0])).to(sceneProps[0].device)

    for i in range(len(sceneProps)):

        sceneStarts = torch.cumsum(sceneProps[i]*shotNb[i],dim=0).long()

        sceneStarts = sceneStarts[sceneStarts < shotNb[i]]

        binary[i][sceneStarts] = 1

    return binary

def frame_to_shots(dataset,pathToXml,scenesF):
    ''' Computes scene boundaries relative with shot index instead of frame index '''

    shotsF = xmlToArray(pathToXml)

    #Computing scene boundaries with shot number instead of frame
    scenesS = []

    shotInd = 0
    sceneInd = 0

    currShotStart = 0

    while shotInd<len(shotsF) and sceneInd<len(scenesF):

        if shotsF[shotInd,0] <= scenesF[sceneInd,1] and scenesF[sceneInd,1] <= shotsF[shotInd,1]:

            #This condition is added just to prevent a non-sense line to be written at the end
            if currShotStart<=shotInd:
                scenesS.append([currShotStart,shotInd])

            currShotStart = shotInd+1

            sceneInd += 1
        else:
            shotInd+=1

    return scenesS

def shots_to_frames(pathToXml,scenesS):
    ''' Computes scene boundaries file with frame index instead of shot index '''

    shotsF = xmlToArray(pathToXml)

    scenes_startF = shotsF[:,0][scenesS[:,0].astype(int)]
    scenes_endF = shotsF[:,1][scenesS[:,1].astype(int)]

    scenesF = np.concatenate((scenes_startF[:,np.newaxis],scenes_endF[:,np.newaxis]),axis=1)

    return scenesF

def xmlToArray(xmlPath):
    ''' Read the shot segmentation for a video

    If the shot segmentation does not exist in .xml at the path indicated, \
    this function look for the segmentation in csv file, in the same folder.

     '''

    if os.path.exists(xmlPath):
        #Getting the shot bounds with frame number
        tree = ET.parse(xmlPath).getroot()
        shotsF = tree.find("content").find("body").find("shots")
        frameNb = int(shotsF[-1].get("fduration"))+int(shotsF[-1].get("fbegin"))

        shotsF = list(map(lambda x:int(x.get("fbegin")),shotsF))
        shotsF.append(frameNb)

        shotsF = np.array(shotsF)
        shotsF = np.concatenate((shotsF[:-1,np.newaxis],shotsF[1:,np.newaxis]-1),axis=1)

        return shotsF
    else:
        return np.genfromtxt(xmlPath.replace(".xml",".csv"))

def coverage(gt,pred):
    ''' Computes the coverage of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the mean coverage of the predicted scene segmentation

    '''

    cov_gt_array = np.zeros(len(gt))
    for i,scene in enumerate(gt):

        cov_pred_array = np.zeros(len(pred))
        for j,scene_pred in enumerate(pred):

            cov_pred_array[j] = inter(scene,scene_pred)/leng(scene)
        cov_gt_array[i] = cov_pred_array.max()
        #cov_gt_array[i] *= leng(scene)/(gt[-1,1]+1)

    #print(cov_gt_array)
    #return cov_gt_array.sum()
    return cov_gt_array.mean()

def leng(scene):
    ''' The number of shot in an interval, i.e. a scene '''

    return scene[1]-scene[0]+1

def overflow(gt,pred):
    ''' Computes the overflow of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the mean overflow of the predicted scene segmentation

    '''

    ov_gt_array = np.zeros(len(gt))
    for i,scene in enumerate(gt):



        #if scene[0] == 3 and scene[1] == 21:
        #    print(scene)
        #print("Ground truth scene :",i)
        #print(scene)
        ov_pred_array = np.zeros(len(pred))
        for j,scene_pred in enumerate(pred):
            #if scene[0] == 3 and scene[1] == 21:
            #    print(scene_pred,minus(scene_pred,scene),(inter(scene_pred,gt[i-1])>0),(inter(scene_pred,gt[i+1])>0))
            #ov_pred_array[j] = minus(scene_pred,scene)*(inter(scene_pred,gt[i-1])>0)*(inter(scene_pred,gt[i+1])>0)
            #ov_pred_array[j] = minus(scene_pred,scene)*(inter(scene_pred,gt[i-1])+inter(scene_pred,gt[i+1])>0)
            ov_pred_array[j] = minus(scene_pred,scene)*min(1,inter(scene_pred,scene))

        #if scene[0] == 3 and scene[1] == 21:
        #    print("ov_pred_array",ov_pred_array,(leng(gt[i-1])+leng(gt[i+1])))
        if i>0 and i<len(gt)-1:
            ov_gt_array[i] = min(ov_pred_array.sum()/(leng(gt[i-1])+leng(gt[i+1])),1)
        elif i == 0:
            ov_gt_array[i] = min(ov_pred_array.sum()/(leng(gt[i+1])),1)
        elif i == len(gt)-1:

            ov_gt_array[i] = min(ov_pred_array.sum()/(leng(gt[i-1])),1)

        #ov_gt_array[i] *= leng(scene)/(gt[-1,1]+1)

        #print(leng(scene),gt[-1,1],leng(gt[0]),leng(gt[-1]))

    #print(ov_gt_array)
    #return ov_gt_array.sum()
    return ov_gt_array.mean()

def IoU(gt,pred):
    ''' Computes the Intersection over Union of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the IoU of the predicted scene segmentation with the ground-truth

    '''

    #The IoU is first computed relative to the ground truth and then relative to the prediction
    return 0.5*(IoU_oneRef(gt,pred)+IoU_oneRef(pred,gt))

def IoU_oneRef(sceneCuts1,sceneCuts2):
    ''' Compute the IoU of a segmentation relative another '''

    #Will store the IoU of every scene from sceneCuts1 with every scene from sceneCuts2
    iou = np.zeros((len(sceneCuts1),len(sceneCuts2),2))

    iou_mean = 0
    for i in range(len(sceneCuts1)):

        iou = np.zeros(len(sceneCuts2))
        for j in range(len(sceneCuts2)):
            iou[j] = inter(sceneCuts1[i],sceneCuts2[j])/union(sceneCuts1[i],sceneCuts2[j])

        iou_mean += iou.max()
    iou_mean /= len(sceneCuts1)

    return iou_mean

def union(a,b):
    ''' The union between two intervals '''

    return b[1]-b[0]+1+a[1]-a[0]+1-inter(a,b)

def inter(a,b):
    ''' The intersection between two intervals '''

    if b[0] > a[1] or a[0] > b[1]:
        return 0
    else:
        return min(a[1],b[1])-max(a[0],b[0])+1

def IoU_par(gt,pred):
    ''' Computes the Intersection over Union of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the IoU of the predicted scene segmentation with the ground-truth

    '''

    #The IoU is first computed relative to the ground truth and then relative to the prediction
    return 0.5*(IoU_oneRef_par(gt,pred)+IoU_oneRef_par(pred,gt))

def IoU_oneRef_par(sceneCuts1,sceneCuts2):
    iou = inter_par(sceneCuts1.unsqueeze(1),sceneCuts2.unsqueeze(0))/union_par(sceneCuts1.unsqueeze(1),sceneCuts2.unsqueeze(0))

    return torch.max(iou,dim=1)[0].mean()

def union_par(a,b):

    return b[:,:,1]-b[:,:,0]+1+a[:,:,1]-a[:,:,0]+1-inter_par(a,b)

def inter_par(a,b):

    return (1-((b[:,:,0] > a[:,:,1])+(a[:,:,0] > b[:,:,1])>=1)).float()*(torch.min(a[:,:,1],b[:,:,1])-torch.max(a[:,:,0],b[:,:,0])+1)

def minus(a,b):
    ''' the interval a minus the interval b '''

    totalLen = 0
    bVal = np.arange(int(b[0]),int(b[1])+1)

    for shotInd in range(int(a[0]),int(a[1])+1):
        if not shotInd in bVal:
            totalLen += 1

    return totalLen

def computeDED(segmA,segmB):
    """ Computes the differential edit distance.

    Args:
        - segmA (array) a scene segmentation in the binary format. There is one binary digit per shot.\
         1 if the shot starts a new scene. 0 else.
        - segmB (array) another scene segmentation in the same format as segmA.
     """
    segmA,segmB = torch.cumsum(segmA,dim=-1),torch.cumsum(segmB,dim=-1)

    ded = 0

    #For each example in the batch
    for i in range(len(segmA)):

        #It is required that segmA is the sequence with the greatest number of scenes
        if segmB[i].max() > segmA[i].max():
            segmA[i],segmB[i] = segmB[i],segmA[i]
        else:
            segmA[i],segmB[i] = segmA[i],segmB[i]
        occMat = torch.zeros((torch.max(segmB[i])+1,torch.max(segmA[i])+1))
        for j in range(len(segmA[i])):
            occMat[segmB[i][j],segmA[i][j]] += 1

        costMat = torch.max(occMat)-occMat

        assign = sp.optimize.linear_sum_assignment(costMat)

        correctAssignedShots = np.array([occMat[p[0],p[1]] for p in zip(assign[0],assign[1])]).sum()

        ded += (len(segmB[i])-correctAssignedShots)/len(segmB[i])

    return ded/len(segmA)

def bbcAnnotDist(annotFold,modelExpId,modelId,modelEpoch):

    #The number and names of episodes
    epiNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),sorted(glob.glob(annotFold+"/scenes/annotator_0/*.txt"))))
    epNb = len(epiNames)
    #The number of annotators
    annotNb = len(list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),sorted(glob.glob(annotFold+"/scenes/annotator_*/")))))
    distMat = -np.ones((epNb,annotNb+1,annotNb+1))
    distMat = {"DED":distMat.copy(),"IoU":distMat.copy(),"F-score":distMat.copy()}

    for i in range(epNb):

        shotNb = np.genfromtxt("../data/bbc/{}/result.csv".format(i)).shape[0]

        for j in range(annotNb+1):

            for k in range(annotNb+1):

                #This function will set the decision threshold of the model to make it predict aproximately
                #as much scene as the human annotator did. It is necessary to adjust the threshold when using
                #this metric because it favors models that predicts a few scenes (0 scenes gives a perfect scores)
                segmJ,segmK = readBothSeg(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,k,annotNb,shotNb)
                distMat["DED"][i,j,k] = computeDED(torch.tensor(segmJ).unsqueeze(0),torch.tensor(segmK).unsqueeze(0))

                segmJ = binaryToSceneBounds(ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb))
                segmK = binaryToSceneBounds(ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb))

                over = overflow(np.array(segmJ),np.array(segmK))
                cover = coverage(np.array(segmJ),np.array(segmK))
                f_score = 2*cover*(1-over)/(cover+1-over)
                distMat["F-score"][i,j,k] = f_score

                distMat["IoU"][i,j,k] = (IoU_oneRef(np.array(segmJ),np.array(segmK))+IoU_oneRef(np.array(segmK),np.array(segmJ)))/2

        for metric in distMat.keys():
            plotHeatMapWithValues(distMat[metric][i],"../vis/bbc_annot{}_ep{}_w{}.png".format(metric,i,modelId))

    for metric in distMat.keys():
        plotHeatMapWithValues(distMat[metric].mean(axis=0),"../vis/bbc_annot{}_allEp_w{}.png".format(metric,modelId))

def readBothSeg(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,k,annotNb,shotNb):

    if j==annotNb:
        segmK = ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb)
        segmJ = ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb,segmK.sum())
    elif k==annotNb:
        segmJ = ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb)
        segmK = ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb,segmJ.sum())
    else:
        segmJ = ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,j,annotNb),shotNb)
        segmK = ToBinary(getPath(annotFold,modelExpId,modelId,modelEpoch,i,epiNames,k,annotNb),shotNb)

    return segmJ,segmK

def getPath(annotFold,modelExpId,modelId,modelEpoch,epiInd,epiNames,annotationInd,annotatorNb):

    if annotationInd < annotatorNb:
        return annotFold+"/scenes/annotator_{}/{}.txt".format(annotationInd,epiNames[epiInd])
    else:
        return "../results/{}/{}_epoch{}_{}.csv".format(modelExpId,modelId,modelEpoch,epiInd)

def plotHeatMapWithValues(mat,path):
    # Limits for the extent
    x_start = 0.0
    x_end = mat.shape[0]
    y_start = 0
    y_end = mat.shape[1]

    extent = [x_start, x_end, y_start, y_end]

    size = mat.shape[0]

    # The normal figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, extent=extent, origin='lower', interpolation='None', cmap='viridis')

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
            ax.text(text_x, text_y, label, color='black', ha='center', va='center',fontsize='xx-large')

    fig.colorbar(im)
    fig.savefig(path)
    plt.close()

def ToBinary(segmPath,shotNb,targetNbScene=None):
    ''' Read and convert a scene segmentation from the format given in the BBC dataset (list of starts on one line)
    into the binary format (list of 0 and 1, with one where the scene is changing).

    It also read segmentation in the model format, which is the format produced after the eval function of trainVal.py

    '''
    segm = np.genfromtxt(segmPath)

    if np.isnan(segm).any():
        segm = np.genfromtxt(segmPath,delimiter=",")[:-1]
        binary = np.zeros(shotNb)
        binary[segm.astype(int)] = 1
        binary[0] = 0
        return binary.astype(int)
    else:
        if not targetNbScene is None:
            #Finding the threshold to get a number of scene as close as possible from the desired number
            thres = 1
            nbScene = 0
            while nbScene <  targetNbScene:
                thres -= 0.05
                precNbScene = nbScene
                nbScene = (segm[:,1]>thres).sum()

            if np.abs(precNbScene-targetNbScene)<np.abs(nbScene-targetNbScene):
                thres += 0.05
        else:
            thres = 0.5

        return (segm[:,1]>thres).astype(int)

def continuousIoU(batch,gt):

    ious = torch.empty(len(batch),requires_grad=False)
    for i,output in enumerate(batch):

        firstInd = torch.tensor([0.0]).to(gt.device)
        endInd = torch.tensor([1.0]).to(gt.device)

        output = torch.cat((firstInd,output),dim=0)
        output = torch.cumsum(output,dim=0)*len(gt[i])

        output = output[output < len(gt[i])]

        output = torch.cat((output,endInd*len(gt[i])),dim=0)

        ends = (output[:-1]+9*output[1:])/10
        ends[-1] = len(gt[i])

        starts = output[:-1]

        scenes_pred = torch.cat((starts.unsqueeze(1),ends.unsqueeze(1)),dim=1)

        scenes_gt = torch.tensor(binaryToSceneBounds(gt[i])).float().to(gt.device)

        ious[i] = IoU_par(scenes_gt,scenes_pred)

    return ious.mean()

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
    shotFrames = xmlToArray("../data/{}/{}/result.xml".format(dataset))

    cap = cv2.VideoCapture(videoPath)
    print(cap)
    shotCount = 0

    widProp = 0.6
    heigProp = 0.1
    heigShifProp = 0.1

    i=0
    success=True
    imgWidth,imgHeigth = None,None

    fps = getVideoFPS(videoPath)
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

def getVideoFPS(videoPath):
    ''' Get the number of frame per sencond of a video.'''

    return float(pims.Video(videoPath)._frame_rate)

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
    featPaths = sorted(glob.glob("../results/{}/{}/*_{}.csv".format(exp_id,videoName,model_id)),key=modelBuilder.findNumbers)

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

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch*.csv".format(exp_id,model_id)),key=modelBuilder.findNumbers))

    if firstEpoch is None:
        firstEpoch = modelBuilder.findNumbers(resFilePaths[0][resFilePaths[0].find("epoch")+5:].split("_")[0])

    if lastEpoch is None:
        lastEpoch = modelBuilder.findNumbers(resFilePaths[-1][resFilePaths[-1].find("epoch")+5:].split("_")[0])

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

            resFilePaths = sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,k+firstEpoch)),key=modelBuilder.findNumbers)

            iou_mean,iou_pred_mean,iou_gt_mean,f_sco_mean = 0,0,0,0

            for j,path in enumerate(resFilePaths):

                fileName = os.path.basename(os.path.splitext(path)[0])
                videoName = videoNameDict[path]

                gt = load_data.getGT(dataset_test,videoName).astype(int)
                #gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset_test,videoName)).astype(int)
                scores = np.genfromtxt(path)[:,1]

                pred = (scores > thres).astype(int)

                metrDict = binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))


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

def evalModel_leaveOneOut(exp_id,model_id,model_name,dataset_test,epoch,firstThres,lastThres):

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,epoch)),key=modelBuilder.findNumbers))
    videoNameDict = buildVideoNameDict(dataset_test,0,1,resFilePaths)
    resFilePaths = np.array(list(filter(lambda x:x in videoNameDict.keys(),resFilePaths)))

    thresList = np.arange(firstThres,lastThres,step=(lastThres-firstThres)/10)

    #Store the value of the f-score of for video and for each threshold
    metTun = {}
    metEval = {"IoU":    np.zeros(len(resFilePaths)),\
               "F-score":np.zeros(len(resFilePaths)),\
               "DED":    np.zeros(len(resFilePaths))}

    metDef = {"IoU":    np.zeros(len(resFilePaths)),\
              "F-score":np.zeros(len(resFilePaths)),\
              "DED":    np.zeros(len(resFilePaths))}

    for j,path in enumerate(resFilePaths):

        fileName = os.path.basename(os.path.splitext(path)[0])
        videoName = videoNameDict[path]

        metEval["F-score"][j],metDef["F-score"][j] = findBestThres_computeFsco(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"F-score")
        metEval["IoU"][j],metDef["IoU"][j] = findBestThres_computeFsco(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"IoU")
        metEval["DED"][j],metDef["DED"][j] = findBestThres_computeFsco(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,"DED")

    printHeader = not os.path.exists("../results/{}_metrics.csv".format(dataset_test))

    with open("../results/{}_metrics.csv".format(dataset_test),"a") as text_file:
        if printHeader:
            print("Model,Threshold tuning,F-score,IoU,DED",file=text_file)

        print("\multirow{2}{*}{"+model_name+"}"+"&"+"Yes"+"&"+formatMetr(metEval["F-score"])+"&"+formatMetr(metEval["IoU"])+"&"+formatMetr(metEval["DED"])+"\\\\",file=text_file)
        print("&"+"No"+"&"+formatMetr(metDef["F-score"])+"&"+formatMetr(metDef["IoU"])+"&"+formatMetr(metDef["DED"])+"\\\\",file=text_file)
        print("\hline",file=text_file)

    print("Best F-score : ",str(round(metEval["F-score"].mean(),2)),"Default F-score :",str(round(metDef["F-score"].mean(),2)),\
          "Best IoU :",str(round(metEval["IoU"].mean(),2)),"Default IoU :",str(round(metDef["IoU"].mean(),2)))

def findBestThres_computeFsco(path,videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,metric):
    _,thres = bestThres(videoName,resFilePaths,thresList,dataset_test,videoNameDict,metTun,metric)

    gt = load_data.getGT(dataset_test,videoName).astype(int)
    scores = np.genfromtxt(path)[:,1]

    pred = (scores > thres).astype(int)
    metr_dict = binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))

    pred = (scores > 0.5).astype(int)
    def_metr_dict = binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))

    return metr_dict[metric],def_metr_dict[metric]

def formatMetr(metricValuesArr):

    return "$"+str(round(metricValuesArr.mean(),2))+" \pm "+str(round(metricValuesArr.std(),2))+"$"

def bestThres(videoToEvalName,resFilePaths,thresList,dataset,videoNameDict,metTun,metric):

    optiFunc = np.min if metric == "DED" else np.max
    argOptiFunc = np.argmin if metric == "DED" else np.argmax

    metr_list = np.zeros(len(thresList))

    for i,thres in enumerate(thresList):

        mean = 0
        for j,path in enumerate(resFilePaths):

            if videoNameDict[path] != videoToEvalName:

                key = videoToEvalName+str(thres)+metric
                if not key in metTun.keys():

                    gt = load_data.getGT(dataset,videoNameDict[path]).astype(int)
                    scores = np.genfromtxt(path)[:,1]
                    pred = (scores > thres).astype(int)
                    metrDict = binaryToAllMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))
                    metTun[key] = metrDict[metric]

                mean += metTun[key]

        metr_list[i] = mean/(len(resFilePaths)-1)

    return optiFunc(metr_list),thresList[argOptiFunc(metr_list)]

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
                                    to evaluate.')

    argreader.parser.add_argument('--model_name',type=str,metavar="NAME",help='The name of the model as will appear in the latex table produced by the --eval_model_leave_one_out argument.')

    argreader.parser.add_argument('--bbc_annot_dist',type=str,nargs=4,help='To comute the differential edit distance (DED) between annotators of the BBC dataset and one model. \
                                                                            It requires to have already evaluated the model on the bbc database.\
                                                                            The values of the arg are the following :\
                                                                            - is the path to the BBC annnotation folder downloaded.\
                                                                            - the exp_id of the model \
                                                                            - the id of the model \
                                                                            - the epoch at which the model has been evaluated.')

    argreader.parser.add_argument('--results_table',action="store_true",help='To write the metric value for several models. The arguments that must be set are \
                                                                            --exp_ids, --model_ids, --thres_list, --epoch_list and --dataset_test')

    argreader.parser.add_argument('--exp_id_list',type=str,nargs="*",help='The list of model experience ids (useful for the --results_table argument')
    argreader.parser.add_argument('--model_id_list',type=str,nargs="*",help='The list of model ids (useful for the --results_table argument')
    argreader.parser.add_argument('--thres_list',type=float,nargs="*",help='The list of decision threshold (useful for the --results_table argument')
    argreader.parser.add_argument('--epoch_list',type=int,nargs="*",help='The list of epoch at which is model is evaluated (useful for the --results_table argument')

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
        evalModel_leaveOneOut(args.exp_id,args.model_id,args.model_name,args.dataset_test,epoch,thresMin,thresMax)
    if args.bbc_annot_dist:
        bbcAnnotDist(args.bbc_annot_dist[0],args.bbc_annot_dist[1],args.bbc_annot_dist[2],args.bbc_annot_dist[3])
    if args.results_table:
        resultTables(args.exp_id_list,args.model_id_list,args.thres_list,args.epoch_list,args.dataset_test)

if __name__ == "__main__":
    main()
