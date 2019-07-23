
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
                #print(repr_tsne.shape)
                plt.figure()
                plt.title("T-SNE view of feature from {}".format(videoName))
                plt.scatter(repr_tsne[:,0],repr_tsne[:,1], zorder=2,color=cmap[colorInds])
                plt.savefig("../vis/{}/{}_model{}_tsne.png".format(exp_id,videoName,model_id))
            else:
                print("\tT-sne already done")
        else:
            print("\tFeature for video {} does not exist".format(videoName))

def compGT(exp_id,metric,thres):
    ''' Evaluate all the models of an experiment on all the video parsed by this model

    Args:
    - exp_id (str): the experience name
    - metric (str): can only be 'IoU' for now
    - thres (float): the score threshold to determine which score is sufficient to say \
                    that there is a scene change

    '''

    modelIniPaths = glob.glob("../models/{}/*.ini".format(exp_id))
    metricFunc = globals()[metric]

    csv = "modelId,"+metric+"\n"

    for modelIniPath in modelIniPaths:

        modelId = os.path.basename(modelIniPath.replace(".ini",""))

        sceneCutsPathList = glob.glob("../results/{}/{}_*.csv".format(exp_id,modelId))
        metricsArr=np.zeros(len(sceneCutsPathList))
        for i,sceneCutsPath in enumerate(sceneCutsPathList):

            vidName = "_".join(os.path.basename(sceneCutsPath).split("_")[:-1])

            gtCuts = np.genfromtxt("../data/{}_cuts.csv".format(vidName))
            sceneCuts = (np.genfromtxt(sceneCutsPath) > thres)

            metricsArr[i] = metricFunc(gtCuts,sceneCuts)

        metrics_mean = metricsArr.mean()
        metrics_std = metricsArr.std()

        csv += modelId+","+str(metrics_mean)+"\pm"+str(metrics_std)+"\n"

    with open("../results/{}/{}.csv".format(exp_id,metric),"w") as csvFile:
        print(csv,file=csvFile)

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

        #print("pred",pred_bounds)
        #print("targ",targ_bounds)

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

def binaryToFullMetrics(pred,target):
    ''' Computes the IoU of a predicted scene segmentation using a gt and a prediction encoded in binary format

    This computes IoU relative to prediction and to ground truth and also computes the mean of the two. \

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

        #print("pred",pred_bounds)
        #print("targ",targ_bounds)

        if len(targ_bounds) > 1:
            predBounds.append(pred_bounds)
            targBounds.append(targ_bounds)

    iou,iou_pred,iou_gt,over,cover = 0,0,0,0,0
    for pred,targ in zip(predBounds,targBounds):

        iou_pred += IoU_oneRef(np.array(targ),np.array(pred))
        iou_gt += IoU_oneRef(np.array(pred),np.array(targ))
        iou += iou_pred*0.5+iou_gt*0.5
        over += overflow(np.array(targ),np.array(pred))
        cover += coverage(np.array(targ),np.array(pred))

    iou_pred /= len(targBounds)
    iou_gt /= len(targBounds)
    iou /= len(targBounds)
    over /= len(targBounds)
    cover /= len(targBounds)

    f_score = 2*cover*(1-over)/(cover+1-over)

    return iou,iou_pred,iou_gt,f_score

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

def plotScore(exp_id,model_id,exp_id_init,model_id_init,dataset_test,test_part_beg,test_part_end):

    resFilePaths = sorted(glob.glob("../results/{}/{}_epoch*.csv".format(exp_id,model_id)))

    videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset_test)))))
    videoPaths = list(filter(lambda x:x.find(".xml") == -1,videoPaths))
    videoPaths = list(filter(lambda x:os.path.isfile(x),videoPaths))
    videoPaths = np.array(videoPaths)[int(test_part_beg*len(videoPaths)):int(test_part_end*len(videoPaths))]
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    for path in resFilePaths:

        videoName = None
        for candidateVideoName in videoNames:
            if "_"+candidateVideoName.replace("__","_")+".csv" in path:
                videoName = candidateVideoName

        if not videoName is None:
            fileName = os.path.basename(os.path.splitext(path)[0])

            scores = np.genfromtxt(path)[:,1]

            #gt = np.zeros(len(scores))
            #gtScenes = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset_test,videoName))
            #gtSceneStarts = np.array(frame_to_shots(dataset_test,"../data/{}/{}/result.xml".format(dataset_test,videoName),gtScenes))[:,0].astype(int)
            #gt[gtSceneStarts] = 1

            gt = load_data.getGT(dataset_test,videoName)

            fig = plt.figure(figsize=(30,5))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            legHandles = []

            #Plot the ground truth transitions
            ax1.vlines(gt.nonzero(),0,1,linewidths=3,color='gray')

            #Plot the decision threshold
            ax1.hlines([0.5],0,len(scores),linewidths=3,color='red')

            #Plot the scores
            legHandles += ax1.plot(np.arange(len(scores)),scores,color="blue",label="Scene change score")

            #Getting the epoch using the path
            epoch = modelBuilder.findNumbers(path[path.find("epoch")+5:].split("_")[0])

            #Plot the distance between the successive trained features
            legHandles = plotFeat(exp_id,model_id,videoName,len(scores),"orange","Distance between shot trained representations",legHandles,ax2)
            legHandles = plotFeat(exp_id_init,model_id_init,videoName,len(scores),"green","Distance between untrained shot representations",legHandles,ax2)

            legend = fig.legend(handles=legHandles, loc='center right' ,title="")
            fig.gca().add_artist(legend)

            plt.savefig("../vis/{}/{}.png".format(exp_id,fileName))
            sys.exit(0)
        else:
            raise ValueError("Unkown video : ",path)

def plotFeat(exp_id,model_id,videoName,nbShots,color,label,legHandles,ax):
    featPaths = sorted(glob.glob("../results/{}/{}/*_{}.csv".format(exp_id,videoName,model_id)),key=modelBuilder.findNumbers)
    feats = np.array(list(map(lambda x:np.genfromtxt(x),featPaths)))
    dists = np.sqrt(np.power(feats[:-1]-feats[:1],2).sum(axis=1))
    legHandles += ax.plot(np.arange(nbShots-1)+1,dists,color=color,label=label)

    return legHandles

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
        print(thres)

        for k in range(lastEpoch-firstEpoch+1):

            resFilePaths = sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,k+firstEpoch)),key=modelBuilder.findNumbers)

            iou_mean,iou_pred_mean,iou_gt_mean,f_sco_mean = 0,0,0,0

            print("Epoch",k+firstEpoch,len(resFilePaths)," videos")

            for j,path in enumerate(resFilePaths):

                fileName = os.path.basename(os.path.splitext(path)[0])
                videoName = videoNameDict[path]

                gt = load_data.getGT(dataset_test,videoName)
                #gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset_test,videoName)).astype(int)
                scores = np.genfromtxt(path)[:,1]

                pred = (scores > thres)

                iou,iou_pred,iou_gt,f_sco = binaryToFullMetrics(pred[np.newaxis,:],gt[np.newaxis,:])

                iou_mean += iou
                iou_pred_mean += iou_pred
                iou_gt_mean += iou_gt
                f_sco_mean += f_sco

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
    print("Best threshold for IoU: ",thresList[bestThresInd])
    print("Best epoch : ",bestEpochInd+firstEpoch)
    print("Best IoU's : ")
    print(iouArr[bestThresInd])

    bestInd = np.argmax(fscoArr)
    bestThresInd,bestEpochInd = bestInd//iouArr.shape[1],bestInd%iouArr.shape[1]
    print("Best threshold for F-score : ",thresList[bestThresInd])
    print("Best epoch : ",bestEpochInd+firstEpoch)
    print("Best F-scores : ")
    print(fscoArr[bestThresInd])

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--comp_gt',action='store_true',help='To compare the performance of models in an experiment with the ground truth. The --exp_id argument\
                                    must be set. ')

    argreader.parser.add_argument('--compt_gt_true_baseline',nargs=2,type=str,help='To compare the performance of the true segmentation sent by the IBM reseacher \
                                    compared to ground truth. The first value is the path to the folder containing the segmentation for each video.\
                                    The second value is the outpath where the metrics for each video will be written.\
                                    The dataset argument must also be set.')

    argreader.parser.add_argument('--frame',action='store_true',help='To compute the metrics at the frame level.')

    argreader.parser.add_argument('--metric',type=str,default="IoU",metavar='METRIC',help='The metric to use. Can only be \'IoU\' for now.')

    argreader.parser.add_argument('--tsne',action='store_true',help='To plot t-sne representation of feature extracted. Also plots the representation of a video side by side to make an image. \
                                    The --exp_id, --model_id, --frames_per_shot, --seed and --dataset_test arguments should be set.')

    argreader.parser.add_argument('--score_vis_video',type=str,help='To plot the scene change score on the video itself. Requires the --dataset_test and --exp_id arguments to be set. The value is a path to a result file.')
    argreader.parser.add_argument('--plot_cat_repr',type=str,help=' The value must a path to a folder containing representations vector as CSV files.')
    argreader.parser.add_argument('--conv_sco_plot',type=str,help='To plot the frequency response of the 1D filter learned by a model filtering its scores. The value is the path to the weight file.')

    argreader.parser.add_argument('--plot_score',type=str,nargs=2,help='To plot the scene change probability of produced by a model for all the videos processed by this model during validation for all epochs.\
                                                                            The --model_id argument must be set, along with the --exp_id, --dataset_test, --test_part_beg  and --test_part_end arguments.\
                                                                            The values of this arg are the exp_id and the model_id of the model not trained on scene change detection (i.e. the model with the ImageNet weights)')

    argreader.parser.add_argument('--eval_model',type=float,nargs=4,help='To evaluate a model and plot the mean IoU as a function of the score threshold.\
                                    The --model_id argument must be set, along with the --exp_id, --dataset_test, --test_part_beg  and --test_part_end arguments. \
                                    The values of this args are the epochs at which to start and end the plot, followed by the minimum and maximum decision threshold \
                                    to evaluate.')

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.comp_gt:
        compGT(args.exp_id,args.metric)
    if args.compt_gt_true_baseline:
        comptGT_trueBaseline(args.compt_gt_true_baseline[0],args.exp_id,args.dataset_test,args.compt_gt_true_baseline[1],args.frame)
    if args.tsne:
        tsne(args.dataset_test,args.exp_id,args.model_id,args.seed,args.frames_per_shot)
    if args.score_vis_video:
        scoreVis_video(args.dataset_test,args.exp_id,args.score_vis_video)
    if args.plot_cat_repr:
        plotCatRepr(args.plot_cat_repr)
    if args.conv_sco_plot:
        convScoPlot(args.conv_sco_plot)
    if args.plot_score:
        plotScore(args.exp_id,args.model_id,args.plot_score[0],args.plot_score[1],args.dataset_test,args.test_part_beg,args.test_part_end)
    if args.eval_model:
        epochStart = int(args.eval_model[0])
        epochEnd = int(args.eval_model[1])
        thresMin = args.eval_model[2]
        thresMax = args.eval_model[3]
        evalModel(args.exp_id,args.model_id,args.dataset_test,args.test_part_beg,args.test_part_end,epochStart,epochEnd,thresMin,thresMax)

if __name__ == "__main__":
    main()
