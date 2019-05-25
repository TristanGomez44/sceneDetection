
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

import subprocess

def tsne(dataset,exp_id,model_id,seed,nb_scenes=10):

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

                gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,videoName)).astype(int)
                cmap = cm.rainbow(np.linspace(0, 1, gt.sum()+1))

                reps = np.array(list(map(lambda x:np.genfromtxt(x),repFilePaths)))

                gt = gt[:len(reps)]

                repr_tsne = TSNE(n_components=2,init='pca',random_state=1,learning_rate=20).fit_transform(reps)

                plt.figure()
                plt.title("T-SNE view of feature from {}".format(videoName))
                plt.scatter(repr_tsne[:,0],repr_tsne[:,1], zorder=2,color=cmap[np.cumsum(gt)])
                plt.savefig("../vis/{}/{}_model{}_tsne.png".format(exp_id,videoName,model_id))
            else:
                print("\tT-sne already done")
        else:
            print("\tFeature for video does not {} exist".format(videoName))

def compGT(exp_id,metric,thres):

    modelIniPaths = glob.glob("../models/{}/*.ini".format(exp_id))
    metricFunc = globals()[metric]

    csv = "modelId,"+metric+"\n"

    for modelIniPath in modelIniPaths:

        modelId = os.path.basename(modelIniPath.replace(".ini",""))

        sceneCutsPathList = glob.glob("../results/{}/{}_*.csv".format(exp_id,modelId))
        print(sceneCutsPathList)
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

def comptGT_trueBaseline(truebaseFold,exp_id,dataset,resPath,frame):

    resCSV = "video,coverage,overflow,F-score,IoU\n"

    for trueBaseCutsPath in sorted(glob.glob(truebaseFold+"/*.csv")) :

        truebasecuts = np.genfromtxt(trueBaseCutsPath)

        if len(truebasecuts.shape) < 2:
            truebasecuts = [truebasecuts]

        fileName = os.path.basename(trueBaseCutsPath)
        if frame:
            pos = fileName.find("_frames_truebasecuts.csv")
        else:
            pos = fileName.find("_truebasecuts.csv")

        vidName = fileName[:pos]

        if frame:
            gt = np.genfromtxt("../data/{}/annotations/{}_frames_scenes.csv".format(dataset,vidName))
        else:
            gt = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName))

        cov = coverage(gt,truebasecuts)
        over = overflow(gt,truebasecuts)
        f_score = 2*cov*(1-over)/(cov+(1-over))

        iou = IoU(gt,truebasecuts)

        resCSV +=vidName+","+str(cov)+","+str(over)+","+str(f_score)+","+str(iou)+"\n"
        print(vidName+","+str(cov)+","+str(over)+","+str(f_score)+","+str(iou))

    with open(resPath,"w") as text_file:
        print(resCSV,file=text_file)

def binaryToMetrics(pred,target):

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

def binaryToSceneBounds(scenesBinary):

    sceneBounds = []
    currSceneStart=0

    for i in range(len(scenesBinary)):

        if scenesBinary[i]:
            sceneBounds.append([currSceneStart,i-1])
            currSceneStart = i

    sceneBounds.append([currSceneStart,len(scenesBinary)-1])

    return sceneBounds

def frame_to_shots(dataset,vidName,scenesF):
    ''' Computes scene boundaries file with shot index instead of frame index '''

    shotsF = xmlToArray(dataset,vidName)

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

def shots_to_frames(dataset,vidName,scenesS):

    ''' Computes scene boundaries file with shot index instead of frame index '''

    shotsF = xmlToArray(dataset,vidName)

    print(scenesS)
    print(shotsF.shape)

    scenes_startF = shotsF[:,0][scenesS[:,0].astype(int)]
    scenes_endF = shotsF[:,1][scenesS[:,1].astype(int)]

    scenesF = np.concatenate((scenes_startF[:,np.newaxis],scenes_endF[:,np.newaxis]),axis=1)
    #sys.exit(0)

    return scenesF

def xmlToArray(dataset,vidName):

    return xmlToArray("../data/{}/{}/result.xml".format(dataset,vidName))

def xmlToArray(xmlPath):

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

def removeHolesScenes(scenesS):
    ''' Some scene boundaries file can be inconsistent : a scene can finish at frame/shot x
    and the next scene can start at frame/shot y>x+1, leading to a few frame/shot belonging to no scene.
    This function solves this by putting the scene cut in the middle of the hole. '''

    holesDetected = False
    for i in range(len(scenesS)-1):

        if scenesS[i,1] + 1 < scenesS[i+1,0]:

            middleInd = (scenesS[i,1] + scenesS[i+1,0])//2

            scenesS[i,1] = middleInd
            scenesS[i+1,0] = middleInd+1

            holesDetected = True

    return scenesS

def coverage(gt,pred):

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

    return scene[1]-scene[0]+1

def overflow(gt,pred):

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

    return 0.5*(IoU_oneRef(gt,pred)+IoU_oneRef(pred,gt))

def IoU_oneRef(sceneCuts1,sceneCuts2):

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

    return b[1]-b[0]+1+a[1]-a[0]+1-inter(a,b)

def inter(a,b):

    if b[0] > a[1] or a[0] > b[1]:
        return 0
    else:
        return min(a[1],b[1])-max(a[0],b[0])+1

def minus(a,b):

    totalLen = 0
    bVal = np.arange(int(b[0]),int(b[1])+1)

    for shotInd in range(int(a[0]),int(a[1])+1):
        if not shotInd in bVal:
            totalLen += 1

    return totalLen

def scoreVis_video(dataset,exp_id,resFilePath,nbScoToPlot=11):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    resFile = np.genfromtxt(resFilePath)
    frameInds = resFile[:,:-1]
    scores = resFile[:,-1]

    splitResFileName = os.path.splitext(os.path.basename(resFilePath))[0].split("_")

    i =0
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
    shotFrames = xmlToArray("../data/{}/{}/result.xml".format(dataset,videoName))

    cap = cv2.VideoCapture(videoPath)

    shotCount = 0

    widProp = 0.6
    heigProp = 0.1
    heigShifProp = 0.1

    i=0
    success=True
    imgWidth,imgHeigth = None,None

    fps = getVideoFPS(videoPath)

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
    subprocess.call("ffmpeg -i {} 2>info.txt".format(videoPath),shell=True)
    with open('info.txt', 'r') as infoFile:
        infos = infoFile.read()
    fps = None
    for line in infos.split("\n"):

        if line.find("fps") != -1:
            for info in line.split(","):
                if info.find("fps") != -1:
                    fps = round(float(info.replace(" ","").replace("fps","")))
    if not fps:
        raise ValueError("FPS info not found in info.txt")

    return fps

def scoreVis_frames(dataset,resFilePath):

    resFile = np.genfromtxt(resFilePath)

    frameInds = resFile[:,:-1]
    scores = resFile[:,-1]

    videoName = os.path.splitext(resFilePath)[0].split("_")[2:]
    videoName = "_".join(videoName)

    videoPath = list(filter(lambda x:x.find("wav")==-1,glob.glob("../data/"+dataset+"/"+videoName+"*.*")))[0]

    video = pims.Video(videoPath)

    nbCols = int(np.sqrt(len(resFile[0])-1))+1
    nbRows = nbCols

    exp_id = os.path.dirname(resFilePath).split("/")[-1]
    model_id = os.path.basename(os.path.splitext(resFilePath)[0].split("_")[0])

    for i,score in enumerate(scores):

        imgList = []

        for frameInd in frameInds[i]:

            frame = video[frameInd]
            frame = frame.reshape(frame.shape[1],frame.shape[0],3)
            imgList.append(frame)

        nbImageAdded = 0
        bigImage = None
        for j in range(nbRows):

            if nbImageAdded<len(imgList):
                row = imgList[nbImageAdded]
                nbImageAdded += 1

                for k in range(nbCols):
                    if nbImageAdded<len(imgList):
                        row = np.concatenate((row,imgList[nbImageAdded]),axis=1)
                        nbImageAdded += 1

                if j==0:
                    bigImage = row
                else:
                    bigImage = np.concatenate((bigImage,row),axis=0)


        heigth = 60
        width = heigth//2
        offX = 10
        offY = offX

        cv2.rectangle(bigImage, (offX,offY), (offX+int(width*2),offY+heigth), (0,0,0),thickness=-1)
        cv2.rectangle(bigImage, (offX,offY+int(heigth*score)), (offX+width,offY+heigth), (255,255,255),thickness=-1)
        cv2.rectangle(bigImage, (offX,offY), (offX+width,offY+heigth), (255,0,0),thickness=2)

        for j in range(6):
            cv2.line(bigImage,(offX+width-5,offY+heigth*j//5),(offX+width+5,offY+heigth*j//5),(255,0,0),2)
            cv2.putText(bigImage,str(10*round((5-j)*2)),(offX+width+10,offY+int(j*0.2*heigth)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),2,cv2.LINE_AA)


        bigImage = Image.fromarray(bigImage)
        bigImage.save("../vis/{}/{}_{}_{}.png".format(exp_id,model_id,videoName,frameInds[i,0]))



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

    argreader.parser.add_argument('--tsne',action='store_true',help='To plot t-sne representation of feature extracted. The --exp_id, --model_id, --seed and --dataset_test arguments should\
                                    be set.')

    argreader.parser.add_argument('--score_vis_frames',type=str,help='To plot the image used to make decisions and their respective score. Requires the --dataset_test argument to be set. The value is a path to a result file.')
    argreader.parser.add_argument('--score_vis_video',type=str,help='To plot the scene change score on the video itself. Requires the --dataset_test and --exp_id arguments to be set. The value is a path to a result file.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.comp_gt:
        compGT(args.exp_id,args.metric)
    if args.compt_gt_true_baseline:
        comptGT_trueBaseline(args.compt_gt_true_baseline[0],args.exp_id,args.dataset_test,args.compt_gt_true_baseline[1],args.frame)
    if args.tsne:
        tsne(args.dataset_test,args.exp_id,args.model_id,args.seed)
    if args.score_vis_frames:
        scoreVis_frames(args.dataset_test,args.score_vis_frames)
    if args.score_vis_video:
        scoreVis_video(args.dataset_test,args.exp_id,args.score_vis_video)

if __name__ == "__main__":
    main()
