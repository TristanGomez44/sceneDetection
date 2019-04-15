
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
def plotSceneBounds(simMatPath, modelCutPath,outPath):

    sceneCuts = np.genfromtxt(modelCutPath)

    simMat = np.genfromtxt(simMatPath)

    plt.figure()
    plt.imshow(simMat.astype(int), cmap='gray', interpolation='nearest')

    for sceneCut in sceneCuts:
        plt.plot([sceneCut-10,sceneCut+10],[sceneCut+10,sceneCut-10],"-",color="red")

    plt.xlim(0,len(simMat))
    plt.ylim(len(simMat),0)
    plt.savefig(outPath)

def compGT(exp_id,metric):

    modelIniPaths = glob.glob("../models/{}/*.ini".format(exp_id))
    metricFunc = globals()[metric]

    csv = "modelId,"+metric+"\n"

    for modelIniPath in modelIniPaths:

        modelId = os.path.basename(modelIniPath.replace(".ini",""))

        sceneCutsPathList = glob.glob("../results/{}/*_{}.csv".format(exp_id,modelId))
        print(sceneCutsPathList)
        metricsArr=np.zeros(len(sceneCutsPathList))
        for i,sceneCutsPath in enumerate(sceneCutsPathList):

            vidName = "_".join(os.path.basename(sceneCutsPath).split("_")[:-1])

            gtCuts = np.genfromtxt("../data/{}_cuts.csv".format(vidName))
            sceneCuts = np.genfromtxt(sceneCutsPath)

            metricsArr[i] = metricFunc(gtCuts,sceneCuts)

        metrics_mean = iou.mean()
        metrics_std = iou.std()

        csv += modelId+","+str(metrics_mean)+"\pm"+str(metrics_std  )+"\n"

    with open("../results/{}/{}.csv".format(exp_id,metric),"w") as csvFile:
        print(csv,file=csvFile)

def comptGT_trueBaseline(truebaseFold,exp_id,dataset,resPath,frame):

    resCSV = "video,coverage,overflow,F-score\n"

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

        resCSV +=vidName+","+str(cov)+","+str(over)+","+str(f_score)+"\n"
        print(vidName+","+str(cov)+","+str(over)+","+str(f_score))

    with open(resPath,"w") as text_file:
        print(resCSV,file=text_file)

def binaryToMetrics(pred,target,seqLen):

    predBounds = []
    targBounds = []

    for i in range(len(pred)):

        pred_bounds = binaryToSceneBounds(pred[i][:seqLen[i]])
        targ_bounds = binaryToSceneBounds(target[i][:seqLen[i]])

        print("pred",pred_bounds)
        print("targ",targ_bounds)

        if len(targ_bounds) > 1:
            predBounds.append(pred_bounds)
            targBounds.append(targ_bounds)

    cov_val,overflow_val = 0,0
    for pred,targ in zip(predBounds,targBounds):

        cov_sam = coverage(np.array(targ),np.array(pred))
        overf_sam = overflow(np.array(targ),np.array(pred))

        cov_val += cov_sam
        overflow_val += overf_sam

    cov_val /= len(targBounds)
    overflow_val /= len(targBounds)

    return cov_val,overflow_val

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

    #Getting the shot bounds with frame number
    tree = ET.parse("../data/{}/{}/result.xml".format(dataset,vidName)).getroot()
    shotsF = tree.find("content").find("body").find("shots")
    frameNb = int(shotsF[-1].get("fduration"))+int(shotsF[-1].get("fbegin"))

    shotsF = list(map(lambda x:int(x.get("fbegin")),shotsF))
    shotsF.append(frameNb)

    shotsF = np.array(shotsF)
    shotsF = np.concatenate((shotsF[:-1,np.newaxis],shotsF[1:,np.newaxis]-1),axis=1)

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

    #scenesS = removeHolesScenes(np.array(scenesS))
    #scenesS = removeHolesScenes(scenesS)

    return scenesS

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

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--plot_scenebounds',nargs=3,type=str,metavar='RESFILE',help='To plot the scene boundaries found by a model in an experiment. \
                                    The argument value is the path to the similarity matrix. The second value is the path to the cut found by the model and the last is the path\
                                    to the output image file.')

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

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_scenebounds:
        plotSceneBounds(args.plot_scenebounds[0],args.plot_scenebounds[1],args.plot_scenebounds[2])

    if args.comp_gt:
        compGT(args.exp_id,args.metric)
    if args.compt_gt_true_baseline:
        comptGT_trueBaseline(args.compt_gt_true_baseline[0],args.exp_id,args.dataset_test,args.compt_gt_true_baseline[1],args.frame)
    if args.tsne:

        tsne(args.dataset_test,args.exp_id,args.model_id,args.seed)

if __name__ == "__main__":
    main()
