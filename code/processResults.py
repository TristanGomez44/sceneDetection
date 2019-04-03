
from args import ArgReader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
        iou=np.zeros(len(sceneCutsPathList))
        for i,sceneCutsPath in enumerate(sceneCutsPathList):

            vidName = "_".join(os.path.basename(sceneCutsPath).split("_")[:-1])

            gtCuts = np.genfromtxt("../data/{}_cuts.csv".format(vidName))
            sceneCuts = np.genfromtxt(sceneCutsPath)

            iou[i] = metricFunc(gtCuts,sceneCuts)

        iou_mean = iou.mean()
        iou_std = iou.std()

        csv += modelId+","+str(iou_mean)+"\pm"+str(iou_std)+"\n"

    with open("../results/{}/{}.csv".format(exp_id,metric),"w") as csvFile:
        print(csv,file=csvFile)

def IoU_oneRef(sceneCuts1,sceneCuts2):

    #Will store the IoU of every scene from sceneCuts1 with every scene from sceneCuts2
    iou = np.zeros((len(sceneCuts1),len(sceneCuts2),2))

    sceneCuts1 = np.concatenate((sceneCuts1[:-1,np.newaxis],sceneCuts1[1:,np.newaxis]),axis=1)
    sceneCuts2 = np.concatenate((sceneCuts2[:-1,np.newaxis],sceneCuts2[1:,np.newaxis]),axis=1)

    iou_mean = 0
    for i in range(len(sceneCuts1)):

        iou = np.zeros(len(sceneCuts2))
        for j in range(len(sceneCuts2)):
            iou[j] = inter(sceneCuts1[i],sceneCuts2[j])/union(sceneCuts1[i],sceneCuts2[j])

        iou_mean += iou.max()

    iou_mean /= len(sceneCuts1)

    return iou_mean

def IoU(gt,pred):

    return 0.5*(IoU_oneRef(gt,pred)+IoU_oneRef(pred,gt))

def union(a,b):

    return b[1]-b[0]+a[1]-a[0]-inter(a,b)

def inter(a,b):

    if b[0] > a[1] or a[0] > b[1]:
        return 0
    else:
        return min(a[1],b[1])-max(a[0],b[0])

def main(argv=None):


    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--plot_scenebounds',nargs=3,type=str,metavar='RESFILE',help='To plot the scene boundaries found by a model in an experiment. \
                                    The argument value is the path to the similarity matrix. The second value is the path to the cut found by the model and the last is the path\
                                    to the output image file.')

    argreader.parser.add_argument('--comp_gt',action='store_true',help='To compare the performance of models in an experiment with the ground truth. The --exp_id argument\
                                    must be set. ')

    argreader.parser.add_argument('--metric',type=str,default="IoU",metavar='METRIC',help='The metric to use. Can only be \'IoU\' for now.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_scenebounds:
        plotSceneBounds(args.plot_scenebounds[0],args.plot_scenebounds[1],args.plot_scenebounds[2])

    if args.comp_gt:
        compGT(args.exp_id,args.metric)



if __name__ == "__main__":
    main()
