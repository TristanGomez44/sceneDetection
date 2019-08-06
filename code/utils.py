import pims
import numpy as np
import xml.etree.ElementTree as ET
import os
import processResults

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

def framesToShot(scenesBounds,shotBounds):
    ''' Convert a list of scene bounds expressed with frame index into a list of scene bounds expressed with shot index

    Args:
    - scenesBounds (array): the scene bounds expressed with frame index
    - shotBounds (array): the shot bounds expressed with frame index

    Returns:
    - gt (array): the scene bounds expressed with shot index

    '''

    gt = np.zeros(len(shotBounds))
    sceneInd = 0

    gt[np.power(scenesBounds[:,0][np.newaxis,:]-shotBounds[:,0][:,np.newaxis],2).argmin(axis=0)] = 1
    gt[0] = 0

    return gt

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

def getPath(annotFold,modelExpId,modelId,modelEpoch,epiInd,epiNames,annotationInd,annotatorNb):

    if annotationInd < annotatorNb:
        return annotFold+"/scenes/annotator_{}/{}.txt".format(annotationInd,epiNames[epiInd])
    else:
        return "../results/{}/{}_epoch{}_{}.csv".format(modelExpId,modelId,modelEpoch,epiInd)

def toBinary(segmPath,shotNb,**kwargs):
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

        if kwargs["fine_tuned_thres"] == True:
            _,thres = processResults.bestThres(kwargs["videoName"],kwargs["resFilePaths"],kwargs["thresList"],kwargs["dataset"],\
                              kwargs["videoNameDict"],kwargs["metTun"],kwargs["metric"],kwargs["annotatorGT"])
            print(kwargs["metric"],thres)
        else:
            thres = 0.5

        return (segm[:,1]>thres).astype(int)

def getVideoFPS(videoPath):
    ''' Get the number of frame per sencond of a video.'''

    return float(pims.Video(videoPath)._frame_rate)

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

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
