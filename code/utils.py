import pims
import numpy as np
import xml.etree.ElementTree as ET
import os 
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

def getPath(annotFold,modelExpId,modelId,modelEpoch,epiInd,epiNames,annotationInd,annotatorNb):

    if annotationInd < annotatorNb:
        return annotFold+"/scenes/annotator_{}/{}.txt".format(annotationInd,epiNames[epiInd])
    else:
        return "../results/{}/{}_epoch{}_{}.csv".format(modelExpId,modelId,modelEpoch,epiInd)

def toBinary(segmPath,shotNb,targetNbScene=None):
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

def getVideoFPS(videoPath):
    ''' Get the number of frame per sencond of a video.'''

    return float(pims.Video(videoPath)._frame_rate)
