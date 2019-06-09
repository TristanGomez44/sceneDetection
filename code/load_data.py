
import shotDetect
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np
import glob
import modelBuilder
import torch

import subprocess

import soundfile as sf
import math
from PIL import Image
from torchvision import transforms

import vggish_input
import processResults
import pims
import time
import torchvision

import warnings
warnings.filterwarnings('ignore',module=".*av.*")

def getMiddleFrames(dataset,audioLen=1):
    #Get the middles frames of each shot for all video in a dataset
    #Require the shot to be pre-computed.

    videoPathList = glob.glob("../data/{}/*.*".format(dataset))
    videoPathList = list(filter(lambda x:x.find(".wav")==-1,videoPathList))

    for videoPath in videoPathList:

        dirname = "../data/{}/{}/".format(dataset,os.path.splitext(os.path.basename(videoPath))[0])
        if not os.path.exists(dirname+"/middleFrames/"):
            #print(videoPath)

            videoName = os.path.splitext(os.path.basename(videoPath))[0]

            os.makedirs(dirname+"/middleFrames/")

            #Middle frame index extration
            print("\t Middle frame extraction")
            tree = ET.parse("{}/result.xml".format(dirname)).getroot()
            shots = tree.find("content").find("body").find("shots")

            frameNb = int(shots[-1].get("fduration"))+int(shots[-1].get("fbegin"))
            fps = float(tree.find("content").find("head").find("media").find("fps").text)

            #Extract the audio of the video
            audioPath = os.path.dirname(videoPath)+"/"+videoName+".wav"
            audioInfStr = tree.find("content").find("head").find("media").find("codec").find("audio").text

            audioSampleRate = int(audioInfStr.split(",")[1].replace(",","").replace("Hz","").replace(" ",""))
            audioBitRate = int(audioInfStr.split(",")[4].replace(",","").replace("kb/s","").replace(" ",""))*1000

            if not os.path.exists(audioPath):
                command = "ffmpeg -i {} -ab {} -ac 2 -ar {} -vn {}".format(videoPath,audioBitRate,audioSampleRate,audioPath)
                subprocess.call(command, shell=True)

            bounds = list(map(lambda x:int(x.get("fbegin")),shots))
            bounds.append(frameNb)
            keyFrameInd = []

            for i,shot in enumerate(bounds):
                if i < len(bounds)-1:
                    keyFrameInd.append((bounds[i]+bounds[i+1])//2)

            if dataset == "OVSD":
                sceneBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,videoName),delimiter='\t')
            else:
                raise ValueError("Unknown dataset :".format(dataset))

            cap = cv2.VideoCapture(videoPath)
            success = True
            i = 0
            scene=0
            newSceneShotIndexs = []
            keyFrameInd = np.array(keyFrameInd)

            #Opening the sound file
            audioData, fs = sf.read(audioPath)
            #audioData= np.transpose(audioData,[1,0])

            while success:
                success, imageRaw = cap.read()
                if i in keyFrameInd:
                    if scene < len(sceneBounds):
                        if sceneBounds[scene,1] < i:
                            print("Scene ",scene,"shot",np.where(keyFrameInd == i)[0][0],"frame",i,"time",(i//fps)//60,"m",(i//fps)%60,"s")
                            newSceneShotIndexs.append(np.where(keyFrameInd == i)[0][0])
                            scene += 1

                    cv2.imwrite(dirname+"/middleFrames/frame"+str(i)+".png",imageRaw)

                    #Writing the audio sample
                    time = i/fps
                    pos = time*audioSampleRate
                    interv = audioLen*audioSampleRate/2
                    sampleToWrite = audioData[int(round(pos-interv)):int(round(pos+interv)),:]
                    fullArray = np.zeros((int(round(pos+interv))-int(round(pos-interv)),sampleToWrite.shape[1]))
                    fullArray[:len(sampleToWrite)] = sampleToWrite

                    sf.write(dirname+"/middleFrames/frame"+str(i)+".wav",fullArray,audioSampleRate,subtype='PCM_16',format="WAV")

                i += 1

            #This binary mask indicates if a shot is the begining of a new scene or not
            sceneTransition = np.zeros((len(keyFrameInd)))

            sceneTransition[newSceneShotIndexs] = 1

            print(sceneTransition.nonzero())

            np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,videoName),sceneTransition)

class PairLoader():

    def __init__(self,dataset,batchSize,imgSize,propStart,propEnd,shuffle,audioLen,resizeImage):

        self.batchSize = batchSize
        self.videoPathLists = list(filter(lambda x:x.find(".wav") ==-1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPathLists = list(filter(lambda x:x.find(".xml") == -1,self.videoPathLists))

        nbVid = len(self.videoPathLists)
        self.videoPathLists = self.videoPathLists[int(nbVid*propStart):int(nbVid*propEnd)]
        print("Nb videos :",len(self.videoPathLists))
        self.dataset=dataset
        self.targetDict = {}
        self.shuffle = shuffle
        for videoPath in self.videoPathLists:
            vidName = os.path.basename(os.path.splitext(videoPath)[0])
            self.targetDict[vidName] = getGT(dataset,vidName)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imgSize),transforms.ToTensor(),normalize])

        self.audioLen = audioLen

        self.video = None

    def __iter__(self):

        self.audioDict = {}
        self.videoDict = {}
        self.fpsDict = {}
        self.fsDict = {}

        self.vidPaths = []
        self.vidNames = []
        self.anchList,self.posList,self.negList = [],[],[]
        self.anchTargList,self.posTargList,self.negTargList = [],[],[]

        for videoPath in self.videoPathLists:

            vidName = os.path.basename(os.path.splitext(videoPath)[0])
            #print(vidName)
            shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
            shotInds = np.arange(len(shotBounds))
            shotBounds = torch.tensor(shotBounds[shotInds.astype(int)]).float()
            frameInds = torch.distributions.uniform.Uniform(shotBounds[:,0], shotBounds[:,1]+1).sample().long()

            sceneInds = np.cumsum(self.targetDict[vidName])

            #Building anchor tensor
            anchList,anchTargList = np.array(frameInds),sceneInds
            anchorAndTarget = np.concatenate((anchList[:,np.newaxis],anchTargList[:,np.newaxis].astype(str)),axis=1)

            #Building positive tensor
            posAndTarget = np.zeros_like(anchorAndTarget)
            for i in range(int(anchTargList[-1])+1):
                imageTargScene = anchorAndTarget[anchorAndTarget[:,1].astype(float) == i]
                np.random.shuffle(imageTargScene)
                posAndTarget[anchorAndTarget[:,1].astype(float) == i] = imageTargScene

            posList,posTargList = posAndTarget.transpose()

            #Buiding negative tensor
            negAndTarget = np.zeros_like(anchorAndTarget)

            #print(anchorAndTarget)
            for i in range(int(anchTargList[-1])+1):

                imageTargDiff = anchorAndTarget[anchorAndTarget[:,1].astype(float) != i]
                np.random.shuffle(imageTargDiff)

                nbRep = 1+len(negAndTarget[anchorAndTarget[:,1].astype(float) == i])//len(imageTargDiff)

                negAndTarget[anchorAndTarget[:,1].astype(float) == i] = imageTargDiff.repeat(nbRep,axis=0)[:(anchorAndTarget[:,1].astype(float) == i).sum()]

            negList,negTargList = negAndTarget.transpose()

            self.vidPaths.extend([videoPath for i in range(len(frameInds))])
            self.vidNames.extend([vidName for i in range(len(frameInds))])
            self.anchList.extend(anchList)
            self.posList.extend(posList)
            self.negList.extend(negList)
            self.anchTargList.extend(anchTargList)
            self.posTargList.extend(posTargList)
            self.negTargList.extend(negTargList)

        self.zipped = np.concatenate((np.array(self.vidPaths)[:,np.newaxis],np.array(self.vidNames)[:,np.newaxis],\
                                 np.array(self.anchList)[:,np.newaxis],np.array(self.posList)[:,np.newaxis],np.array(self.negList)[:,np.newaxis],\
                                 np.array(self.anchTargList)[:,np.newaxis],np.array(self.posTargList)[:,np.newaxis],np.array(self.negTargList)[:,np.newaxis]),axis=1)

        if self.shuffle:

            np.random.shuffle(self.zipped)

        _,self.vidNames,self.anchList,self.posList,self.negList,self.anchTargList,self.posTargList,self.negTargList = self.zipped.transpose()

        self.batchNb = len(self.vidNames)//self.batchSize

        self.currInd = 0
        return self

    def __next__(self):

        if self.currInd >= len(self.anchList):
            raise StopIteration

        batchSize = min(self.batchSize,len(self.anchList)-self.currInd)

        videoNames = self.vidNames[self.currInd:self.currInd+batchSize]

        if self.audioLen > 0:
            audioTens1 = torch.cat(list(map(lambda x: self.readAudio(x,2),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
            audioTens2 = torch.cat(list(map(lambda x: self.readAudio(x,3),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
            audioTens3 = torch.cat(list(map(lambda x: self.readAudio(x,4),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        else:
            audioTens1,audioTens2,audioTens3 = None,None,None

        batchTens1 = torch.cat(list(map(lambda x: self.readImage(x,2),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        batchTens2 = torch.cat(list(map(lambda x: self.readImage(x,3),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        batchTens3 = torch.cat(list(map(lambda x: self.readImage(x,4),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)

        targ1 = self.anchTargList[self.currInd:self.currInd+batchSize]
        targ2 = self.posTargList[self.currInd:self.currInd+batchSize]
        targ3 = self.negTargList[self.currInd:self.currInd+batchSize]

        self.currInd += self.batchSize

        return videoNames,batchTens1,batchTens2,batchTens3,\
                          audioTens1,audioTens2,audioTens3,\
                          torch.tensor(targ1.astype(int)),torch.tensor(targ2.astype(int)),torch.tensor(targ3.astype(int))

    def readImage(self,x,i):

        if not x[0] in self.videoDict.keys():
            self.videoDict[x[0]] = pims.Video(x[0])

        #self.video = pims.Video(x[0])

        #print(x)
        #print(self.videoDict[x[0]])
        #print(x[0])

        return self.preproc(self.videoDict[x[0]][int(x[i])]).unsqueeze(0)
        #return torch.zeros((3,300,300)).unsqueeze(0)

    def readAudio(self,x,i):

        audioPath = os.path.splitext(x[0])[0]+".wav"

        if not audioPath in self.audioDict.keys():
            tree = ET.parse("../data/{}/{}/result.xml".format(self.dataset,x[1])).getroot()
            fps = float(tree.find("content").find("head").find("media").find("fps").text)
            audioData, fs = sf.read(audioPath,dtype='int16')
            self.audioDict[audioPath] = audioData

            self.fpsDict[audioPath] = fps
            self.fsDict[audioPath] = fs

        audioData = readAudio(self.audioDict[audioPath],int(x[i]),self.fpsDict[audioPath],self.fsDict[audioPath],self.audioLen)

        return torch.tensor(vggish_input.waveform_to_examples(audioData/32768.0,self.fsDict[audioPath])).unsqueeze(0).float()

class Sampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self, nb_videos,nbTotalShots,nbShotPerSeq):
        self.nb_videos = nb_videos

        self.length = nbTotalShots//nbShotPerSeq
    def __iter__(self):
        return iter(torch.randint(self.nb_videos,(self.length,)))

    def __len__(self):
        return self.length

def collateSeq(batch):

    res = list(zip(*batch))

    res[0] = torch.cat(res[0],dim=0)
    if not res[1][0] is None:
        res[1] = torch.cat(res[1],dim=0)
    res[2] = torch.cat(res[2],dim=0)

    return res

class SeqTrDataset(torch.utils.data.Dataset):
    '''
    The dataset to sample sequence of frames from videos

    When the method __getitem__(i) is called, the dataset select some shots from the video i, select frame(s) from each shot, shuffle them, regroup them by scene and
    returns them. It can also returns the MFCC coefficient of a sound extract for each shot.

    Args:
    - dataset (str): the name of the dataset to use
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - lMin (int): the minimum length of a sequence during training
    - lMax (int): the maximum length of a sequence during training
    - imgSize (tuple): a tuple containing (in order) the width and size of the image
    - audioLen (int): the length of the audio extract for each shot
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - framesPerShot (int): the number of frames to use to reprensent each shot
    - exp_id (str): the name of the experience
    - max_shots (int): the total number of shot to process before stopping the training epoch. The youtube_large dataset contains a million shots, which is why this argument is useful.
    '''

    def __init__(self,dataset,propStart,propEnd,lMin,lMax,imgSize,audioLen,resizeImage,framesPerShot,exp_id,max_shots):

        super(SeqTrDataset, self).__init__()

        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = list(filter(lambda x:x.find(".xml") == -1,self.videoPaths))
        self.videoPaths = list(filter(lambda x:os.path.isfile(x),self.videoPaths))

        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.lMin,self.lMax = lMin,lMax
        self.dataset = dataset
        self.audioLen = audioLen
        self.framesPerShot = framesPerShot
        self.nbShots = 0
        self.exp_id = exp_id

        if max_shots != -1:
            self.nbShots = max_shots
        else:
            for videoPath in self.videoPaths:
                videoFold = os.path.splitext(videoPath)[0]
                self.nbShots += len(processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold))))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.RandomResizedCrop(imgSize),transforms.ToTensor(),normalize])

        self.FPSDict = {}

    def __len__(self):
        return self.nbShots

    def __getitem__(self,vidInd):

        data = torch.zeros(self.framesPerShot*self.lMax,3,self.imgSize[0],self.imgSize[1])
        targ = torch.zeros(self.lMax)
        vidNames = []

        vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

        if not self.videoPaths[vidInd] in self.FPSDict.keys():
            self.FPSDict[self.videoPaths[vidInd]] = processResults.getVideoFPS(self.videoPaths[vidInd],self.exp_id)

        fps = processResults.getVideoFPS(self.videoPaths[vidInd],self.exp_id)

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
        shotInds = np.arange(len(shotBounds))

        gt = getGT(self.dataset,vidName)

        #The scene number of each shot
        gt = np.cumsum(gt)

        #Permutate the scenes
        gt = self.permuteScenes(gt)

        #Select some shots
        #zipped = np.concatenate((shotInds[:,np.newaxis],gt[:,np.newaxis]),axis=1)

        try:
            zipped = np.concatenate((shotInds[:,np.newaxis],gt[:,np.newaxis]),axis=1)
        except ValueError:
            print(vidName,len(shotInds),len(gt))
            sys.exit(0)

        np.random.shuffle(zipped)
        zipped = zipped[:self.lMax]
        if len(zipped) < self.lMax:

            repeatedShotInd = np.random.randint(len(zipped),size=self.lMax-len(zipped))
            zipped = np.concatenate((zipped,zipped[repeatedShotInd]),axis=0)

        #If the shots are not sorted, each shot is very likely to be followed by a shot from a different scene
        #Sorting them balance this effect.
        zipped = zipped[zipped[:,1].argsort()]

        shotInds,gt = zipped.transpose()

        #A boolean array indicating if a selected shot in preceded by a shot from a different scene
        gt[1:] = (gt[1:] !=  gt[:-1])
        gt[0] = 0

        shotBounds = torch.tensor(shotBounds[shotInds.astype(int)]).float()
        frameInds = torch.distributions.uniform.Uniform(shotBounds[:,0], shotBounds[:,1]+1).sample((self.framesPerShot,)).long()

        frameInds = frameInds.transpose(dim0=0,dim1=1)
        frameInds = frameInds.contiguous().view(-1)

        video = pims.Video(self.videoPaths[vidInd])

        arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
        self.preprocAudio = transforms.Compose([arrToExamp])

        try:
            frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)
        except IndexError:
            print(vidName,frameInds.max(),processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName)).max(),gt.max())
            sys.exit(0)

        if self.audioLen > 0:
            audioData, fs = sf.read(os.path.splitext(self.videoPaths[vidInd])[0]+".wav")
            audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))
            audioSeq = audioSeq.unsqueeze(0).float()
        else:
            audioSeq = None

        return frameSeq.unsqueeze(0),audioSeq,torch.tensor(gt).float().unsqueeze(0),vidName

    def permuteScenes(self,gt):
        randScenePerm = np.random.permutation(int(gt.max()+1))
        gt_perm = np.zeros_like(gt)
        for j in range(len(gt)):
            gt_perm[j] = randScenePerm[gt[j]]
        gt = gt_perm
        return gt

class TestLoader():
    '''
    The dataset to sample sequence of frames from videos. As the video contains a great number of shot,
    each video is processed through several batches and each batch contain only one sequence.

    Args:
    - evalL (int): the length of a sequence in a batch. A big value will reduce the number of batches necessary to process a whole video
    - dataset (str): the name of the dataset
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - imgSize (tuple): a tuple containing (in order) the width and size of the image
    - audioLen (int): the length of the audio extract for each shot
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - framesPerShot (int): the number of frames to use to reprensent each shot
    - exp_id (str): the name of the experience
    '''

    def __init__(self,evalL,dataset,propStart,propEnd,imgSize,audioLen,resizeImage,framesPerShot,exp_id):
        self.evalL = evalL
        self.dataset = dataset
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = list(filter(lambda x:x.find(".xml") == -1,self.videoPaths))
        self.videoPaths = list(filter(lambda x:os.path.isfile(x),self.videoPaths))
        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.framesPerShot = framesPerShot

        self.exp_id = exp_id

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(imgSize),transforms.ToTensor(),normalize])

        self.audioLen = audioLen
        self.nbShots =0
        for videoPath in self.videoPaths:
            videoFold = os.path.splitext(videoPath)[0]
            self.nbShots += len(processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold))))

    def __iter__(self):
        self.videoInd = 0
        self.shotInd = 0
        self.sumL = 0
        return self

    def __next__(self):

        if self.videoInd == len(self.videoPaths):
            raise StopIteration

        L = self.evalL
        self.sumL += L

        videoPath = self.videoPaths[self.videoInd]
        video = pims.Video(videoPath)

        vidName = os.path.basename(os.path.splitext(videoPath)[0])

        fps = processResults.getVideoFPS(videoPath,self.exp_id)

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
        shotInds =  np.arange(self.shotInd,min(self.shotInd+L,len(shotBounds)))

        frameInds = self.regularlySpacedFrames(shotBounds[shotInds]).reshape(-1)

        arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
        self.preprocAudio = transforms.Compose([arrToExamp])

        frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)

        if self.audioLen > 0:
            audioData, fs = sf.read(os.path.splitext(videoPath)[0]+".wav")
            audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))
            audioSeq = audioSeq.unsqueeze(0).float()
        else:
            audioSeq = None

        gt = getGT(self.dataset,vidName)[self.shotInd:self.shotInd+L]

        if shotInds[-1] + 1 == len(shotBounds):
            self.shotInd = 0
            self.videoInd += 1
        else:
            self.shotInd += L

        return frameSeq.unsqueeze(0),audioSeq,torch.tensor(gt).float().unsqueeze(0),vidName,torch.tensor(frameInds)

    def regularlySpacedFrames(self,shotBounds):
        ''' Select several frame indexs regularly spaced in each shot '''

        frameInds = ((np.arange(self.framesPerShot)/self.framesPerShot)[np.newaxis,:]*(shotBounds[:,1]-shotBounds[:,0])[:,np.newaxis]+shotBounds[:,0][:,np.newaxis]).astype(int)

        return frameInds

def readAudio(audioData,i,fps,fs,audio_len):
    ''' Select part of an audio track
    Args:
    - audioData (array) the full waveform
    - i (int) the frame index at which the extract should be centered
    - fps (int): the number of frams per second
    - audio_len (float): the length of the audio extract, in seconds

    Returns:
    - fullArray (array): the audio extract

    '''

    time = float(i)/float(fps)
    pos = time*fs
    interv = audio_len*fs/2
    sampleToWrite = audioData[int(round(pos-interv)):int(round(pos+interv)),:]
    fullArray = np.zeros((int(round(pos+interv))-int(round(pos-interv)),sampleToWrite.shape[1]))
    fullArray[:len(sampleToWrite)] = sampleToWrite

    return fullArray

def getGT(dataset,vidName):
    ''' For one video, returns a list of 0,1 indicating for each shot if it's the first shot of a scene or not.

    This function computes the 0,1 list and save if it does not already exists.

    Args:
    - dataset (str): the dataset name
    - vidName (str): the video name. It is the name of the videol file minux the extension.
    Returns:
    - gt (array): a list of 0,1 indicating for each shot if it's the first shot of a scene or not.

    '''

    if not os.path.exists("../data/{}/annotations/{}_targ.csv".format(dataset,vidName)):

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(dataset,vidName))

        scenesBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName))
        gt = framesToShot(scenesBounds,shotBounds)
        np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName),gt)
    else:
        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName))

    return gt.astype(int)

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

def main():

    #getMiddleFrames("OVSD")

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    def testLoader(trainLoader,nbBatch_max=20,countFirst=True):
        nbBatch = 0
        time_start = time.time()
        timeList = np.zeros(nbBatch_max-(not countFirst))
        for batch in trainLoader:
            #print(batch[0][0].size(),batch[0][1].size(),batch[0][2].size(),batch[0][3])

            if not (nbBatch==0 and not countFirst):
                timeList[nbBatch-(not countFirst)] = time.time()-time_start

            nbBatch += 1
            time_start = time.time()
            if nbBatch == nbBatch_max:
                break
        end_time = time.time()

        print("Time per batch : ",timeList.mean(),timeList.std(),timeList)

    train_dataset = SeqTrDataset(4,"Holly2",0,0.9,25,35,(298,298),1,False,1,"TestOfPytorchLoader")
    sampler = Sampler(len(train_dataset.videoPaths))

    for i in [4,8]:
        trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                          batch_size=4,
                          sampler=sampler,
                          collate_fn=collateSeq, # use custom collate function here
                          pin_memory=True,
                          num_workers=i)

        print("num_workers",i)
        testLoader(trainLoader,countFirst=False)

    trainLoader = TrainLoader(4,"Holly2",0,0.9,25,35,(298,298),1,False,1,"TestOfPytorchLoader")
    testLoader(trainLoader)

if __name__ == "__main__":
    main()
