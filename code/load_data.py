
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

class TrainLoader():

    def __init__(self,batchSize,dataset,propStart,propEnd,lMin,lMax,imgSize,audioLen,resizeImage,framesPerShot,exp_id):

        self.batchSize = batchSize
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = list(filter(lambda x:x.find(".xml") == -1,self.videoPaths))

        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.lMin,self.lMax = lMin,lMax
        self.dataset = dataset
        self.audioLen = audioLen
        self.framesPerShot = framesPerShot
        self.nbShots = 0
        self.exp_id = exp_id

        for videoPath in self.videoPaths:
            videoFold = os.path.splitext(videoPath)[0]
            self.nbShots += len(processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold))))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imgSize),transforms.ToTensor(),normalize])

    def __iter__(self):

        self.sumL = 0
        return self

    def __next__(self):

        if self.sumL*self.batchSize > self.nbShots:
            raise StopIteration

        l = np.random.randint(self.lMin,self.lMax+1)
        self.sumL += l

        vidInds = self.nonRepRandInt(len(self.videoPaths),self.batchSize)

        data = torch.zeros(self.batchSize,self.framesPerShot*l,3,self.imgSize[0],self.imgSize[1])
        targ = torch.zeros(self.batchSize,l)
        vidNames = []

        if self.audioLen > 0:
            audio = torch.zeros(self.batchSize,self.framesPerShot*l,1,int(96*self.audioLen),64).float()
        else:
            audio = None

        for i,vidInd in enumerate(vidInds):

            vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

            fps = processResults.getVideoFPS(self.videoPaths[vidInd],self.exp_id)

            shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
            shotInds = np.arange(len(shotBounds))

            gt = getGT(self.dataset,vidName)

            #The scene number of each shot
            gt = np.cumsum(gt)

            #Permutate the scenes
            gt = self.permuteScenes(gt)

            #Select some shots
            zipped = np.concatenate((shotInds[:,np.newaxis],gt[:,np.newaxis]),axis=1)
            np.random.shuffle(zipped)
            zipped = zipped[:l]

            if len(zipped) < l:

                repeatedShotInd = np.random.randint(len(zipped),size=l-len(zipped))
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

            frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)

            if self.audioLen > 0:
                audioData, fs = sf.read(os.path.splitext(self.videoPaths[vidInd])[0]+".wav")
                audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))
                audio[i] = audioSeq

            data[i] = frameSeq
            targ[i] = torch.tensor(gt.astype(float).astype(int))
            vidNames.append(vidName)

        return data,audio,targ,vidNames

    def sampleAFrame(self,x):
        return x[np.random.randint(0,len(x))]

    def nonRepRandInt(self,nbMax,size):
        ints = np.arange(0,nbMax)
        np.random.shuffle(ints)
        ints = ints[:size]
        return ints

    def permuteScenes(self,gt):
        randScenePerm = np.random.permutation(int(gt.max()+1))
        gt_perm = np.zeros_like(gt)
        for j in range(len(gt)):
            gt_perm[j] = randScenePerm[gt[j]]
        gt = gt_perm
        return gt

class TestLoader():

    def __init__(self,evalL,dataset,propStart,propEnd,imgSize,audioLen,resizeImage,framesPerShot,exp_id):
        self.evalL = evalL
        self.dataset = dataset
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = list(filter(lambda x:x.find(".xml") == -1,self.videoPaths))

        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.framesPerShot = framesPerShot

        self.exp_id = exp_id

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imgSize),transforms.ToTensor(),normalize])

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

        #for j in range(len(frameInds)):
        #    img = video[frameInds[j]]
        #    cv2.imwrite('../vis/testImgPims_{}_{}.png'.format(j,shotInds[j]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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

        frameInds = ((np.arange(self.framesPerShot)/self.framesPerShot)[np.newaxis,:]*(shotBounds[:,1]-shotBounds[:,0])[:,np.newaxis]+shotBounds[:,0][:,np.newaxis]).astype(int)

        return frameInds

    def middleFrames(self,shotBounds):
        starts = np.concatenate((shotBounds[:,0],[shotBounds[-1,1]]),axis=0)
        return (starts[1:]+starts[:-1])//2

def readAudio(audioData,i,fps,fs,audio_len):
    time = i/fps
    pos = time*fs
    interv = audio_len*fs/2
    sampleToWrite = audioData[int(round(pos-interv)):int(round(pos+interv)),:]
    fullArray = np.zeros((int(round(pos+interv))-int(round(pos-interv)),sampleToWrite.shape[1]))
    fullArray[:len(sampleToWrite)] = sampleToWrite

    return fullArray

def getGT(dataset,vidName):
    if not os.path.exists("../data/{}/annotations/{}_targ.csv".format(dataset,vidName)):

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(dataset,vidName))

        scenesBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName))
        gt = framesToShot(scenesBounds,shotBounds)
        np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName),gt)
    else:
        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName))

    return gt.astype(int)

def framesToShot(scenesBounds,shotBounds):

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


    '''
    trainLoad = TrainSeqLoader(1,"OVSD",0,0.5,10,20,(299,299))
    for i,(data,targ) in enumerate(trainLoad):

        print(type(data))
        print(data.size(),targ.size())

        for j,img in enumerate(data[0]):

            img = 255*np.array(((img-img.min())/(img.max()-img.min())).permute(1,2,0))

            cv2.imwrite("../vis/trainVis_{}_{}.png".format(i,j),img)

        sys.exit(0)
    '''

    trainLoad = TrainLoader(1,"OVSD",0,0.5,10,20,(299,299),1,True,3)
    for i,(data,audio,targ,vidName) in enumerate(trainLoad):
        print(data.size(),audio.size(),targ.size(),vidName)

        break

    '''

    testLoad = TestLoader(20,"OVSD",0,0.5,(299,299),1,True,3)
    for i,(data,audio,targ,vidName) in enumerate(testLoad):
        print(data.size(),audio.size(),targ.size(),vidName)

        break
    '''


    '''
    pairLoad = PairLoader("OVSD",16,(299,299),0,0.5,True,1)
    for videoNames,anch,pos,neg,anchAudio,posAudio,negAudio,targ1,targ2,targ3 in pairLoad:
        print(videoNames,anch.size(),anchAudio.size(),targ1.size())
        sys.exit(0)
    '''

if __name__ == "__main__":
    main()
