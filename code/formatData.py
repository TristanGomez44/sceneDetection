import numpy as np
from args import ArgReader
import os
import glob
import processResults
import cv2
import subprocess
import video2Tensor
import vggish_input
from torchvision import transforms
import xml.etree.ElementTree as ET
import soundfile as sf
import torch
import modelBuilder
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
from skimage.transform import resize
import sys
import shotdetect
import shutil
import pims

from torch.distributions.gamma import Gamma

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--dataset', type=str, metavar='N',help='The dataset')
    argreader.parser.add_argument('--merge_videos',type=str, metavar='EXT',help='Accumulate the clips from a folder to obtain one video per movie in the dataset. The value \
                                    is the extension of the video files, example : \'avi\'.')
    argreader.parser.add_argument('--format_youtube',action='store_true',help='For the youtube and the youtube_large datasets. Put the clips into separate folder')
    argreader.parser.add_argument('--format_bbc',type=str,metavar='EXT',help='Format the bbc dataset. The value is the extension of the video file. E.g : \"--format_bbc mp4\".')
    argreader.parser.add_argument('--format_bbc2',nargs=2,type=str,metavar='EXT',help='Format the bbc season 2 dataset. As there is no annotation, artificial annotation are built.\
                                                                                       The first value is the extension of the video file. E.g : \"--format_bbc2 mp4\". The second value\
                                                                                       is the average length of the scene desired in number of shots.')

    argreader.parser.add_argument('--format_ovsd',action='store_true',help='Format the OVSD dataset')
    argreader.parser.add_argument('--shot_thres',type=float, default=0.1,metavar='EXT',help='The detection threshold for the shot segmentation done by ffmpeg')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.format_youtube and args.dataset == "youtube":

        #Removing non-scene video (montages, trailer etc.)
        videoPaths = sorted(glob.glob("../data/{}/*.*".format(args.dataset)))

        videoToRm = list(filter(lambda x: os.path.basename(x).find("Movie_Clip") == -1 \
                                      and os.path.basename(x).find("Movie Clip") == -1,videoPaths))

        for videoPath in videoToRm:
            try:
                os.remove(videoPath)
            except IsADirectoryError:
                pass

        removeBadChar(args.dataset)

        #Grouping clip by movie
        for videoPath in videoPaths:
            print(videoPath)

            movieName = os.path.splitext(os.path.basename(videoPath))[0].split("_-_")[0]
            movieName = ''.join([i for i in movieName if not i.isdigit()])

            folder = "../data/{}/{}".format(args.dataset,movieName)

            if not os.path.exists(folder):
                os.makedirs(folder)

            targetPath = folder+"/"+os.path.basename(videoPath)

            os.rename(videoPath,targetPath)

    if args.format_youtube and args.dataset == "youtube_large":

        #Removing non-scene video (montages, trailer etc.)
        videoPaths = sorted(glob.glob("../data/youtube_large/*.*"))

        videoToRm = list(filter(lambda x: os.path.basename(x).find("Movie_CLIP") == -1 \
                                      and os.path.basename(x).find("Movie CLIP") == -1 \
                                      and os.path.basename(x).find("Movieclips") == -1,videoPaths))

        for videoPath in videoToRm:
            try:
                os.remove(videoPath)
            except IsADirectoryError:
                pass

        removeBadChar("youtube_large")

        videoPaths = sorted(glob.glob("../data/youtube_large/*.mp4"))

        movieDict = {}
        descrNotFound = []
        for video in videoPaths:

            if os.path.exists(os.path.splitext(video)[0]+".description") and os.path.basename(video).lower().find("mashup") == -1:
                with open(os.path.splitext(video)[0]+".description","r") as text_file:
                    descr = text_file.read()

                descr = descr.split("\n")[0]

                #descr = descr.replace("  "," ")

                if descr.find("movie clips:") != -1:
                    movieName = descr[:descr.find("movie clips:")-1]
                elif descr.find(" Movie Clip :") != -1:
                    movieName = descr[:descr.find(" Movie Clip :")-1]
                elif descr.find(" - ") != -1:
                    movieName = descr[:descr.find(" - ")]
                elif descr.find(":") != -1:
                    movieName = descr[:descr.find(":")]
                else:
                    movieName = descr[:descr.find("-")-1]

                if not movieName in movieDict.keys():
                    movieDict[movieName] = [video]
                else:
                    movieDict[movieName].append(video)
            else:
                descrNotFound.append(os.path.splitext(video)[0]+".description")

        #Removing scene without description
        if not os.path.exists("../data/nodescr_youtube_large"):
            os.makedirs("../data/nodescr_youtube_large")
        for descrPath in descrNotFound:
            videoPath = descrPath.replace(".description",".mp4")
            os.rename(videoPath,"../data/nodescr_youtube_large/"+os.path.basename(videoPath))

        #The folder in which the scenes with badly formated description will be put,
        if not os.path.exists("../data/baddescr_youtube_large"):
            os.makedirs("../data/baddescr_youtube_large")

        #Grouping the scenes by movie
        for i,movieName in enumerate(movieDict.keys()):

            #Removing scene with a badly formated description
            if len(movieDict[movieName]) == 1:
                for scene in movieDict[movieName]:
                    os.rename(scene,"../data/baddescr_youtube_large/"+os.path.basename(scene))
            else:
                if not os.path.exists("../data/youtube_large/{}".format(movieName)):
                    os.makedirs("../data/{}".format(movieName))
                for scene in movieDict[movieName]:
                    os.rename(scene,"../data/youtube_large/{}/{}".format(movieName,os.path.basename(scene)))

        #Puting the descriptions in a folder
        if not os.path.exists("../data/descr_youtube_large"):
            os.makedirs("../data/descr_youtube_large")
        for descr in sorted(glob.glob("../data/youtube_large/*.description")):
            os.rename(descr,"../data/descr_youtube_large/{}".format(os.path.basename(descr)))

        #Removing bad characters in movie name:
        for movieFold in sorted(glob.glob("../data/youtube_large/*/")):
            os.rename(movieFold,removeBadChar_filename(movieFold))

        for video in sorted(glob.glob("../data/youtube_large/*/*")):
            os.rename(video,removeBadChar_filename(video))

    if args.merge_videos:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not os.path.exists("../data/{}/annotations".format(args.dataset)):
            os.makedirs("../data/{}/annotations".format(args.dataset))

        videoFoldPaths = sorted(glob.glob("../data/{}/*/".format(args.dataset)))
        videoFoldPaths = list(filter(lambda x:x.find("annotations") == -1,videoFoldPaths))

        for videoFoldPath in videoFoldPaths:
            print(videoFoldPath)

            accVidPath = videoFoldPath[:-1]+"_tmp.{}".format(args.merge_videos)

            if not os.path.exists(accVidPath.replace("_tmp","")):

                accumulatedVideo = None
                gt = []
                startFr = -1
                accAudioData = None

                fileToCat = ''

                clips = sorted(glob.glob(videoFoldPath+"/*.{}".format(args.merge_videos)))
                clips = list(filter(lambda x: x.find("_cut.") ==-1,clips))

                for k,videoPath in enumerate(clips):
                    print("\t",videoPath)

                    cap = cv2.VideoCapture(videoPath)
                    extractAudio(videoPath)
                    accAudioData,fs = accumulateAudio(videoPath.replace(".{}".format(args.merge_videos),".wav"),accAudioData)

                    #Getting the number of frames of the video
                    subprocess.call("ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 {} > nbFrames.txt".format(videoPath),shell=True)

                    nbFrames = np.genfromtxt("nbFrames.txt")

                    fps = processResults.getVideoFPS(videoPath)

                    if args.dataset == "youtube_large":
                        stopFrame = nbFrames-32*fps
                    else:
                        stopFrame = nbFrames

                    gt.append([startFr+1,stopFrame])
                    startFr = stopFrame

                    if args.dataset == "youtube_large":
                        #Remove the end landmark with ffpmeg
                        subprocess.call("ffmpeg -v error -y -i {} -t {} -vcodec copy -acodec copy {}".format(videoPath,stopFrame/fps,videoPath.replace(".mp4","_cut.mp4")),shell=True)
                        fileToCat += "file \'{}\'\n".format(os.path.basename(videoPath.replace(".mp4","_cut.mp4")))
                    else:
                        fileToCat += "file \'{}\'\n".format(os.path.basename(videoPath))

                with open(videoFoldPath+"/fileToCat.txt","w") as text_file:
                    print(fileToCat,file=text_file)

                #Concatenate the videos
                subprocess.call("ffmpeg -v error -safe 0 -f concat -i {} -c copy -an {}".format(videoFoldPath+"/fileToCat.txt",accVidPath.replace("_tmp","")),shell=True)

                if args.dataset == "youtube_large":
                    #Removing cut file created by ffmpeg
                    for cutClipPath in sorted(glob.glob(videoFoldPath+"/*cut.{}".format(args.merge_videos))):
                        os.remove(cutClipPath)

                wavFilePath = accVidPath.replace(".{}".format(args.merge_videos),".wav")
                if len(accAudioData.shape) == 1:
                    sf.write(wavFilePath,accAudioData[:,np.newaxis],fs)
                else:
                    sf.write(wavFilePath,accAudioData,fs)

                vidName = os.path.basename(os.path.splitext(accVidPath.replace("_tmp",""))[0])

                np.savetxt("../data/{}/annotations/{}_scenes.txt".format(args.dataset,vidName),gt)

                os.rename(accVidPath.replace(".{}".format(args.merge_videos),".wav"),accVidPath.replace(".{}".format(args.merge_videos),".wav").replace("_tmp",""))

            #Detecting shots
            vidName = os.path.basename(os.path.splitext(accVidPath.replace("_tmp",""))[0])
            if not os.path.exists("../data/{}/{}/result.csv".format(args.dataset,vidName)):

                shotBoundsFrame = detect_format_shots(accVidPath.replace("_tmp",""),args.shot_thres)
                np.savetxt("../data/{}/{}/result.csv".format(args.dataset,vidName),shotBoundsFrame)

            #Remove the temporary wav files:
            for wavFilePath in sorted(glob.glob(videoFoldPath+"/*.wav")):
                os.remove(wavFilePath)

            #Remove the videos which shot detection is bad i.e. video with a detected shot number inferior to their scene number (there's only a few videos in this case)
            resPath = "../data/youtube_large/{}/result.xml".format(vidName)
            res = processResults.xmlToArray(resPath)

            if res.shape[0] < len(glob.glob(os.path.dirname(resPath)+"/*.mp4")) or len(res.shape) == 1:

                if not os.path.exists("../data/youtBadShotDet/"):
                    os.makedirs("../data/youtBadShotDet/")

                shutil.move(os.path.dirname(resPath),"../data/youtBadShotDet/")
                shutil.move(os.path.dirname(resPath)+".mp4","../data/youtBadShotDet/")

    if args.format_bbc:

        videosPaths = sorted(glob.glob("../data/PlanetEarth/*.{}".format(args.format_bbc)))

        #This creates the bbc folder and the annotation folder in it
        if not os.path.exists("../data/bbc/annotations"):
            os.makedirs("../data/bbc/annotations")

        rawshotFilePaths = sorted(glob.glob("../data/PlanetEarth/annotations/shots/*.txt"))
        rawSceneFilePaths = sorted(glob.glob("../data/PlanetEarth/annotations/scenes/annotator_0/*"))

        for i,path in enumerate(videosPaths):

            print(path,rawshotFilePaths[i])

            vidName = str(i)

            newPath = path.replace("PlanetEarth","bbc")
            newPath = os.path.dirname(newPath)+"/"+vidName+".{}".format(args.format_bbc)

            videoFold = os.path.splitext(newPath)[0]

            if not os.path.exists(videoFold):
                os.makedirs(videoFold)

            if not os.path.exists(newPath):
                #Copy video
                shutil.copyfile(path,newPath)

            #Extract audio
            extractAudio(newPath)

            if not os.path.exists(videoFold+"/result.csv"):
                #Extract shots
                rawShotCSV = np.genfromtxt(rawshotFilePaths[i])
                shotCSV = removeHoles(rawShotCSV)
                nbFrames = getNbFrames(newPath)
                shotCSV[-1,1] = nbFrames-1

                np.savetxt(videoFold+"/result.csv",shotCSV)
            else:
                shotCSV = np.genfromtxt(videoFold+"/result.csv")

            #Extract scenes:
            starts = np.genfromtxt(rawSceneFilePaths[i],delimiter=",")[:-1]
            print(starts)

            shotNb = len(shotCSV)
            ends = np.concatenate((starts[1:]-1,[shotNb-1]),axis=0)
            starts,ends = starts[:,np.newaxis],ends[:,np.newaxis]
            #The scene boundaries expressed with shot index
            scenesS = np.concatenate((starts,ends),axis=1)
            #The scene boundaries expressed with frame index
            scenesF = processResults.shots_to_frames("../data/bbc/{}/result.xml".format(vidName),scenesS)

            np.savetxt("../data/bbc/annotations/{}_scenes.txt".format(vidName),scenesF)

    if args.format_bbc2:

        torch.manual_seed(0)

        videosPaths = sorted(glob.glob("../data/PlanetEarth2/*.{}".format(args.format_bbc2[0])))

        #This creates the bbc folder and the annotation folder in it
        if not os.path.exists("../data/bbc2/annotations"):
            os.makedirs("../data/bbc2/annotations")

        #rawshotFilePaths = sorted(glob.glob("../data/PlanetEarth2/annotations/shots/*.txt"))
        #rawSceneFilePaths = sorted(glob.glob("../data/PlanetEarth2/annotations/scenes/annotator_0/*"))

        for i,path in enumerate(videosPaths):

            print(path)

            vidName = str(i)

            newPath = path.replace("PlanetEarth2","bbc2")
            newPath = os.path.dirname(newPath)+"/"+vidName+".{}".format(args.format_bbc2[0])

            videoFold = os.path.splitext(newPath)[0]

            if not os.path.exists(videoFold):
                os.makedirs(videoFold)

            #Copy video
            if not os.path.exists(newPath):
                shutil.copyfile(path,newPath)

            fps = processResults.getVideoFPS(path)
            frameNb = int(float(pims.Video(path)._duration)*fps)-1

            #Extract shots
            print("../data/bbc2/{}/result.csv".format(vidName))
            if not os.path.exists("../data/bbc2/{}/result.csv".format(vidName)):
                shotCSV = detect_format_shots(path,args.shot_thres,frameNb,fps)
                if not os.path.exists("../data/bbc2/{}/".format(vidName)):
                    os.makedirs("../data/bbc2/{}/".format(vidName))
                np.savetxt("../data/bbc2/{}/result.csv".format(vidName),shotCSV)
            else:
                shotCSV = np.genfromtxt(videoFold+"/result.csv")

            shotNb = len(shotCSV)

            #Randomly generates scenes:
            starts = generateRandomScenes(shotNb,float(args.format_bbc2[1]))
            print(starts,shotNb)
            ends = np.concatenate((starts[1:]-1,[shotNb-1]),axis=0)
            starts,ends = starts[:,np.newaxis],ends[:,np.newaxis]
            #The scene boundaries expressed with shot index
            scenesS = np.concatenate((starts,ends),axis=1)
            #The scene boundaries expressed with frame index
            scenesF = processResults.shots_to_frames("../data/bbc2/{}/result.xml".format(vidName),scenesS)

            np.savetxt("../data/bbc/annotations/{}_scenes.txt".format(vidName),scenesF)

    if args.format_ovsd:

        videosPaths = sorted(glob.glob("../data/OVSD/*.*"),key=lambda x:os.path.basename(x).lower())
        videosPaths = list(filter(lambda x:x.find(".wav") == -1,videosPaths))

        rawSceneFilePaths = sorted(glob.glob("../data/OVSD/annotations/*_scenes.txt"),key=lambda x:os.path.basename(x).lower())

        for i,path in enumerate(videosPaths):

            print(path,rawSceneFilePaths[i])

            vidName = os.path.basename(os.path.splitext(path)[0])

            fps = processResults.getVideoFPS(path)
            frameNb = int(float(pims.Video(path)._duration)*fps)-1

            #Removing holes in scenes segmentation
            rawSceneCSV = np.genfromtxt(rawSceneFilePaths[i])
            sceneCSV = removeHoles(rawSceneCSV)
            sceneCSV[-1,1] = frameNb
            np.savetxt(rawSceneFilePaths[i],sceneCSV)

            #Extract shots
            if not os.path.exists("../data/OVSD/{}/result.csv".format(vidName)):
                shotBoundsFrame = detect_format_shots(path,args.shot_thres,frameNb,fps)
                if not os.path.exists("../data/OVSD/{}/".format(vidName)):
                    os.makedirs("../data/OVSD/{}/".format(vidName))
                np.savetxt("../data/OVSD/{}/result.csv".format(vidName),shotBoundsFrame)

            #Extract audio
            extractAudio(path)

def getNbFrames(path):
    pimsVid = pims.Video(path)
    fps = float(pimsVid._frame_rate)
    nbFrames = int(float(pimsVid._duration)*fps)
    return nbFrames

def generateRandomScenes(shotNb,mean,var=0.5):

    scale = var/mean
    shape = mean/scale

    gam = Gamma(shape, 1/scale)

    #Generating scene starts index by first generating scene lengths
    starts = torch.cat((torch.tensor([0]),torch.cumsum(gam.rsample((shotNb,)).int(),dim=0)),dim=0)
    #Probably to many scenes have been generated. Removing those above the movie limit
    starts = starts[starts < shotNb]

    return starts

def detect_format_shots(path,shot_thres,frameNb,fps):

    vidName = os.path.basename(os.path.splitext(path)[0])

    #Detecting shots
    shotBoundsTime = shotdetect.extract_shots_with_ffprobe(path,threshold=shot_thres)

    #Shot boundaries as a function of frame index instead of time
    shotBoundsFrame = (np.array(shotBoundsTime)*fps).astype(int)

    starts = np.concatenate(([0],shotBoundsFrame),axis=0)
    ends = np.concatenate((shotBoundsFrame-1,[frameNb]),axis=0)
    shotBoundsFrame = np.concatenate((starts[:,np.newaxis],ends[:,np.newaxis]),axis=1)

    return shotBoundsFrame

def removeHoles(csvFile):

    starts = csvFile[:,0]

    end = csvFile[-1,1]
    ends = np.concatenate((starts[1:]-1,[end]),axis=0)[:,np.newaxis]

    csvFile = np.concatenate((starts[:,np.newaxis],ends),axis=1)

    return csvFile

def common_str(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

def removeBadChar_filename(filename):
    return filename.replace(" ","_").replace("(","").replace(")","").replace("'","").replace("$","").replace(",","_").replace("&","").replace(":","")

def removeBadChar_list(videoPaths):
    for videoPath in videoPaths:
        targetPath = removeBadChar_filename(videoPath)

        if os.path.exists(targetPath) and (not os.path.isdir(videoPath)):
            os.remove(videoPath)
        else:
            os.rename(videoPath,targetPath)

def removeBadChar(dataset):

    #Removing bad characters
    videoPaths = sorted(glob.glob("../data/{}/*.*".format(dataset)))
    removeBadChar_list(videoPaths)

def extractAudio(videoPath):
    #Extracting audio
    videoSubFold =  os.path.splitext(videoPath)[0]
    audioPath = videoSubFold+".wav"

    if not os.path.exists(audioPath):

        command = "ffmpeg -loglevel panic -i {} -vn {}".format(videoPath,audioPath)
        subprocess.call(command, shell=True)

def accumulateAudio(audioPath,accAudioData):

    if accAudioData is None:
        audioData, fs = sf.read(audioPath)
        accAudioData = audioData
    else:
        audioData, fs = sf.read(audioPath)
        accAudioData = np.concatenate((accAudioData,audioData),axis=0)

    return accAudioData,fs

if __name__ == "__main__":
    main()
