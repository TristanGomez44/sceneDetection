
from args import ArgReader
import os
import glob
import shutil
import subprocess

import torch
from torch.distributions.gamma import Gamma
import numpy as np
import cv2
import shotdetect
import pims
import h5py
import cv2
import utils
import processResults

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--dataset', type=str, metavar='N',help='The dataset')
    argreader.parser.add_argument('--merge_videos',type=str, metavar='EXT',help='Accumulate the clips from a folder to obtain one video per movie in the dataset. The value \
                                    is the extension of the video files, example : \'avi\'.')

    argreader.parser.add_argument('--compute_only_gt',action='store_true',help='To compute only gt when using --merge_videos')

    argreader.parser.add_argument('--format_youtube',action='store_true',help='For the youtube_large datasets. Put the clips into separate folder')
    argreader.parser.add_argument('--format_bbc',type=str,metavar='EXT',help='Format the bbc dataset. The value is the extension of the video file. E.g : \"--format_bbc mp4\".')

    argreader.parser.add_argument('--detect_shot_bbc',action='store_true',help='To detect BBC shot using ffmpeg and not rely on the shot segmentation provided in the bbc dataset.')

    argreader.parser.add_argument('--format_bbc2',nargs=2,type=str,metavar='EXT',help='Format the bbc season 2 dataset. As there is no annotation, artificial annotation are built.\
                                                                                       The first value is the extension of the video file. E.g : \"--format_bbc2 mp4\". The second value\
                                                                                       is the average length of the scene desired in number of shots.')

    argreader.parser.add_argument('--format_ovsd',action='store_true',help='Format the OVSD dataset')
    argreader.parser.add_argument('--shot_thres',type=float, default=0.1,metavar='EXT',help='The detection threshold for the shot segmentation done by ffmpeg')
    argreader.parser.add_argument('--format_ally_mcbeal',type=str,metavar="EXT",help='Format the Ally McBeal dataset. \
                                            The value is the extension of the video file. E.g : \"--format_ally_mcbeal avi\".')

    argreader.parser.add_argument('--format_rai',action='store_true',help='Format the RAI dataset.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

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
                    os.makedirs("../data/youtube_large/{}".format(movieName))
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

        if not os.path.exists("../data/{}/annotations".format(args.dataset)):
            os.makedirs("../data/{}/annotations".format(args.dataset))

        #Collecting video folders where are stored all the clips
        videoFoldPaths = sorted(glob.glob("../data/{}/*/".format(args.dataset)))
        videoFoldPaths = list(filter(lambda x:x.find("annotations") == -1,videoFoldPaths))

        for videoFoldPath in videoFoldPaths:
            print(videoFoldPath)

            #The temporary path to the concatenated video
            catVidPath = videoFoldPath[:-1]+"_tmp.{}".format(args.merge_videos)

            #Concatenate all the videos and build the ground truth file
            if (not os.path.exists(catVidPath.replace("_tmp",""))) or args.compute_only_gt:
                processVideo(catVidPath,videoFoldPath,args.merge_videos,args.compute_only_gt,args.dataset,vidExt=args.merge_videos)

            nbFrames = getNbFrames(catVidPath.replace("_tmp",""))
            fps = utils.getVideoFPS(catVidPath.replace("_tmp",""))

            #Detecting shots
            vidName = os.path.basename(os.path.splitext(catVidPath.replace("_tmp",""))[0])
            if not os.path.exists("../data/{}/{}/result.csv".format(args.dataset,vidName)):

                shotBoundsFrame = detect_format_shots(catVidPath.replace("_tmp",""),args.shot_thres,nbFrames,fps)
                np.savetxt("../data/{}/{}/result.csv".format(args.dataset,vidName),shotBoundsFrame)

            removeBadShotVideos(args.dataset,vidName)

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
                nbFrames = getNbFrames(newPath)

                #Extract shots
                if args.detect_shot_bbc:
                    fps = utils.getVideoFPS(newPath)
                    shotCSV = detect_format_shots(newPath,args.shot_thres,nbFrames,fps)
                else:
                    rawShotCSV = np.genfromtxt(rawshotFilePaths[i])
                    shotCSV = removeHoles(rawShotCSV)

                shotCSV[-1,1] = nbFrames-1

                np.savetxt(videoFold+"/result.csv",shotCSV)
            else:
                shotCSV = np.genfromtxt(videoFold+"/result.csv")

            if not os.path.exists("../data/bbc/annotations/{}_scenes.txt".format(vidName)):

                #Extract scenes:
                starts = np.genfromtxt(rawSceneFilePaths[i],delimiter=",")[:-1]
                shotNb = len(shotCSV)
                ends = np.concatenate((starts[1:]-1,[shotNb-1]),axis=0)
                starts,ends = starts[:,np.newaxis],ends[:,np.newaxis]
                #The scene boundaries expressed with shot index
                scenesS = np.concatenate((starts,ends),axis=1)
                #The scene boundaries expressed with frame index
                scenesF = utils.shots_to_frames("../data/bbc/{}/result.csv".format(vidName),scenesS)
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

            fps = utils.getVideoFPS(path)
            frameNb = round(float(pims.Video(path)._duration)*fps)

            #Extract shots
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
            ends = np.concatenate((starts[1:]-1,[shotNb-1]),axis=0)
            starts,ends = starts[:,np.newaxis],ends[:,np.newaxis]
            #The scene boundaries expressed with shot index
            scenesS = np.concatenate((starts,ends),axis=1)
            #The scene boundaries expressed with frame index
            scenesF = utils.shots_to_frames("../data/bbc2/{}/result.csv".format(vidName),scenesS)

            np.savetxt("../data/bbc2/annotations/{}_scenes.txt".format(vidName),scenesF)

    if args.format_ovsd:

        videosPaths = sorted(glob.glob("../data/OVSD/*.*"),key=lambda x:os.path.basename(x).lower())
        videosPaths = list(filter(lambda x:x.find(".wav") == -1,videosPaths))

        rawSceneFilePaths = sorted(glob.glob("../data/OVSD/annotations/*_scenes.txt"),key=lambda x:os.path.basename(x).lower())

        for i,path in enumerate(videosPaths):

            print(path,rawSceneFilePaths[i])

            vidName = os.path.basename(os.path.splitext(path)[0])

            fps = utils.getVideoFPS(path)

            if hasattr(pims.Video(path),"_len"):
                frameNb = pims.Video(path)._len
            else:
                frameNb = round(float(pims.Video(path)._duration)*fps)

            #Removing holes in scenes segmentation
            rawSceneCSV = np.genfromtxt(rawSceneFilePaths[i])
            sceneCSV = removeHoles(rawSceneCSV)

            if sceneCSV[-1,1] < frameNb-1 - fps:
                sceneCSV = np.concatenate((sceneCSV,np.array([[sceneCSV[-1,1]+1,frameNb-1]])),axis=0)
            else:
                sceneCSV[-1,1] = frameNb-1

            np.savetxt(rawSceneFilePaths[i],sceneCSV)

            #Extract shots
            if not os.path.exists("../data/OVSD/{}/result.csv".format(vidName)):
                shotBoundsFrame = detect_format_shots(path,args.shot_thres,frameNb,fps)
                if not os.path.exists("../data/OVSD/{}/".format(vidName)):
                    os.makedirs("../data/OVSD/{}/".format(vidName))
                np.savetxt("../data/OVSD/{}/result.csv".format(vidName),shotBoundsFrame)
            else:
                shotBoundsFrame = np.genfromtxt("../data/OVSD/{}/result.csv".format(vidName))
                shotBoundsFrame[-1,1] = frameNb-1
                np.savetxt("../data/OVSD/{}/result.csv".format(vidName),shotBoundsFrame)

    if args.format_ally_mcbeal:

        videoPaths = sorted(glob.glob("../data/AllyMcBeal/*.{}".format(args.format_ally_mcbeal)))
        rawAnnotationFilePaths = sorted(glob.glob("../data/AllyMcBeal/Ally_McBeal.Annotations-1.1/*.pio"))

        if not os.path.exists("../data/allymcbeal/annotations"):
            os.makedirs("../data/allymcbeal/annotations")

        for i,videoPath in enumerate(videoPaths):
            print(videoPath,rawAnnotationFilePaths[i])

            videoName = i
            newVideoPath = "../data/allymcbeal/{}.{}".format(videoName,args.format_ally_mcbeal)
            videoFold = os.path.splitext(newVideoPath)[0]

            #Copy video
            if not os.path.exists(newVideoPath):
                shutil.copyfile(videoPath,newVideoPath)

            if not os.path.exists(videoFold):
                os.makedirs(videoFold)

            fps = utils.getVideoFPS(newVideoPath)
            frameNb = round(float(pims.Video(newVideoPath)._duration)*fps)

            #Extract shots
            tripletToInterv(rawAnnotationFilePaths[i],"shots",fps,frameNb,videoFold+"/result.csv")
            #Extract scenes
            tripletToInterv(rawAnnotationFilePaths[i],"scenes",fps,frameNb,"../data/allymcbeal/annotations/{}_scenes.txt".format(i))

    if args.format_rai:
        videoPaths = sorted(glob.glob("../data/RAIDataset/videos/*.mp4"),key=utils.findNumbers)
        rawAnnotationFilePaths = sorted(glob.glob("../data/RAIDataset/scenes_*.txt"),key=utils.findNumbers)

        if not os.path.exists("../data/rai/annotations"):
            os.makedirs("../data/rai/annotations")

        for i,videoPath in enumerate(videoPaths):
            print(videoPath,rawAnnotationFilePaths[i])

            vidName = i
            newVideoPath = "../data/rai/{}.mp4".format(vidName)
            videoFold = os.path.splitext(newVideoPath)[0]

            #Copy video
            if not os.path.exists(newVideoPath):
                shutil.copyfile(videoPath,newVideoPath)

            if not os.path.exists(videoFold):
                os.makedirs(videoFold)

            fps = utils.getVideoFPS(newVideoPath)
            frameNb = round(float(pims.Video(newVideoPath)._duration)*fps)

            #Extract shots
            if not os.path.exists("../data/rai/{}/result.csv".format(vidName)):
                shotBoundsFrame = detect_format_shots(newVideoPath,args.shot_thres,frameNb,fps)
                np.savetxt("../data/rai/{}/result.csv".format(vidName),shotBoundsFrame)
            else:
                shotBoundsFrame = np.genfromtxt("../data/rai/{}/result.csv".format(vidName))
                if shotBoundsFrame[-1,1] != frameNb-1:
                    shotBoundsFrame[-1,1] = frameNb-1
                    np.savetxt("../data/rai/{}/result.csv".format(vidName),shotBoundsFrame)

            scenes = np.genfromtxt("../data/RAIDataset/scenes_{}.txt".format(vidName+1))
            scenes[-1,1] = frameNb

            np.savetxt("../data/rai/annotations/{}_scenes.txt".format(vidName),scenes-1)

def removeBadShotVideos(dataset,vidName):
    #Remove the videos which shot detection is bad i.e. video with a detected shot number inferior to their scene number (there's only a few videos in this case)
    resPath = "../data/{}/{}/result.csv".format(dataset,vidName)
    res = np.genfromtxt(resPath)

    if res.shape[0] < len(glob.glob(os.path.dirname(resPath)+"/*.mp4")) or len(res.shape) == 1:

        if not os.path.exists("../data/youtBadShotDet/"):
            os.makedirs("../data/youtBadShotDet/")

        shutil.move(os.path.dirname(resPath),"../data/youtBadShotDet/")
        shutil.move(os.path.dirname(resPath)+".mp4","../data/youtBadShotDet/")

def processVideo(catVidPath,videoFoldPath,videoExtension,compute_only_gt,dataset,vidExt):

    gt = []
    startFr = -1

    fileToCat = ''

    clips = sorted(glob.glob(videoFoldPath+"/*.{}".format(videoExtension)))
    clips = list(filter(lambda x: x.find("_cut.") ==-1,clips))

    totalFrameNb = 0
    for k,videoPath in enumerate(clips):
        totalFrameNb,fileToCat = processScene(videoPath,dataset,totalFrameNb,compute_only_gt,fileToCat,gt,vidExt)

    if not compute_only_gt:
        with open(videoFoldPath+"/fileToCat.txt","w") as text_file:
            print(fileToCat,file=text_file)

        #Concatenate the videos
        subprocess.call("ffmpeg -v error -safe 0 -f concat -i {} -c copy -an {}".format(videoFoldPath+"/fileToCat.txt",catVidPath.replace("_tmp","")),shell=True)

        if dataset == "youtube_large" or dataset == "bbcEarth":
            #Removing cut file created by ffmpeg
            for cutClipPath in sorted(glob.glob(videoFoldPath+"/*cut.{}".format(videoExtension))):
                os.remove(cutClipPath)

    vidName = os.path.basename(os.path.splitext(catVidPath.replace("_tmp",""))[0])

    gt = np.array(gt).astype(int)
    gt[-1,1] = getNbFrames(catVidPath.replace("_tmp",""))-1
    np.savetxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName),gt)

def processScene(videoPath,dataset,totalFrameNb,compute_only_gt,fileToCat,gt,vidExt):

    print("\t",videoPath)

    #Getting the number of frames of the video
    nbFrames = getNbFrames(videoPath)

    if dataset == "youtube_large":
        fps = utils.getVideoFPS(videoPath)
        stopFrame = nbFrames-32*fps
    elif dataset == "bbcEarth":
        fps = utils.getVideoFPS(videoPath)
        stopFrame = nbFrames-20*fps
    else:
        stopFrame = nbFrames

    gt.append([totalFrameNb,totalFrameNb+stopFrame])
    totalFrameNb += stopFrame+1

    if not compute_only_gt:
        if dataset == "youtube_large":
            #Remove the end landmark with ffpmeg
            subprocess.call("ffmpeg -v error -y -i {} -t {} -vcodec copy -acodec copy {}".format(videoPath,stopFrame/fps,videoPath.replace("."+vidExt,"_cut."+vidExt)),shell=True)
            fileToCat += "file \'{}\'\n".format(os.path.basename(videoPath.replace("."+vidExt,"_cut."+vidExt)))
        elif dataset == "bbcEarth":

            if fps == 25:
                subprocess.call("ffmpeg -v error -y -i {} -t {} -vf scale=634:360 -an {}".format(videoPath,stopFrame/fps,videoPath.replace("."+vidExt,"_cut."+vidExt)),shell=True)
            else:
                subprocess.call("ffmpeg -v error -r 25 -y -i {} -t {} -vf scale=634:360 -an {}".format(videoPath,stopFrame/fps,videoPath.replace("."+vidExt,"_cut."+vidExt)),shell=True)
            fileToCat += "file \'{}\'\n".format(os.path.basename(videoPath.replace("."+vidExt,"_cut."+vidExt)))
        else:
            fileToCat += "file \'{}\'\n".format(os.path.basename(videoPath))

    return totalFrameNb,fileToCat

def tripletToInterv(h5FilePath,segKey,fps,frameNb,savePath):
    """ Convert from the format found in the h5py files (i.e. the Ally McBeal annotations \
    into the intervals format """

    tripletList = h5py.File(h5FilePath, 'r')["timeline"]["segmentation"][segKey]

    intervCSV = []
    for triplet in tripletList:
        start,end = triplet[0]/triplet[2],(triplet[0]+triplet[1])/triplet[2]
        intervCSV.append([start,end])

    intervCSV = (np.array(intervCSV)*fps).astype(int)

    if intervCSV[0,0] != 0:
        intervCSV = np.concatenate(([[0,intervCSV[0,0]-1]],intervCSV),axis=0)
    if intervCSV[-1,1] != frameNb -1:
        intervCSV = np.concatenate((intervCSV,[[intervCSV[-1,1]+1,frameNb-1]]),axis=0)

    intervCSV = removeHoles(intervCSV)

    np.savetxt(savePath,intervCSV)

def getNbFrames(path):

    try:
        pimsVid = pims.Video(path)
        if hasattr(pimsVid,"_frame_rate"):
            fps = float(pimsVid._frame_rate)
            nbFrames = int(float(pimsVid._duration)*fps)
        else:
            nbFrames = int(pimsVid._len)
        return nbFrames
    except OSError:

        res = subprocess.check_output("ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 {}".format(path),shell=True)
        if res == "N/A":
            res = subprocess.check_output("ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 {}".format(path),shell=True)

        return int(res)

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
    ends = np.concatenate((shotBoundsFrame-1,[frameNb-1]),axis=0)
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

        if targetPath != videoPath:
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
