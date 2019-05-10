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

CV_DEF_FPS = 30

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--csv_path', type=str, metavar='N',help='To format the csv file send by the IBM researcher. The value is the path to the csv.')
    argreader.parser.add_argument('--annot_fold', type=str, metavar='N',help='To format the annotations file. The value is the path to the annotations folder')
    argreader.parser.add_argument('--base_fold', type=str, metavar='N',help='To format the baseline file. The value is the path to the baseline folder')
    argreader.parser.add_argument('--dataset', type=str, metavar='N',help='The dataset')
    argreader.parser.add_argument('--merge_videos',type=str, metavar='EXT',help='Accumulate the clips from a folder to obtain one video per movie in the dataset. The value \
                                    is the extension of the video files, example : \'avi\'.')
    argreader.parser.add_argument('--format_youtube',action='store_true',help='Put the clips into separate folder')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.csv_path:
        csv = np.genfromtxt(args.csv_path,dtype=str,delimiter=",")

        for j in range(len(csv[0])):

            #Computing scene boundaries with frame number
            scenesF = csv[1:,j]
            scenesF = scenesF[scenesF != ''].astype(int)
            scenesF = np.concatenate(([-1],scenesF),axis=0)

            lastFrameInd = np.genfromtxt("../data/{}/annotations/{}_frames_scenes.csv".format(args.dataset,csv[0,j]))[-1,1]+1

            #Getting the predicted scenes bounds
            scenesF = np.concatenate((scenesF,[lastFrameInd]),axis=0)
            scenesF = np.concatenate((scenesF[:-1,np.newaxis]+1,scenesF[1:,np.newaxis]),axis=1)

            filePath = os.path.dirname(args.csv_path)+"/"+csv[0,j]+"_frames_truebasecuts.csv"
            np.savetxt(filePath,scenesF)

            scenesS = processResults.frame_to_shots(args.dataset,csv[0,j],scenesF)

            filePath = os.path.dirname(args.csv_path)+"/"+csv[0,j]+"_truebasecuts.csv"
            print(filePath)
            np.savetxt(filePath,scenesS)

    if args.annot_fold:

        annotFilePaths = glob.glob(args.annot_fold+"/*_scenes.txt")

        for annotFilePath in annotFilePaths:

            pos = os.path.basename(annotFilePath).find("_scenes.txt")
            vidName = os.path.basename(annotFilePath)[:pos]

            scenesF = np.genfromtxt(annotFilePath)

            #renaming the old annotation file
            newFilePath = args.annot_fold+"/"+vidName+"_frames_scenes.csv"
            os.rename(annotFilePath,newFilePath)

            #scenesS = processResults.frame_to_shots(args.dataset,vidName,scenesF)

            #filePath = args.annot_fold+"/"+vidName+"_scenes.csv"
            #np.savetxt(baseFilePath,scenesS)

    if args.base_fold:

        baseFilePaths = glob.glob(args.base_fold+"/*_baseline.csv")

        for baseFilePath in baseFilePaths:

            pos = os.path.basename(baseFilePath).find("_baseline.csv")
            vidName = os.path.basename(baseFilePath)[:pos]
            #print(vidName)
            scenesF = np.genfromtxt(baseFilePath)
            #print(baseFilePath)
            #renaming the old baseline file
            newFilePath = args.base_fold+"/"+vidName+"_shots_baseline.csv"
            os.rename(baseFilePath,newFilePath)

            scenesS = processResults.shots_to_frames(args.dataset,vidName,scenesF)

            filePath = args.base_fold+"/"+vidName+"_baseline.csv"
            np.savetxt(baseFilePath,scenesS)

    if args.format_youtube:

        #Removing non-scene video (montages, trailer etc.)
        videoPaths = sorted(glob.glob("../data/{}/*.*".format(args.dataset)))
        print(videoPaths)
        videoToRm = list(filter(lambda x:os.path.basename(x).find("Trailer") != -1 \
                                      or os.path.basename(x).find("MASHUP") != -1\
                                      or os.path.basename(x).find("Tribute") != -1,videoPaths))

        for videoPath in videoToRm:
            os.remove(videoPath)

        #Removing bad characters
        videoPaths = sorted(glob.glob("../data/{}/*.*".format(args.dataset)))
        for videoPath in videoPaths:
            if videoPath.find(" ") != -1 or videoPath.find("(") != -1 or videoPath.find(")") != -1:
                os.rename(videoPath,videoPath.replace(" ","_").replace("(","").replace(")","").replace("'",""))

        #Grouping clip by movie
        for videoPath in videoPaths:
            print(videoPath)

            if args.dataset != "youtube_large":
                movieName = os.path.splitext(os.path.basename(videoPath).split("___")[-1].split("_Movie_Clip")[0])[0]
                movieName = movieName.split("_(")[0]
            else:
                movieName = os.path.splitext(os.path.basename(videoPath))[0].split("_-_")[0]

            folder = "../data/{}/{}".format(args.dataset,movieName)

            if not os.path.exists(folder):
                os.makedirs(folder)

            targetPath = folder+"/"+os.path.basename(videoPath)

            os.rename(videoPath,targetPath)

    if args.merge_videos:

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
                accAudioData = None

                for videoPath in sorted(glob.glob(videoFoldPath+"/*.{}".format(args.merge_videos))):
                    print("\t",videoPath)

                    cap = cv2.VideoCapture(videoPath)
                    print("Extracting audio")
                    audioData = extractAudio(videoPath,videoFoldPath)
                    print("Accumulating audio")
                    accAudioData,fs = accumulateAudio(videoPath.replace(".{}".format(args.merge_videos),".wav"),accAudioData)

                    print("Getting number of frames")
                    #Getting the number of frames of the video
                    subprocess.call("ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 {} > nbFrames.txt".format(videoPath),shell=True)
                    nbFrames = np.genfromtxt("nbFrames.txt")
                    print("{} has {} frames".format(os.path.basename(videoPath),nbFrames))

                    subprocess.call("ffmpeg -i {} 2>info.txt".format(videoPath),shell=True)
                    with open('info.txt', 'r') as infoFile:
                        infos = infoFile.read()

                    fps = round(float(infos.split("\n")[18].split(",")[5].replace(" ","").replace("fps","")))

                    print("{} has {} frame per second".format(os.path.basename(videoPath),fps))

                    if args.dataset == "youtube_large":
                        if videoPath.find("Movie_CLIP") != -1:
                            stopFrame = nbFrames-32*fps
                        elif videoPath.find("Movieclips") != -1:
                            stopFrame = nbFrames-12*fps
                        else:
                            raise ValueError("Unkown title format for video",videoPath)
                    else:
                        stopFrame = nbFrames

                    i=0
                    success=True
                    while success:
                        success, imageRaw = cap.read()

                        if success:
                            if not accumulatedVideo:
                                if (not args.img_width) or (not args.img_heigth):
                                    accumulatedVideo = cv2.VideoWriter(accVidPath, fourcc, 30, (imageRaw.shape[0],imageRaw.shape[1]))
                                else:
                                    accumulatedVideo = cv2.VideoWriter(accVidPath, fourcc, 30, (args.img_width,args.img_heigth))

                            #Removing the black areas in the video and the "movie clip" symbol in the bottom
                            if args.dataset == "youtube_large":
                                imageRaw = imageRaw[50:-50]

                            imageRaw = (resize(imageRaw,(args.img_width,args.img_heigth,3),anti_aliasing=True,mode='constant')*255).astype("uint8")

                            accumulatedVideo.write(imageRaw)

                            if i==0:
                                gt.append(1)
                            else:
                                gt.append(0)

                            i += 1

                            if i>=stopFrame:
                                success = False

                sf.write(accVidPath.replace(".{}".format(args.merge_videos),".wav"),accAudioData,fs)
                sys.exit(0)

                lastFram = len(gt)
                gt = np.array(gt).nonzero()[0]

                gt = np.concatenate((gt,[lastFram]),axis=0)

                gt = np.concatenate((gt[:-1,np.newaxis],gt[1:,np.newaxis]-1),axis=1)

                gt[-1,1] += 1

                vidName = os.path.basename(os.path.splitext(accVidPath.replace("_tmp",""))[0])

                np.savetxt("../data/{}/annotations/{}_scenes.txt".format(args.dataset,vidName),gt)
                accumulatedVideo.release()

                os.rename(accVidPath,accVidPath.replace("_tmp",""))
                os.rename(accVidPath.replace(".{}".format(args.merge_videos),".wav"),accVidPath.replace(".{}".format(args.merge_videos),".wav").replace("_tmp",""))

                #Detecting shots
                #if not os.path.exists(videoFoldPath+"/result.xml"):
            vidName = os.path.basename(os.path.splitext(accVidPath.replace("_tmp",""))[0])
            if not os.path.exists("../data/{}/{}/result.csv".format(args.dataset,vidName)):
                shotBoundsTime = shotdetect.extract_shots_with_ffprobe(accVidPath.replace("_tmp",""))
                shotBoundsFrame = (np.array(shotBoundsTime)*CV_DEF_FPS).astype(int)

                frameNb = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(args.dataset,vidName))[-1,1]
                starts = np.concatenate(([0],shotBoundsFrame),axis=0)
                ends =  np.concatenate((shotBoundsFrame-1,[frameNb]),axis=0)
                shotBoundsFrame = np.concatenate((starts[:,np.newaxis],ends[:,np.newaxis]),axis=1)
                np.savetxt("../data/{}/{}/result.csv".format(args.dataset,vidName),shotBoundsFrame)
                        #subprocess.run("shotdetect -i "+accVidPath.replace("_tmp","")+" -o "+videoFoldPath+" -f -l -m",shell=True)

def extractAudio(videoPath,videoFoldPath):
    #Extracting audio
    videoSubFold =  os.path.splitext(videoPath)[0]
    audioPath = videoSubFold+".wav"

    if not os.path.exists(audioPath):

        subprocess.run("ffprobe "+videoPath+" 2> tmp.txt",shell=True)

        audioInfStr = None
        with open("tmp.txt") as metadata:
            for line in metadata:
                if line.find("Audio:") != -1:
                    audioInfStr = line
        if not audioInfStr:
            raise ValueError("No audio data found in ffprobe output")

        audioSampleRate = int(audioInfStr.split(",")[1].replace(",","").replace("Hz","").replace(" ",""))
        audioBitRate = int(audioInfStr.split(",")[4].replace(",","").replace("kb/s","").replace(" ","").replace("(default)\n",""))*1000
        command = "ffmpeg -loglevel panic -i {} -ab {} -ac 2 -ar {} -vn {}".format(videoPath,audioBitRate,audioSampleRate,audioPath)
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
