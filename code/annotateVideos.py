import argparse
import glob
import os
import numpy as np
import cv2
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read videos and add the scene number on the image. It uses a ground truth')

    parser.add_argument('--video_folder', metavar='PATH',help='Path leading to the video folder',type=str)
    parser.add_argument('--annotations_folder', metavar='PATH',help='Path leading to the annotations folder',type=str)
    parser.add_argument('--truebase_folder', metavar='PATH',help='Path leading to the true baseline folder',type=str)
    parser.add_argument('--implbase_folder', metavar='PATH',help='Path leading to the implemented baseline folder',type=str)

    args = parser.parse_args()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for videoPath in glob.glob(args.video_folder+"/*.*"):

        if videoPath.find("annot") == -1:
            print(videoPath)

            vidName,ext = os.path.splitext(os.path.basename(videoPath))

            if not os.path.exists(args.video_folder+"/"+vidName+"_annot"+ext):

                annotationPath = args.annotations_folder +"/"+vidName+"_frames_scenes.csv"
                annotations = np.genfromtxt(annotationPath)

                trueBasePath = args.truebase_folder+"/"+vidName+"_frames_truebasecuts.csv"
                trueBase = np.genfromtxt(trueBasePath)

                implBasePath = args.implbase_folder+"/"+vidName+"_baseline.csv"
                implBase = np.genfromtxt(implBasePath)

                cap = cv2.VideoCapture(videoPath)

                success = True
                i = 0
                scene = 0
                sceneBase = 0
                sceneImplBase = 0
                annotatedVideo = None
                while success:
                    success, imageRaw = cap.read()

                    if not annotatedVideo:
                        annotatedVideo = cv2.VideoWriter(args.video_folder+"/"+vidName+"_annot"+ext, fourcc, 30, (imageRaw.shape[1],imageRaw.shape[0]))
                        print("Loading the video with",annotatedVideo.get(cv2.CAP_PROP_FPS),"fps")

                    cv2.rectangle(imageRaw, (0,0), (240,120), (255,255,255),thickness=-1)
                    cv2.putText(imageRaw,"Scene "+str(scene), (10, 40),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
                    cv2.putText(imageRaw,"Baseline : Scene "+str(sceneBase), (10, 60),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
                    cv2.putText(imageRaw,"Impl Baseline : Scene "+str(sceneImplBase), (10, 80),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)

                    if scene < len(annotations):
                        if i > annotations[scene,1]:
                            scene += 1

                    if sceneBase < len(trueBase):
                        if i > trueBase[sceneBase,1]:
                            sceneBase += 1

                    if sceneImplBase < len(implBase):
                        if i > implBase[sceneImplBase,1]:
                            sceneImplBase += 1

                    if i > trueBase[-1,1] and i > annotations[-1,1] and i > implBase[-1,1]:
                        success = False

                    annotatedVideo.write(imageRaw)

                    i += 1

                annotatedVideo.release()
