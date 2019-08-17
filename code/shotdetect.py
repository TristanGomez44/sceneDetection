from __future__ import division

import subprocess
import sys
import os
def extract_shots_with_ffprobe(src_video, threshold=0.1):
    """
    uses ffprobe to produce a list of shot
    boundaries (in seconds)

    Args:
        src_video (string): the path to the source
            video
        threshold (float): the minimum value used
            by ffprobe to classify a shot boundary

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """

    videoName = os.path.splitext(os.path.basename(src_video))[0]
    datasetName = os.path.dirname(src_video).split("/")[-1]

    subprocess.call("ffprobe -show_frames -of compact=p=0 -f lavfi movie=" + src_video + ",select=\'gt(scene\," + str(threshold) + ")\' > shotBoundsRaw_{}_{}.txt".format(datasetName,videoName), shell=True)

    with open("shotBoundsRaw_{}_{}.txt".format(datasetName,videoName), 'r') as output_file:
        output = output_file.read()

    boundaries = extract_boundaries_from_ffprobe_output(output)

    return boundaries

def extract_boundaries_from_ffprobe_output(output):
    """
    extracts the shot boundaries from the string output
    producted by ffprobe

    Args:
        output (string): the full output of the ffprobe
            shot detector as a single string

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """
    boundaries = []
    for line in output.split('\n')[:-1]:
        boundary = float(line.split('|')[4].split('=')[-1])
        boundaries.append(boundary)
    return boundaries
