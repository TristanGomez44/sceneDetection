import requests
import json
import torch
from args import ArgReader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
import argparse
import re
import time
import load_data
import metrics
import matplotlib.cm as cm

def main(argv=None):

    parser = argparse.ArgumentParser(description='Generate scores produced by an random model on a dataset.')

    parser.add_argument('--seed', metavar='NB',help='The seed to generate the scores',type=int,default=1)
    parser.add_argument('--model_id', metavar='ID',help='The ID of the random model',type=str)
    parser.add_argument('--exp_id', metavar='ID',help='The exp id of the random model',type=str)
    parser.add_argument('--dataset', metavar='ID',help='The dataset',type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    videoFoldList = sorted(glob.glob("../data/{}/*/".format(args.dataset)))
    videoFoldList = list(filter(lambda x:x.find("annotation") == -1,videoFoldList))
    unif = torch.distributions.uniform.Uniform(0, 1)

    for videoFold in videoFoldList:

        nbShots = len(np.genfromtxt(videoFold+"/result.csv"))
        scores = unif.sample((nbShots,)).unsqueeze(1)

        inds = torch.arange(nbShots).unsqueeze(1).float()

        videoName = videoFold.split("/")[-2]

        np.savetxt("../results/{}/{}_epoch0_{}.csv".format(args.exp_id,args.model_id,videoName),torch.cat((inds,scores),dim=1))

if __name__ == "__main__":
    main()
