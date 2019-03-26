import random
from args import ArgReader
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--dataset_id',default="test",type=str,metavar='DATAID',help='The name of the csv file generated.')
    argreader.parser.add_argument('--nb_shots',default=100,type=int,metavar='NB',help='The number of shots in the video')
    argreader.parser.add_argument('--nb_scenes',default=5,type=int,metavar='NB',help='The number of scene in the video')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    np.random.seed(args.seed)

    sceneCuts = np.arange(args.nb_shots)
    np.random.shuffle(sceneCuts)

    sceneCuts = sceneCuts[:args.nb_scenes]
    sceneCuts.sort()

    sceneCuts = np.concatenate(([0],sceneCuts,[args.nb_shots]))

    simMatrix = np.ones((args.nb_shots,args.nb_shots))

    for i in range(len(sceneCuts)-1):
        print(i,sceneCuts[i],sceneCuts[i+1])
        simMatrix[sceneCuts[i]:sceneCuts[i+1],sceneCuts[i]:sceneCuts[i+1]] = 0

    np.savetxt("../results/{}.csv".format(args.dataset_id),simMatrix)
    np.savetxt("../data/{}_cuts.csv".format(args.dataset_id),sceneCuts)

    plt.imshow(simMatrix.astype(int), cmap='gray', interpolation='nearest')
    plt.xlim(0,len(simMatrix))
    plt.ylim(len(simMatrix),0)
    plt.savefig("../vis/{}.png".format(args.dataset_id))

if __name__ == "__main__":
    main()
