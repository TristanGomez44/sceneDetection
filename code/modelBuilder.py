import sys
import torchvision
from torchvision.models.inception import inception_v3
import cv2
import glob
import torch
from skimage.transform import resize
import numpy as np
class DiagBlock():

    def __init__(self,cuda=True,batchSize=16):
        self.incep = inception_v3(pretrained=True)

        self.cuda = cuda
        self.batchSize = batchSize

        if cuda:
            self.incep = self.incep.cuda()

    def detectDiagBlock(self,imagePathList):

        def preprocc(x):
            x = cv2.imread(x)
            x = resize(x, (299, 299,3), anti_aliasing=True)
            x = torch.tensor(x).cuda().permute(2,0,1).float().unsqueeze(0)

            return x

        #The modulo term increases the total number of batch by one in case
        #One last small batch is required
        nbBatch = len(imagePathList)//self.batchSize + (len(imagePathList) % self.batchSize == 0)

        for i in range(nbBatch):
            print(i)

            indMax = min((i+1)*self.batchSize,len(imagePathList))

            imageList = list(map(preprocc,imagePathList[i*self.batchSize:indMax]))

            videoExtr = torch.cat(imageList)

            featList = self.incep(videoExtr)

            print(featList)
def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))


def main():

    imagePathList = np.array(sorted(glob.glob("../data/big_buck_bunny_480p_surround-fix/middleFrames/*"),key=findNumbers),dtype=str)

    diagBlock = DiagBlock(cuda=True)

    diagBlock.detectDiagBlock(imagePathList)
if __name__ == "__main__":
    main()
