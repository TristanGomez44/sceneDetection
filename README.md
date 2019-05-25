# Scene detection with CNN-RNN

This repo contains python scripts to train and evaluate a CNN-RNN to detect scene change.

## Instalation

First clone this git. Then install conda and the dependencies with the following command :

```
conda env create -f environment.yml
```

### Data bases :

The datasets should be folders in the "data" folder. When you want to use a dataset, create a folder and put the dataset in it.

#### HollyWood2 :

    Go to the following link : https://www.di.ens.fr/~laptev/actions/hollywood2/ and download the scenes : Scene samples (25Gb) in a folder called "Holly2" in the folder "data".
    Untar the dowloaded videos and put the video files in the Holly2 folder.
    Group the videos coming from the same movie in separate folders. This an operation you have to do at hand (There is a thousand videos but they are numbered by movie : the videos 1 to x come from the movie A, the video x+1 to y come from the movie B, etc.)
    Use the script formatData.py pour merge the videos and create the annotations :

'''
python formatData.py --dataset Holly2 --merge_videos avi
'''

The value of the --merge_videos argument is "avi" because the videos of this dataset are in the avi format. This operation should take some time. The dataset is ready when it is done !


#### OVSD

#### 


## Usage

### Scripts you have to use :

The script you have to use are the following :

- trainVal.py :

If you want to train a model called 'testModel' in an experiment called 'testExperience' during 30 epochs with the other parameters left with their default value, simply type :

```
python trainVal.py -c model.config --exp_id testExperience --model_id testModel --epochs 30
```

The argument -c model.config allows the script to read the config file model.config which contains default value for all the arguments. All the arguments are detailed in the script args.py

- processResults.py

### Other scripts

The other scripts are the following :

- args.py : Defines the arguments.
- load_data.py : Contains the classes defining the loader to train the siamese network and the CNN-RNN
- modelBuilder.py : Contains the classes definining the siamese model and the CNN-RNN
- formatData.py : the script to format the data.
- processResults.py : contains functions to compute metrics and visualise the results of training. This is the script to use to evaluate the model on a test dataset.
