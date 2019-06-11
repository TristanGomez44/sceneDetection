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
Use the script formatData.py pour merge the videos, extract the audio and create the annotations :

'''
python formatData.py --dataset Holly2 --merge_videos avi
'''

The value of the --merge_videos argument is "avi" because the videos of this dataset are in the avi format. This operation should take some time. The dataset is ready when it is done !


#### Youtube

Dowload the videos with youtube-dl in a folder called "youtube" in the "data" folder :

'''
youtube-dl -f 18 -o "%(title)s.%(ext)s" https://www.youtube.com/playlist?list=PLSdQjuD0Brw4R2I2jdLadTS6WPNcimfG6
'''

The -f 18 argument download the video in a 640x360 mp4 format.

Once the dataset is downloaded, group the videos by movies with the formatData.py script :

'''
python formatData.py --dataset youtube --format_youtube
'''

Then, merge the videos, extract the sound and create the annotations with the same script, different arguments :

'''
python formatData.py --dataset youtube --merge_videos mp4
'''

#### Youtube Large

Dowload the videos with youtube-dl in a folder called "youtube" in the "data" folder :

'''
youtube-dl -f 18 -ciw -o "%(title)s.%(ext)s" -v https://www.youtube.com/user/movieclips/
'''

The -f 18 argument download the video in a 640x360 mp4 format.

Once the dataset is downloaded, group the videos by movies with the formatData.py script :

'''
python formatData.py --dataset youtube_large --format_youtube
'''

Then, merge the videos, extract the sound and create the annotations with the same script, different arguments :

'''
python formatData.py --dataset youtube_large --merge_videos mp4 --write-description
'''

The --write-description also download the descriptions of the videos, necessary to find from which movie every video comes from.

There is 2000 movies and 25 000 clips, so this should take a while (several days...).

The format of the description varies from one video to another so there are some video (aproximately 200) that are just rejected by
the algorithm and put in a folder "nodescr_youtube_large" in the "data" folder. The description are also put in a folder "descr_youtube_large",
alors in the "data" folder.

### OVSD

The links to download each video from the OVSD dataset can be found on the IBM website : http://www.research.ibm.com/haifa/projects/imt/video/Video_DataSetTable.shtml

Once the video are downloaded, put them in a folder called 'OVSD' in the 'data' folder. Also download the annotations (the button 'Full dataset meta-data download' at the bottom of the page) and put the folder containing them in the 'OVSD' folder as well. Ensure that the video name are the same than the annotations file. For example if a video is named 'filmA.avi' then its annotation should be named 'filmeA_scenes.txt'.

## Usage

### Scripts you have to use :

The script you have to use are the following :

#### The trainVal.py script :

If you want to train a model called 'testModel' in an experiment called 'testExperience' during 30 epochs with the other parameters left with their default value, simply type :

```
python trainVal.py -c model.config --exp_id testExperience --model_id testModel --epochs 30
```

The argument -c model.config allows the script to read the config file model.config which contains default value for all the arguments. All the arguments are detailed in the script args.py

If you want to train a siamese network to differentiate image coming from different scenes, type :

```
python trainVal.py -c model.config --exp_id siamese --model_id siam1 --train_siam
```

To visualise the metrics evolution during the training, you can use tensorboardX :

```
tensorboad --logdir=../results/expName
```

Where 'expName' is the name of the experiment. You should then be able to open your navigator and go to the adress indicated.

#### The processResults.py script :
This contains functions to compute metrics and visualise the results of training. To plot the scores that a model produced on the video, on the video itself.

```
 python processResults.py -c model.config --score_vis_video ../results/keepLearning3/moreLayers_epoch99_Big_fish.csv --exp_id keepLearning3 --dataset_test Holly2
```

Here the score produced by the model "moreLayers" at epoch 99 (from experience 'keepLearning3') on the video 'BigFish' (from dataset 'Holly2') will be plot on the video 'BigFish'.
The resulting video will be written at the path "../vis/keepLearning3/moreLayers_Big_fish_score.mp4".


To plot the t-sne representation of the shot produced by a model from all the video in a dataset, use the following command :

```
python processResults.py -c model.config --exp_id testExperience --model_id testModel --seed 1 --dataset_test testDataset --test_part_beg 0 --test_part_end 0.5 --tsne
```

Here, the videos from the first half of the dataset testDataset will be processed.


## Reproduce the results :

Here, are the scripts to train the models with the different ideas I have proposed :

```
python trainVal.py -c model.config --exp_id improvements  --model_id baseline        --epochs 100
python trainVal.py -c model.config --exp_id improvements  --model_id tempCnn         --epochs 100 --batch_size 4 --temp_model resnet50 --feat resnet50
python trainVal.py -c model.config --exp_id improvements  --model_id longerSeq       --epochs 100 --l_min 55 --l_max 65
python trainVal.py -c model.config --exp_id improvements  --model_id softLoss        --epochs 60  --batch_size 4 --soft_loss True --soft_loss_width 4
python trainVal.py -c model.config --exp_id improvements  --model_id softLossAnneal  --epochs 100 --batch_size 4 --soft_loss True --soft_loss_width 10,9,8,7,6,5,4,3,2,1
python trainVal.py -c model.config --exp_id improvements  --model_id FrameAtt        --epochs 100 --batch_size 1 --frames_per_shot 4
python trainVal.py -c model.config --exp_id improvements  --model_id VisualAndAudio  --epochs 100 --batch_size 1  --temp_model resnet50 --feat resnet50 --feat_audio vggish

```


### Other scripts

The other scripts are the following :

- args.py : Defines the arguments.
- load_data.py : Contains the classes defining the loader to train the siamese network and the CNN-RNN
- modelBuilder.py : Contains the classes definining the siamese model and the CNN-RNN
- formatData.py : the script to format the data.
