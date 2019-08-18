
mkdir ../data/Holly2

#Download the videos
wget ftp://ftp.irisa.fr/local/vistas/actions/Hollywood2-scenes.tar.gz -P ../data

#If the link is broken, go to the following link : https://www.di.ens.fr/~laptev/actions/hollywood2/ and download the scenes by clicking on the link : "Scene samples (25Gb)".
#Untar the dowloaded videos :

tar xzf ../data/Hollywood2-scenes.tar.gz -C ../data/

#Put the videos in the "AVIClipsScenes" folder in the "Holly2" folder :

mv ../data/Hollywood2/AVIClipsScenes/*.avi ../data/Holly2/

#Group the videos coming from the same movie in separate folders using this script :

./sortHollyVideos.sh

#Use the script formatData.py pour merge the videos, extract the audio and create the annotations :

python formatData.py --dataset Holly2 --merge_videos avi

#The value of the --merge_videos argument is "avi" because the videos of this dataset are in the avi format. \
#This operation should take some time. The dataset is ready when it is done !
