mkdir ../data/rai

#Download the videos
wget http://imagelab.ing.unimore.it/files/RaiSceneDetection.zip -P ../data

unzip ../data/RaiSceneDetection.zip -d ../data/

#Format them
python formatData.py --format_rai
