mkdir ../data/youtube_large

#Dowload the videos with youtube-dl in a folder called "youtube" in the "data" folder :
youtube-dl -f 18 -ciw -o "../data/youtube_large/%(title)s.%(ext)s" -v https://www.youtube.com/user/movieclips/

#The -f 18 argument download the video in a 640x360 mp4 format.

#Once the dataset is downloaded, group the videos by movies with the formatData.py script :

python formatData.py --dataset youtube_large --format_youtube

#Then, merge the videos and create the annotations with the same script, different arguments :

python formatData.py --dataset youtube_large --merge_videos mp4 --write-description

#The --write-description also download the descriptions of the videos, necessary to find from which movie every video comes from.

#There is 2000 movies and 25 000 clips, so this should take a while (2 or 3 days...). You can stop the download and go to the next
#step if you don't want to wait, things will work even if not all the videos are downloaded.

#The format of the description varies from one video to another so there are some video (aproximately 200) that are rejected
# and put in a folder "nodescr_youtube_large" in the "data" folder. The description are also put in a folder "descr_youtube_large",
#alors in the "data" folder.
