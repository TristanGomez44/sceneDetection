import requests
import json

API_KEY = 'c76f02b9c8629ae09a178d293639d04c'

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
from matplotlib.ticker import StrMethodFormatter

def findYear(dataset,videoName):
    #Find the release year of a movie using the file name of the videos belonging to the same movie
    #Some movie have their respective videos containing in their filenames the reslease year
    #If we find a sequence of digits comon to all videos file name
    #it might be the release year or some part of the move title

    yearFound = False
    videoPaths = sorted(glob.glob("../data/{}/{}/*.mp4".format(dataset,videoName)))

    candidFound = False
    i=0
    indStart = 0
    lastStrMatch = None
    titleSkiped = False

    while i< len(videoPaths) and not yearFound:
        groupObj = re.search("\d{4}", videoPaths[i][indStart:])

        #If theres not match, the release year is not indicated in the filename
        if groupObj is None:
            i=len(videoPaths)
        else:

            strMatch = groupObj.group(0)

            #If the string match is found in the title, it is not the year
            #The search will start again after the position of the match
            if strMatch in videoName and not titleSkiped:
                indStart += videoPaths[i][indStart:].find(strMatch)+len(strMatch)
                titleSkiped = True
            else:
                if not lastStrMatch is None:
                    #If the string has also been find in another video extract, it must be the year
                    if strMatch == lastStrMatch:
                        yearFound = True
                    else:
                        lastStrMatch = strMatch
                else:
                    lastStrMatch = strMatch

                i+=1

    if not yearFound:
        return None
    else:
        return int(strMatch)

def saveDict(path,infosDict):

    with open(path, 'w') as json_file:
      json.dump(infosDict, json_file)

def updateInfoDict(infosDict,res,videoName):

    movieDict = {}
    if "genres" in res.keys():
        movieDict["genres"] = list(map(lambda x:x['name'],res["genres"]))
    if "production_countries" in res.keys():
        movieDict["production_countries"] = list(map(lambda x:x['name'],res["production_countries"]))
    if "release_date" in res.keys() and res["release_date"] != "":
        movieDict["year"] = res["release_date"].split("-")[0]
    if "revenue" in res.keys() and res["revenue"] != 0:
        movieDict["revenue"] = res["revenue"]

    infosDict[videoName] = movieDict

def chooseResult(res,dataset,videoName):

    #If we are not able to find a result with a release date that match the year we want
    #We just use the first result of the requests
    resInd = 0
    if len(res["results"]) > 1:
        releaseYear = findYear(dataset,videoName)

        #Searching among the results for the movie released at the year found
        if not releaseYear is None:
            i=0
            movieFound = False
            while i<len(res["results"]) and not movieFound:
                releaseDate = res["results"][i]["release_date"]
                if releaseDate != "" and int(releaseDate.split("-")[0]) == releaseYear:
                    movieFound = True
                else:
                    i+=1
            if movieFound:
                resInd = i
    return resInd

def updateOccDict(info,infoOccDict):
    if info in infoOccDict.keys():
        infoOccDict[info] +=1
    else:
        infoOccDict[info] =1

def infoBarplot(infosDict,infoKey,sortKey,dataset,xlabel,maxMod=20):
    #This will hold the number of occurences of each modality
    infoOccDict = {}

    for movie in infosDict.keys():
        if infoKey in infosDict[movie].keys():
            if isinstance(infosDict[movie][infoKey], list):
                for info in infosDict[movie][infoKey]:
                    updateOccDict(info,infoOccDict)
            else:
                updateOccDict(infosDict[movie][infoKey],infoOccDict)

    #Sorting the modalities order by frequency
    sortOccTupList = sorted([(info,infoOccDict[info]) for info in infoOccDict.keys()],key=sortKey)[:maxMod]

    infos,occ = zip(*sortOccTupList)

    infos = [info.replace("United States of America","USA").replace("United Kingdom","UK") for info in infos]

    print("Number of bars : ",len(infoOccDict.keys()))

    plt.figure()
    plt.bar(np.arange(len(infos)),occ)
    plt.ylabel("Number of occurences")
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(infos)),infos, rotation=45,ha="right")
    plt.tight_layout()
    plt.savefig("../vis/{}_{}.png".format(dataset,infoKey))

def infoHistogramPlot(infosDict,infoKey,dataset,dataType,xlabel,plotRange=None,nbBins=30):

    #This will hold the number of occurences of each modality
    infoOccDict = {}

    #Sorting the modalities order by frequency
    values = [dataType(infosDict[movie][infoKey]) for movie in infosDict.keys() if infoKey in infosDict[movie].keys()]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(values,bins=nbBins,range=plotRange)
    plt.xlabel(xlabel)
    plt.ylabel("Number of occurences")
    if not plotRange is None:
        plt.xticks(np.arange(11)*plotRange[1]/10,((np.arange(11)*plotRange[1]/10)/1000000).astype(int), rotation=45,ha="right")
    plt.tight_layout()

    plt.savefig("../vis/{}_{}.png".format(dataset,infoKey))

def main(argv=None):

    parser = argparse.ArgumentParser(description='Process the youtube_large dataset to plot histograms about its content, like genre or release year distribution, using the IMDB \
                                                    API.')

    #Collecting all files
    videoPaths = list(filter(lambda x:os.path.isfile(x),sorted(glob.glob("../data/youtube_large/*.*"))))
    #Remove the wav file
    videoPaths = list(filter(lambda x:x.find(".wav") == -1,videoPaths))
    #Isolates the movie name
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    if os.path.exists('../data/youtube_large_infos.json'):
        with open('../data/youtube_large_infos.json') as json_file:
            infosDict = json.load(json_file)
    else:
        infosDict = {}

    #Check if some movie infos have not yet been requested
    for j,videoName in enumerate(videoNames):

        if not videoName in infosDict.keys():

            print(videoName)

            request = requests.get('https://api.themoviedb.org/3/search/movie?api_key={}&query={}'.format(API_KEY,videoName.replace("_","+")))

            res = request.text
            d = json.loads(res)

            if "results" not in d.keys():

                #We have probably made too much request in a too shot amount of time
                #so lets save the results of the request done until now
                saveDict('../data/youtube_large_infos.json',infosDict)

                raise ValueError("No result in d : ",d)

            if len(d["results"]) > 0:

                resInd = chooseResult(d,"youtube_large",videoName)

                idMovie = d["results"][resInd]["id"]

                request = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(idMovie,API_KEY))
                res = request.text
                d = json.loads(res)

                #Add release year, genres and country of the movie in the dictionnary
                updateInfoDict(infosDict,d,videoName)

                if (j+1)%20 == 0:
                    saveDict('../data/youtube_large_infos.json',infosDict)
                    time.sleep(3)

    saveDict('../data/youtube_large_infos.json',infosDict)

    print("Number of movies : ",len(infosDict.keys()))

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 10}

    plt.rc('font', **font)
    plt.rc('axes', labelsize=20)


    infoBarplot(infosDict,"genres",lambda x:-x[1],"youtube_large","Genres")
    infoBarplot(infosDict,"production_countries",lambda x:-x[1],"youtube_large","Production countries")
    infoHistogramPlot(infosDict,"year","youtube_large",int,"Release year")
    infoHistogramPlot(infosDict,"revenue","youtube_large",float,"Movie revenue (in M$)",(0,1000000000),60)

if __name__ == "__main__":
    main()
