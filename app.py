from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import os
import csv
import sys
import re
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
from surprise import SVD, Dataset
from surprise.model_selection import cross_validate
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader



app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	class MovieLens:
	    movieID_to_name = {}
	    name_to_movieID = {}
	    ratingsPath = "ratings.csv"
	    moviesPath = "movies.csv"
	    def loadMovieLensLatestSmall(self):
	        # Look for files relative to the directory we are running from
	        # print(sys.argv[0], 'check')
	        # os.chdir(os.path.dirname(sys.argv[0]))
	        ratingsDataset = 0
	        self.movieID_to_name = {}
	        self.name_to_movieID = {}

	        reader = Reader(line_format='user item rating timestamp',
	                        sep=',', skip_lines=1)

	        ratingsDataset = Dataset.load_from_file(
	            self.ratingsPath, reader=reader)

	        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
	                movieReader = csv.reader(csvfile)
	                next(movieReader)  # Skip header line
	                for row in movieReader:
	                    movieID = int(row[0])
	                    movieName = row[1]
	                    self.movieID_to_name[movieID] = movieName
	                    self.name_to_movieID[movieName] = movieID

	        return ratingsDataset

	    def getUserRatings(self, user):
	        userRatings = []
	        hitUser = False
	        with open(self.ratingsPath, newline='') as csvfile:
	            ratingReader = csv.reader(csvfile)
	            next(ratingReader)
	            for row in ratingReader:
	                userID = int(row[0])
	                if (user == userID):
	                    movieID = int(row[1])
	                    rating = float(row[2])
	                    userRatings.append((movieID, rating))
	                    hitUser = True
	                if (hitUser and (user != userID)):
	                    break

	        return userRatings

	    def getPopularityRanks(self):
	        ratings = defaultdict(int)
	        rankings = defaultdict(int)
	        with open(self.ratingsPath, newline='') as csvfile:
	            ratingReader = csv.reader(csvfile)
	            next(ratingReader)
	            for row in ratingReader:
	                movieID = int(row[1])
	                ratings[movieID] += 1
	        rank = 1
	        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
	            rankings[movieID] = rank
	            rank += 1
	        return rankings

	    def getGenres(self):
	        genres = defaultdict(list)
	        genreIDs = {}
	        maxGenreID = 0
	        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
	            movieReader = csv.reader(csvfile)
	            next(movieReader)  # Skip header line
	            for row in movieReader:
	                movieID = int(row[0])
	                genreList = row[2].split('|')
	                genreIDList = []
	                for genre in genreList:
	                    if genre in genreIDs:
	                        genreID = genreIDs[genre]
	                    else:
	                        genreID = maxGenreID
	                        genreIDs[genre] = genreID
	                        maxGenreID += 1
	                    genreIDList.append(genreID)
	                genres[movieID] = genreIDList
	        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
	        for (movieID, genreIDList) in genres.items():
	            bitfield = [0] * maxGenreID
	            for genreID in genreIDList:
	                bitfield[genreID] = 1
	            genres[movieID] = bitfield

	        return genres

	    def getYears(self):
	        p = re.compile(r"(?:\((\d{4})\))?\s*$")
	        years = defaultdict(int)
	        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
	            movieReader = csv.reader(csvfile)
	            next(movieReader)
	            for row in movieReader:
	                movieID = int(row[0])
	                title = row[1]
	                m = p.search(title)
	                year = m.group(1)
	                if year:
	                    years[movieID] = int(year)
	        return years

	    def getMiseEnScene(self):
	        mes = defaultdict(list)
	        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
	            mesReader = csv.reader(csvfile)
	            next(mesReader)
	            for row in mesReader:
	                movieID = int(row[0])
	                avgShotLength = float(row[1])
	                meanColorVariance = float(row[2])
	                stddevColorVariance = float(row[3])
	                meanMotion = float(row[4])
	                stddevMotion = float(row[5])
	                meanLightingKey = float(row[6])
	                numShots = float(row[7])
	                mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
	                   meanMotion, stddevMotion, meanLightingKey, numShots]
	        return mes

	    def getMovieName(self, movieID):
	        if movieID in self.movieID_to_name:
	            return self.movieID_to_name[movieID]
	        else:
	            return ""

	    def getMovieID(self, movieName):
	        if movieName in self.name_to_movieID:
	            return self.name_to_movieID[movieName]
	        else:
	            return 0

	def BuildAntiTestSetForUser(testSubject, trainset):
	    fill = trainset.global_mean

	    anti_testset = []

	    u = trainset.to_inner_uid(str(testSubject))

	    user_items = set([j for (j, _) in trainset.ur[u]])
	    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
	                             i in trainset.all_items() if
	                             i not in user_items]
	    return anti_testset

	# Receives the input query from form
	if request.method == 'POST':

		testSubject1 = request.form['namequery']

		df= pd.read_csv("data/names_dataset.csv")

		testSubject=df[df['name']==testSubject1]['index'].values[0]


		res = ""

		ml = MovieLens()

		data = ml.loadMovieLensLatestSmall()

		userRatings = ml.getUserRatings(testSubject)
		loved = []
		hated = []
		for ratings in userRatings:
		    if (float(ratings[1]) > 4.0):
		        loved.append(ratings)
		    if (float(ratings[1]) < 3.0):
		        hated.append(ratings)

		# res += "\nUser, " + testSubject + ", loved these movies:\n"
		print('check', ml.getUserRatings(testSubject))
		for ratings in loved:
			res += ml.getMovieName(ratings[0]) + '\n'
		# res += "\n...and didn't like these movies:\n"
		for ratings in hated:
		    res += ml.getMovieName(ratings[0]) + '\n'

		trainSet = data.build_full_trainset()
		algo = SVD()
		algo.fit(trainSet)

		testSet = BuildAntiTestSetForUser(testSubject, trainSet)
		predictions = algo.test(testSet)

		recommendations = []

		res += "\nWe recommend:\n"
		for userID, movieID, actualRating, estimatedRating, _ in predictions:
		    intMovieID = int(movieID)
		    recommendations.append((intMovieID, estimatedRating))

		recommendations.sort(key=lambda x: x[1], reverse=True)

		for ratings in recommendations[:10]:
		    res += ml.getMovieName(ratings[0]) + '\n'


		namequery = request.form['namequery']
		data = [namequery]
		my_prediction = res
		# print(my_prediction)
	return render_template('results.html', prediction = my_prediction, name = namequery)


if __name__ == '__main__':
	app.run(debug=True)
