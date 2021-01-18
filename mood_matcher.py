# importing libraries

#this takes a while because it makes requests to spotify and it limits the number of requests per second

from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import numpy as np
import spotipy.util as util

import matplotlib as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

scope = 'playlist-modify-public' # for making new playlists

scope1 = "user-library-read" # for accessing one's songs

# setting everything up

# (you'll need to get set up with Spotify's API first)

SPOTIPY_CLIENT_ID = # enter here
SPOTIPY_CLIENT_SECRET = # enter here
SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:9090'

# change to your username below

token = util.prompt_for_user_token('Viraj ramakrishnan', scope, client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI)
token1 = util.prompt_for_user_token('Viraj ramakrishnan', scope1, client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI)

sp = spotipy.Spotify(auth=token)
sp1 = spotipy.Spotify(auth=token1)

# access song stats 

def get_track_features(song_id):
	meta = sp.track(song_id)
	features = sp.audio_features(song_id)

	# meta
	name = meta['name']
	album = meta['album']['name']
	artist = meta['album']['artists'][0]['name']
	release_date = meta['album']['release_date']
	length = meta['duration_ms']
	popularity = meta['popularity']

	# features
	acousticness = features[0]['acousticness']
	danceability = features[0]['danceability']
	energy = features[0]['energy']
	instrumentalness = features[0]['instrumentalness']
	liveness = features[0]['liveness']
	loudness = features[0]['loudness']
	speechiness = features[0]['speechiness']
	tempo = features[0]['tempo']
	time_signature = features[0]['time_signature']

	track = [name, album, artist, release_date, length, song_id, float(popularity) / 100, acousticness, danceability, energy, instrumentalness, liveness, speechiness, float(tempo)/100]
	return track

# converts data into a numpy matrix
def playlist_to_matrix(creator, playlist_id):
	playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]

	# this is the song title, album, etc...
	info_matrix = np.empty((len(playlist), 6), dtype='object')

	# these are all the stats of the songs
	result_matrix = np.empty((len(playlist), 8), dtype=np.float64)

	#adding every song into the matrix
	index = 0

	for track in playlist:
		track_id = track["track"]["id"]

		track_features = get_track_features(track_id)

		for i in range(8):
			result_matrix[index][i] = np.array(track_features)[i + 6]

		for i in range(6):
			info_matrix[index][i] = np.array(track_features)[i]

		index += 1

	return (info_matrix, result_matrix)

# least squares algorithm

def least_squares(A, b):

	return np.linalg.inv(A.T @ A) @ A.T @ b

# mixed playlist function for creating playlist using least squares
# not quite what we are looking for, least squares is perhaps not the best, as we want
#each song to be a good fit, rather than the 'linear combination' of them being a good fit

def mixed_playlist(info_matrix, result_matrix, target_vector):

	x_hat = least_squares(result_matrix.T, target_vector)

	print("Performed least squares analysis")

	result_array = [list(np.concatenate((info_matrix[i], np.array(x_hat[i])), axis=None)) for i in range(len(x_hat))]

	result_array.sort(reverse=True, key=lambda x: x[4])

	final_playlist = result_array[:20]

	return final_playlist

# this function takes the data and makes it into vectors. It takes the dot product of the song vectors and the desired song vector
# the songs with the highest dot product (of the vectors, after making their magnitudes 1) are the ones which are the most similar

def precision_playlist(info_matrix, result_matrix, target_vector):

	# finds (a.b)//(||a||.||b||) = cos(theta) of angle between songs
	# big cos(theta) means small angle means similar songs

	def angle(v1, v2):
		return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

	# add the angle to the data we have
	similarity_matrix = [list(info_matrix[i]) + list(result_matrix[i]) + [angle(result_matrix[i], target_vector)] for i in range(len(info_matrix))]

	# sort the data by that angle
	similarity_matrix.sort(reverse=True, key=lambda x: x[14])

	return similarity_matrix

# user input section
def metric_collection():

	print("Hey, this program creates playlists based on your music preferences.")
	print("In the following questions, we are going to guage what kind of music you like.")
	print("Unless otherwise stated, the scale goes from 0 to 1, with a higher decimal indicating a higher preference.")
	print()

	popularity_level = float(input("How popular is the music you want (0 to 1)? "))
	danceability_level = float(input("How dancable is the music you want (0 to 1)? "))
	acousticness_level = float(input("How acoustic is the music you want (0 to 1)? "))
	energy_level = float(input("How energetic is the music you want (0 to 1)? "))
	instrumentalness_level = float(input("How instrumental is the music you want (0 to 1)? "))
	liveness_level = float(input("How 'live' is the music you want (0 to 1)? "))
	speechiness_level = float(input("How much speech do you want (0 to 1)? "))
	tempo_level = float(input("How fast is the music you want (input a tempo in beats per minute)? "))

	return np.array([popularity_level, danceability_level, acousticness_level, energy_level, instrumentalness_level, liveness_level, speechiness_level, float(tempo_level)/100])

def k_means_analysis(num_songs, target_playlist_creator, target_playlist_id):

	# getting liked songs
	liked_songs = sp1.current_user_saved_tracks()['items']

	liked_song_len = len(liked_songs)


	# this is the song title, album, etc...
	info_matrix = np.empty((liked_song_len, 6), dtype='object')

	# these are all the stats of the songs
	result_matrix = np.empty((liked_song_len, 8), dtype=np.float64)

	# convert it into numpy array 

	index = 0

	for track in liked_songs:
		track_id = track["track"]["id"]

		track_features = get_track_features(track_id)

		for i in range(8):
			result_matrix[index][i] = np.array(track_features)[i + 6]

		for i in range(6):
			info_matrix[index][i] = np.array(track_features)[i]

		index += 1

	# need to record mean and std_dev of each metric to reconstruct actual centroids

	std_dev_stats = np.std(result_matrix, axis=0)
	mean_stats = np.mean(result_matrix, axis=0)


	# standardize (i.e. get z-scores)
	scaler = StandardScaler()
	scaled_result_matrix = scaler.fit_transform(result_matrix)

	# do k-means analysis

	best_coeff = -1
	best_k = 2
	best_centroids = None
	
	# for different numbers of centroids
	# find best number of centroids, and then use those results

	print("doing k-means analysis")


	for k in range(2, 11):
		kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=None)

		print("trying with k = " + str(k))

		kmeans.fit(scaled_result_matrix)

		score = silhouette_score(scaled_result_matrix, kmeans.labels_)

		print("score = " + str(score))

		if score > best_coeff:
			best_coeff = score 
			best_k = k
			best_centroids = kmeans.cluster_centers_


	# un-standardize centroid vectors

	# z = (x - mean)/std_dev

	# x = z * std_dev + mean


	for i in range(best_k):

		for j in range(8):

			best_centroids[i][j] = best_centroids[i][j] * std_dev_stats[j] + mean_stats[j]

	print("here are the song cluster centers")

	print("number of clusters: " + str(best_k))

	print(best_centroids)

	print()

	print("processing target playlist")

	# load up target playlist from which we'll be choosing songs

	target_info_matrix, target_result_matrix = playlist_to_matrix(target_playlist_creator, target_playlist_id)

	# make list of songs that are closest to cluster centers

	print("collecting best songs from target playlist")

	combo_playlist = []

	for centroid in best_centroids:

		combo_playlist.extend(precision_playlist(target_info_matrix, target_result_matrix, centroid)[:num_songs//best_k])

	# remove duplicates, return

	print("removing duplicates")

	cleaned_combo_playlist = [combo_playlist[i] for i in range(len(combo_playlist)) if combo_playlist[i] not in combo_playlist[:i]]

	print("playlist length: " + str(len(cleaned_combo_playlist)))

	return cleaned_combo_playlist


"""

	# for testing purposes

	popularity_level = 0.1
	danceability_level = 0.9
	acousticness_level = 0.1
	energy_level = 0.9
	instrumentalness_level = 0.9
	liveness_level = 0.1
	speechiness_level = 0.1
	tempo_level = 150

"""


def metrics_to_playlist(num_songs, your_id, target_playlist_creator, target_playlist_id, manual=False, name="playlist highlights"):

	# data processing

	# make the user input into a vector (which we can dot with the songs)

	if manual: # manual is the option where you input what you want
		target_vector = metric_collection()

		# here's where you enter the username of the creator of the playlist and the id of the playlist, I set up for the nye one

		# this is the target playlist from which we will be selecting songs
		info_matrix, result_matrix = playlist_to_matrix(target_playlist_creator, target_playlist_id)

		print("Transformed data to matrix form")

		
		# here is the list of songs to be made into a playlist

		precision_result = precision_playlist(info_matrix, result_matrix, target_vector)[:num_songs]

	else: # this is the option where we use k-means to judge music taste

		# make a playlist based on your taste 
		# this will take in all your liked songs

		precision_result = k_means_analysis(num_songs, target_playlist_creator, target_playlist_id)

	track_ids = []

	# print result, with the cosine of the angle as 'percentage fit'
	print("Song | Album | Artist | percentage fit")

	for song in precision_result:
		print(song[:3] + [round(song[-1] * 100, 2)])

		track_ids.append(song[5])

	print()
	print(track_ids)
	print("Creating new playlist...")

	sp.user_playlist_create(your_id, name, public=True, description='1 2 3 testing testing?')

	playlist_list = sp.user_playlists(your_id, limit=50, offset=0)

	desired_playlist_id = [playlist['id'] for playlist in playlist_list['items'] if playlist['name'] == name][0]

	print("Adding new tracks to playlist")

	sp.user_playlist_add_tracks(your_id, desired_playlist_id, track_ids, position=None)

	print("Playlist created!")

# change details below

metrics_to_playlist(40, your_user_id_here, target_playlist_creator, target_playlist_id, manual=False, name="tasteful melodies for k-means analyses")


