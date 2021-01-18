# mood-matcher
Program that uses your Spotify liked songs to determine your taste, and then will pick out songs you may like from a playlist.

This program uses k-means analysis to find how your music taste is distributed. Spotify's API provides a set of metrics for every song, so we can use this to model every song as a vector. After finding the centroids that best model the data, the program finds the songs in the target playlist that are the most like the centroids (i.e. those with a small angle between it and the centroid vector).

You can also manually select the vector you want to match the songs in the playlist against.
