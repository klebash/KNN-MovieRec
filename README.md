**MovieRec-Pearson**
MovieRec-Pearson is a movie recommendation system built using collaborative filtering techniques with Pearson correlation to identify similarities between movies. This repository provides a comprehensive framework for predicting movie ratings based on the preferences of similar users.

**Features**
-Data Preprocessing: Filters out movies and users with less than 5 ratings to ensure quality data.
-Training and Testing Splits: Splits the dataset into training and testing sets with customizable training set size.
-Rating Matrix: Constructs a user-movie rating matrix, filling missing values with the mean rating for each movie.
-Similarity Calculation: Utilizes Pearson correlation to compute similarities between movies, ensuring accurate neighbor selection by removing self-similarity.
-K-Nearest Neighbors: Allows users to specify the number of neighbors (k) for predictions.
-Prediction Methods: Offers both mean rating and weighted mean rating prediction methods based on the ratings of similar movies.
-Performance Metrics: Calculates precision, recall, and sum of squared errors (SSE) to evaluate the performance of the recommendation system.

