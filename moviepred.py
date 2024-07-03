import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split

df = pd.read_csv('ratings.csv')
k_neigh = int(input("Insert number of neighbour's:"))
train_data_pres = int(input("Insert Presetage of the training set (Example input : 80)"))/100

movie_counts = df['movieId'].value_counts()
df = df[df['movieId'].isin(movie_counts[movie_counts >= 5].index)]
user_counts = df['userId'].value_counts()
df = df[df['userId'].isin(user_counts[user_counts >= 5].index)]


train_data, test_data = train_test_split(df, test_size=0.1,train_size=train_data_pres, random_state=42)
rating_matrix  = train_data.pivot_table(index='movieId',columns='userId',values='rating')

rating_matrix = rating_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

def remove_diagonal(df):
    n = df.shape[0]
    np.fill_diagonal(df.values, np.nan)
    df.fillna(0, inplace=True)
    return df
pearson_corr = rating_matrix.corr(method='pearson')
pearson_corr = remove_diagonal(pearson_corr)

def similarity_movies(movie_id,k_neigh):
    movie_corr = pearson_corr.loc[movie_id]
    top_k = movie_corr.nlargest(k_neigh)
    return top_k

def mean_pred(nearest_neighbors_ratings):
    mean_rating  = 0
    for rating in nearest_neighbors_ratings.values:
        mean_rating += rating
    mean_rating  = mean_rating/k_neigh
    return mean_rating

def weigthed_mean_pred(nearest_neighbors_ratings,nearest_neighbors):
    similarity_sum = nearest_neighbors.sum()
    weighted_mean_rating  = 0
    for rating,similarity in zip(nearest_neighbors_ratings.values,nearest_neighbors.values):
        weighted_mean_rating += (rating * similarity)
    weighted_mean_rating  = weighted_mean_rating/similarity_sum
    
    return weighted_mean_rating

def neigh_ratings(userId,movieId,k):
    nearest_neighbors = similarity_movies(movieId,k)
    nearest_neighbors_ratings = rating_matrix.loc[userId,nearest_neighbors.index]
    mean_pre = mean_pred(nearest_neighbors_ratings)
    weigthed_mean_pre  = weigthed_mean_pred(nearest_neighbors_ratings,nearest_neighbors)
    return mean_pre,weigthed_mean_pre

Predictions_mean = []
Predictions_weighted_mean = []
TP_mean = 0
FP_mean = 0
TP_wmean = 0
FP_wmean = 0
FN_mean = 0
FN_wmean = 0 

mean = 0
w_mean = 0 
for index,row in test_data.iterrows():
    mean,w_mean = neigh_ratings(row['userId'],row['movieId'],k_neigh)
    if mean >= 3:
        if row['rating'] >= 3:
            TP_mean += 1 
        else :
            FP_mean += 1
            
            
    if w_mean >= 3:
        if row['rating'] >= 3:
            TP_wmean += 1 
        else :
            FP_wmean += 1
    if mean < 3:
        if row['rating'] >= 3:
            FN_mean += 1 

    if w_mean < 3:
        if row['rating'] >= 3:
            FN_wmean += 1 

    Predictions_mean.append([row['userId'],row['movieId'],mean])
    Predictions_weighted_mean.append([row['userId'],row['movieId'],w_mean])
m_pre = pd.DataFrame(Predictions_mean, columns=['userId', 'movieId', 'rating'])
w_pre = pd.DataFrame(Predictions_weighted_mean, columns=['userId', 'movieId', 'rating'])

def calc_precision(TP,FP):
    return TP/(TP+FP)
def calc_recall(TP,FN):
    return TP / (TP + FN)
def calc_SSE(acttual,predicted):
    SSE = np.sum((acttual - predicted)**2)
    return SSE
mean_pre = calc_precision(TP_mean,FP_mean)
mean_recall = calc_recall(TP_mean,FN_mean)
mean_SSE = calc_SSE(test_data['rating'],m_pre['rating'])

wmean_pre = calc_precision(TP_wmean,FP_wmean)
wmean_recall = calc_recall(TP_wmean,FN_wmean)
wmean_SSE = calc_SSE(test_data['rating'],w_pre['rating'])

print("\n\n")
print("Mean:")
print(mean_pre)
print(mean_recall)
print(mean_SSE)

print("\n\n")
print("WEIGHTED Mean:")
print(wmean_pre)
print(wmean_recall)
print(wmean_SSE)
