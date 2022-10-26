import pandas as pd
from sklearn.metrics.pairwise import paired_distances,cosine_similarity
import numpy as np

movies = pd.read_csv("./ml-latest-small/movies.csv")
rates = pd.read_csv("./ml-latest-small/ratings.csv")

# movies 留下 movieId與genres(電影分類)，rate 留下userId與movieId
# 使用movieId 串 movies.csv ratings.csv
# rating應該很有用!?
movies.drop("title", axis=1, inplace=True)
rates.drop(["rating","timestamp"], axis=1, inplace=True)
df = pd.merge(rates, movies, on="movieId") # usrId, movieId, genres


# 建電影特徵，對genres 做one-hot encoding
oneHot = movies["genres"].str.get_dummies("|")
movie_arr = pd.concat([movies, oneHot], axis=1)
movie_arr.drop("genres", axis=1, inplace=True)
movie_arr.set_index("movieId", inplace=True)

# 建使用者特徵，user對每個電影的平均分數
oneHot = df["genres"].str.get_dummies("|")
user_arr = pd.concat([df, oneHot], axis=1)
user_arr.drop(["movieId", "genres"], axis=1, inplace=True)
user_arr = user_arr.groupby("userId").mean()
# print(user_arr.head(5))

# 使用者電影餘弦相似度
similarity = cosine_similarity(user_arr.values, movie_arr.values)
similarity_mtx = pd.DataFrame(similarity, index=user_arr.index, columns=movie_arr.index)
# similarity_mtx.to_csv("ml-latest-small/cosine_similarity.csv")

# 取得查詢user最相似的前n部電影
def get_the_most_similar_movies(usrId, num):
    vec = similarity_mtx.loc[usrId].values
    sorted_idx = np.argsort(-vec)[:num]
    return list(similarity_mtx.columns[sorted_idx])

# 取得查詢電影最相似的前n位使用者

def get_the_most_similar_users(movieId, num):
    movie_vec = similarity_mtx[movieId].values
    sorted_idx = np.argsort(-movie_vec)[:num]
    return list(similarity_mtx.index[sorted_idx])

if __name__ == "__main__":
    usrId = 10
    movieId = 46976
    num = 5
    similar_movie = get_the_most_similar_movies(usrId, num)
    similar_user = get_the_most_similar_users(movieId, num)
    # print(get_the_most_similar_movies(usrId, num))
    # print(get_the_most_similar_users(movieId, num))

    movies = pd.read_csv("./ml-latest-small/movies.csv")
    df_movie = pd.DataFrame({f'依照餘弦相似度，推薦給使用者{usrId}的{num}部電影': movies[movies.movieId.isin(similar_movie)].title[:num]}).reset_index()
    df_movie.drop("index", axis=1, inplace=True)
    df_usr = pd.DataFrame({f'依照餘弦相似度，可能喜歡{movieId}的{num}位使用者': rates[rates.userId.isin(similar_user)].userId.unique()[:num]}).reset_index()
    df_usr.drop("index", axis=1, inplace=True)
    pd.concat([df_movie,df_usr],axis=1).to_csv("result.csv")