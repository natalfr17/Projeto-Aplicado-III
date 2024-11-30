import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Carregar o arquivo JSON usando pandas
# Ler as primeiras 1000 linhas
chunks = pd.read_json('goodreads_reviews_dedup.json', lines=True, chunksize = 1000)
for c in chunks:
    print(c.info())
    break

n_ratings = len(c)
n_books = len(c['book_id'].unique())
n_users = len(c['user_id'].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique bookId's: {n_books}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per book: {round(n_ratings/n_books, 2)}")

user_freq = c[['user_id', 'book_id']].groupby(
    'user_id').count().reset_index()
user_freq.columns = ['user_id', 'n_ratings']
print(user_freq.head())

# Encontrar os livros com as menores e maiores avaliações:
mean_rating = c.groupby('book_id')[['rating']].mean()
# Livros com as menores avaliações
lowest_rated = mean_rating['rating'].idxmin()
c.loc[c['book_id'] == lowest_rated]
# Livros com as maiores avaliações
highest_rated = mean_rating['rating'].idxmax()
c.loc[c['book_id'] == highest_rated]
# Mostrar o número de pessoas que avaliaram o livro com maior nota
c[c['book_id'] == highest_rated]
# Mostrar o número de pessoas que avaliaram o livro com menor nota
c[c['book_id'] == lowest_rated]

print(f"Mean rating: {mean_rating}")
print(f"Lowest rated: {lowest_rated}")
print(f"Highestrated: {highest_rated}")

# Agora, criamos a matriz usuário-item usando a matriz esparsa CSR do SciPy
from scipy.sparse import csr_matrix

def create_matrix(df):
    
    N = len(df['user_id'].unique())
    M = len(df['book_id'].unique())
    
    # Mapear IDs para índices
    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    book_mapper = dict(zip(np.unique(df["book_id"]), list(range(M))))
    
    # Mapear índices para IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
    book_inv_mapper = dict(zip(list(range(M)), np.unique(df["book_id"])))
    
    user_index = [user_mapper[i] for i in df['user_id']]
    book_index = [book_mapper[i] for i in df['book_id']]

    X = csr_matrix((df["rating"], (book_index, user_index)), shape=(M, N))
    
    return X, user_mapper, book_mapper, user_inv_mapper, book_inv_mapper
    
X, user_mapper, book_mapper, user_inv_mapper, book_inv_mapper = create_matrix(c)

print(X)

"""
Encontrar livros similares usando KNN
"""
from sklearn.neighbors import NearestNeighbors
def find_similar_books(book_id, X, k, metric='cosine', show_distance=False):
    
    neighbour_ids = []
    
    book_ind = book_mapper[book_id]
    book_vec = X[book_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    book_vec = book_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(book_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(book_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

book_id = 2

similar_ids = find_similar_books(book_id, X, k=10)

for i in similar_ids:
    print(i)


