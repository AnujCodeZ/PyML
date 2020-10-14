import re
import csv
import tqdm
import numpy as np
from typing import NamedTuple
from collections import Counter, defaultdict

from PCA.main import PCA


# Simply returns what's popular
# users_interest must be a list of lists of users interests
popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests)

def most_popular_new_interests(user_interests, max_results):
    suggestions = [(interest, frequency)
                   for interest, frequency in popular_interests.most_common()
                   if interest not in user_interests]
    return suggestions

# User-Based Collaborative Filtering
unique_interests = sorted({interest
                           for user_interests in users_interests
                           for interest in user_interests})

def make_user_interest_vector(user_interests):
    return [1 if interest in user_interests else 0
            for user_interests in users_interests]

user_interest_vectors = np.array([make_user_interest_vector(user_interests)
                         for user_interests in users_interests])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_i in user_interest_vectors]
                     for interest_vector_j in user_interest_vectors]

def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity)
             for other_user_id, similarity in enumerate(user_similarities[user_id])
             if user_id != other_user_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)

def user_based_suggestions(user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
    
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)
    
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_vectors]
                        for j, _ in enumerate(unique_interests)]

interest_similarities = [[cosine_similarity(user_vetor_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)

def item_based_suggestions(user_id, include_current_interests):
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_vectors[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity
    
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)
    
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

# Matrix Factorization

MOVIES = 'ml-100k/u.item'
RATINGS = 'ml-100k/u.data'

class Rating(NamedTuple):
    user_id: str
    movie_id: str
    rating: float

with open(MOVIES, encoding='iso-8859-1') as f:
    reader = csv.reader(f, delimiter='|')
    movies = {movie_id: title for movie_id, title, *_ in reader}

with open(RATINGS, encoding='iso-8859-1') as f:
    reader = csv.reader(f, delimeter='\t')
    ratings = [Rating(user_id, movie_id, float(rating))
               for user_id, movie_id, rating, _ in reader]

np.random.seed(0)
np.random.shuffle(ratings)

split1 = int(len(ratings) * 0.7)
split2 = int(len(ratings) * 0.85)

train = ratings[:split1]
validation = ratings[split1:split2]
test = ratings[split2:]

avg_rating = sum(rating.rating for rating in train) / len(train)
baseline_error = sum((rating.rating - avg_rating) ** 2
                     for rating in test) / len(test)

EMBEDDING_DIM = 2

user_ids = {rating.user_id for rating in ratings}
movie_ids = {rating.movie_id for rating in ratings}

user_vectors = {user_id: np.random.randn(EMBEDDING_DIM)
                for user_id in user_ids}
movie_vectors = {movie_id: np.random.randn(EMBEDDING_DIM)
                for movie_id in movie_ids}

def loop(dataset, learning_rate=None):
    with tqdm.tqdm(dataset) as t:
        loss = 0.0
        for i, rating in enumerate(t):
            movie_vector = movie_vectors[rating.movie_id]
            user_vector = user_vectors[rating.user_id]
            predicted = np.dot(user_vector, movie_vector)
            error = predicted - rating.rating
            loss += error ** 2
            
            if learning_rate is not None:
                user_gradient = [error * u_j for u_j in user_vector]
                movie_gradient = [error * m_j for m_j in movie_vector]
                
                for j in range(EMBEDDING_DIM):
                    user_vector[j] -= learning_rate * user_gradient[j]
                    movie_vector[j] -= learning_rate * movie_gradient[j]
            
            t.set_description(f'avg loss: {loss / (i + 1)}')

learning_rate = 0.05
for epoch in range(20):
    learning_rate *= 0.9
    print(epoch, learning_rate)
    loop(train, learning_rate=learning_rate)
    loop(validation, learning_rate=learning_rate)
loop(test)

original_vectors = [vector for vector in movie_vectors.values()]
pca = PCA(EMBEDDING_DIM)
components = pca.fit(original_vectors)

ratings_by_movie = defaultdict(list)
for rating in ratings:
    ratings_by_movie[rating.movie_id].append(rating.rating)

vectors = [
    (movie_id,
     sum(ratings_by_movie[movie_id]) / len(ratings_by_movie[movie_id]),
     movies[movie_id],
     vector)
    for movie_id, vector in zip(movie_vectors.keys(), 
                                pca.transform(original_vectors))
]