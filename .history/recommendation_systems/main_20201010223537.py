import numpy as np
from collections import Counter, defaultdict


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
