import re
import numpy as np
from typing import NamedTuple
from collections import defaultdict


def tokenize(text):
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

class Message(NamedTuple):
    text: str
    is_spam: bool
    
class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.tokens = set()
        self.token_spam_counts = defaultdict(int)
        self.token_ham_counts = defaultdict(int)
        self.spam_messages = self.ham_messages = 0
    
    def train(self, messages):
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
                
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1
    
    def _probabilities(self, token):
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
        
        return p_token_spam, p_token_ham
    
    def predict(self, text):
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            
            if token in text_tokens:
                log_prob_if_spam += np.log(prob_if_spam)
                log_prob_if_ham += np.log(prob_if_ham)
            else:
                log_prob_if_spam += np.log(1.0 - prob_if_spam)
                log_prob_if_ham += np.log(1.0 - prob_if_ham)
                
        prob_if_spam = np.exp(log_prob_if_spam)
        prob_if_ham = np.exp(log_prob_if_ham)
        
        return prob_if_spam / (prob_if_spam + prob_if_ham)

# Testing
messages = [Message('spam rules', is_spam=True),
            Message('ham rules', is_spam=False),
            Message('hello ham', is_spam=False)]

model = NaiveBayesClassifier()
model.train(messages)

text = 'hello spam'
print(model.predict(text))
                