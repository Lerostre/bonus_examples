from collections import Counter
from abc import ABC, abstractmethod


class SearchBase(ABC):
    """
    Abstract class with the most favored functions implemented
    in gensim.KeyedVectors. Used for Tfidf and W2V subclasses
    ---
    corpus:
        list of tokenized texts
    min_count:
        minimal word count for it to remain in cleaned text
    """
    
    def __init__(self, corpus=None, min_count=1):
        
        self.min_count = min_count
        try:
            self.build_vocab(corpus)
        except:
            pass
        
    def build_vocab(self, corpus=None):
        """
        Basic method for updating all vocab stats and parameters,
        such as _counter, _corpus, key_to_index, etc. Can be used
        explicitly or during class initialization
        """
        # extracting words with counter >= min_count first
        try:
            counter = Counter()
            for text in corpus:
                counter.update(text)
        except:
            raise ValueError("No corpus provided")
        counter = counter.most_common()
        self._counter = {word: count for word, count in counter if count >= self.min_count}
        # corpus is used later for dataloader, but might be too heavy, drop if necessary
        self._corpus = corpus

        # building mapping dicts
        self.vocab_size = len(self._counter)
        self.key_to_index = {x: y for x, y in zip(self._counter,
                                                  range(self.vocab_size))}
        self.index_to_key = {y: x for x, y in self.key_to_index.items()}
        self.corpus_size = corpus.shape[0]
    
    def __getitem__(self, word):
        """Qol method for easy vector mainpulating"""
        return self.get_vector(word)
    
    def get_vector(self, word):
        """
        Under the hood implementation of __getitem__ with all the
        necessary checks and exceptions
        """
        if hasattr(self, "vectors"):
            try:
                return self.vectors[self.key_to_index[word], :]
            except:
                return NotImplementedError(f"Model vocab has no word {word}")
        else:
            return NotImplementedError("Abstract class has no vectors")
        
    def most_similar(self, word, topn=10, **dist_kwargs):
        """
        Method for retrieving topn most similar tokens for a given word.
        Calculates all distances, sorts them in descending order,
        return the result as np.ndarray
        ---
        word:
            token to compare with all the trained vocab
        topn:
            number of most common words to return
        dist_kwargs:
            other distance related parameters, such as similarity function
            for CustomTfidf
        ---
        return np.ndarray of size (2, topn) : [..., [token_n, similarity_n]]
        """
        distances = self._distance_to_all(word, **dist_kwargs)
        # the first one is the word itself more often than not
        most_similar = np.argsort(-distances)[1:topn+1]
        similar_pairs = zip(np.array(list(self.key_to_index))[most_similar], distances[most_similar])
        return np.array(list(similar_pairs))
    
    def distance(self, word1, word2, **dist_kwargs):
        """
        Common method for measuring distance between model vectors,
        given a predefined _distance_func
        """
        return self._distance_func(self[word1], self[word2], **dist_kwargs)
    
    def _distance_to_all(self, word, **dist_kwargs):
        """
        Common method for measuring distance between a token and all the
        vectors in the model. Scaled to [0, 1] to facilitate comparison
        """
        distances = self._distance_func(self[word], self.vectors, **dist_kwargs)
        distances -= distances.min()
        distances /= (distances.max() - distances.min())
        return distances
    
    @staticmethod
    def cosine_distance(w1, w2):
        """
        Static method for computing cosine similarity between
        two vectors w1 and w2
        """
        w1, w2 = w1.copy(), w2.copy()
        sim = w1.dot(w2.T) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        return sim
    
    @abstractmethod
    def _distance_func(self):
        """Model-specific distance function for vectors"""
        raise NotImplementedError("No distance measuring method defined")
        
    @abstractmethod
    def preprocess(self, query):
        """Model-specific preprocess function for query"""
        raise NotImplementedError("No preprocessing method defined")
        
    @abstractmethod
    def predict(self, query):
        """Model-specific prediction function for query"""
        raise NotImplementedError("No prediction method defined")