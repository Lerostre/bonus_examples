from collections import Counter
from scipy import sparse
from .models_abstract import SearchBase
from .search_utils import query_preprocess
import numpy as np


class CustomTfidf(SearchBase):
    """
    Basic class for matrix-based Tfidf
    """
    
    def __init__(self, use_idf=True):
        super().__init__()
        self.use_idf = use_idf
        
        self.__func_map = {
            "jaccard": lambda x, y: self.jaccard_distance(x, y),
            "euclidean": lambda x, y: self.euclidean_distance(x, y),
            "cosine": lambda x, y: self.cosine_distance(x, y)
        }
        
    def fit(self, corpus):
        """
        Fit collects key_to_index dictionary and computes tf and idf on a given corpus
        """
        self.build_vocab(corpus)
        self.vector_size = self.corpus_size
        
        tf, idf, tfidf, vectors = self.__get_tf_idf(corpus)
        self.tf = tf
        self.idf = idf
        self.tfidf = tfidf
        self.vectors = vectors
        
        self._corpus = None
        self._counter = None

        return self
    
    def __get_tf_idf(self, corpus):
        """
        Common function for computing tf_idf score. Uses optimized sparse matrix
        formats to maximize time and memory efficiency. Tf is essentially
        word count for text, idf is a log of inverse sum of all tf. Allows
        {'jaccard', 'cosine', 'euclidean'} distance functions
        """
        
        tf = sparse.lil_array((self.vocab_size, corpus.shape[0]), dtype=np.uint8)

        for col, text in (enumerate(corpus)):
            word_count = Counter(text)
            for word in word_count: # этим заполняем индексы в массиве
                if word in self.key_to_index:
                    tf[self.key_to_index[word], col] = word_count[word]
                    
        tf = tf.tocsr() # лил матрицы считаются очень быстро, но в вычислениях страдают
        if self.use_idf:
            idf = np.log1p(self.corpus_size / tf.sum(1))  # idf не бывает спарс
            # оказывается можно очень изящно умножить
            tfidf = tf.copy().astype(np.float32)
            tfidf.data *= idf[tf.indices]
            vectors = tfidf
        else:
            tfidf = None
            vectors = tf
            
        return tf, idf, tfidf, vectors
    
    def transform(self, corpus):
        """Computes and returns tfidf on a different (the same in our case) corpus"""
        
        _, _, _, vectors = self.__get_tf_idf(corpus)
        return vectors
    
    def fit_transform(self, corpus):
        """Fit and transform on the same corpus"""
        self.fit(corpus)
        return self.vectors
    
    def preprocess(self, query):
        """Uses common function for preprocess with sparse matrix specifics"""
        return query_preprocess(query)[np.newaxis, :]
    
    def predict(self, processed_query, simfunc="jaccard"):
        """Prediction is computed for a particular distance function, returns distances"""
        query_tf = self.transform(processed_query)
        return self._distance_func(query_tf.T, self.vectors.T, simfunc=simfunc)
    
    def get_vector(self, word):
        """Picks word index row from sparse matrix"""
        return self.vectors[[self.key_to_index[word]], :]
    
    def _distance_func(self, left, right, simfunc="jaccard"):
        """Alias for picking a function from func dict, returns result"""
        return self.__func_map[simfunc](left, right)
    
    @staticmethod
    def jaccard_distance(w1, w2):
        """Jaccard distance between two sparse vectors"""
        w1, w2 = w1.astype(bool), w2.astype(bool)
        iou = (w1.multiply(w2)).sum(axis=1)
        aou = w2.copy()
        aou.data += w1[:, w2.indices]
        aou.data = aou.data.squeeze()
        aou = aou.sum(axis=1)
        return np.nan_to_num(iou / aou)
    
    @staticmethod
    def cosine_distance(w1, w2):
        """Cosine similarity between two sparse vectors"""
        w1, w2 = w1.copy(), w2.copy()
        sim = w1.dot(w2.T) / (sparse.linalg.norm(w1) * sparse.linalg.norm(w2, axis=1))
        return sim
    
    @staticmethod
    def euclidean_distance(w1, w2):
        """Euclidean distance between two sparse vectors"""
        w1, w2 = w1.copy(), w2.copy()
        w2.data -= w1[:, w2.indices]
        w2.data = w2.data.squeeze()
        return sparse.linalg.norm(w2, axis=1)   
    
    def _distance_to_all(self, word, **dist_kwargs):
        """Distance to all vectors in vocab with sparse matrix specifics"""
        dists = super()._distance_to_all(word, **dist_kwargs)
        if isinstance(dists, np.ndarray):
            return dists
        else:
            return dists.data


class CustomBM25(CustomTfidf):
    """
    Matrix-based BM25. Only requires a few new constant values,
    such as k_1 and b. Has no transform method, self.vectors usage
    is best avoided
    """
    
    def __init__(self, k_1=1.2, b=0.75):
        super().__init__(use_idf=True)
        self.k_1 = k_1
        self.b = b
        
    def fit(self, corpus, *args, **kwargs):
        """Same as Tfidf fit with a couple of new parameters"""
        super().fit(corpus, *args, **kwargs)
        self.tfidf = None
        self.doc_lens = self.tf.sum(axis=0)
        self.avgdl = np.vectorize(len)(corpus).mean()
        
    def transform(self, corpus=None):
        raise Exception("BM25 has no transform method")
        
    def predict(self, processed_query):
        """
        BM25 prediction. Different from Tfidf as it doesn't use self.vectors.
        Compatible with search function nonetheless
        """
        bm25 = 0
        indexing = [self.key_to_index[q] for q in processed_query[0] if q in self.key_to_index]
        if len(indexing) == 0:
            # no known words => random query
            return np.random.permutation(np.arange(self.corpus_size))
        else:
            # tfidf formula without avgdl
            tf = self.tf[indexing]
            idf = self.idf[indexing][:, np.newaxis]
            up = tf * (self.k_1 + 1) * idf
            down = tf + self.k_1*(1 - self.b + self.b*self.doc_lens)
            bm25 += up/down
        
        return np.array(bm25.sum(0))