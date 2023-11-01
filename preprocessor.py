import re
import nltk
import inspect
import swifter

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from string import punctuation
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize, TreebankWordTokenizer


def auto_assign(func):
    signature = inspect.signature(func)

    def wrapper(*args, **kwargs):
        instance = args[0]
        bind = signature.bind(*args, **kwargs)
        for param in signature.parameters.values():
            if param.name != 'self':
                if param.name in bind.arguments:
                    setattr(instance, param.name, bind.arguments[param.name])
                if param.name not in bind.arguments and param.default is not param.empty:
                    setattr(instance, param.name, param.default)
        return func(*args, **kwargs)

    wrapper.__signature__ = signature

    return wrapper
    

tqdm.pandas()
class LanguagePipeline(BaseEstimator, TransformerMixin):
    """
    Handmade class for performing all the usual (and not as much) preprocessing
    for English, including lowercasing, lemmatization and whatnot, all the
    parameters should be self-explanatory. Has the same functionality as sklearn transformers
    ---
    columns: list of columns or indices to transform
    lang: text language, affects lemmatizing and stemming
    lower: might be redundant for capsed messages, like tweets
    lemmatize: does not always improve performance
    stemming: same problems as above
    remove_stopwords: pronouns or negation, might prove useful e.g for sentiment
    remove_numbers: numbers are rarely useful overall
    remove_punctuation: marks like exclamations can help in sentiment analysis
    fix_spelling: mistakes might hinder training
    normalize: same motivation as above
    regex_to_remove: optional regex to remove from text
    tokenizer: has to be a class with predefined tokenize method, nltk treebank by default
    lemmatizer: option for lemmatization, WordNet or Mystem are used if None
    n_jobs: number of parallel threads (will slow down if the dataset is not big enough)
    verbose: level of verbosity, 0 or 1
    ---
    returns transformed dataset
    """
    
    @auto_assign
    def __init__(self,
                 lang: str = "english",
                 lower: bool = False,
                 lemmatize: bool = False,
                 stemming: bool = False,
                 remove_stopwords: bool = False,
                 remove_contractions: bool = False,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = False,
                 fix_spelling: bool = False,
                 normalize: bool = False,
                 regex_to_remove: str = "",
                 tokenizer: object = TreebankWordTokenizer(),
                 lemmatizer: object = None,
                 n_workers=4,
                 verbose: int = 0
                ):
        
        if self.stemming:
            # lemmatization is pointless if stemming
            if self.lemmatize:
                self.lemmatize = False
        # most of the transforms can be written in one line, but some require additional imports
        # function order is crucial
        self._lambda_dict = {
            # general transforms
            "lower": lambda x: x.lower(),
            "normalize": self._normalize,
            "remove_contractions": self._remove_contractions,
            "fix_spelling": self._fix_spelling,
            # regex transforms, for some reason, faster, if done separatelly
            "remove_numbers": lambda x: self.regex_remove(x, r"\d+"),
            "remove_punctuation": lambda x: self.regex_remove(x, r"[^\w\s]"),
            "regex_to_remove": self.regex_remove,
            # list transforms
            "tokenizer": self.tokenizer.tokenize,
            "lemmatize": self._lemmatize,
            "stemming": self._stemming,
            "remove_stopwords": self._remove_stopwords
        }
        
    # functions with imports
    def _normalize(self, x):
        from unidecode import unidecode
        return unidecode(x, errors="preserve")
    
    def _remove_contractions(self, x):
        import contractions
        return contractions(x)
    
    def _fix_spelling(self, x):
        from textblob import TextBlob
        return TextBlob(x).correct()
    
    def _lemmatize(self, x):
        # lemmatization is intentionally by one word, too long otherwise
        if self.lang == "english":
            if not self.lemmatizer:
                nltk.download("wordnet", quiet=True)
#                     todo: add spacy functionality
#                     self.nlp = spacy.load("en_core_web_sm")
                self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
            lemmatize = lambda x: self.lemmatizer.lemmatize(x)
        elif self.lang == "russian":
            from pymystem3 import Mystem
            if not self.lemmatizer:
                self.lemmatizer = Mystem()
            lemmatize = lambda x: self.lemmatizer.lemmatize(x)[0]
        return [lemmatize(y) for y in x]
    
    def _stemming(self, x):
        self.stemmer = SnowballStemmer(self.lang)
        return [self.stemmer.stem(y) for y in x]
    
    def _remove_stopwords(self, x):
        from nltk.corpus import stopwords
        nltk.download("stopwords", quiet=True)
        self.stopwords = stopwords.words(self.lang)
        return [word for word in x if word not in self.stopwords]
    
    # operations must always contain tokenize, but it must not always come last
    def __tokenization_insert(self, operations):
        temp, n = [], len(operations)
        for i in range(n):
            if operations[n-i-1] in ["lemmatize", "stemming", "remove_stopwords"]:
                # find tokenizer order
                temp.append(operations.pop())
            elif operations[n-i-1] == "tokenizer":
                # stop if tokenizer in list
                return operations+temp
        operations.append("tokenizer")
        return operations + temp
        
    def regex_remove(self, string, regex=None):
        """Regex transform is unique, as it can be defined differently each time"""
        if not regex:
            regex = self.regex_to_remove
        # empty string is bad <= bad,string --re.sub("")-> badstring, not bad string
        return re.sub(regex, ' ', string)
            
    def fit(self, X, y=None, columns=[]):
        """
        There is no need to fit or train, fit is used for assigning output data type
        and raising exceptions. Specified columns are there for inplace transform
        """
        if type(X) == pd.Series:
            X = pd.DataFrame(X)
        self.X_type = type(X)
        # todo: fix types mess
        if len(columns) == 0:
            if self.X_type == np.ndarray:
                self.columns = np.arange(X.shape[1])
            elif self.X_type == pd.DataFrame:
                self.columns = X.columns
            else:
                raise Exception("X is not a pd.DataFrame or np.array")
        else:
            self.columns = columns
        return self
    
    def simple_transform(self, string, transform):
        """
        Transform from the lambda function dictionary. Might be used separately with any given
        option from __init__. Elementary unit of a general transform. Alias of _lambda_dict for public use
        """
        return self._lambda_dict[transform](string)
            
    def general_transform(self, string, transforms=[]):
        """
        Applies all transforms chosen during the transformer initialization
        or passed as `transforms` argument to a given text. Always returns tokenized output
        """
        operations = list(self._lambda_dict.keys()) if transforms == [] else transforms
        # tokenization is always a necessary step
        operations = self.__tokenization_insert(operations)
        # no transforms means applying transformer parameters
        for operation in operations:
            if transforms == []:
                string = self.simple_transform(string, operation) if self.__dict__[operation] else string
            else:
                string = self.simple_transform(string, operation)
            
        # remove trailing spaces
        string = re.sub('\s+', ' ', " ".join(string))
    
        return string
    
    def transform(self, X, y=None):
        """Generic transform function for transforming datasets"""
        # vectorize: 243ms, map: 133ms, lc: 129ms
        # apply seems to be the cleanest option overall, as transform can be applied to different columns
        # todo: add verbose, merge with general_transform
        if type(X) == pd.Series:
            X = pd.DataFrame(X)
        if self.X_type == pd.DataFrame:
            X[self.columns] = X[self.columns].swifter.set_npartitions(self.n_workers).applymap(self.general_transform)
        elif self.X_type == np.ndarray:
            # todo: добавить? распределение для нампая
            vfunc = np.vectorize(self.general_transform)
            X[:, self.columns] = vfunc(X[:, self.columns])
        return X
    
    # parallel execution is also available but does not always improve performance
    def parallel_transform(self, X, y=None, n_jobs=4):
        transform_splits = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.transform)(X_split)
            for X_split in np.array_split(X, n_jobs))
        return np.vstack(transform_splits)