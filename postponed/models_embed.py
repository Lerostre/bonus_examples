from collections import Counter
from joblib import Parallel, delayed
from .models_abstract import SearchBase
from .search_utils import query_preprocess
from .algo_utils import sigmoid, softplus


class Word2Vec(SearchBase):
    """
    Custom w2v implementation with numpy
    ---
    corpus: training dataset, list of lists
    vector_size: size of word embeddings
    window_size: n of words from the center to include
    min_count: minimal word count to be included in vocab
    batch_size: loader batch size
    n_negative: number of negative samples
    n_epoches: number of epochs to train
    lr: learning rate for gradient descent
    num_workers: number of parallel threads for train
    schedule: allow decreasing lr depending on epoch progress
    split_counts: if True, each gradient step is multiplied by pair freq
    end_alpha: final lr value for scheduling
    """
    def __init__(self, corpus, vector_size=100, window_size=5,
                 min_count=5, batch_size=32, n_negative=5,
                 n_epoches=1, lr=0.025, num_workers=4,
                 schedule=True, split_counts=False, end_alpha=0.0001):
        super().__init__(corpus=corpus, min_count=min_count)
        
        self.window_size = window_size
        self.batch_size = batch_size
        self.vector_size = vector_size

        self.n_epoches = n_epoches
        if batch_size is None:
            self.batch_size = np.max([len(text) for text in corpus])
        self.n_negative = n_negative
        self.split_counts = split_counts
        
        self.lr = lr
        if schedule:
            self.end_alpha = lr/10 if not end_alpha else end_alpha
        self.schedule = schedule
        if type(lr) == list:
            assert len(lr) == n_epoches
            self.schedule = False
        
        self.num_workers = num_workers
        self.step_losses = np.array([])
        self.epoch_losses = np.array([])
        self.epoch_progress = 0
        self.cur_epoch = 0
        
        self.__get_dataloader()
        self.__generate_noise_dist()
        self.context_W = self.__sample_weights()
        self.center_W = self.__sample_weights()
        
    def __sample_weights(self):
        """Weight distribution as defined in the original paper"""
        uniform = np.random.uniform(-0.5, 0.5, size=(self.vocab_size, self.vector_size))
        return uniform / self.vector_size
        
    def __get_windows(self, window_size):
        """
        Extract all windows from each text, store each pair of
        (context, center) in the counter
        """
    
        dataset = Counter()
        for text in tqdm(self._corpus, desc="Creating dataloader", leave=False):
            text = [x for x in text if x in self.key_to_index]
            for word_idx, _ in enumerate(text):
                window_range = list(range(max(0, word_idx-self.window_size),
                                          min(word_idx+self.window_size+1, len(text))))
                window = [self.key_to_index[text[x]] for x in window_range if x != word_idx]
                target = self.key_to_index[text[word_idx]]

                for context in window:
                    if context != target:
                        dataset[context, target] += 1
                        
        self.pair_set = dataset.copy()
                    
        if self.split_counts:
            # counts facilitate training but are somewhat erratic
            return np.array(list(dataset)), np.array(list(dataset.values()))
        else:
            dataset = np.array([[i, target] for i, target in dataset
                                            for _ in range(dataset[i, target])])
            return (dataset, np.ones(dataset.shape[0]))
        
    def __get_dataloader(self):
        """Construct dataloader from extracted windows"""
        dataset, counts = self.__get_windows(self.window_size)
        self.loader = CustomDataloader(dataset, counts, self.batch_size)
        
    def __generate_noise_dist(self):
        """
        Noise distribution as proposed in the original paper for negative sampling.
        Makes probable words more likely to be sampled
        """
        freq_ratio = {word: count/len(self.key_to_index) for word, count in self._counter.items()}        
        freq_ratio = np.array(sorted(freq_ratio.values(), reverse = True))
        unigram_dist = freq_ratio / freq_ratio.sum() 
        self.__noise_dist = unigram_dist**0.75 / np.sum(unigram_dist**0.75)
        
    def _neg_sample(self, batch):
        """Negative sampler. Allows noise distribution to be specified, uniform otherwise"""
        neg_samples = np.random.choice(np.arange(self.vocab_size),
                                       size=(batch.shape[0], self.n_negative),
                                       p=self.__noise_dist)
        return neg_samples
    
    def __get_next_alpha(self):
        """Scheduling function for lr"""
        
        self.epoch_progress += 1 / len(self.loader.dataset)

        if self.schedule:
            if self.epoch_progress == 0 and cur_epoch == 0:
                self.init_alpha = self.lr
            end_alpha = self.end_alpha
            progress = (self.cur_epoch + self.epoch_progress / len(self.loader.dataset)) / self.n_epoches
            next_alpha = self.lr - (self.lr - self.end_alpha) * progress
            self.lr = max(self.end_alpha, next_alpha)
        
    def training_step(self, batch):
        """
        One step of gradient descent for a batch. Relies on einsums to compute
        row by row dot products. torch.bmm is a faster alternative but requires torch lib.
        The bigger the batch, the faster the train, the lower the quality.
        gensim uses one word at a time, which is faster done with one_train below
        """
        
        center, context, count = batch[:, 0], batch[:, 1], batch[:, 2]
        neg_samples = self._neg_sample(count)
        
        center_emb = self.center_W[center]
        pos_context = self.context_W[context]
        neg_context = self.context_W[neg_samples]
        
        pos_dotprod = (
            np.einsum(
                'ij,ij->i',
                pos_context,
                center_emb,
            )
        )
        neg_dotprod = (
            np.einsum(
                'ikj,ij->ik',
                neg_context,
                center_emb,
            )
        )

        grad_context_pos = (sigmoid(pos_dotprod) - 1)[:, np.newaxis] * center_emb
        self.context_W[context] -= self.lr*(grad_context_pos*count[:, np.newaxis])

        grad_context_neg = np.einsum(
            'ij,ik->ijk',
            sigmoid(neg_dotprod),
            center_emb,
        )
        self.context_W[neg_samples] -= self.lr*(grad_context_neg*count[:, np.newaxis, np.newaxis])

        grad_center_pos = (sigmoid(pos_dotprod) - 1)[:, np.newaxis] * pos_context
        grad_center_neg = np.einsum(
            'ij,ijk->ik',
            sigmoid(neg_dotprod),
            neg_context,
        )
        self.center_W[center] -= self.lr*((grad_center_pos+grad_center_neg) * count[:, np.newaxis])

        loss = softplus(pos_dotprod) + np.sum(softplus(-neg_dotprod))
        
        self.step_losses = np.append(self.step_losses, loss)
        self.epoch_progress += 1 / len(self.loader.dataset)
        
        self.__get_next_alpha()
        return loss
        
    def train(self):
        """Full train on all epochs on all batches. Parallelized with ray"""
        
        trange = tqdm(range(self.n_epoches))
        trange.set_description(f"Epoch: 0, epoch_loss: 0", refresh=True)
        
        for epoch in trange:
    
            if len(self.epoch_losses) != 0:
                epoch_loss = self.epoch_losses[-1]
                trange.set_description(f"Epoch: {epoch}, epoch_loss: {epoch_loss.round(3)}", refresh=True)

            step_range = tqdm(self.loader, position=0,
                              leave=False, total=len(self.loader.dataset))

            if type(self.lr) == list:
                lr = self.lr[epoch]
            else:
                lr = self.lr

            with joblib.parallel_backend('ray', ray_remote_args=dict(num_cpus=4)):
                for batch in step_range:
                    self.training_step(batch)

            self.epoch_losses = np.append(self.epoch_losses, np.mean(self.step_losses))
            self.vectors = self.context_W
            self.cur_epoch += 1
        
    def one_train(self):
        """
        batch_size=1 specific train function. Much faster due to usage of direct
        dot products instead if einsum. Not parallelized yet
        """

        for batch in tqdm(self.loader, total=len(self.loader.dataset)):
            center, context, count = batch[:, 0], batch[:, 1], batch[:, 2]

            neg_samples = self._neg_sample(center)
            v_center = self.center_W[center].squeeze(0)
            v_context = self.context_W[context].squeeze(0)
            v_negative = self.context_W[neg_samples].squeeze(0)

            pos_dotprod = v_center @ v_context
            neg_dotprod = v_center @ v_negative.T

            pos_sigmoid = sigmoid(pos_dotprod)
            neg_sigmoid = sigmoid(neg_dotprod)

            grad_center = (pos_sigmoid - 1)*v_context + (neg_sigmoid[:, np.newaxis]*v_negative).sum(0)
            self.center_W[center] -= self.lr*grad_center

            grad_pos_context = (pos_sigmoid - 1)*v_center
            self.context_W[context] -= self.lr*grad_pos_context

            grad_neg_context = neg_sigmoid[:, np.newaxis]*v_center   
            self.context_W[neg_samples] -= self.lr*grad_neg_context

            loss = np.log1p(pos_sigmoid) + np.log1p(neg_sigmoid).sum()
            
            self.__get_next_alpha()
            
        self.vectors = self.context_W
        
    def get_vector(self, word):
        """
        Under the hood implementation of __getitem__ with all the
        necessary checks and exceptions
        """
        if hasattr(self, "vectors"):
            try:
                return self.vectors[self.key_to_index[word]]
            except:
                return NotImplementedError(f"Model vocab has no word {word}")
            
#     def __getitem__(self, word):
#         return self.get_vector(word)
        
    def __handle_no_vector_in_w2v(self, word):
        """Return zero vector for non-existant words else word vector"""
        if word in self.key_to_index:
            return self[word]
        else:
            return np.zeros_like(self["который"])
        
    def _distance_func(self, left, right):
        """w2v distance function is cosine distance"""
        return self.cosine_distance(left, right)
            
    def predict(self, query):
        """Prediction requires preprocessed corpus sentences"""
        return self._distance_to_all(self.preprocess(query_emb), self.sentences)[0]
    
    def preprocess(self, query):
        """Query preprocessing with handling <unk>"""
        return np.mean([self.__handle_no_vector_in_w2v(t) for t in query_preprocess(query)], 0)
    
    def save(self, filename):
        """Model saving with only the most relevant params"""
        
        self.context_W = None
        self.center_W = None
        self._counter = None
        self.__noise_dist = None
        self.loader = None
        
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            
    def __vectorize_corpus(self):
        self._corpus = self.corpus.apply(lambda x: self.preprocess(" ".join(x)))


class RandomSearch(Word2Vec):
    """Class with random vector embeddings. Here to prove a point"""
    
    def __init__(self, corpus, vector_size=100):
        super().__init__(corpus)
        self.vector_size = vector_size
        self.vectors = np.random.uniform(-1, 1, size=(self.vocab_size, self.vector_size))
        self.min_count = 1


class CustomDataloader:
    """
    Custom dataloader to work with numpy arrays instead of torch tensors
    ---
    dataset:
        prepared array of (context, center) pairs
    counts:
        their respective counts
    batch_size:
        number of pairs in batch
    shuffle:
        shuffle dataset or not
    ---
    returns iterable loader with batch of (centext, center, count)
    """
    
    def __init__(self, dataset, counts, batch_size=2048, shuffle=True):
    
        dataset = np.hstack([dataset, counts[:, np.newaxis]]).astype(np.uint8)
        self.dataset = np.array(np.split(dataset, np.arange(batch_size, len(dataset), batch_size)))
        self.shuffle = shuffle

    def __iter__(self):
        # true shuffle is expensive, index permutation is preferable
        if self.shuffle:
            perm = self.dataset[np.random.permutation(np.arange(self.dataset.shape[0]))]
        else:
            perm = self.dataset
        return iter(list(perm))