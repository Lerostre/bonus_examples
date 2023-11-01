from collections import defaultdict
from IPython.display import clear_output
from sklearn.base import check_X_y, clone
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm
import numpy.ma as ma
import numpy as np
import pandas as pd

class BackwardElimination:
    """
    Custom backward elimination class, permits any possible scorer
    on any possbile dataset, with any possible model, but does not
    allow multiprocessing, yet
    ---
    estimator:
        prefit (subject to change) estimator to tune
    k_features:
         desired number of features left
    verbose:
         verbosity of an elimination process
    """

    def __init__(self, estimator, k_features, verbose=0):

        self.estimator = estimator.copy()
        self.k_features = k_features
        self.log = []
        self.fit_params = defaultdict(dict)
        self.verbose = bool(verbose)

    def fit(self, X, y, test_set=None, eval_set=None, **fit_params):
        """
        Storing the dataset in an object-ignorant format and saving
        current dataset features
        ---
        X, y:
            data and target, must have the same features to be compatible
            with the pretrained estimator, used for tuning if no eval_set is
            specified.
        eval_set:
            validation set for tuning, also used in model fit
        test_set:
            test set for tracking test metrics, not used in tuning
        **kwargs:
            fit parameters for a particular estimator
        """
        if type(X) != pd.DataFrame:
            self.X, self.y = check_X_y(X, y, dtype=None, accept_sparse=True, force_all_finite=False)
            self.features = np.arange(X.shape[1])
        else:
            self.X, self.y = X, y
            self.features = X.columns.values
        self.features_left = self.features.copy()
        self.test_set = test_set
        self.eval_set = eval_set
        self.fit_params.update(fit_params)

    def _eliminate_feature(self):
        X = self.X.loc[:, self.features_left]
        if self.test_set:
            test = self.test_set[0].loc[:, self.features_left]
        if self.eval_set:
            valid = self.eval_set[0].loc[:, self.features_left]
        self.estimator = clone(self.estimator)
        self.estimator.set_params(verbose=0)
        return X, valid, test


    def transform(self, **fp_params):
        """
        Elimination process itself
        ---
        **kwargs:
            feature_importance parameters for a particular estimator
        """

        trange = range(len(self.features) - self.k_features + 1)
        if self.verbose:
            trange = tqdm(trange, desc="Calculating initial score... ")
        X, valid, test = self.X, self.eval_set[0], self.test_set[0]

        for iteration in trange:

            # модель всегда предобучена, или нет?
            model = self.estimator
            # print(self.estimator.get_params())
            if not model.is_fitted():
                print(X.shape)
                # print(X.columns)
                model.fit(X, self.y, eval_set=(valid, self.eval_set[1]), **self.fit_params)

            # трейн есть всегда, валидация, если нет, берётся с трейна, тест только для теста, лол
            scores = defaultdict(dict)
            for score, score_f in [("gini", gini), ("auc-pr", average_precision_score)]:
                scores[score]["train"] = score_f(self.y, model.predict_proba(X)[:, 1])
                if self.eval_set:
                    scores[score]["val"] = score_f(self.eval_set[1],
                                                   model.predict_proba(valid)[:, 1])
                else:
                    scores[score]["val"] = scores[score]["train"]
                if self.test_set:
                    scores[score]["test"] = score_f(self.test_set[1],
                                                    model.predict_proba(test)[:, 1])

            # суммарный скор высчитывается, как среднее между всеми сплитами
            # нужен только для verbose, в тюнинге не участвует
            total_scores = [scores[data] for data in ["train", "val", "test"]]
            summary_score = np.mean(total_scores).round(3)
            if iteration == 0:
                self.initial_score = summary_score
                to_eliminate = None
                # print(f"Initial score: {total_scores}")
            else:
                if self.verbose:
                    score_change = np.round(self.initial_score - summary_score, 3)
                    trange.set_description(f"Eliminating... Score change: {score_change}",
                                           refresh=True)
                # print(f"Eliminated feature '{to_eliminate}'")
                # print(f"Scores: {total_scores}")
            for score_f in scores:
                for subset in scores[score_f]:
                    self.log.append({
                        "subset": subset,
                        "score_f": score_f
                        "score": scores[score_f][subset],
                        "eliminated": to_eliminate,
                        "features_left": self.features_left.size
                    })

            cat_features = np.array(self.estimator.get_params()["cat_features"])
            # f_importance = np.zeros_like(self.features)
            f_importance = model.get_feature_importance(data=Pool(valid,
                                                                  self.eval_set[1],
                                                                  cat_features),
                                                        type="LossFunctionChange")
            # f_importance[np.setdiff1d(np.arange(self.features.size), self.features_left)] = np.nan
            to_eliminate = np.argmin(f_importance)
            features = np.array(model.feature_names_)
            to_eliminate = features[to_eliminate] # для лога
            # to_eliminate = np.argmin(ma.masked_where(f_importance==0, f_importance))
            # print(np.argwhere(features==to_eliminate))
            self.features_left = np.delete(features,
                                           np.argwhere(features==to_eliminate))

            # to_eliminate = self.features[to_eliminate] # для лога
            
            cat_del_mask = np.argwhere(cat_features==to_eliminate)
            # print(cat_features)
            # print(to_eliminate)
            # print(cat_features == to_eliminate))
            
            X, valid, test = self._eliminate_feature()
            if len(cat_del_mask) > 0:
                self.estimator.set_params(cat_features=list(np.delete(cat_features, cat_del_mask)))
            
        self.estimator = model


class CatBoostBE(BackwardElimination):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

    def _eliminate_feature(self):
        self.estimator = clone(self.estimator)
        self.estimator.set_params(verbose=0,
                                  ignored_features=np.setdiff1d(np.arange(self.features.size),
                                                                self.features_left))
        
def gini(y_true, y_pred, sample_weight=None):
    roc_auc = roc_auc_score(y_true, y_pred, sample_weight=sample_weight) 
    return 2 * roc_auc - 1