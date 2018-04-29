import pandas as pd
import numpy as np

import re

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, strip_tags
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# load data
df = pd.read_csv("data/train.csv", encoding="utf-8").head(10000)

# train/test split
yvar = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
xdata = df.comment_text
ydata = df[yvar]
xdata_train, xdata_eval, ydata_train, ydata_eval = train_test_split(xdata, ydata, test_size=0.2, random_state=1)


# return words from corpus
# TODO: also try r"([\w][\w']*\w)"
def tokenize(doc, token=r"(?u)\b\w\w+\b"):
    doc = strip_tags(doc.lower())
    doc = re.compile(r"\s\s+").sub(" ", doc)
    words = re.compile(token).findall(doc)
    return words


# remove stop words
def remove_stop_words(x, stop_words=ENGLISH_STOP_WORDS):
    return [i for i in x if i not in stop_words]


# wrapper for gensim Phraser
COMMON_TERMS = ["of", "with", "without", "and", "or", "the", "a"]
class PhraseTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, common_terms=COMMON_TERMS):
        self.phraser = None
        self.common_terms = common_terms

    def fit(self, X, y=None):
        phrases = Phrases(X, common_terms=self.common_terms)
        self.phraser = Phraser(phrases)
        return self

    def transform(self, X):
        return X.apply(lambda x: self.phraser[x])


# for making tagged documents
# NOTE: can't use FunctionTransformer since TransformerMixin doesn't pass y to fit_transform anymore
class MakeTaggedDocuments(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is not None:
            yvar = list(y.columns)
            tags = y.apply(lambda row: [i for i,j in zip(yvar, row) if j == 1], axis=1)
            return [TaggedDocument(words=w, tags=t) for w,t in zip(X, tags)]
        else:
            return [TaggedDocument(words=w, tags=[]) for w in X]

    def fit_transform(self, X, y):
        return self.transform(X, y)


# wrapper for gensim Doc2Vec
class D2VEstimator(BaseEstimator):

    def __init__(self, min_count=10, alpha=0.025, min_alpha=0.0001, vector_size=200, dm=0, epochs=20):
        self.min_count = min_count
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.vector_size = vector_size
        self.dm = dm
        self.epochs = epochs
        self.yvar = None
        self.model = Doc2Vec(seed=1, hs=1, negative=0, dbow_words=0,
                             min_count=self.min_count, alpha=self.alpha, min_alpha=self.min_alpha,
                             vector_size=self.vector_size, dm=self.dm, epochs=self.epochs)

    def get_tags(self, doc):
        vec = self.model.infer_vector(doc.words, self.model.alpha, self.model.min_alpha, self.model.epochs)
        return dict(self.model.docvecs.most_similar([vec]))

    def fit(self, X, y=None):
        self.model.build_vocab(X)
        self.model.train(X, epochs=self.model.epochs, total_examples=self.model.corpus_count)
        self.model.delete_temporary_training_data()
        self.yvar = list(y.columns)
        return self

    def predict_proba(self, X):
        pred = [self.get_tags(d) for d in X]
        pred = pd.DataFrame.from_records(data=pred)
        return pred[self.yvar]


# blend predictions from multiple models
class Blender(FeatureUnion):

    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        self.transformer_list = transformer_list
        self.scaler_list = [(t, StandardScaler()) for t, _ in transformer_list]
        self.n_jobs = n_jobs
        default_transformer_weights = [1.0 / len(transformer_list) for i, j in transformer_list]
        self.transformer_weights = transformer_weights if transformer_weights else default_transformer_weights

    @property
    def transformer_weights(self):
        return self._transformer_weights

    @transformer_weights.setter
    def transformer_weights(self, values):
        self._transformer_weights = {t[0]: v for t, v in zip(self.transformer_list, values)}

    # don't need to check for fit and transform
    def _validate_transformers(self):
        pass

    # iterator with scalers
    def _iter_ss(self):
        get_weight = (self.transformer_weights or {}).get
        return [(t[0], t[1], s[1], get_weight(t[0])) for t, s in zip(self.transformer_list, self.scaler_list)]

    # also fit scalers
    def fit(self, X, y):
        super(Blender, self).fit(X, y)
        self.scaler_list = [(name, ss.fit(trans.predict_proba(X))) for name, trans, ss, _ in self._iter_ss()]
        return self

    # generate probabilities
    def predict_proba(self, X):
        Xs = [ss.transform(trans.predict_proba(X)) * weight for name, trans, ss, weight in self._iter_ss()]
        return np.sum(Xs, axis=0)


# create pipeline
d2v_pipeline = Pipeline(steps=[
    ('tk', FunctionTransformer(func=lambda x: x.apply(tokenize), validate=False)),
    ('ph', PhraseTransformer()),
    ('sw', FunctionTransformer(func=lambda x: x.apply(remove_stop_words), validate=False)),
    ('doc', MakeTaggedDocuments()),
    ('d2v', D2VEstimator())
])

lr_pipeline = Pipeline(steps=[
    ('cv', CountVectorizer(min_df=5, max_features=50000, lowercase=False, strip_accents='unicode',
                           stop_words='english', analyzer='word')),
    ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
    ('lr', OneVsRestClassifier(LogisticRegression(class_weight="balanced", C=0.1, penalty="l2")))
])

pipeline = Blender(transformer_list=[('d2v', d2v_pipeline), ('lr', lr_pipeline)],
                   transformer_weights=[0.3, 0.7])

# train model
pipeline.fit(xdata_train, ydata_train)

# apply to eval set
ydata_eval_pred = pipeline.predict_proba(xdata_eval)

# calculate auc
auc = [roc_auc_score(ydata_eval[y], ydata_eval_pred[:,i]) for i,y in enumerate(yvar)]
print('Model AUCs: %s' % auc)
print('Avg AUC: %s' % np.mean(auc))
