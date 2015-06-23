# query-based document and keyword recommendation
from __future__ import division
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from collections import Counter
import time
from gensim import corpora, models, similarities
from random import sample
import os.path

class QueryRecommender:
    def __init__(self, num_rec_docs=10, num_rec_keys=10, topic_num=100, num_sample_docs=1000):
        """For initialization"""
        # number of documents to recommend to user each iteration
        self.num_rec_docs = num_rec_docs
        # number of keywords to recommend to user each iteration
        self.num_rec_keys = num_rec_keys
        # dimension of LSI model
        self.topic_num = topic_num
        # set of stopwords for preprocessing
        self.english_stopwords = set(stopwords.words('english'))
        # number of the sample docs, which is a subset of all the documents
        self.num_sample_docs = num_sample_docs

    def read_data(self, t_a_k):
        """Read data of documents and keywords"""
        # t_a_k is a list of all the documents
        # each document represented as a tuple (title, abstract, list of keywords)
        self.t_a_k = t_a_k
        # combine title and abstract for later use
        self.texts = [p[0]+'. '+p[1] for p in t_a_k]
        # record keywords
        self.keywords = [p[2] for p in t_a_k]

    def preprocess(self):
        """Preprocessing for texts"""
        # if preprocessed data already saved in folder tmp, just load it
        print 'Loading preprocessed data from folder /tmp...'
        if os.path.isfile('./tmp/texts_filtered.json'):
            with open('./tmp/texts_filtered.json', 'r') as f:
                self.texts_filtered = json.loads(f.read())
        else:
            print 'No preprocessed data found, creating a new one...'
            # remove hyphen
            self.texts = [text.replace('-', ' ') for text in self.texts]
            # word tokenize
            self.texts_filtered = [nltk.word_tokenize(text) for text in self.texts]
            # remove non-alphabet and lower them
            self.texts_filtered = [[w.lower() for w in text if w.isalpha()] for text in self.texts_filtered]
            # remove stopwords
            self.texts_filtered = [[w for w in text if not w in self.english_stopwords] for text in self.texts_filtered]
            low_fq_word_set = self._low_fq_set()
            # remove words with low frequency  
            self.texts_filtered = [[w for w in text if not w in low_fq_word_set] for text in self.texts_filtered]
            # save the preprocessed data
            with open('./tmp/texts_filtered.json', 'w') as f:
                f.write(json.dumps(self.texts_filtered))

    def get_query(self):
        self.query = raw_input('Please input a query: ')

    def _low_fq_set(self, fq=1):
        """Return the set of words with frequency equal or smaller than fq."""
        # list of all words
        word_all = [w for text in self.texts_filtered for w in text]
        # frequency of all words
        word_fq = Counter(word_all)
        # set of words with frequency equal or smaller than fq
        low_fq_word_set = set([w for w in set(word_all) if word_fq[w]<=fq])
        return low_fq_word_set

    def lsi_index(self):
        """Build LSI model upon the preprocessed texts"""
        # if LSI model already saved in folder tmp, just load it
        print 'Loading LSI model from folder /tmp...'
        if os.path.isfile('./tmp/all.dict') and os.path.isfile('./tmp/all.lsi') and os.path.isfile('./tmp/all.index'):
            self.dictionary = corpora.Dictionary.load('./tmp/all.dict')
            self.lsi = models.LsiModel.load('./tmp/all.lsi')
            self.index = similarities.MatrixSimilarity.load('./tmp/all.index')
        else:
            # use libraries from gensim to build LSI model
            print 'No LSI model found, creating a new one...'
            # build dictionary for all the texts and save it 
            self.dictionary = corpora.Dictionary(self.texts_filtered)
            self.dictionary.save('./tmp/all.dict')
            # transform texts to bag-of-words representation
            corpus = [self.dictionary.doc2bow(text) for text in self.texts_filtered]
            # transform bag-of-words to TF-IDF weights
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]
            # build LSI model to transform TF-IDF representation to vector representation
            self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=self.topic_num)
            self.lsi.save('./tmp/all.lsi')
            # build index for fast access
            self.index = similarities.MatrixSimilarity(self.lsi[corpus])
            self.index.save('./tmp/all.index')

    def sample_docs(self):
        """Based on the similarities of the documents to the given query, sample
        a subset of the documents"""
        # preprocess query
        self.query = self.query.replace('-', ' ')
        self.query = self.query.lower().split()
        self.query = [w for w in self.query if w in set(self.dictionary.values())]
        # transform query to bag-of-words
        query_bow = self.dictionary.doc2bow(self.query)
        # transform query to LSI model
        query_lsi = self.lsi[query_bow]
        # get similarities of the query to all the documents
        # sims is a list of scores ordering as the documents id
        sims = self.index[query_lsi]
        # sort the scores in a descending order
        # sort_sims is a list of tuples, each tuple is (doc_id, score)
        sort_sims = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)

        # only keep the id of the documents that contain all the important words of the query
        # this is just a heuristic way to improve the matching accuracy
        valid_doc_id = set([doc_id for doc_id,doc in enumerate(self.texts_filtered) if self._contain(doc)])
        # sort_sims_valid is a list of doc ids
        sort_sims_valid = [text_id for text_id,score in sort_sims if text_id in valid_doc_id]

        # for a half of the sample docs, get them from the top ranked docs 
        sample_docs_id_top = sort_sims_valid[:int(self.num_sample_docs/2)]
        # get id of the rest docs
        sample_docs_id_remain = list(set(range(len(self.keywords))) - set(sample_docs_id_top))
        # randomly sample docs to get the other half of the sample docs
        sample_docs_id_sample = sample(sample_docs_id_remain, self.num_sample_docs-len(sample_docs_id_top))
        # form the sample docs
        sample_docs_id = sample_docs_id_top + sample_docs_id_sample

        # update the sample docs of the class
        self.sample_docs_id = sample_docs_id
        return sample_docs_id

    def recommend(self):
        """Print the recommended docs and keywords"""
        # get all the keywords from the sample documents
        sample_keys = [kw for doc_id in self.sample_docs_id for kw in self.keywords[doc_id]]
        # get top 100 of them
        # also a very heuristic way
        sample_keys = sample_keys[:100]
        sample_keys = list(set(sample_keys))
        # sample keywords as recommendation
        rec_keys_id = sample(range(len(sample_keys)), self.num_rec_keys)
        rec_keys = [sample_keys[key_id] for key_id in rec_keys_id]
        # call function to print recommended docs and keywords
        self._print_docs_keys(rec_keys)
        return rec_keys


    def _print_docs_keys(self, rec_keys):
        print ''
        print 'Recommended documents:\n'
        i = 1
        for doc_id in self.sample_docs_id[:self.num_rec_docs]:
            print 'Doc %d' % i
            print 'Title: %s' % self.t_a_k[doc_id][0]
            print 'Abstract: %s' % self.t_a_k[doc_id][1]
            print ''
            i += 1

        print ''
        print 'Recommended keywords:\n'
        i = 1
        for keyword in rec_keys:
            print 'Keyword %d' % i
            print keyword
            print ''
            i += 1

    def _contain(self, doc):
        """Check if all important words of the query in the doc."""
        doc_set = set(doc)
        for w in self.query:
            if not w in doc_set:
                return False
        return True


if __name__ == '__main__':
    # a simple test
    # initialize a recommender
    query_rec = QueryRecommender()
    # read documents data from json file
    with open('arxiv_cs_t_a_k.json', 'r') as f:
        t_a_k = json.loads(f.read())

    query_rec.read_data(t_a_k)
    query_rec.preprocess()
    query_rec.lsi_index()
    query_rec.get_query()
    sample_docs_id = query_rec.sample_docs()
    rec_keys = query_rec.recommend()
