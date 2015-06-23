# Linrel-based document and keyword recommendation system
from __future__ import division
import json
import numpy
import random
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import inv
import time
from collections import Counter
from collections import defaultdict
from math import log
random.seed('12')

class LinrelRecommender:
    def __init__(self, num_rec_docs=10, num_rec_keys=10, mu=1, c=0.5):
        """For initialization"""
        # for storing docs that explicitly selected by user
        self.selected_docs = []
        # for storing the feedback scores of the selected docs
        self.feedback_docs = []
        # for storing keywords that explicitly selected by user
        self.selected_keys = []
        # for storing the feedback scores of the selected keywords
        self.feedback_keys = []
        # for storing docs that implicitly selected, also stores its score
        self.impl_feedback_docs = defaultdict(float)
        # for storing keywords that implicitly selected, also stores its score
        self.impl_feedback_keys = defaultdict(float)
        # number of docs for recommendation each iteration
        self.num_rec_docs = num_rec_docs
        # number of keywords for recommendation each iteration
        self.num_rec_keys = num_rec_keys
        # parameter of linrel, regularization term
        self.mu = mu
        # parameter of linrel, exploration rate
        self.c = c

    def read_data(self, t_a_k):
        """Read data of documents and keywords"""
        # t_a_k is a list of all the documents
        # each document represented as a tuple (title, abstract, list of keywords)
        self.t_a_k = t_a_k

    def get_samples(self, sample_docs_id):
        """Get sample docs, which is a subset of all documents"""
        self.sample_docs_id = sample_docs_id
        self.t_a_k_sample = [self.t_a_k[doc_id] for doc_id in sample_docs_id]

        self.titles = [p[0] for p in self.t_a_k_sample]
        self.abstracts = [p[1] for p in self.t_a_k_sample]
        self.keywords = [p[2] for p in self.t_a_k_sample]

        self._preprocess()

    def resample(self, sample_docs_id_pre):
        """Randomly replace the bottom half sample docs"""
        sample_docs_id = sample_docs_id_pre[:int(self.num_doc/2)]
        remain_index = list(set(range(len(self.t_a_k))) - set(sample_docs_id))
        remain_index_sample = random.sample(remain_index, int(self.num_doc/2))
        sample_docs_id += remain_index_sample
        self.get_samples(sample_docs_id)

    def _preprocess(self):
        # mapping keyword to keyword id
        keywords_set = set([k for keyword in self.keywords for k in keyword])
        self.dict_key_id = {}
        for i,k in enumerate(keywords_set):
            self.dict_key_id[k] = i

        # mapping keyword id to keyword
        self.dict_id_key = {v:k for k,v in self.dict_key_id.items()}

        # record the number of documents and keywords
        self.num_doc = len(self.keywords)
        self.num_key = len(keywords_set)

        # mapping doc id to its list of keywords
        self.dict_doc_key={}
        for i,k in enumerate(self.keywords):
            self.dict_doc_key[i] = k

        # mapping keyword to list of document ids
        self.dict_key_doc = defaultdict(list)
        for i,keys in enumerate(self.keywords):
            for key in keys:
                self.dict_key_doc[key].append(i)

        # mapping keyword to number of documents
        keywords_all = [k for keyword in self.keywords for k in keyword]
        self.key_fq = Counter(keywords_all)

        # mapping document id to number of keywords
        self.doc_fq = [len(keyword) for keyword in self.keywords]

        # mapping document id to number of keywords
        self.doc_num_key = [len(keyword) for keyword in self.keywords]

        # mapping keyword to number of documents
        keywords_all = [k for keyword in self.keywords for k in keyword]
        self.key_num_doc = Counter(keywords_all)

    def select_docs(self, score_index_valid):
        """Ask user to select docs"""
        selected_docs = []
        while(1):
            selected_doc_id = raw_input('Please select a doc (type 0 to stop): ')
            if selected_doc_id == '0':
                break
            else:
                selected_docs.append(self.sample_docs_id[score_index_valid[int(selected_doc_id)-1]])
        print ''

        self.selected_docs += selected_docs
        self.feedback_docs += [1.0] * len(selected_docs)

        # implicit feedback
        for doc_id in selected_docs:
            for keyword in self.dict_doc_key[self.sample_docs_id.index(doc_id)]:
                #self.impl_feedback_keys[keyword] += 100.0/self.key_fq[keyword]/self.doc_num_key[self.sample_docs_id.index(doc_id)]
                self.impl_feedback_keys[keyword] = 0.5
        #print '%d implicit keywords found' % len(self.impl_feedback_keys.keys())

    def select_keys(self, rec_keys):
        """Ask user to select keywords"""
        selected_keys = []
        while(1):
            selected_key_id = raw_input('Please select a key (type 0 to stop): ')
            if selected_key_id == '0':
                break
            else:
                selected_keys.append(rec_keys[int(selected_key_id)-1])
        print ''

        self.selected_keys += selected_keys
        self.feedback_keys += [1.0] * len(selected_keys)

        # implicit feedback
        for key in selected_keys:
            for doc_id in self.dict_key_doc[key]:
                #self.impl_feedback_docs[self.sample_docs_id[doc_id]] += 100.0/self.doc_fq[doc_id]/self.key_num_doc[key]
                self.impl_feedback_docs[self.sample_docs_id[doc_id]] = 0.5
        #print '%d implicit docs found' % len(self.impl_feedback_docs.keys())

    def get_X_t_doc(self):
        """Compute the X_t matrix for docs, which records the docs with feedbacks."""
        # remove implicit feedback if already in the explicit feedback or not in
        # the current sample docs
        for doc_id in self.impl_feedback_docs.keys():
            if doc_id in set(self.selected_docs):
                del self.impl_feedback_docs[doc_id]
            if doc_id not in set(self.sample_docs_id):
                del self.impl_feedback_docs[doc_id]
        #print '%d implicit docs remained' % len(self.impl_feedback_docs.keys())

        # matrix X_t as a sparse matrix
        row = []
        col = []
        data = []
        for row_index,doc_id in enumerate(self.selected_docs):
            for col_index in self.dict_doc_key[self.sample_docs_id.index(doc_id)]:
                row.append(row_index)
                col.append(self.dict_key_id[col_index])
                #data.append(log(100.0/self.key_fq[col_index]/self.doc_num_key[self.sample_docs_id.index(doc_id)]))
                data.append(1.0)
        temp_row_index = row_index + 1
        for row_index,doc_id in enumerate(self.impl_feedback_docs.keys()):
            for col_index in self.dict_doc_key[self.sample_docs_id.index(doc_id)]:
                row.append(row_index+temp_row_index)
                col.append(self.dict_key_id[col_index])
                #data.append(log(100.0/self.key_fq[col_index]/self.doc_num_key[self.sample_docs_id.index(doc_id)]))
                data.append(1.0)
        row = numpy.array(row)
        col = numpy.array(col)
        data = numpy.array(data)
        X_t = csr_matrix((data, (row, col)), shape=(len(self.selected_docs)+len(self.impl_feedback_docs.keys()),self.num_key))
        return X_t

    def get_X_t_key(self):
        """Compute the X_t matrix for keywords, which records the keywords with feedbacks."""
        # remove implicit feedback if already in the explicit feedback
        for key in self.impl_feedback_keys.keys():
            if key in set(self.selected_keys):
                del self.impl_feedback_keys[key]
            if key not in set(self.dict_key_doc):
                del self.impl_feedback_keys[key]
        #print '%d implicit keywords remained' % len(self.impl_feedback_keys.keys())
        #print self.impl_feedback_keys.items()

        # matrix X_t as a sparse matrix
        row = []
        col = []
        data = []
        for row_index,key in enumerate(self.selected_keys):
            for col_index in self.dict_key_doc[key]:
                row.append(row_index)
                col.append(col_index)
                #data.append(log(100.0/self.doc_fq[col_index]/self.key_num_doc[key]))
                data.append(1.0)
        if len(self.selected_keys) == 0:
            row_index = -1
        temp_row_index = row_index + 1
        for row_index,key in enumerate(self.impl_feedback_keys.keys()):
            for col_index in self.dict_key_doc[key]:
                row.append(row_index+temp_row_index)
                col.append(col_index)
                #data.append(log(100.0/self.doc_fq[col_index]/self.key_num_doc[key]))
                data.append(1.0)
        row = numpy.array(row)
        col = numpy.array(col)
        data = numpy.array(data)
        X_t = csr_matrix((data, (row, col)), shape=(len(self.selected_keys)+len(self.impl_feedback_keys.keys()),self.num_doc))
        return X_t

    def get_y_t_doc(self):
        """Compute column matrix y_t for docs, which records the feedback scores."""
        # if there is implicit feedbacks for docs
        if len(self.impl_feedback_docs.keys()) > 0:
            num_row = len(self.feedback_docs)+len(self.impl_feedback_docs.keys())
            row = numpy.array(range(num_row))
            col = numpy.array([0] * num_row)
            data1 = numpy.array([1.0] * len(self.feedback_docs))
            #data2 = numpy.array(self.impl_feedback_docs.values()) / max(self.impl_feedback_docs.values())
            data2 = numpy.array(self.impl_feedback_docs.values()) 
            data = numpy.concatenate([data1, data2], axis=1)
            y_t = csr_matrix((data, (row, col)), shape=(num_row,1))
        # if there is no implicit feedbacks for docs
        else:
            row = numpy.array(range(len(self.feedback_docs)))
            col = numpy.array([0] * len(self.feedback_docs))
            data = numpy.array([1.0] * len(self.feedback_docs))
            y_t = csr_matrix((data, (row, col)), shape=(len(self.feedback_docs),1))

        return y_t

    def get_y_t_key(self):
        """Compute column matrix y_t for keywords, which records the feedback scores."""
        # if there is implicit feedbacks for keywords
        if len(self.impl_feedback_keys.keys()) > 0:
            num_row = len(self.feedback_keys)+len(self.impl_feedback_keys.keys())
            row = numpy.array(range(num_row))
            col = numpy.array([0] * num_row)
            data1 = numpy.array([1.0] * len(self.feedback_keys))
            #data2 = numpy.array(self.impl_feedback_keys.values()) / max(self.impl_feedback_keys.values())
            data2 = numpy.array(self.impl_feedback_keys.values()) 
            data = numpy.concatenate([data1, data2], axis=1)
            y_t = csr_matrix((data, (row, col)), shape=(num_row,1))
        # if there is no implicit feedbacks for keywords
        else:
            row = numpy.array(range(len(self.feedback_keys)))
            col = numpy.array([0] * len(self.feedback_keys))
            data = numpy.array([1.0] * len(self.feedback_keys))
            y_t = csr_matrix((data, (row, col)), shape=(len(self.feedback_keys),1))
        return y_t

    def get_X_doc(self):
        """Compute matrix X for all sample docs"""
        sample_docs = range(self.num_doc)
        row = []
        col = []
        data = []
        for row_index,doc_id in enumerate(sample_docs):
            for col_index in self.dict_doc_key[doc_id]:
                row.append(row_index)
                col.append(self.dict_key_id[col_index])
                #data.append(log(100.0/self.key_fq[col_index]/self.doc_num_key[doc_id]))
                data.append(1.0)
        row = numpy.array(row)
        col = numpy.array(col)
        data = numpy.array(data)
        X = csr_matrix((data, (row, col)), shape=(len(sample_docs),self.num_key))
        return X

    def get_X_key(self):
        """Compute matrix X for all keywords of sample docs"""
        sample_keys = [self.dict_id_key[key_id] for key_id in range(self.num_key)]
        row = []
        col = []
        data = []
        for row_index,key in enumerate(sample_keys):
            for col_index in self.dict_key_doc[key]:
                row.append(row_index)
                col.append(col_index)
                #data.append(log(100.0/self.doc_fq[col_index]/self.key_num_doc[key]))
                data.append(1.0)
        row = numpy.array(row)
        col = numpy.array(col)
        data = numpy.array(data)
        X = csr_matrix((data, (row, col)), shape=(self.num_key,self.num_doc))
        return X

    def linrel(self, X_t, y_t, X, mu, c):
        """Linrel algorithm"""
        temp = X_t.T * X_t + mu*identity(X_t.shape[1])
        temp = inv(temp)
        temp = X * temp * X_t.T
        score = (temp*y_t).toarray() + numpy.linalg.norm(temp.toarray(), axis=1).reshape((temp.shape[0],1))
        return score

    def rec_docs(self):
        """Recommend docs to user"""
        X_t = self.get_X_t_doc()
        y_t = self.get_y_t_doc()
        X = self.get_X_doc()

        score = self.linrel(X_t, y_t, X, self.mu, self.c)
        score = score.T.tolist()[0]
        score_index = sorted(range(len(score)), key=lambda k:score[k], reverse=True)

        # make sure the selected docs are not recommended to user again
        score_index_valid = [index for index in score_index if self.sample_docs_id[index] not in set(self.selected_docs) and score[index]>0]

        print ''
        print 'Recommended documents:\n'
        i = 1
        for doc_id in score_index_valid[:self.num_rec_docs]:
            print 'Doc %d' % i
            print 'Title: %s' % self.t_a_k_sample[doc_id][0]
            print 'Abstract: %s' % self.t_a_k_sample[doc_id][1]
            print ''
            i += 1

        sample_docs_id_pre = [self.sample_docs_id[doc_id] for doc_id in score_index]
        return score_index_valid, sample_docs_id_pre

    def rec_keys(self):
        """Recommend keywords to user"""
        X_t = self.get_X_t_key()
        y_t = self.get_y_t_key()
        X = self.get_X_key()

        score = self.linrel(X_t, y_t, X, self.mu, self.c)
        score = score.T.tolist()[0]
        score_index = sorted(range(len(score)), key=lambda k:score[k], reverse=True)

        # make sure the selected keywords are not recommended to user again
        score_index = [index for index in score_index if self.dict_id_key[index] not in set(self.selected_keys) and score[index]>0]
        score_index = score_index[:self.num_rec_keys] 
        rec_keys = [self.dict_id_key[key_id] for key_id in score_index]

        print ''
        print 'Recommended keywords:\n'
        i = 1
        for keyword in rec_keys:
            print 'Keyword %d' % i
            print keyword
            print ''
            i += 1
        return rec_keys

if __name__ == '__main__':
    linrel_rec = LinrelRecommender()
    with open('arxiv_cs_t_a_k.json', 'r') as f:
        t_a_k = json.loads(f.read())
