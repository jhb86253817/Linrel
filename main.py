from query_rec import QueryRecommender
from linrel_rec import LinrelRecommender
import json

# parameters
# number of documents to recommend to user each iteration
num_rec_docs = 10
# number of keywords to recommend to user each iteration
num_rec_keys = 10
# number of the sample docs, which is a subset of all the documents
num_sample_docs = 200
# parameter of linrel, regularization term
mu = 1
# parameter of linrel, exploration rate
c = 0.5

print 'Reading data...'
with open('arxiv_cs_t_a_k.json', 'r') as f:
    t_a_k = json.loads(f.read())

# query-based recommendation
query_rec = QueryRecommender(num_rec_docs=num_rec_docs, num_rec_keys=num_rec_keys, num_sample_docs=num_sample_docs)
query_rec.read_data(t_a_k)
query_rec.preprocess()
query_rec.lsi_index()
query_rec.get_query()
sample_docs_id = query_rec.sample_docs()
rec_keys = query_rec.recommend()

# linrel-based recommendation
linrel_rec = LinrelRecommender(num_rec_docs=num_rec_docs, num_rec_keys=num_rec_keys, mu=mu, c=c)
linrel_rec.read_data(t_a_k)
linrel_rec.get_samples(sample_docs_id)
linrel_rec.select_docs(range(num_sample_docs))
linrel_rec.select_keys(rec_keys)
score_index_valid, sample_docs_id_pre = linrel_rec.rec_docs()
rec_keys = linrel_rec.rec_keys()

while(1):
    linrel_rec.select_docs(score_index_valid)
    linrel_rec.select_keys(rec_keys)
    linrel_rec.resample(sample_docs_id_pre)
    score_index_valid, sample_docs_id_pre = linrel_rec.rec_docs()
    rec_keys = linrel_rec.rec_keys()

