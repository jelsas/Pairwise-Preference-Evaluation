#!/usr/bin/python
'''This script computes several pairwise preference evaluation measures. All the
measures calculated here, as well as the format of the gold-standard preference
data to use as input are described in the publication:

  B. Carterette, P.N. Bennett, O. Chapelle (2008). A Test Collection of Preference Judgments. In Proceedings of the SIGIR 2008 Beyond Binary Relevance: Preferences, Diversity, and Set-Level Judgments Workshop.   Singapore. July, 2008.
  http://research.microsoft.com/en-us/um/people/pauben/papers/sigir-2008-bbr-data-preference-overview.pdf

'''
from __future__ import division
from collections import defaultdict
from itertools import groupby, izip
from operator import itemgetter
from math import log

class PreferenceGraph(object):
  '''A class that encapsulates all the relevance preference information for a
  single query.  This class supports weighted preferences, duplicate document judgements, as well as the ability to mark documents as BAD.'''
  def __init__(self):

    # a set of edges from PREFERRED -> NONPREFERRED
    self.edges = set()

    # a set of all docs that have been marked bad
    self.bad_docs = set()

  @property
  def vertices(self):
    '''The set of all documents in the preference graph. Note: this does not
    include documents judged BAD.'''
    return set(e[0] for e in self.edges) | set(e[1] for e in self.edges)

  @property
  def preferred(self):
    '''The set of documents that have been preferred to any other document.'''
    return set(e[0] for e in self.edges)

  def __str__(self):
    edges_str = '\n  '.join( '%s -> %s (%0.2f)' % e for e in self.edges )
    bad_docs_str = '\n  '.join( self.bad_docs )

    return 'Edges:\n  %s\nBadDocs:\n  %s' % (edges_str, bad_docs_str)

  def add_bad_doc(self, bad_doc):
    '''Adds a document to the set of BAD docs. If the document has already been
    assessed as preferred to any other document, a ValueError is raised.'''
    if bad_doc in self.preferred:
      raise ValueError('Doc %s can not be both preferred & bad' % bad_doc)
    self.bad_docs.add(bad_doc)

  def add_edge(self, from_vertex, to_vertex, weight = 1):
    '''Adds an edge in the preference graph. ValueError is raised if an edge is
    added where the preferred document has previously been added as BAD.'''
    if from_vertex in self.bad_docs:
      raise ValueError('Doc %s can not be both preferred & bad' % from_vertex)
    self.edges.add( (from_vertex, to_vertex, weight) )

  def all_path_lengths(self, transitive=True):
    '''Floyd-Warshall algorithm to find the distance of all minimum-length paths between any two vetices. Returns a dictionary such that d[(i, j)] is the shortest path between nodes i and j. If no such path exists, this key won't be present in the dictionary. This algorithm runs in O(|V|**3).

    If transitive=False, only explicit preferences are assumed, and the F-W
    algorithm is not run.

    Note: BAD documents are not included.'''
    d = dict()
    for (f, t, w) in self.edges:
      d[(f, t)] = w

    if not transitive:
      return d

    vertices = self.vertices
    for k in vertices:
      for i in vertices:
        if i == k: continue
        try:
          d_i_k = d[(i,k)]
        except KeyError:
          # if i doesn't reach k, then no need to proceed
          continue
        for j in vertices:
          if i == j or j == k: continue
          d_k_j = d.get((k, j))
          d_i_j = d.get((i, j))
          # if k->j, we update d[(i,j)], otherwise leave untouched
          if d_k_j is not None:
            d[(i, j)] = min(d_i_j, d_i_k + d_k_j) if d_i_j else d_i_k + d_k_j
    return d

  @classmethod
  def read_pref_file(cls, filename):
    '''Reads a preference file and returns a dict of qid -> PreferenceGraph.

    File should be in the format:
      [qid] [doc1] [doc2] [preference]
    where the [preference] value indicates which document is preferred or BAD.
    The format is described in the above citation. Briefly:

    [preference] ==  2: doc2 = BAD, doc1 must be "NA"
                 == -2: doc1 = BAD, doc2 must be "NA"
                 ==  1: doc2 preferred to doc1
                 == -1: doc1 preferred to doc2
                 ==  0: doc1 and doc2 duplicates.
    '''
    preferences = defaultdict(cls)
    for line in open(filename):
      try:
        qid, from_doc, to_doc, pref = line.lower().strip().split()
      except ValueError, e:
        raise ValueError('Can\'t parse line: \'%s\', %s' % \
            (line.strip(), str(e)))
      pref = int(pref)
      if pref == -1:
        preferences[qid].add_edge(from_doc, to_doc, 1)
      elif pref == 1:
        preferences[qid].add_edge(to_doc, from_doc, 1)
      elif pref == 0:
        # if duplicates, add a zero-weight edge in both directions
        preferences[qid].add_edge(from_doc, to_doc, 0)
        preferences[qid].add_edge(to_doc, from_doc, 0)
      elif pref == -2:
        if to_doc.lower() != 'na':
          raise ValueError('OTHER doc in bad judgements must be "NA"')
        preferences[qid].add_bad_doc(from_doc)
      elif pref == 2:
        if from_doc.lower() != 'na':
          raise ValueError('OTHER doc in bad judgements must be "NA"')
        preferences[qid].add_bad_doc(to_doc)
      else:
        raise ValueError('Can\'t understand preference: %d' % pref)
    return preferences


######### THE FOLLOWING FUNCTIONS ARE THE EVALUATION MEASURES ###########

def count_incorrect(preferred, bad, rank = -1):
  '''Counts the incorrect pairs among the list of bad & preferred ranks. 'preferred' and 'bad' must be in sorted order.'''
  pi, bi, count = 0, 0, 0
  if rank < 0: rank = max(preferred)
  while pi < len(preferred) and preferred[pi] <= rank:
    while bi < len(bad) and bad[bi] <= preferred[pi]:
      bi += 1
    count += bi
    pi += 1
  return count

def num_pref(k):
  '''Returns a function to count the number of judged preferences present in the top k ranked documents. If k < 0, counts judged preferences in all retrieved
  documents. '''
  if k < 0:
    def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
      n_bad = len(bad_docs_ranks) + bad_docs_count_unranked
      n_preferred = len(preferred_ranks) + preferred_count_unranked
      return len(pref_ranks) + n_bad * n_preferred
    return f
  else:
    def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
      n_prefs_above_k = sum(1 for (f, t, w) in pref_ranks if \
                            (f <= k or t <= k) and w > 0)
      n_preferred_above_k = sum(1 for i in preferred_ranks if i <= k)
      n_bad = len(bad_docs_ranks) + bad_docs_count_unranked
      return n_prefs_above_k + n_preferred_above_k * n_bad
    return f

def num_pref_correct(k):
  '''Returns a function to count the number of judged preferences correctly
  ordered present in the top k ranked documents.'''
  def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
    # total prefs @ k
    n_prefs = num_pref(k)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                          preferred_ranks, preferred_count_unranked)
    # incorrect from expressed preferences
    n_incorrect = sum(int(f > t) for (f, t, w) in pref_ranks \
                      if w > 0 and (t <= k or f <= k))
    # incorrect from preferred/bad judgements
    n_incorrect += count_incorrect(preferred_ranks, bad_docs_ranks, k)
    return n_prefs - n_incorrect
  return f

def num_preferred(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
  return len(preferred_ranks) + preferred_count_unranked

def num_bad(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
  return len(bad_docs_ranks) + bad_docs_count_unranked

def ppref(k):
  '''Returns function calculating ppref at the given cutoff (k)'''
  def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
    # total prefs @ k
    n_prefs = num_pref(k)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                          preferred_ranks, preferred_count_unranked)
    if n_prefs == 0: return 0
    n_prefs_correct = num_pref_correct(k)(pref_ranks, bad_docs_ranks,
            bad_docs_count_unranked, preferred_ranks, preferred_count_unranked)

    precision = n_prefs_correct / n_prefs
    return precision
  return f

def rpref(k):
  '''Returns function calculating rpref at the given cutoff (k)'''
  def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
    # total prefs anywhere
    n_prefs = num_pref(-1)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                          preferred_ranks, preferred_count_unranked)
    if n_prefs == 0: return 0
    n_prefs_correct = num_pref_correct(k)(pref_ranks, bad_docs_ranks,
            bad_docs_count_unranked, preferred_ranks, preferred_count_unranked)
    recall = n_prefs_correct / n_prefs
    return recall
  return f

def fpref(k, beta = 1):
  '''Returns function calculating rpref/ppref F measure at the given cutoff'''
  def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
          preferred_ranks, preferred_count_unranked):
    ppref_k = ppref(k)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                          preferred_ranks, preferred_count_unranked)
    rpref_k = rpref(k)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                          preferred_ranks, preferred_count_unranked)
    if ppref_k == 0 and rpref_k == 0:
      return 0
    return (1 + beta**2) * (ppref_k * rpref_k) / (beta**2 * ppref_k + rpref_k)
  return f

def appref(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
           preferred_ranks, preferred_count_unranked):
  # list of (rpref, rank), making sure we include the first rank
  # rpref can only change when we encounter a ranked preferred doc
  if len(preferred_ranks) == 0 or preferred_ranks[0] != 1:
    preferred_ranks = [1] + preferred_ranks
  rprefs = [(i, rpref(i)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
           preferred_ranks, preferred_count_unranked)) for i in preferred_ranks]
  #print rprefs
  # average ppref across ranks where rpref changes.
  rpref_change_ranks = [1] + [rprefs[i][0] for i in xrange(1,len(rprefs)) \
                              if rprefs[i][1] != rprefs[i-1][1]]
  #print rpref_change_ranks
  pprefs = [ ppref(i)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
           preferred_ranks, preferred_count_unranked) \
              for i in rpref_change_ranks ]
  #print pprefs
  return sum(pprefs) / len(rpref_change_ranks)

def wpref(k, w_func = None):
  '''returns a function for calculating wpref@k. assumes uniform preference degree (pref_ij == 1)'''
  def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
           preferred_ranks, preferred_count_unranked):
    if w_func is None:
      weight = lambda f, t: 1.0 / (log(f, 2)+1) if f < t else 0.0
    else:
      weight = w_func
    # iterate through the pref_ranks & tally up the wpref values
    wpref = sum(weight(f, t) for (f, t, w) in pref_ranks if \
                          (f <= k or t <= k) and w > 0)
    # add the wpref for unranked BAD docs. assume bad docs are at rank k+1
    wpref += sum(weight(f, k+1)*bad_docs_count_unranked \
                for f in preferred_ranks)
    return wpref
  return f

def nwpref(k):
  def f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
           preferred_ranks, preferred_count_unranked):
    unnorm = wpref(k)(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                      preferred_ranks, preferred_count_unranked)
    # to normalize the wpref values, we create a weighting function that always
    # counts a document pair regardless of the correct ordering of the docs.
    # TODO: this doesn't really reflect a "perfect" ordering at rank k
    norm_weight = lambda f, t: 1.0 / (log(min(f, t), 2)+1)
    norm = wpref(k, w_func = norm_weight)(pref_ranks, bad_docs_ranks,
                 bad_docs_count_unranked, preferred_ranks,
                 preferred_count_unranked)
    if norm == 0:
      assert unnorm == 0, 'wpref norm = 0, but wpref != 0'
      return 0
    else:
      return unnorm/norm
  return f

def rrpref(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
           preferred_ranks, preferred_count_unranked):
  '''Reciprocal Rank pref. 1/rank of first correctly ordered preferred doc.'''
  correctly_ranked = [f for (f, t, w) in pref_ranks if w > 0 and f < t]
  if correctly_ranked:
    return 1.0 / min(correctly_ranked)
  else:
    return 0.0

######### MAIN: #########################
if __name__=='__main__':
  from optparse import OptionParser
  import sys

  parser = OptionParser(usage='usage: %prog [options] pref_file results_file')
  parser.add_option('-q', action='store_true', dest='per_q',
                    help='print per-query statistics')
  parser.add_option('-v', action='store_true', dest='verbose',
                    help='print lots of debugging info')
  parser.add_option('-i', action='store_true', dest='intransitive',
                    default=False,
                    help='Do not assume transitive preferences')
  (options, args) = parser.parse_args()

  try:
    prefs_file, results_file = args
  except ValueError, e:
    parser.error(e)

  try:
    prefs = PreferenceGraph.read_pref_file(prefs_file)
  except ValueError, e:
    parser.error('Error parsing pref_file \'%s\'\n%s' % (prefs_file, str(e)))
  except IOError, e:
    parser.error('Error reading pref_file \'%s\'\n%s' % (prefs_file, str(e)))

  # All the evaluation measures we calculate.
  # A sequence of tuples (name, function, format)
  eval_measures = (
                    ('num_preference', num_pref(-1),    '%d'),
                    ('num_preferred',  num_preferred,   '%d'),
                    ('num_bad',        num_bad,         '%d'),
                    ('rrpref',         rrpref,          '%0.4f'),
                    ('ppref1',         ppref(1),        '%0.4f'),
                    ('ppref5',         ppref(5),        '%0.4f'),
                    ('ppref10',        ppref(10),       '%0.4f'),
                    ('rpref1',         rpref(1),        '%0.4f'),
                    ('rpref5',         rpref(5),        '%0.4f'),
                    ('rpref10',        rpref(10),       '%0.4f'),
                    ('fpref1',         fpref(1),        '%0.4f'),
                    ('fpref5',         fpref(5),        '%0.4f'),
                    ('fpref10',        fpref(10),       '%0.4f'),
                    ('wpref1',         wpref(1),        '%0.4f'),
                    ('wpref5',         wpref(5),        '%0.4f'),
                    ('wpref10',        wpref(10),       '%0.4f'),
                    ('nwpref1',        nwpref(1),       '%0.4f'),
                    ('nwpref5',        nwpref(5),       '%0.4f'),
                    ('nwpref10',       nwpref(10),      '%0.4f'),
                    ('APpref' ,        appref,          '%0.4f'),
                  )
  label_len = max(len(x[0]) for x in eval_measures)+2

  def read_results_file(filename):
    '''Reads a TREC format results file. This function is a generator yielding
    the tuples (qid, [list of docs in rank order]).'''
    def parse_line(line):
      try:
        qid, _, docname, _, score, _ = line.lower().strip().split(None, 5)
      except ValueError, e:
        parser.error('Error parsing results_file %s\nCan\'t parse line: \'%s\', %s' % \
            (filename, line.strip(), str(e)))
      return (qid, docname, float(score))

    input = (parse_line(line) for line in open(filename))
    for (q, q_data) in groupby(input, itemgetter(0)):
      q_data = sorted(q_data, key=itemgetter(2), reverse=True)
      yield (q, [d for (q, d, s) in q_data])

  summary_measures = defaultdict(float)
  num_q = 0
  for (q, docs) in read_results_file(results_file):
    q_prefs = prefs[q]
    if options.verbose:
      print 'Query: %s' % q
      print q_prefs
    if len(q_prefs.edges) == 0:
      if options.verbose: print 'skipping q', q, 'no edges'
      continue
    num_q += 1
    trans_prefs = q_prefs.all_path_lengths(transitive=not options.intransitive)
    # map preferences & bad docs to ranks instead of docids
    doc_rank = dict( (doc, i+1) for (i, doc) in enumerate(docs) )
    pref_ranks = [ \
          (doc_rank.get(f), doc_rank.get(t), w) \
                          for ( (f, t), w) in trans_prefs.iteritems() \
                          if f in doc_rank and t in doc_rank]

    bad_docs_ranks = sorted(doc_rank[d] for d in q_prefs.bad_docs \
                            if d in doc_rank)
    bad_docs_count_unranked = len(q_prefs.bad_docs) - len(bad_docs_ranks)

    preferred = set( f for ((f, t), w) in trans_prefs.iteritems() if w > 0 )
    preferred_ranks = sorted( doc_rank[d] for d in preferred \
                                if d in doc_rank )
    preferred_count_unranked = len(preferred) - len(preferred_ranks)

    for (eval_name, eval_f, fmt) in eval_measures:
      m = eval_f(pref_ranks, bad_docs_ranks, bad_docs_count_unranked,
                 preferred_ranks, preferred_count_unranked)
      if options.per_q:
        print '%s\t%s\t%s' % (eval_name.ljust(label_len), q, fmt % m)
      summary_measures[eval_name] += m

  print '%s\tall\t%d' % ('num_q'.ljust(label_len), num_q)
  if num_q > 0:
    for (eval_name, _, fmt) in eval_measures:
      m = summary_measures[eval_name]
      print 'm%s\tall\t%0.04f' % (eval_name.ljust(label_len), (m / num_q) )
