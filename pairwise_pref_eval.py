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
import sys

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

    if self.bad_docs:
      bad_docs_str = '\n  '.join( self.bad_docs )
      return 'Edges:\n  %s\nBadDocs:\n  %s' % (edges_str, bad_docs_str)
    else:
      return 'Edges:\n  %s' % edges_str

  def add_bad_doc(self, bad_doc):
    '''Adds a document to the set of BAD docs. If the document has already been
    assessed as preferred to any other document, it is silently ignored.'''
    if bad_doc in self.preferred:
      pass
    else:
      self.bad_docs.add(bad_doc)

  def add_edge(self, from_vertex, to_vertex, weight = 1):
    '''Adds an edge in the preference graph. If an edge is added where the 
    preferred document has previously been added as BAD, the edge is silently 
    ignored.'''
    if from_vertex in self.bad_docs:
      pass
    else:
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

class ResultPreferences(object):
  '''Encapsulates all the preference data associated with retrieval results.
  This class handles mapping document ids's in the PreferenceGraph to document
  ranks.'''

  UNRANKED = sys.maxint

  def __init__(self, ordered_docs, pref_graph, transitive=True):
    all_prefs = q_prefs.all_path_lengths(transitive=transitive)
    doc_rank = dict( (doc, i+1) for (i, doc) in enumerate(ordered_docs) )

    # the preferences (f, t, w) where (f, t) are ranks, (w) is a weight
    # f or t can be UNRANKED if they aren't ranked
    self.pref_ranks = [ \
          (doc_rank.get(f, self.UNRANKED), doc_rank.get(t, self.UNRANKED)) \
                          for ( (f, t), w) in all_prefs.iteritems() if w > 0]
    self.count_preferred_unranked = len(set(f for ((f, t), w) \
                                        in all_prefs.iteritems() \
                                        if f not in doc_rank))
    self.duplicates = [ \
          (doc_rank.get(f, self.UNRANKED), doc_rank.get(t, self.UNRANKED)) \
                          for ( (f, t), w) in all_prefs.iteritems() if w == 0]

    # the ranks of judged bad documents. can be UNRANKED if they aren't ranked
    self.bad_docs_ranks = sorted(doc_rank.get(d, self.UNRANKED) \
                                for d in q_prefs.bad_docs)
    self.count_bad_unranked = len(set(d for d in q_prefs.bad_docs \
                                  if d not in doc_rank))

    # generate preference pairs between all preferred docs & all bad docs
    # ranks of preferred ranked docs
    self.preferred_ranks = sorted(set(f for (f, t) in self.pref_ranks \
                                if f != self.UNRANKED))

    # add pairs of ranked preferred w/ unranked BAD
    self.pref_ranks = self.pref_ranks + \
        [(f, t) for f in self.preferred_ranks for t in self.bad_docs_ranks]

    # num. of pairs of preferred-bad docs that are both unranked
    count_both_unranked = self.count_bad_unranked*self.count_preferred_unranked

    # add pairs of unranked preferred, unranked BAD
    self.pref_ranks = self.pref_ranks + \
        [(self.UNRANKED, self.UNRANKED)]*count_both_unranked

  def preferred_docs_above(self, k):
    '''Returns the rank of preferred documents ranked at k or above. If k < 0,
    returns ranks of preferred documents ranked anywhere.'''
    if k < 0:
      return sorted(set(f for (f, t) in self.pref_ranks if f < self.UNRANKED))
    else:
      return sorted(set(f for (f, t) in self.pref_ranks if f <= k))

  def bad_docs_above(self, k):
    '''Returns the ranks of bad documents ranked above k. If k < 0, returns
    the ranks of all bad documents retrieved.'''
    if k < 0:
      return [d for d in self.bad_docs_ranks if d < self.UNRANKED]
    else:
      return [d for d in self.bad_docs_ranks if d <= k]

  def pref_pairs_above(self, k):
    '''Returns the list of preference pairs where either document is above
    or equal to rank k. If k < 0, returns all preference pairs where either
    document is ranked.'''
    if k < 0:
      return [(f, t) for (f, t) in self.pref_ranks \
                  if f < self.UNRANKED or t < self.UNRANKED]
    else:
      return [(f, t) for (f, t) in self.pref_ranks if f <= k or t <= k]

  def __str__(self):
    ranks = set(f for (f, t) in self.pref_ranks) | \
            set(t for (f, t) in self.pref_ranks)
    if len(ranks) == 0: return 'NA'
    if self.UNRANKED in ranks: ranks.remove(self.UNRANKED)
    ranks = sorted(ranks)
    preferred_to_non = set( self.pref_ranks )
    # build a matrix M of rank x rank w/ each cell >, <, =, B
    m = []
    for (i, ri) in enumerate(ranks):
      r = []
      for rj in ranks:
        if ri == rj:
          r.append('#')      # diagonal == '#'
        elif (ri, rj) in preferred_to_non:
          r.append('>')      # row preferred to column
        elif (rj, ri) in preferred_to_non:
          r.append('<')      # column preferred to row
        elif (ri, rj) in self.duplicates or (rj, ri) in self.duplicates:
          r.append('=')      # duplicates
        else:
          r.append(' ')      # no information
      # add unranked info
      # cols: INF< INF> INF=
      inf_gt = sum(1 for (f, t) in self.pref_ranks \
                          if f == ri and t == self.UNRANKED)
      inf_lt = sum(1 for (f, t) in self.pref_ranks \
                          if t == ri and f == self.UNRANKED)
      inf_eq = sum(1 for (f, t) in self.duplicates \
                          if (f == ri and t == self.UNRANKED) or \
                             (t == ri and f == self.UNRANKED))
      r = r + [str(inf_lt), str(inf_gt), str(inf_eq)]
      m.append(r)
    headers = [str(x) for x in ranks] + ['INF<', 'INF>', 'INF=']
    cell_widths = [len(x)+1 for x in headers]
    s = ' '*max(cell_widths) + \
                ''.join(x.ljust(w) for (x, w) in izip(headers, cell_widths))
    for i in xrange(len(headers[:-3])):
      r = ranks[i]
      h = headers[i]
      si = '%s%s' % (h.ljust(max(cell_widths)),
                    ''.join(x.ljust(w) for (x, w) in izip(m[i], cell_widths)))
      s = '\n'.join((s, si))
    # add a row showing INF-INF preferences
    inf_inf_prefs = sum(f == t == self.UNRANKED for (f, t) in self.pref_ranks)
    s = '%s\n%s' % (s, 'INF<INF: %d'.rjust(sum(cell_widths)) % inf_inf_prefs)
    return s

######### THE FOLLOWING FUNCTIONS ARE THE EVALUATION MEASURES ###########

def count_correct(rank_prefs, rank = -1):
  '''Counts the number of correct pairs at or above rank. If rank < 0, counts
  correct pairs retrieved at any rank.'''
  if rank < 0:
    return sum(1 for (f, t) in rank_prefs.pref_ranks \
              if f < t and f < rank_prefs.UNRANKED)
  else:
    return sum(1 for (f, t) in rank_prefs.pref_ranks if f < t and f <= rank)

def count_incorrect(rank_prefs, rank = -1):
  '''Counts the number of incorrect pairs at or above rank. If rank < 0, counts
  incorrect pairs retrieved at any rank.'''
  if rank < 0:
    return sum(1 for (f, t) in rank_prefs.pref_ranks \
              if f > t and t < rank_prefs.UNRANKED)
  else:
    return sum(1 for (f, t) in rank_prefs.pref_ranks if f > t and t <= rank)

def num_preferred(rank_prefs):
  '''Returns the number of documents that have ever been preferred to another
  document, whether or not those documents were retrieved'''
  return len(rank_prefs.preferred_docs_above(-1)) + \
              rank_prefs.count_preferred_unranked

def num_preferred_unranked(rank_prefs):
  '''Returns the number of documents that have ever been preferred to another
  document and were not retrieved'''
  return rank_prefs.count_preferred_unranked

def num_bad(rank_prefs):
  '''Returns the number of documents judged bad, whether or not those documents
  were retrieved'''
  return len(rank_prefs.bad_docs_above(-1)) + \
              rank_prefs.count_bad_unranked

def num_pref(rank_prefs):
  return len(rank_prefs.pref_ranks)

def num_pref_ranked(k):
  '''Returns a function to count the number of judged preferences where either
  document is present in the top k ranked documents. If k < 0, counts judged
  preferences in all retrieved documents. Includes implied preference between
  preferred & bad documents'''
  def f(rank_prefs):
    return len(rank_prefs.pref_pairs_above(k))
  return f

def num_pref_correct(k):
  '''Returns a function to count the number of judged preferences correctly
  ordered present in the top k ranked documents.'''
  def f(rank_prefs):
    return count_correct(rank_prefs, k)
  return f

def ppref(k):
  '''Returns function calculating ppref at the given cutoff (k)'''
  def f(rank_prefs):
    # total prefs @ k
    n_prefs = num_pref_ranked(k)(rank_prefs)
    if n_prefs == 0: return 0
    n_prefs_correct = num_pref_correct(k)(rank_prefs)
    precision = n_prefs_correct / n_prefs
    return precision
  return f

def rpref(k):
  '''Returns function calculating rpref at the given cutoff (k)'''
  def f(rank_prefs):
    # total prefs anywhere
    n_prefs = len(rank_prefs.pref_ranks)
    if n_prefs == 0: return 0
    n_prefs_correct = num_pref_correct(k)(rank_prefs)
    recall = n_prefs_correct / n_prefs
    return recall
  return f

def fpref(k, beta = 1):
  '''Returns function calculating rpref/ppref F measure at the given cutoff'''
  def f(rank_prefs):
    ppref_k = ppref(k)(rank_prefs)
    rpref_k = rpref(k)(rank_prefs)
    if ppref_k == 0 and rpref_k == 0:
      return 0
    return (1 + beta**2) * (ppref_k * rpref_k) / (beta**2 * ppref_k + rpref_k)
  return f

def appref(rank_prefs):
  '''Calculates the APpref, which is ppref@k averaged over the ranks of all the
  documents that have ever been preferred, including the the UNRANKED
  preferred documents.

  Note: This calculation differs from the description of APpref in the above
  citation, but tends to produce more sensible results when preferred documents
  are not retrieved by the system.'''
  total_preferred = num_preferred(rank_prefs)
  if total_preferred == 0:
    return 0
  ranks_to_eval = rank_prefs.preferred_ranks
  ppref_sum = sum(ppref(i)(rank_prefs) for i in rank_prefs.preferred_ranks) + \
                ppref(rank_prefs.UNRANKED)(rank_prefs) * \
                  rank_prefs.count_preferred_unranked
  return ppref_sum / num_preferred(rank_prefs)

def ppref_max(rank_prefs):
  '''Calculates the maximum ppref over all ranks'''
  if rank_prefs.preferred_ranks:
    return max(ppref(k)(rank_prefs) for k in rank_prefs.preferred_ranks)
  else:
    return 0.0

def rpref_max(rank_prefs):
  '''Calculates the maximum rpref over all ranks'''
  if rank_prefs.preferred_ranks:
    return max(rpref(k)(rank_prefs) for k in rank_prefs.preferred_ranks)
  else:
    return 0.0

def wpref(k, w_func = None):
  '''Returns a function for calculating wpref@k. assumes uniform preference degree
  (pref_ij == 1)'''
  def f(rank_prefs):
    if w_func is None:
      weight = lambda f, t: 1.0 / (log(f, 2)+1) if f < t else 0.0
    else:
      weight = w_func
    # iterate through the pref_ranks & tally up the wpref values. this includes
    # bad documents
    return sum(weight(f, t) for (f, t) in rank_prefs.pref_ranks \
                if f <= k or t <= k)
  return f

def nwpref(k):
  '''Returns a function calculating normalized wpref@k. Normalization assumes a
  perfect ranking -- i.e. every preference observed is correctly ranked.
  TODO(jelsas): this is probably not the correct way to normalize, but works
  for now'''
  def f(rank_prefs):
    unnorm = wpref(k)(rank_prefs)
    # to normalize the wpref values, we create a weighting function that always
    # counts a document pair regardless of the correct ordering of the docs.
    # TODO: this doesn't really reflect a "perfect" ordering at rank k
    norm_weight = lambda f, t: 1.0 / (log(min(f, t), 2)+1)
    norm = wpref(k, w_func = norm_weight)(rank_prefs)
    if norm == 0:
      assert unnorm == 0, 'wpref norm = 0, but wpref != 0'
      return 0
    else:
      return unnorm/norm
  return f

def rrpref(rank_prefs):
  '''Reciprocal Rank pref. 1/rank of first correctly ordered preferred doc.'''
  correctly_ranked = [f for (f, t) in rank_prefs.pref_ranks if f < t]
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
                    ('num_pref_ranked',num_pref_ranked(-1),'%d'),
                    ('num_pref_total', num_pref,           '%d'),
                    ('num_preferred',  num_preferred,      '%d'),
                    ('num_preferred_unrk',num_preferred_unranked,'%d'),
                    ('num_bad',        num_bad,            '%d'),
                    ('rrpref',         rrpref,             '%0.4f'),
                    ('ppref1',         ppref(1),           '%0.4f'),
                    ('ppref5',         ppref(5),           '%0.4f'),
                    ('ppref10',        ppref(10),          '%0.4f'),
                    ('ppref25',        ppref(25),          '%0.4f'),
                    ('ppref50',        ppref(50),          '%0.4f'),
                    ('pprefMax',       ppref_max,          '%0.4f'),
                    ('rpref1',         rpref(1),           '%0.4f'),
                    ('rpref5',         rpref(5),           '%0.4f'),
                    ('rpref10',        rpref(10),          '%0.4f'),
                    ('rpref25',        rpref(25),          '%0.4f'),
                    ('rpref50',        rpref(50),          '%0.4f'),
                    ('rprefMax',       rpref_max,          '%0.4f'),
                    ('fpref1',         fpref(1),           '%0.4f'),
                    ('fpref5',         fpref(5),           '%0.4f'),
                    ('fpref10',        fpref(10),          '%0.4f'),
                    ('wpref1',         wpref(1),           '%0.4f'),
                    ('wpref5',         wpref(5),           '%0.4f'),
                    ('wpref10',        wpref(10),          '%0.4f'),
                    ('nwpref1',        nwpref(1),          '%0.4f'),
                    ('nwpref5',        nwpref(5),          '%0.4f'),
                    ('nwpref10',       nwpref(10),         '%0.4f'),
                    ('APpref' ,        appref,             '%0.4f'),
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
    if len(q_prefs.edges) == 0 and len(q_prefs.bad_docs) == 0:
      if options.verbose: print 'skipping q', q, 'no edges'
      continue
    num_q += 1
    r_pref = ResultPreferences(docs, q_prefs, not options.intransitive)

    if options.per_q and options.verbose: print r_pref

    for (eval_name, eval_f, fmt) in eval_measures:
      m = eval_f(r_pref)
      if options.per_q:
        print '%s\t%s\t%s' % (eval_name.ljust(label_len), q, fmt % m)
      summary_measures[eval_name] += m

  print '%s\tall\t%d' % ('num_q'.ljust(label_len), num_q)
  if num_q > 0:
    for (eval_name, _, fmt) in eval_measures:
      m = summary_measures[eval_name]
      print 'm%s\tall\t%0.04f' % (eval_name.ljust(label_len), (m / num_q) )
