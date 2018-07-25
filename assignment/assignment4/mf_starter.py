# Using graphlab create to implement SGD for matrix factorization
import graphlab
from graphlab import SFrame
from graphlab import SGraph
import numpy as np
import matplotlib.pyplot as plt

# Reads matrix in mtx format into a numpy array
def read_matrix(filename):
  f = open(filename)
  f.readline()
  n, m = map(int, f.readline().split())[:2]
  M = np.zeros((n+ 1, m + 1))
  for line in f:
    i, j, v = line.split(' ')
    M[int(i),int(j)] = float(v)
  return M

k = 5
M = read_matrix('data/SubSampleTrainingSet.txt')
n, m = M.shape
val = read_matrix('data/SubSampleValidationSet.txt')
test = read_matrix('data/SubSampleTestSet.txt')


global lambda_u, lambda_v
def sgd_update(src, edge, dst):
  eta = .1
  global lambda_u, lambda_v
  Lu = np.array(src['factors'])
  Rv = np.array(dst['factors'])
  ruv = edge['rating']
  # TODO: Your code goes in here. Implement the Stochastic Gradient Descent
  # update. Remember that src is the user node, dst is the movie node.
  # Lu and Rv are kept as row vectors, but don't worry about that.
  # You should update 
  # src['factors'] = NEW value for Lu
  # dst['factors'] = NEW value for Rv
  # Before returning from this function

  return (src, edge,dst)

# Example usage:
# rmse_train[0.1] = [r1, r2, r3, ...]
# where 0.1 is the regularization parameter (lambda),
# r1 is the RMSE on the training data after 1 pass over the data,
# r2 is the RMSE on the training data after 2 passes over the data,
# etc
rmse_train = {}
# Same thing, but for validation data
rmse_val = {}
# Same thing, but for test data
rmse_test = {}
lambs = [0,0.001,0.01,0.1,1]


# You should not have to edit any of the code below, except to plot figures.
for l in lambs:
  g = graphlab.load_graph('data/training_graph.sgraph')
  n, m = M.shape
  L = np.ones((n + 1, k))
  R = np.ones((k, m + 1))
  lambda_u = lambda_v = l
  rmse_train[l] = []
  rmse_val[l] = []
  rmse_test[l] = []
  # Get the initial rmse, before we do anything
  rmse = np.sqrt(sum((M[M.nonzero()] - L.dot(R)[M.nonzero()]) ** 2) / len(M[M.nonzero()]))
  rmse_train[l].append(rmse)
  rmse = np.sqrt(sum((val[val.nonzero()] - L.dot(R)[val.nonzero()]) ** 2) / len(val[val.nonzero()]))
  rmse_val[l].append(rmse)
  rmse = np.sqrt(sum((test[test.nonzero()] - L.dot(R)[test.nonzero()]) ** 2) / len(test[test.nonzero()]))
  rmse_test[l].append(rmse)
  print 'Lambda = ' , l
  # Each iteration here is a pass over the dataset
  for i in range(10):
    # This runs SGD over the training data
    g = g.triple_apply(sgd_update, 'factors')

    # This updates a copy of L and R in np.array format. These have the proper
    # dimensions (n+1,k) and (k, m+1). The +1 is just because the users start at
    # 1, while numpy arrays are indexed starting at 0.
    u = g.get_vertices(fields={'user':1})
    L[np.array(u['__id'], dtype=int)] = np.array(u['factors'])
    m = g.get_vertices(fields={'user':0})
    mids = [x[1:] for x in m['__id']]
    R[:, np.array(mids, dtype=int)] = np.array(m['factors']).T

    # This measures the RMSE in the training, validation and test sets.
    rmse = np.sqrt(sum((M[M.nonzero()] - L.dot(R)[M.nonzero()]) ** 2) / len(M[M.nonzero()]))
    rmse_train[l].append(rmse)
    rmse = np.sqrt(sum((val[val.nonzero()] - L.dot(R)[val.nonzero()]) ** 2) / len(val[val.nonzero()]))
    rmse_val[l].append(rmse)
    rmse = np.sqrt(sum((test[test.nonzero()] - L.dot(R)[test.nonzero()]) ** 2) / len(test[test.nonzero()]))
    rmse_test[l].append(rmse)
