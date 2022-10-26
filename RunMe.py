# Code for Label Informed Attributed Network Embedding
#
#   Copyright 2017, Xiao Huang and Jundong Li.
#   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $

import numpy as np
Dataset = 'BlogCatalog'
if str(Dataset) == str('BlogCatalog'):
    scipy.io.loadmat('BlogCatalog.mat')
    alpha1 = 43
    alpha2 = 36
    numiter = 5
    delta1 = 0.97
    delta2 = 1.6
else:
    if str(Dataset) == str('Flickr'):
        scipy.io.loadmat('Flickr.mat')
        alpha1 = 10 ** 0.8
        alpha2 = 100
        numiter = 4
        delta1 = 0.3
        delta2 = 2.3

d = 100

G = Network
A = Attributes
n,__ = G.shape

G[np.arange[1,n ** 2+n + 1,n + 1]] = 1
Y = []
LabelIdx = unique(Label)

for n_Label_i in np.arange(1,len(LabelIdx)+1).reshape(-1):
    Y = np.array([Y,Label == LabelIdx(n_Label_i)])

Y = Y * 1
Indices = randi(20,n,1)

Group1 = find(Indices <= 16)

Group2 = find(Indices >= 17)

## Training group
G1 = sparse(G(Group1,Group1))

A1 = sparse(A(Group1,:))

Y1 = sparse(Y(Group1,:))

## Test group
A2 = sparse(A(Group2,:))

GC1 = sparse(G(Group1,:))

GC2 = sparse(G(Group2,:))

## Label Informed Attributed Network Embedding (Supervised)
print('Label informed Attributed Network Embedding (LANE), 5-fold with 100% of training is used:')
H1 = LANE_fun(G1,A1,Y1,d,alpha1,alpha2,numiter)

H2 = delta1 * (GC2 * pinv(pinv(H1) * GC1)) + delta2 * (A2 * pinv(pinv(H1) * A1))

F1macro1,F1micro1 = Performance(H1,H2,Label(Group1,:),Label(Group2,:))
## Unsupervised Attributed Network Embedding (LANE w/o Label)
print('Unsupervised Attributed Network Embedding (LANE w/o Label):')
if str(Dataset) == str('BlogCatalog'):
    # Parameters of BlogCatalog in Unsupervised
    beta1 = 8
    beta2 = 0.1
    numiter = 3
    delta1 = 1.4
    delta2 = 1
else:
    if str(Dataset) == str('Flickr'):
        # Parameters of BlogCatalog in Unsupervised
        beta1 = 0.51
        beta2 = 0.1
        numiter = 2
        delta1 = 0.55
        delta2 = 2.1

H1 = LANE_fun(G1,A1,d,beta1,beta2,numiter)

H2 = delta1 * (GC2 * pinv(pinv(H1) * GC1)) + delta2 * (A2 * pinv(pinv(H1) * A1))

F1macro2,F1micro2 = Performance(H1,H2,Label(Group1,:),Label(Group2,:))