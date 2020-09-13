import torch
import subprocess
import time
import numpy as np
import pymeshedup_c

# https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
g = [
        (0,1,3),
        (0,3,3),
        (1,2,4),
        (2,0,3),
        (2,3,1),
        (2,4,2),
        (3,4,2),
        (3,5,6),

        (4,2,1),
        (4,6,1),
        (5,6,9)
    ]

if False:
    mfmc = pymeshedup_c.MaxFlowMinCut(7)
    mfmc.setSourceSink(0,6)
    for a,b,c in g:
        mfmc.addEdge(a,b,c)

else:
    g = np.random.randint(0,20, size=(100,3))
    mfmc = pymeshedup_c.MaxFlowMinCut(20)
    mfmc.setSourceSink(0,20-1)
    for a,b,c in g:
        mfmc.addEdge(a,b,c)

print(' - Running')
mfmc.run()
print('********************')
print(' - MaxFlow: {}'.format(mfmc.maxFlow))
print(' - S:', mfmc.minCutS)
print(' - T:', mfmc.minCutT)
print(' - cut edges:', mfmc.getMinCutEdges())
print('********************')

with open('mfmc.dot','w') as fp:
    print(mfmc.printViz(), file=fp)
print(' - created mfmc.dot')
subprocess.getoutput('dot -Tjpg mfmc.dot -o mfmc.jpg')
print(' - created mfmc.jpg')
