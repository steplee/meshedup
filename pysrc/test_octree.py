import torch
import numpy as np
import pymeshedup_c

from OpenGL.GL import *
from OpenGL.GLUT import *
from .gl_stuff import *

depth = 15
pts = np.random.uniform(0,1,size=(4000,3)).astype(np.float32)
tree = pymeshedup_c.IntegralOctree(depth,pts)
print(pts)

def search_it(loc, depth):
    print( ' - [search {}, {}] [get {}]'.format(loc,depth,repr(tree.searchNode(loc,depth))))


def print_children(n,recurse=False,depth=0):
    for ii in range(8):
        i,j,k = ii&1,(ii&2)>>1,(ii&4)>>2
        c = n.child(ii)
        if c is not None: print(' '*depth, ' - {}{}{}: {}'.format(i,j,k,repr(c)), sep='')
        if c is not None and recurse: print_children(c,recurse, depth+2)

def find_ids(node):
    if node is None: return []
    ids = []
    if node.id >= 0: ids.append(node.id)
    for i in range(8): ids.extend(find_ids(node.child(i)))
    return ids

def find_locs(node):
    if node is None: return []
    locs = []
    if node.id >= 0: locs.append(node.loc)
    for i in range(8): locs.extend(find_locs(node.child(i)))
    return locs

search_it((0,0,0),0)
search_it((0,0,0),1)
search_it((0,1,0),1)

root = tree.searchNode((0,0,0),0)
print_children(root, True)

found_ids = sorted(find_ids(root))
assert found_ids == list(range(len(pts)))
print(' - found all ids')


locs = find_locs(root)
for loc in locs:
    n = tree.searchNode(loc, depth)
    assert all(n.loc == loc)
print(' - searched all nodes by location successfully')
print(tree)

app = OctreeApp((1600,1000))
app.init(True)
glEnable(GL_CULL_FACE)
for i in range(100000):
    app.updateCamera(.01)
    app.render()
    glColor4f(0,0,1,.5)

    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
    #tree.render(10)
    tree.render2()

    glBegin(GL_LINES)
    s = 2
    glColor4f(1,0,0,0)
    glVertex3f(0,0,0)
    glColor4f(1,0,0,1)
    glVertex3f(s,0,0)
    glColor4f(0,1,0,0)
    glVertex3f(0,0,0)
    glColor4f(0,1,0,1)
    glVertex3f(0,s,0)
    glColor4f(0,0,1,0)
    glVertex3f(0,0,1)
    glColor4f(0,0,1,1)
    glVertex3f(0,0,s)
    glEnd()

    glEnableClientState(GL_VERTEX_ARRAY)
    if pts is not None:
        glColor4f(.6, .6, .99, .8)
        glPointSize(2)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        #glEnableClientState(GL_COLOR_ARRAY)
        #glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(pts))
        #glDisableClientState(GL_COLOR_ARRAY)

    time.sleep(.008)
    glutSwapBuffers()
    glutPostRedisplay()
    glutMainLoopEvent()
    glFlush()
