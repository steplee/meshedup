import torch
import numpy as np
import pymeshedup_c

from OpenGL.GL import *
from OpenGL.GLUT import *

from .gl_stuff import *
from .data import get_dc_lidar

depth = 12
pts = get_dc_lidar({'stride':8})['pts']
tree = pymeshedup_c.IntegralOctree(depth,pts)

app = OctreeApp((1000,1000))
app.init(True)
glEnable(GL_CULL_FACE)

tstMesh = pymeshedup_c.IndexedMesh()
tstMesh.verts = np.random.randn(50,3).astype(np.float32)
tstMesh.inds = np.arange(0,50).astype(np.uint32)
tstMesh.mode = GL_LINES
tstMesh.bake(False)
tstMesh.bake(False)

for i in range(100000):
    app.updateCamera(.01)
    app.render()
    glColor4f(0,0,1,.5)

    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
    #tree.render(10)
    tree.render2()

    draw_gizmo(2)

    glEnableClientState(GL_VERTEX_ARRAY)
    if pts is not None:
        glColor4f(.6, .6, .99, .8)
        glPointSize(2)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        #glEnableClientState(GL_COLOR_ARRAY)
        #glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(pts))
        #glDisableClientState(GL_COLOR_ARRAY)

    glColor4f(1.,0,0,1)
    tstMesh.render()

    time.sleep(.008)
    glutSwapBuffers()
    glutPostRedisplay()
    glutMainLoopEvent()
    glFlush()
