import torch
import time
import numpy as np
import pymeshedup_c

from OpenGL.GL import *
from OpenGL.GLUT import *

from .gl_stuff import *
from .data import get_dc_lidar

app = OctreeApp((1000,1000))
app.init(True)
glEnable(GL_CULL_FACE)
#glDisable(GL_CULL_FACE)

#pts = np.random.uniform(0,1,size=(50000,3)).astype(np.float32)
pts = get_dc_lidar({'stride':16})['pts']

dt_opts = pymeshedup_c.DTOpts()
dt = pymeshedup_c.DelaunayTetrahedrialization(dt_opts)
st = time.time()
dt.run(pts)
print(' - dt took {:1f}s'.format(time.time()-st))
print(' - DT Mesh:')
dt.mesh.print()
dt.mesh.bake(True)

for i in range(100000):
    app.updateCamera(.01)
    app.render()

    glEnable(GL_BLEND)

    draw_gizmo(2)

    glColor4f(0,0,1,.5)

    glEnableClientState(GL_VERTEX_ARRAY)
    if pts is not None:
        glColor4f(.6, .6, .99, .8)
        glPointSize(1)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        #glEnableClientState(GL_COLOR_ARRAY)
        #glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(pts))
        #glDisableClientState(GL_COLOR_ARRAY)

    glColor4f(1.,1,1,.3)
    dt.mesh.render()

    time.sleep(.008)
    glutSwapBuffers()
    glutPostRedisplay()
    glutMainLoopEvent()
    glFlush()
