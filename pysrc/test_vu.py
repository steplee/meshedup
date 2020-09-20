import torch
import time
import numpy as np
import pymeshedup_c

from OpenGL.GL import *
from OpenGL.GLUT import *

from .gl_stuff import *
from .data import get_dc_lidar

def make_tex_from_img(img):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    #img = np.copy(img,'C')
    print( ' - creating tex for img ', img.shape)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1],img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    assert glGetError() == 0
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    assert glGetError() == 0
    glBindTexture(GL_TEXTURE_2D, 0)
    del img
    return tex

DC = True
RENDER_COMPLEX = True
RENDER_COMPLEX = False

if not DC:
    pts = np.random.uniform(0,1,size=(900,3)).astype(np.float32)
    #elev = np.random.uniform(0,1,size=(512,512)).astype(np.float32)
    elev = np.ones((512,512),dtype=np.float32)*.5
else:
    STRIDE = 9
    meta = get_dc_lidar({'stride':STRIDE})
    pts = meta['pts']
    #elev = np.random.uniform(0,.1,size=(512,512)).astype(np.float32)
    #elev = elev * .000001 + .01

    # Form elev by some sparse tensor magic.
    if STRIDE < 15: res = 1024
    elif STRIDE < 32: res = 512+256
    else: res = 512
    ptsPix = (pts * res).astype(np.int64)
    coo = torch.from_numpy(ptsPix[:,:2])
    val = torch.from_numpy(pts[:,2])
    one = torch.ones_like(val)
    x = torch.cuda.sparse.FloatTensor(coo.T, val).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one).coalesce()
    x.values().copy_(x.values()/cnt.values()) # Now we have average values
    elev = x.to_dense().cpu().numpy()
    elev = cv2.medianBlur(elev, 5)
    del x, cnt, coo, val

dt_opts = pymeshedup_c.DTOpts()
dt_opts.createMesh = False
dt = pymeshedup_c.DelaunayTetrahedrialization(dt_opts)
st = time.time()
dt.run(pts)
print(' - dt took {:1f}s'.format(time.time()-st))
#dt.mesh.bake(True)
import gc
gc.collect()

vu = pymeshedup_c.VuMeshing(dt,1)
vu.runWithElevationMap(elev)



app = OctreeApp((1000,1000))
app.init(True)
glEnable(GL_CULL_FACE)
#glDisable(GL_CULL_FACE)

if DC:
    uvs = np.copy(vu.mesh.verts[:, :2], 'C')
    uvs[:,1] = 1. - uvs[:,1]
    vu.mesh.uvs = uvs
    vu.mesh.tex = make_tex_from_img(meta['img'])
print(' - Vu Mesh:')
vu.mesh.print()

vu.mesh.bake(True)
if RENDER_COMPLEX: vu.assignmentMesh.bake(True)

for i in range(100000):
    app.updateCamera(.01)
    app.render()

    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    draw_gizmo(2)

    glColor4f(0,0,1,.5)


    glColor4f(1.,1,1,1)
    vu.mesh.render()
    glLineWidth(1.0)
    if RENDER_COMPLEX: vu.assignmentMesh.render()
    glLineWidth(1.0)

    '''
    if pts is not None:
        glColor4f(.6, .6, .99, .15)
        glEnableClientState(GL_VERTEX_ARRAY)
        glPointSize(1)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        glPointSize(1)
        glDrawArrays(GL_POINTS, 0, len(pts))
        glDisableClientState(GL_VERTEX_ARRAY)
    '''

    time.sleep(.008)
    glutSwapBuffers()
    glutPostRedisplay()
    glutMainLoopEvent()
    glFlush()
