import torch, torch.nn.functional as F, torch.nn as nn
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
    img = np.copy(img,'C')
    print( ' - creating tex for img ', img.shape)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1],img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    assert glGetError() == 0
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    assert glGetError() == 0
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

def show_normalized(x, name='a',time=0):
    from matplotlib.cm import cubehelix as inferno
    if isinstance(x,torch.Tensor): x = x.cpu().numpy()
    x = x.squeeze().astype(np.float32)
    assert x.ndim == 2
    x = x - np.quantile(x,.01)
    #x = np.clip(x / x.max(),0,1)
    x = np.clip(x / np.quantile(x,.99),0,1)
    x = (inferno(x)[...,:3]*255).astype(np.uint8)[...,[2,1,0]]
    cv2.imshow(name,x)
    cv2.waitKey(time)

def Laplacian(device=None):
    s = nn.Conv2d(1,2,3,padding=1, bias=False)
    s.requires_grad_(False)
    s.weight.data = torch.cuda.FloatTensor([
        [[[1,1,1], [1,-8,1], [1,1,1]]],
        #[[[0,1,0], [1,-4,1], [0,1,0]]],
        ]).to(device=device)
    return s

def optimize_mrf_pytoch(elev_, bad):
    bad = torch.from_numpy(bad).cuda()
    print(elev_.shape, elev_.dtype)
    elev = torch.from_numpy(elev_).clone()
    elev = torch.autograd.Variable(elev, requires_grad=False).cuda()
    x = torch.from_numpy(elev_).clone().cuda()
    #x = x + torch.randn_like(x) * .01
    x = torch.autograd.Variable(x, requires_grad=True)
    opt = torch.optim.Adam([x], lr=.0003)
    #opt = torch.optim.SGD([x], lr=1009)
    median = torch.from_numpy(cv2.medianBlur(elev_,5)).cuda()
    print(x)
    lap = Laplacian(x.device)
    #for i in range(1000):
    for i in range(1000):
        d_cost = abs(x - elev) #** 2
        #d_cost = d_cost * (1-bad)
        if False:
            k = 3
            s_cost = torch.min( abs(x - F.max_pool2d(x.unsqueeze(0).unsqueeze(0),k,1,k//2)[0,0]).clamp(0,.01),
                                abs(x + F.max_pool2d(-x.unsqueeze(0).unsqueeze(0),k,1,k//2)[0,0]).clamp(0,.01)) * 9
            #s_cost += abs(lap(x.unsqueeze(0).unsqueeze(0)).sum(0).sum(0)).clamp(0,.04)
            s_cost += abs(lap(x.unsqueeze(0).unsqueeze(0)).mean(0).mean(0))
        else:
            s_cost = F.conv2d(x.unsqueeze(0).unsqueeze(0), torch.eye(9,device=x.device).reshape(9,1,3,3))[0]
            s_cost = s_cost.max(0).values - s_cost.min(0).values
            #s_cost = s_cost.pow(2).mean() * 20
            s_cost = s_cost.mean()
        cost = d_cost + s_cost
        cost = cost * (1-bad)
        cost = cost.mean()
        cost.backward()
        opt.step()
        print(cost.item())
        opt.zero_grad()
        if i % 50 == 0:
            dimg = torch.cat((
                #elev.detach().cpu()[512:-2,2:512],
                #x.detach().cpu()[512:-2,2:512]), 1).numpy()
                elev.detach().cpu(), x.detach().cpu()), 1).numpy()
            show_normalized(dimg,time=1)
    return x.cpu().detach().numpy()

DC = True
RENDER_COMPLEX = True

if not DC:
    pts = np.random.uniform(0,1,size=(900,3)).astype(np.float32)
    elev = np.random.uniform(0,1,size=(512,512)).astype(np.float32)
    #elev = np.ones((512,512),dtype=np.float32)*.5
else:
    STRIDE = 2
    meta = get_dc_lidar({'stride':STRIDE})
    pts = meta['pts']
    #elev = np.random.uniform(0,.1,size=(512,512)).astype(np.float32)
    #elev = elev * .000001 + .01

    # Form elev by some sparse tensor magic.
    #if STRIDE < 15: res = 1024
    #elif STRIDE < 32: res = 512+256
    #else: res = 512
    res = 512
    #res = 256
    res = 1024
    ptsPix = (pts * res).astype(np.int64)
    coo = torch.from_numpy(ptsPix[:,:2])
    val = torch.from_numpy(pts[:,2])
    one = torch.ones_like(val)
    x = torch.cuda.sparse.FloatTensor(coo.T, val).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one).coalesce()
    x.values().copy_(x.values()/cnt.values()) # Now we have average values
    elev = np.copy(x.to_dense().cpu().numpy(),'C')
    badElev = np.copy((cnt.to_dense().cpu()==0).to(torch.float32).numpy(),'C')
    #badElev = np.copy(((cnt.to_dense().cpu()==0)*0).to(torch.float32).numpy(),'C')
    #show_normalized(badElev, 'BAD')
    #elev = cv2.medianBlur(elev, 5)



if True:
    final = optimize_mrf_pytoch(elev,badElev)
    #final = cv2.medianBlur(elev,5)
    surfacer = pymeshedup_c.EnergySurfacing2d()
    surfacer.output = final
    Z_MULT = 1
else:
    Z_MULT = 400 * (res//512)
    elev = elev * Z_MULT
    print(' - elev:', elev)
    surfacer = pymeshedup_c.EnergySurfacing2d()
    surfacer.dataShape = pymeshedup_c.SHAPE_ABSOLUTE
    surfacer.dataBoundaryCost = 99999
    surfacer.smoothMult = 4
    surfacer.runWithElevationMap(elev, badElev)
    final = np.copy(surfacer.output, 'C')

st = time.time()
#surfacer.make_mesh()
surfacer.make_mesh_simplified()
print(' - make_mesh took {:.1f}s'.format(time.time()-st))
#elev1 = np.copy(surfacer.output, 'C')
#surfacer.runWithElevationMap(elev1, ((badElev==1)&(elev1==elev)).astype(np.float32))
elev_ = cv2.medianBlur(elev, 3)
dimg = np.hstack((elev_,final)) / Z_MULT
while dimg.shape[0] > 1024: dimg = cv2.pyrDown(dimg)
show_normalized(dimg, 'medianFilter(orig)/output', 1)
show_normalized(abs(elev-final), '|orig-output|')

print(' - Done')
#sys.exit(0)


app = OctreeApp((1000,1000))
app.init(True)
glEnable(GL_CULL_FACE)
#glDisable(GL_CULL_FACE)

if DC:
    uvs = np.copy(surfacer.mesh.verts[:, :2], 'C')
    #uvs[:,1] = 1. - uvs[:,1]
    uvs[:,0] = 1. - uvs[:,0]
    uvs[:,[0,1]] = uvs[:,[1,0]]
    surfacer.mesh.uvs = uvs
    surfacer.mesh.tex = make_tex_from_img(meta['img'])
print(' - Mesh:')
surfacer.mesh.print()

surfacer.mesh.bake(True)

for i in range(100000):
    app.updateCamera(.01)
    app.render()

    glDisable(GL_BLEND)
    #glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    draw_gizmo(2)

    glColor4f(0,0,1,.5)


    glColor4f(1.,1,1,1)
    surfacer.mesh.render()
    glLineWidth(1.0)
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

