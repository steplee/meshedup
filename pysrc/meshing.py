import torch, torch.nn as nn, torch.nn.functional as F
import cv2, time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

#from .meshing import get_dc_lidar_data, run_with_data
from .data import get_dc_lidar

from matplotlib.cm import inferno

from triangle import triangulate, compare
import matplotlib.pyplot as plt

def Sobel(device=None):
    s = nn.Conv2d(1,2,3,padding=1, bias=False)
    s.requires_grad_(False)
    s.weight.data = torch.cuda.FloatTensor([
        [[[1,0,-1], [2,0,-2], [1,0,-1]]],
        [[[1,2,1], [0,0,0],  [-1,-2,-1]]],
        ]).to(device=device)
    return s
def Laplacian(device=None):
    s = nn.Conv2d(1,2,3,padding=1, bias=False)
    s.requires_grad_(False)
    s.weight.data = torch.cuda.FloatTensor([
        #[[[1,1,1], [1,-8,1], [1,1,1]]],
        [[[0,1,0], [1,-4,1], [0,1,0]]],
        ]).to(device=device)
    return s

def show_normalized(x, name='a',time=0):
    if isinstance(x,torch.Tensor): x = x.cpu().numpy()
    x = x.squeeze().astype(np.float32)
    assert x.ndim == 2
    x = x - x.min()
    x = x / x.max()
    x = (inferno(x)[...,:3]*255).astype(np.uint8)[...,[2,1,0]]
    cv2.imshow(name,x)
    cv2.waitKey(time)

def explore2(pts, ptsPix):
    coo = ptsPix[:,:2]
    val = pts[:,2]
    one = torch.ones_like(val)
    x = torch.cuda.sparse.FloatTensor(coo.T, val).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one).coalesce()
    x.values().copy_(x.values()/cnt.values()) # Now we have average values
    show_normalized(x.to_dense().cpu(),'x',0)

    y = x.to_dense().cpu().numpy()
    y = cv2.medianBlur(y, 5)
    response = abs(cv2.Laplacian(y,cv2.CV_32F))
    y0 = cv2.Sobel(y,cv2.CV_32F,1,0)
    y1 = cv2.Sobel(y,cv2.CV_32F,0,1)
    response = np.sqrt(y0**2 + y1**2) * 1000
    show_normalized(response,'response',0)

    #response = cv2.pyrDown(response)
    response = cv2.pyrDown(response)
    verts = np.argwhere(response>np.quantile(response,.8))
    verts = verts * 2
    #verts = verts * 2
    edges=None

    from triangle import triangulate, compare
    import matplotlib.pyplot as plt
    A = dict(vertices=verts)
    T = triangulate(A)
    compare(plt, A, T)
    plt.show()
    pass

# segment from watershed ->
# extract connected components per level ->
# triangulate resulting planes using triangle ->
# extrude each level above the one underneath.
def explore3(pts, ptsPix, baseRes, pix2meter):
    coo = ptsPix[:,:2]
    val = pts[:,2]
    one = torch.ones_like(val)
    x = torch.cuda.sparse.FloatTensor(coo.T, val).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one).coalesce()

    x.values().copy_(x.values()/cnt.values()) # Now we have average values
    elev = x.to_dense().cpu()
    elevq = (x.to_dense().cpu() * baseRes).to(torch.int32)
    show_normalized(elev,'elev',0)

    # Good starts with where the are samples, and removed as watershed progresses
    good = cnt.to_dense().cpu(); good = (good>0)
    show_normalized(good.cpu(),'good',0)

    min_e, max_e = np.quantile(elevq, .005), np.quantile(elevq, .995)
    resolution = 4
    MIN_CC_SIZE = 16
    # Watershed + CCs
    lvl = np.zeros_like(good).astype(np.float32)
    for ii,z in enumerate(range(int(max_e+.5), int(min_e), -resolution)):
        #on = (elevq <= z) & (elevq > z-resolution) & (good)
        if z == int(min_e):
            on = (good)
        else:
            on = (elevq > z-resolution) & (good)
        #cv2.imshow('on',on.to(torch.uint8).cpu().numpy()*200); cv2.waitKey(0)
        nccs,ccs = cv2.connectedComponents(on.cpu().numpy().astype(np.uint8))
        #cc_colors = cv2.cvtColor(np.stack( (ccs,np.ones_like(ccs)*255,np.ones_like(ccs)*150) , -1 ).astype(np.uint8), cv2.COLOR_HSV2RGB)
        #cv2.imshow('ccs',cc_colors); cv2.waitKey(0)

        sizes,bins = np.histogram(ccs, nccs) # Might be wrong
        good_sizes = np.argwhere(sizes > MIN_CC_SIZE).squeeze()
        taken = np.zeros_like(ccs)
        for lbl in good_sizes:
            if lbl > 0:
                taken[ccs==lbl] = 1
                lvl[ccs==lbl] = ii+1
        good = (good & (~taken)).to(torch.bool)
        #cv2.imshow('taken',taken*200); cv2.waitKey(0)

        '''
        vertices = []
        regions = []
        for lbl in good_sizes:
            if lbl > 0:
                verts = np.argwhere(ccs==lbl)
                vertices.extend(verts)
                v = verts[0]
                regions.append([*(v+.01), lbl, 0])
        holes = [[0.1,0.1]]
        vertices = np.stack(vertices).astype(np.float32)
        A = dict(vertices=vertices, regions=regions, holes=holes)
        T = triangulate(A)
        compare(plt, A, T)
        plt.show()
        '''
        for lbl in good_sizes:
            if lbl > 0:
                verts = np.argwhere(ccs==lbl)
                v = verts[0]
                regions = [[*(v+.01), lbl, 0]]
                holes = [[0.1,0.1]]
                A = dict(vertices=verts, regions=regions, holes=holes)
                T = triangulate(A)
                compare(plt, A, T)
                plt.show()


    lvl_colors = ((lvl.astype(np.float32) / (lvl.max()+1)) * 195).astype(np.uint8)
    lvl_colors = cv2.cvtColor(np.stack( (lvl_colors,np.ones_like(lvl_colors)*255,np.ones_like(lvl_colors)*250) , -1 ), cv2.COLOR_HSV2BGR)
    lvl_colors[lvl==0] = 0
    cv2.imshow('lvls',lvl_colors); cv2.waitKey(0)


# watershed and insert points below ->
# run MC.
def explore4(pts, ptsPix, baseRes, pix2meter):
    pass

# Create RBF points above surface (not along normals, since they are all ~Z=1)
# Run surface-following 'wavefront' meshing algo on the field defined by RBF
# http://mesh.brown.edu/DGP/pdfs/Carr-sg2001.pdf
def explore4(pts, ptsPix, baseRes, pix2meter):
    pass

def explore5(pts, ptsPix, baseRes, pix2meter):
    coo = ptsPix[:,:2]
    val = pts[:,2]
    one = torch.ones_like(val)
    x = torch.cuda.sparse.FloatTensor(coo.T, val).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one).coalesce()

    x.values().copy_(x.values()/cnt.values()) # Now we have average values
    elev = x.to_dense().cpu()

    d = Sobel(elev.device)(elev.unsqueeze(0).unsqueeze(0))[0].norm(dim=0)
    #d = Laplacian(elev.device)(elev.unsqueeze(0).unsqueeze(0))[0].norm(dim=0)
    #d = Laplacian(d.device)(d.unsqueeze(0).unsqueeze(0))[0].abs()
    on = (d > 15/pix2meter).to(torch.float32)
    show_normalized(on, 'on', 0)



# Takes a sparse tensor, adds new entries
# where there looks to be a discontinuity along XY plane.
def fill_pts(height0, pts, eps, res, sparseSize):
    height = height0 * res
    #d = Sobel(height.device)(height.unsqueeze(0).unsqueeze(0))[0].norm(dim=0)
    d = Laplacian(height.device)(height.unsqueeze(0).unsqueeze(0))[0,0].abs()
    #print('d',d.cpu().numpy())
    #print(' - eps', eps)
    hmin, hmax = height.min().item(), height.max().item()
    print(' - min/max height {} {} ({} slices)'.format(hmin,hmax,hmax-hmin))
    isEdge = (d>.7)
    show_normalized(isEdge.to(torch.float32))

    newCoo = []

    hmin, hmax = int(hmin), int(hmax)
    for z in range(hmin, hmax):
        occ = (height > z) * (isEdge)
        coo = torch.where(occ==True)
        coo = torch.stack((*coo,torch.empty_like(coo[0]).fill_(z)))
        newCoo.append(coo)
        print(' - z {}: {} pts'.format(z,coo.shape[1]))
        print(coo)
        #show_normalized(occ.to(torch.float32))

    newCoo = torch.cat(newCoo,1).to(height.device)
    val = torch.ones(newCoo.size(1), dtype=torch.float32, device=newCoo.device)
    #newPts = torch.cuda.sparse.FloatTensor(newCoo,val, sparseSize)

    out = torch.cuda.sparse.FloatTensor(
            torch.cat((pts.indices(),newCoo),1),
            torch.cat((pts.values(),val)), sparseSize).coalesce()

    #out = torch.cat((pts,newPts)).coalesce()
    nnew = 1. - (len(pts.values())/len(out.values()))
    print(' - have {} new points (from {}, {:.1f}% new)'.format(
        len(val), len(pts.values()), 100*nnew))

    return out

def main():
    #meta = get_dc_lidar_data({'stride':2})
    #res = 1024//1

    meta = get_dc_lidar({'stride':2,'qq':.2})
    res = 1*512//1
    sparseSize = torch.Size((res,res,res))

    pts0 = meta['pts']
    pts    = (torch.from_numpy(pts0).cuda())
    ptsPix = (pts * res).to(torch.int64)
    print(pts, ptsPix)

    #explore1(pts,ptsPix)
    #explore2(pts,ptsPix)
    explore3(pts,ptsPix, res, meta['pix2meter'])
    #explore5(pts,ptsPix, res, meta['pix2meter'])
    return

    # Create (dense) 2d heightmap
    coo = ptsPix[:,:2]
    val = pts[:,2]
    one = torch.ones_like(val)
    hx = torch.cuda.sparse.FloatTensor(coo.T, val, sparseSize[:2]).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one, sparseSize[:2]).coalesce()
    hx.values().copy_(hx.values()/cnt.values()) # Now we have average values
    hx = hx.to_dense().cpu().numpy()
    hx = cv2.medianBlur(hx, 5)
    hx = torch.from_numpy(hx).cuda()

    # Create (sparse) 3d pts
    coo = ptsPix
    val = torch.ones(coo.size(0), dtype=torch.float32, device=coo.device)
    one = torch.ones_like(val)
    px = torch.cuda.sparse.FloatTensor(coo.T, val, sparseSize).coalesce()
    cnt = torch.cuda.sparse.FloatTensor(coo.T, one, sparseSize).coalesce()
    px.values().copy_(px.values()/cnt.values()) # Now we have average values

    # Add more 3d pts wherever there is a discontinuity (e.g. up the side of a building)
    #eps = meta['maxEdge'] / res # Epsilon is grid size (convert meters to 'pix')
    #eps = 1 / res # Epsilon is grid size (convert meters to 'pix')
    eps = 1 # Epsilon is grid size (convert meters to 'pix')
    px = fill_pts(hx, px, eps, res, sparseSize)

    # Create negative points.

    pts1 = ((px.indices().to(torch.float32) / res) + (.0/res)).cpu().numpy()
    meta['pts'] = pts1.T
    meta['vals'] = np.ones(len(meta['pts']),dtype=np.float32)

    meta['maxDepth'] = 8

    print(px.indices().to(torch.float32))
    print(px.indices().to(torch.float32)/res)
    #print(torch.histc(px.values()))
    run_with_data(meta)



if __name__ == '__main__':
    main()
