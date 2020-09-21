import torch
import time
import gc
import pymeshedup_c
import numpy as np

def get_tensor_type(device,dtype):
    mod = torch.cuda.sparse if device.type == 'cuda' else torch.sparse
    if dtype == torch.float32: return mod.FloatTensor
    if dtype == torch.int64: return mod.LongTensor
    if dtype == torch.int8 or dtype == torch.uint8: return mod.ByteTensor
    assert False

def make_averaged_sparse_tensor(inds, vals, size=None):
    assert vals.ndimension() == 2
    ones = torch.ones((inds.size(1),1), device=inds.device)
    # Last value dim will hold the count of that index (that we will divide by)
    vals = torch.cat( (vals,ones), 1 )
    T = torch.cuda.sparse.FloatTensor if inds.device.type=='cuda' else torch.sparse.FloatTensor
    if size is None:
        tval = T(inds,vals).coalesce()
    else:
        size_ = torch.Size((*size[:3],size[3]+1))
        tval = T(inds,vals,size_).coalesce()
    size = torch.Size((*tval.size()[:3], tval.size(3)-1))
    return T(tval.indices(), tval._values()[:,:-1].div_(tval._values()[:,-1:]), size)._coalesced_(True)

def shift_tensor(a, dx,dy,dz):
    inds = a.indices()
    vals = a.values()
    I,T = (torch.cuda.LongTensor,torch.cuda.sparse.FloatTensor) \
          if inds.device.type=='cuda' else (torch.LongTensor,torch.sparse.FloatTensor)
    inds = inds+I((dx,dy,dz)).view(3,-1)

    # We only support square sizes, because it is slightly more efficient.
    assert a.size(0) == a.size(1) and a.size(1) == a.size(2)
    # Is there a faster way to do this? We must drop out-of-bounds indices+values.
    good = ((inds<a.size(0)) & (inds>0)).all(0)
    inds = inds[:, good]
    vals = vals[good]
    return T(inds, vals, a.size())

# TODO: We can map a->b or b->8a, which is what I do with this, but is less efficient.
def scale_tensor_replicate(a, val=1):
    inds,vals = a.indices(), a.values()
    I,T = (torch.cuda.LongTensor,torch.cuda.sparse.FloatTensor) \
          if inds.device.type=='cuda' else (torch.LongTensor,torch.sparse.FloatTensor)
    offs = I((1,0,0, 0,1,0, 0,0,1, 1,1,0, 1,1,1, 0,1,1, 0,0,0, 1,0,1)).view(8,3).T.unsqueeze_(2).contiguous()
    inds = ((inds * 2).unsqueeze_(1) + offs).view(3,-1)
    assert a.size(0) == a.size(1) and a.size(1) == a.size(2)
    good = (inds<a.size(0)).all(0)
    inds = inds[:, good]
    vals = torch.ones((inds.size(1),vals.size(1)),device=vals.device).fill_(val)
    size = torch.Size((a.size(0)*2, a.size(1)*2, a.size(2)*2, *a.size()[3:]))
    #return T(inds, vals, size).coalesce()
    return T(inds, vals, size)._coalesced_(True)

def cat_tensor(tup):
    inds = torch.cat(tuple(t.indices() for t in tup),1)
    vals = torch.cat(tuple(t.values() for t in tup),0)
    T = torch.cuda.sparse.FloatTensor if inds.device.type=='cuda' else torch.sparse.FloatTensor
    return T(inds,vals,tup[0].size()).coalesce()
def balance_octree(lvls):
    for ii in range(len(lvls)-1):
        a,b = lvls[ii], lvls[ii+1]
        acc_missing = []
        # My strategy here is to map the lower-res to higher res and replicate each lower-res 8 times
        # Then, we add that tensor to the shifted hi-res one and check for neighbors in it.
        # TODO: Vectorize the outer-loop
        # TODO: I think this is wrong, you must also add to *same* level depending on LSB
        ba1 = scale_tensor_replicate(b)
        for d in np.kron(np.eye(3,dtype=np.int64),(1,-1)).T:
            aa0 = shift_tensor(a, *d)._coalesced_(True)
            aa0._values().fill_(1.)
            aa1 = scale_tensor_replicate(b)
            aa1._values().fill_(1.)
            #aa = cat_tensor((aa0,aa1))
            aa = (aa0 + aa1)
            #aa._values().fill_(1)
            da = (a - a*aa).coalesce()
            ma = torch.where((da.values()!=0).all(1))[0] # a's indices that are missing neighbors.
            #ma = torch.where(((a - a*aa)._coalesced_(True)._values()!=0).all(1))[0] # a's indices that are missing neighbors.
            miss = da.indices()[:,ma] + torch.from_numpy(d).to(a.device).view(3,1) # The actual coordinates that are missing.
            miss = miss[:, ((miss>0) & (miss<a.size(0))).all(0)]
            acc_missing.append(miss)

        m_inds = torch.cat(acc_missing,1)
        m_vals = torch.ones( (m_inds.size(1), a.values().size(1)) , dtype=a.values().dtype, device=a.device)
        if True:
            missing = torch.cuda.sparse.FloatTensor(m_inds,m_vals, a.size()).coalesce()
            print(' - lvl',ii,'missing',missing.values().size(0))
            if missing.values().size(0)>10:
                l = missing.values().size(0)//2
                print(' - First missing:', missing.indices()[:,l])
                x,y,z = missing.indices()[:,l]
                print('     - that ->', a[x,y,z])
                print('     - that neighborhood')
                for dd in (-1,1): print('     ', a[x+dd,y,z])
                for dd in (-1,1): print('     ', a[x,y+dd,z])
                for dd in (-1,1): print('     ', a[x,y,z+dd])

        # For any missing cells, average them and add them to next level.
        m_inds = torch.cat(acc_missing,1) // 2
        m_vals = torch.ones( (m_inds.size(1), a.values().size(1)) , dtype=a.values().dtype, device=a.device)
        new_cells = make_averaged_sparse_tensor(m_inds, m_vals, b.size())
        # un-averaged addition is okay here because we *know* the new_cells do not collide with b
        #lvls[ii+1] = (new_cells + self.lvls[ii+1])._coalesced_(True)
        lvls[ii+1] = (new_cells + b).coalesce()
        print(' - Done updating next level.')
        gc.collect()

        # TODO: Still must handle propogating average values.

    return missing


# First, builds structure.
# Second, does 'point aggregation', which never goes outside of the structure.
#
# Point Aggregation relies on shifting points and sparse pointwise multplication, and can be done in batches
# if memory is a concern.
class OctreeTwoPhase:
    def __init__(self, pts, vals, maxDepth):
        self.lvls = [None] * maxDepth
        self.maxDepth = maxDepth


        st = time.time()
        for i in range(0,maxDepth):
            if i == 0:
                inds_i = (pts.t() * (1<<maxDepth)).to(torch.int64)
                invScale_i = torch.ones((inds_i.size(1),1), device=pts.device, dtype=torch.int8) << (maxDepth-i)
            else:
                inds_i = self.lvls[i-1].indices() // 2
                invScale_i = torch.ones((inds_i.size(1),1), device=pts.device, dtype=torch.int8) << (maxDepth-i)
            sz = (1<<(maxDepth-i),)*3
            self.lvls[i] = make_averaged_sparse_tensor(inds_i,invScale_i, torch.Size((*sz,invScale_i.size(1))))
            del inds_i, invScale_i
            gc.collect()
            print(' - {:2d}: size {:8d}^3, {:8d} nnz'.format(i,self.lvls[i].size(0),len(self.lvls[i].values())))

        if pts.size(0)>=1e9:   print(' - indexed {:.1f}B pts in {}s'.format(pts.size(0)*1e-9,time.time()-st))
        elif pts.size(0)>=1e6: print(' - indexed {:.1f}M pts in {}s'.format(pts.size(0)*1e-6,time.time()-st))
        elif pts.size(0)>=1e3: print(' - indexed {:.1f}K pts in {}s'.format(pts.size(0)*1e-3,time.time()-st))


    # Makes it so that adjacent nodes never differ by more than one depth
    # Keeps track of the 'scale' (see Ummenhofer 2017)
    def balance(self):
        pass

    def mesh_it(self):
        pass

    def __len__(self):
        return sum([l.values().size(0) for l in self.lvls])


def precompute_tris():
    neigh = {}
    tris = []
    '''
    for side in range(2):
        for i in range(2):
            #if i == 0
        for j in range(2):
        for k in range(2):
    '''

try:
    __IPYTHON__
except:
    if __name__ == '__main__':
        DEVICE = torch.device('cpu')
        DEVICE = torch.device('cuda')
        #pts = torch.rand(10000000, 3, dtype=torch.float32, device=DEVICE)
        #pts = torch.rand(1000000, 3, dtype=torch.float32, device=DEVICE)
        pts = torch.rand(100000, 3, dtype=torch.float32, device=DEVICE)
        pts[:,2].mul_(pts[:,2])
        #pts = torch.rand(10000, 3, dtype=torch.float32, device=torch.device('cuda'))
        vals = torch.randn(pts.size(0),1, device=pts.device).to(torch.float32)

        o = OctreeTwoPhase(pts,vals, 14)

        '''
        pymeshedup_c.tensorOctreeToMesh(o.lvls)
        #print(o.lvls[4])
        #print(o.lvls[13])
        print('TRUE VALUE',o.lvls[10][0,0,0])
        print('TRUE VALUE',o.lvls[10][63,63,63])
        print('TRUE VALUE',o.lvls[10][5,5,5])
        print('TRUE VALUE',o.lvls[10][50,50,50])

        print(' - Balancing.')
        pymeshedup_c.tensorOctreeBalance(o.lvls)
        '''

        print(' - Size before balance:', len(o))
        balance_octree(o.lvls)
        print(' - Size after  balance:', len(o))

