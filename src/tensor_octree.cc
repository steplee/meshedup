
#include "mesh.h"
#include <torch/extension.h>


// Store the left-most index that holds 'i' in the argument 'lo'
// TODO: Phase 2 could be more efficient.
// TODO: Not tested entirely
void binary_search_left(int64_t i, const int64_t* arr, int& lo, int hi) {
  int mid = (lo+hi)/2;
  while (lo<hi and arr[mid] != i) {
    if (i > arr[mid]) lo = mid+1, mid = (lo+hi)/2;
    else hi = mid, mid = (lo+hi)/2;
  }
  lo = mid;
  if (arr[mid] != i)
    lo = -1;
  else {
    while (lo>0 and arr[lo-1]==i) lo--;
    int step = 1;
    while (arr[lo-step] == i and step>0) {
      lo = lo - step;
      if (arr[lo-step] == i) step *= 2;
      else step /= 2;
    }
    if (arr[lo] != i) exit(1);
  }
}

// Store the left and right-most indices in lo and hi.
// TODO: Phase 2 could be more efficient.
// TODO: Not tested entirely
// NOTE: lo0 and hi0 actually change and always record the previous lo&hi, which
// makes phase 2 faster, since it bisects (possibly) much less!
void binary_search_both(int64_t i, const int64_t* arr, int& lo, int& hi) {
  int lo0 = lo;
  int hi0 = hi;
  int mid = (lo+hi)/2;
  while (lo<hi and arr[mid] != i) {
    if (i > arr[mid]) {
      lo0 = lo;
      lo = mid+1;
      mid = (lo+hi)/2;
    }
    else {
      hi0 = hi;
      hi = mid;
      mid = (lo+hi)/2;
    }
  }

  if (arr[mid] != i) lo=-1,hi=-1;
  else {
    // We have outer if-statements in the common case that lo=hi
    lo = mid;
    if (lo>lo0 and arr[lo-1] == i)
      while (lo>lo0) {
        int mid1 = (lo0+lo) / 2;
        if (arr[mid1] == i) lo = mid1;
        else lo0 = mid1+1;
        //std::cout << " - finding lo: " << lo << " " << lo0 << " : " << mid1 << " v " << arr[mid1] << "\n";
      }

    hi = mid;
    if (hi<hi0 and arr[hi+1] == i)
      while (hi<hi0) {
        int mid1 = (hi+hi0) / 2;
        if (arr[mid1] == i) hi = mid1+1;
        else hi0 = mid1;
        //std::cout << " - finding hi: " << hi << " " << hi0 << " : " << mid1 << " v " << arr[mid1] << "\n";
      }

    // Naive O(n) iteration (which may actually be faster for e.g. Z-index search!)
    //while (lo>lo0 and arr[lo-1]==i) lo--;
    //while (hi<hi0 and arr[hi-1]==i) hi++;
  }
}

int locate_index(int i, int j, int k, const int64_t* coo_d, int nnz, int lo=0, int hi=-1) {
  if (hi == -1) hi = nnz;
  binary_search_both(i, coo_d+(nnz*0), lo,hi);
  if (lo == -1 or hi == -1) return -1;
  binary_search_both(j, coo_d+(nnz*1), lo,hi);
  if (lo == -1 or hi == -1) return -1;
  binary_search_left(k, coo_d+(nnz*2), lo,hi);
  return lo;
}
int locate_index(int i, int j, int k, torch::Tensor& coo_t) {
  int nnz = coo_t.size(1);
  int64_t* coo_d = (int64_t*) coo_t.data_ptr();
#if 0
  int lo=0, hi=nnz;
  binary_search_left(i, coo_d+(nnz*0), lo,hi);
  hi = lo; while(coo_d[hi] == i and hi<nnz) hi++;
  //std::cout << " - lo/hi " <<  lo << " " << hi << " inds " << coo_d[lo] << " " << coo_d[hi] << "\n";

  binary_search_left(j, coo_d+(nnz*1), lo,hi);
  hi = lo; while((coo_d+(nnz*1))[hi] == j and hi<nnz) hi++;
  //std::cout << " - lo/hi " <<  lo << " " << hi << " inds " << (coo_d+nnz*1)[lo] << " " << (coo_d+nnz*1)[hi] << "\n";

  binary_search_left(k, coo_d+(nnz*2), lo,hi);
  //std::cout << " - lo/hi " <<  lo << " " << hi << " inds " << (coo_d+nnz*2)[lo] << " " << (coo_d+nnz*2)[hi] << "\n";
  // hi is now gauranteed to be low, no need to check.
#else
  int lo=0, hi=nnz;
  binary_search_both(i, coo_d+(nnz*0), lo,hi);
  if (lo == -1 or hi == -1) return -1;
  binary_search_both(j, coo_d+(nnz*1), lo,hi);
  if (lo == -1 or hi == -1) return -1;
  binary_search_left(k, coo_d+(nnz*2), lo,hi);
  if (lo == -1) return -1;
#endif

  //std::cout << " - found idx " <<  lo << "\n";
  return lo;
}

// Assumes the input octree is 'balanced'
IndexedMesh tensorOctreeToMesh(std::vector<torch::Tensor>& ts) {
  IndexedMesh mesh;
  // TODO use openmp

  int gh = ts[0].size(0);
  int gw = ts[0].size(1);
  int gd = ts[0].size(2);


  auto& t = ts[10];
  auto val_t = t._values().cpu();
  int vd = val_t.ndimension() == 2 ? val_t.size(1) : 1;
  auto coo_t = t._indices().cpu();
  int nnz = coo_t.size(1);
  int64_t* coo_d = (int64_t*) coo_t.data_ptr();
  float* val_d = (float*) val_t.data_ptr();
  std::cout << " - have sparse tensor with " << nnz << " nnz.\n";

  for (int ii = 0; ii < nnz; ii++) {
    int64_t x = coo_d[nnz*0+ii];
    int64_t y = coo_d[nnz*1+ii];
    int64_t z = coo_d[nnz*2+ii];
    float v = val_d[ii*vd];
    //std::cout << " - " << x << " " << y << " " << z << " : " << v << "\n";
  }

  int idx = locate_index(0,0,0, coo_t);
  std::cout << " FOUND VALUE " << val_d[idx] << "\n";
  idx = locate_index(63,63,63, coo_t);
  std::cout << " FOUND VALUE " << val_d[idx] << "\n";
  idx = locate_index(5,5,5, coo_t);
  std::cout << " FOUND VALUE " << val_d[idx] << "\n";
  idx = locate_index(50,50,50, coo_t);
  std::cout << " FOUND VALUE " << val_d[idx] << "\n";

  {
    int N = 10'000'000;
    std::cout << "searching " << N << " times.\n";
    auto& t = ts[2];
    for (int i=0; i<N; i++) {
      int x = rand() % t.size(0);
      int y = rand() % t.size(0);
      int z = rand() % t.size(0);
      int idx = locate_index(x,y,z, coo_t);
      if (rand() % 1000000 == 0)
        std::cout << " " << x << " " << y << " " << z << "\n";
    }
  }

  // For every cell
  // Get top left neighbors
  // Assert balanced
  // Build geometry: (potentially add verts) (add tris)
  for (int ii=0; ii<nnz; ii++) {
    int i = coo_d[nnz*0+ii], j = coo_d[nnz*1+ii], k = coo_d[nnz*2+ii];
    // If the neighbor on our level is missing, it must be present on next up, else have none.
    // If we have 2 neighbors on one face, create 2 tris
    // If we have 1 neighbors on one face, create 2 tris
    // If we have 0 neighbors on one face, connect the perpindicular plane.
  }


  return mesh;
}

void tensorOctreeBalance(std::vector<torch::Tensor>& ts) {
  int L = ts.size();
  for (int l=0; l<L-1; l++) {
    // Current level.
    auto tc = ts[l].indices().cpu();
    auto tv = ts[l].values().cpu();
    auto nnz = tc.size(1);
    int64_t* coo_d = (int64_t*) tc.data_ptr();
    float* val_d = (float*) tv.data_ptr();
    // Next level (half res)
    auto tc2 = ts[l+1].indices().cpu();
    auto tv2 = ts[l+1].values().cpu();
    auto nnz2 = tc2.size(1);
    int64_t* coo_d2 = (int64_t*) tc2.data_ptr();
    float* val_d2 = (float*) tv2.data_ptr();
    int S = ts[l].size(0);

    RowMatrix newVals(1000, tv.size(1));
    std::vector<int64_t> newInds;
    // Note: we can either AVERAGE duplicates or PREVENT.
    // Average by using familiar cnt+sparse tensor (or just unordered_map + cnt)
    // Prevent by using unordered_set check.
    std::unordered_set<int64_t> seen; // XXX: Can only handle 2^21 vals

    std::cout << " doing lvl " << l << std::endl;
    for (int ii=0; ii<nnz; ii++) {
      int i = coo_d[nnz*0+ii], j = coo_d[nnz*1+ii], k = coo_d[nnz*2+ii];

      // Not worrying too much about optimizing the code. I trust the compiler will make LuTs and such.
      for (int q=0; q<6; q++) {
        int a,b,c;
        if (q==0) a = i-1 , b = j   , c = k;
        if (q==1) a = i   , b = j-1 , c = k;
        if (q==2) a = i   , b = j   , c = k-1;
        if (q==3) a = i+1 , b = j   , c = k;
        if (q==4) a = i   , b = j+1 , c = k;
        if (q==5) a = i   , b = j   , c = k+1;
        if (a<0 or b<0 or c<0 or a>=S or b>=S or c>=S) continue;

        int ni = locate_index(a,b,c, coo_d,nnz);
        // If neighbor doesn't exist, its (2x) cell should in the next level.
        if (ni == -1) {
          int aa = a>>1, bb = b>>1, cc = c>>1;
          int ni2 = locate_index(aa,bb,cc, coo_d2,nnz2);
          if (ni2 == -1) {
            seen.insert( (int64_t(aa)<<42) | (int64_t(bb)<<21) | (int64_t(cc)) );
            //std::cout << " - inserting new cell " << (a>>1) << " " << (b>>1) << " " << (c>>1) << "\n";
          }
        }
      }
    }
    std::cout << " done lvl " << l << " (" << seen.size() << " new)\n";

    // Add our new data to the next level, the coalesce() it.
    // Next iter will use the updated results.
  }
}
