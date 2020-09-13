#pragma once

#include "common.h"

struct IntegralOctree;
struct IndexedMesh;

using Triangle = Eigen::Matrix<scalar, 3,3, Eigen::RowMajor>;


Vector3 VertexInterp(scalar isolevel, Vector3 p1,Vector3 p2, scalar valp1, scalar valp2);
//int Polygonise(Gridcell grid, scalar isolevel, Triangle *triangles);
int Polygonise(Gridcell grid, scalar isolevel, Vector3 vertlist[12], int& usedVertMask, int tris[5][3]);


#if 0
IndexedMesh meshOctree(Octree& oct, scalar isolevel);
IndexedMesh meshOctreeSurfaceNet(Octree& oct, scalar isolevel);


// Given a triangular mesh, average the face normals to get vertex normals.
//void computeVertexNormals(Octree& tree, IndexedMesh& mesh);

// For every qpt, approximate the normal by searching the tree+treePts for top
// few neighbors and taking the least important principal axis.
RowMatrix computePointSetNormals(const Octree& tree, RowMatrixCRef treePts, RowMatrixCRef qpts);


// For every point+normal, return up-to two new points at both ends of the (unoriented) normal.
// Points are NOT created if the closest existing point to them is not the one owning the normal (this
// is to prevent intersections)
// Example in 1D:
//   o    =>  x--o--x
//   but
//   o-o  =>  x--o-o--x (inner negative is not created)
RowMatrix getNegativePoints(
    const Octree& tree,
    RowMatrixCRef treePts,
    RowMatrixCRef treePtNormals,
    float normalLength);
#endif
