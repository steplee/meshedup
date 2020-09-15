#pragma once

#include "common.h"
#include "mesh.h"

// A simpler approach where we construct a mesh from a 2d elevation map.
// So the surface will be restricted to '2.5d'
struct SurfaceFollowingMeshing {
  SurfaceFollowingMeshing();


  // If we observe a certain threshold in the gradient, we step.
  float discontinuityThresh;

  void runWithElevationMap(RowMatrixCRef elev);

  IndexedMesh mesh;
};


void SurfaceFollowingMeshing::runWithElevationMap(RowMatrixCRef elev) {
}
