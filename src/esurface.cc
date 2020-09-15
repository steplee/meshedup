#pragma once

#include "common.h"
#include "mesh.h"
#include "dt.h"

struct EnergySurfacing {
  EnergySurface();

  // We spread out a few points along both sides of the normal.
  // How far we travel along the normal is controlled by anchorOffsets,
  // the cost of assigning this anchor is controlled by anchorCosts.
  Eigen::VectorXf anchorOffsets;
  Eigen::VectorXf anchorCosts;

  void runWithElevationMap(RowMatrixCRef elev);

  IndexedMesh mesh;
  IndexedMesh costMesh;
};

