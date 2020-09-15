#pragma once

#include "common.h"
#include "mesh.h"

#include <mrf.h>
#include <GCoptimization.h>

#include <iostream>

//
// Labelling:
// Scheme 1:
//    0: predict point is the min of the neighbors
//    1: predict point is the center value.
//    2: predict point is the max of the neighbors
// Scheme 2:
//    0-9: predict that neighbors value
//

constexpr int SHAPE_QUADRATIC = 1;
constexpr int SHAPE_ABSOLUTE  = 2;
struct EnergySurfacing2d {
  EnergySurfacing2d();
  ~EnergySurfacing2d();

  void runWithElevationMap(RowMatrixCRef elev, RowMatrixCRef bad);

  //MRF::CostVal centerCost    = 0.;
  //MRF::CostVal offCenterCost = 0.;
  int dataShape = SHAPE_QUADRATIC;
  MRF::CostVal dataBoundaryCost = 9999;
  MRF::CostVal smoothMult = 1;
  MRF::CostVal smoothMult2 = 1;
  int numLabels = 9;

  int w,h;
  MRF::CostVal* dataTerm = nullptr;
  MRF::CostVal* smoothTerm = nullptr;

  IndexedMesh mesh;
  IndexedMesh costMesh;

  RowMatrix output;

  void create_nrg(RowMatrixCRef elev, RowMatrixCRef bad);
};
