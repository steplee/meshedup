#pragma once

#include "common.h"
#include "mesh.h"
#include "dt.h"

// High Accuracy and Visibility-Consistent Dense Multiview Stereo
// Hoang-Hiep Vu, Patrick Labatut, Jean-Philippe Pons, and Renaud Keriven
// http://islab.ulsan.ac.kr/files/announcement/441/PAMI-2012%20High%20Accuracy%20and%20Visibility-Consistent%20Dense%20Multiview%20Stereo.pdf
struct VuMeshing {
  VuMeshing(std::shared_ptr<DelaunayTetrahedrialization> dt, float alphaVis);

  std::shared_ptr<DelaunayTetrahedrialization> dt;
  float alphaVis;


  // Carry out an approach similar to the paper, except we only have one camera, and it is orthographic,
  // and it is looking nadir.
  // I believe doing the s-t cut is overkill here, but it will allow me to add priors (?)
  //
  // Note: It is assumed the pointcloud's X and Y axes range 0-1 and that elev is square.
  //
  void runWithElevationMap(RowMatrixCRef elev);

  IndexedMesh mesh;
  IndexedMesh assignmentMesh;
};
