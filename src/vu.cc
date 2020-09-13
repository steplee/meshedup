#include "vu.h"
#include "mfmc.h"
#include <Eigen/Dense>

static void appendRow3(RowMatrix& m, const Vector3& x, int i) {
  if (i >= m.rows()) m.conservativeResize(m.rows()*2, Eigen::NoChange);
  m.row(i) = x;
}
static void appendRow4(RowMatrix& m, const Vector4& x, int i) {
  if (i >= m.rows()) m.conservativeResize(m.rows()*2, Eigen::NoChange);
  m.row(i) = x;
}

// This is 'Facet'. Was hard to find...
// https://doc.cgal.org/latest/TDS_3/classTriangulationDataStructure__3.html#ad6a20b45e66dfb690bfcdb8438e9fcae

VuMeshing::VuMeshing(std::shared_ptr<DelaunayTetrahedrialization> dt, float alphaVis)
  : dt(dt), alphaVis(alphaVis)
{
}

void VuMeshing::runWithElevationMap(RowMatrixCRef elev) {
#if 0
  // NOTE: CGAL does not provide a ray/cell-complex walk function,
  // so you should search for the closest ray/convex-hull intersection, then walk tetrahedra efficiently.
  // I will bypass this by instead checking tetrahedral NOT points.
  // So the loop below is not used, rather I just walk tetrahedra and check if above.
  // TODO: Write the explained intersection then walk algorithm.
  //
  // Vertices will be tetrahedra, edges will be facets
  // For every point in the cloud
  //   For every tetrahedra above it
  //     connect it to src
  //   For every tetrahedra below it
  //     connect to sink.
  // Then solve for s-t cut.
  // Then get all facets that connect s-t nodes (these are the cut edges)

  auto& T = dt->T;
  // Although our vertices are labeled with ints, our cells are not, so we store the mapping here.
  std::unordered_map<Tds::Cell_handle, int32_t> cellIds;
  int ii = 0;
  for (auto it = T.finite_cells_begin(), end = T.finite_cells_end(); it != end; ++it)
    cellIds[it] = ii++;

  MaxFlowMinCut mfmc(dt->T.number_of_cells());

  for (auto v = T.finite_vertices_begin(), end = T.finite_vertices_end(); v != end; ++v) {
    // cgal stores one cell, we must walk to the others.
    std::vector<Tds::Cell_handle> cs;
    T.finite_incident_cells(v, std::back_inserter(cs));
  }
#endif

  // See the above note, this is a simplified version without walking the ray-complex intersection,
  // instead I just consider each tetrahedra in turn.
  //
  // Vertices will be tetrahedra, edges will be facets
  // For every tetrahedra in the DT
  //    If it is above surface: connect to source
  //    If it is below surface: connect to sink
  // Then solve for s-t cut.
  // Then get all facets that connect s-t nodes (these are the cut edges)

  // This should give a *TRIVIAL* s-t cut!

  auto& T = dt->T;
  MaxFlowMinCut mfmc(T.number_of_cells());
  int src_ii = 0;
  int sink_ii = -1;
  int alphaVisInt = 1;

  // Although our vertices are labeled with ints, our cells are not, so we store the mapping here.
  std::unordered_map<Tds::Cell_handle, int32_t> cellIds;
  std::unordered_set<int32_t> exteriorCells;
  int cell_cntr = 1;
  int seen_interior = 0;
  int seen_exterior = 0;
  for (auto c = T.finite_cells_begin(), end = T.finite_cells_end(); c != end; ++c) {
    cellIds[c] = cell_cntr;
    Vector3 cell_position = Vector3::Zero();
    bool is_exterior = false;

    for (int vi=0; vi<4; vi++) {
      auto v = c->vertex(vi);
      if (T.is_infinite(v)) is_exterior = true;
      else {
        auto pt = v->point();
        cell_position += Vector3(pt[0],pt[1],pt[2]);
      }
    }

    cell_position /= 4.;
    int y = cell_position(0) * elev.rows();
    int x = cell_position(1) * elev.cols();
    float elev_at_center = elev(y,x);
    if (cell_position(2) > elev_at_center)
      is_exterior = true;

    if (is_exterior) {
      mfmc.addEdge(0, cell_cntr, alphaVisInt);
      seen_exterior++;
      exteriorCells.insert(cell_cntr);
    } else {
      // If sink hasn't been set, do it.
      if (sink_ii == -1) sink_ii = cell_cntr;
      else mfmc.addEdge(cell_cntr, sink_ii, alphaVisInt);
      seen_interior++;
    }
    cell_cntr++;
  }

  std::cout << " - saw (" << seen_interior << " in) (" << seen_exterior << " out)\n";
  mfmc.setSourceSink(src_ii,sink_ii);
  mfmc.run();
  mfmc.deallocateSome(); // Save memory

  // Now create mesh from the cut-set.
  std::unordered_map<int32_t, int32_t> seenVerts;
  RowMatrix vv(1000,3);
  //mesh.verts = dt->pts;
  // Could make this faster by walking cells and stopping where all one-side.
#if 0
  for (auto f = T.finite_facets_begin(), end = T.finite_facets_end(); f != end; ++f) {
    auto f0 = *f;
    auto f1 = T.mirror_facet(f0);
    Tds::Cell_handle cell0 = f0.first;
    Tds::Cell_handle cell1 = f1.first;
    // Don't consider already seen faces (Note: erase this check, just check based on s&t)
    int32_t cellId0 = cellIds[cell0],
            cellId1 = cellIds[cell1];
    //std::cout << " - checking facet " << cellId0 << " " << cellId1 << "\n";
    /*
    if (mfmc.minCutEdges.find(Edge{cellId0,cellId1}) != mfmc.minCutEdges.end() or
        mfmc.minCutEdges.find(Edge{cellId1,cellId0}) != mfmc.minCutEdges.end()) {
    */
    int s0 = mfmc.minCutS.find(cellId0) != mfmc.minCutS.end();
    int s1 = mfmc.minCutS.find(cellId1) != mfmc.minCutS.end();
    //if (s1 == true and s0 == false) {
    if (exteriorCells.find(cellId0)==exteriorCells.end() and exteriorCells.find(cellId1)!=exteriorCells.end()) {
      int32_t tri[3];
      int tri_i = 0;
      // Add the triangle (will be the three verts not equal to the facet id)
      for (int vi=0; vi<4; vi++) {
        if (vi != f0.second) {
          auto v = cell0->vertex(vi);
          //tri[tri_i++] = v->info();
          int v_idx = 0;
          int v_orig_idx = v->info();
          // Do not copy unused verts.
          if (seenVerts.find(v_orig_idx) == seenVerts.end()) {
            v_idx = seenVerts.size();
            seenVerts[v_orig_idx] = v_idx;
            appendRow3(vv, dt->pts.row(v_orig_idx), v_idx);
          } else v_idx = seenVerts[v_orig_idx];
          tri[tri_i++] = v_idx;
        }
      }
      //for (int i=0; i<3; i++)
      Vector3 aa = mesh.verts.row(tri[0]);
      Vector3 bb = mesh.verts.row(tri[1]);
      Vector3 cc = mesh.verts.row(tri[2]);
      if (f0.second == 0 or f0.second == 2)
      { mesh.inds.push_back(tri[0]); mesh.inds.push_back(tri[1]); mesh.inds.push_back(tri[2]); }
      else
      { mesh.inds.push_back(tri[1]); mesh.inds.push_back(tri[0]); mesh.inds.push_back(tri[2]); }
    }
  }
#endif
  for (auto it = T.finite_cells_begin(), end = T.finite_cells_end(); it != end; ++it) {
    int32_t cellId0 = cellIds[it];
    if (exteriorCells.find(cellId0) == exteriorCells.end()) {
      for (int k=0; k<4; k++) {
        Tds::Cell_handle n = it->neighbor(k);
        if (exteriorCells.find(cellIds[n]) != exteriorCells.end()) {
          int32_t tri[3]; int tri_i = 0;
          for (int vi=0; vi<4; vi++) {
            if (vi != k) {
              auto v = it->vertex(vi);
              //tri[tri_i++] = v->info();
              int v_idx = 0;
              int v_orig_idx = v->info();
              // Do not copy unused verts.
              if (seenVerts.find(v_orig_idx) == seenVerts.end()) {
                v_idx = seenVerts.size();
                seenVerts[v_orig_idx] = v_idx;
                appendRow3(vv, dt->pts.row(v_orig_idx), v_idx);
              } else v_idx = seenVerts[v_orig_idx];
              tri[tri_i++] = v_idx;
            }
          }
          if (k == 0 or k == 2)
          { mesh.inds.push_back(tri[0]); mesh.inds.push_back(tri[1]); mesh.inds.push_back(tri[2]); }
          else
          { mesh.inds.push_back(tri[1]); mesh.inds.push_back(tri[0]); mesh.inds.push_back(tri[2]); }
        }
      }
    }
  }
  mesh.verts = vv.topRows(seenVerts.size());

  std::cout << " - created mesh:\n";



  std::cout << " - done.\n";

  // Create the 'assignment' mesh.
  //assignmentMesh.mode = GL_POINTS;
  assignmentMesh.mode = GL_LINES;
  RowMatrix pts_(1000,3);
  RowMatrix colors_(1000,4);
  int pt_ii = 0;
  for (auto it = T.finite_cells_begin(), end = T.finite_cells_end(); it != end; ++it) {
    int32_t cellId0 = cellIds[it];

    Vector3 center_pos = Vector3::Zero();
    for (int i=0; i<4; i++) center_pos += Vector3 { it->vertex(i)->point()[0], it->vertex(i)->point()[1], it->vertex(i)->point()[2] };
    Vector4 color;
    float alpha = .02;
    if (mfmc.minCutS.find(cellId0) == mfmc.minCutS.end()) color = Vector4 { 1., 0., 0., alpha };
    else color = Vector4 { 0., 1., 0., alpha };

    /*
    appendRow3(pts_, center_pos/4., pt_ii);
    appendRow4(colors_, color, pt_ii);
    assignmentMesh.inds.push_back(pt_ii);
    pt_ii++;
    */
    appendRow3(pts_, center_pos/4., pt_ii);
    for (int i=0; i<4; i++) appendRow3(pts_, Vector3 { it->vertex(i)->point()[0], it->vertex(i)->point()[1], it->vertex(i)->point()[2] }, pt_ii+1+i);
    for (int i=0; i<5; i++) appendRow4(colors_, color, pt_ii+i);
    assignmentMesh.inds.push_back(pt_ii); assignmentMesh.inds.push_back(pt_ii+1);
    assignmentMesh.inds.push_back(pt_ii); assignmentMesh.inds.push_back(pt_ii+2);
    assignmentMesh.inds.push_back(pt_ii); assignmentMesh.inds.push_back(pt_ii+3);
    assignmentMesh.inds.push_back(pt_ii); assignmentMesh.inds.push_back(pt_ii+4);
    pt_ii += 5;
  }
  assignmentMesh.verts = pts_.topRows(pt_ii);
  assignmentMesh.colors = colors_.topRows(pt_ii);
}
