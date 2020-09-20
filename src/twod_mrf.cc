#include "twod_mrf.h"
#include <Eigen/Dense>


EnergySurfacing2d::EnergySurfacing2d() {
}

EnergySurfacing2d::~EnergySurfacing2d() {
  if (dataTerm) delete[] dataTerm;
  if (smoothTerm) delete[] smoothTerm;
}

void EnergySurfacing2d::create_nrg(RowMatrixCRef elev, RowMatrixCRef bad) {
  double size = static_cast<double>(sizeof(MRF::CostVal) * h * w) / (1024.*1024.);
  std::cout << " - creating energy function (grid " << h << " " << w << ") (labels " << numLabels << ") (total " << size << "MB)" << std::endl;
  std::cout << " - creating data terms (mem " << size << "MB)" << std::endl;

  if (dataTerm) delete[] dataTerm;
  if (smoothTerm) delete[] smoothTerm;

  dataTerm = new MRF::CostVal[elev.rows()*elev.cols()*numLabels];
  for (int y=0; y<h; y++) {
    for (int x=0; x<w; x++) {
      for (int k=0; k<3; k++) {
        for (int l=0; l<3; l++) {
          int dy = k-1, dx = l-1;


          if (x+dx < 0 or x+dx >= w or y+dy < 0 or y+dy >= h)
            dataTerm[y*w*9+x*9+k*3+l] = dataBoundaryCost;
          //else if (bad(y,x) > 0 or bad(y+dy,x+dx) > 0) dataTerm[y*w*9+x*9+k*3+l] = dataBoundaryCost;
          //else if ((dy == 0 and dx == 0 and bad(y,x) > 0) or (bad(y,x)==0 and bad(y+dy,x+dx)>0))
          else if (bad(y+dy,x+dx)>0)
            dataTerm[y*w*9+x*9+k*3+l] = dataBoundaryCost;
          else {
            auto d = elev(y,x) - elev(y+dy,x+dx);
            //auto d = (dataTerm[y*w*9+x*9 +1*3+1] - dataTerm[y*w*9+x*9+dy*3+dx]);
            //if (dataShape == SHAPE_QUADRATIC) dataTerm[y*w*9+x*9 +1*3+1] = d*d;
            //if (dataShape == SHAPE_ABSOLUTE) dataTerm[y*w*9+x*9 +1*3+1] = std::abs(d);
            if (dataShape == SHAPE_QUADRATIC) dataTerm[y*w*9+x*9 +k*3+l] = d*d;
            if (dataShape == SHAPE_ABSOLUTE) dataTerm[y*w*9+x*9 +k*3+l] = std::abs(d);

          }
          if (y > w/2-3 and y < w/2+3 and x > w/2-3 and x < w/2+3)
            std::cout << " - data term " << y << " " << x << " " << dy << " " << dx << " " << dataTerm[y*w*9+x*9+k*3+l] << "\n";
        }
      }
    }
  }

  // TODO: These should be spatially varying!
  /*
  size = static_cast<double>(sizeof(MRF::CostVal) * numLabels * numLabels) / (1024.*1024.);
  smoothTerm = new MRF::CostVal[numLabels*numLabels];
  std::cout << " - creating smoothness terms (mem " << size << "MB)" << std::endl;
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {
        for (int l=0; l<3; l++) {
          int d = std::abs((i-1)*(k-1)) + std::abs((j-1)*(l-1));
          smoothTerm[i*3*3*3+j*3*3+k*3+l] = d * smoothMult;
        }
      }
    }
  }
  std::cout << " - creating smoothness terms:\n";
  for (int i=0; i<9; i++) {
    for (int j=0; j<9; j++) {
      std::cout << " " << smoothTerm[i*9+j] << " ";
    }
    std::cout << "\n";
  }
  */


}


RowMatrixCRef* g_bad=nullptr;
RowMatrixCRef* g_elev=nullptr;
MRF::CostVal smoothFn(int a, int b, int i, int j) {
  int w = g_bad->cols(), h = g_bad->rows();
  int ax = a % w, ay = a / w;
  int bx = b % w, by = b / w;

  // Cost is not a <=> b, but a+label_i <=> b+label_j
  ax = ax + (i%3) - 1;
  bx = bx + (j%3) - 1;
  ay = ay + (i/3) - 1;
  by = by + (j/3) - 1;
  //MRF::CostVal d = std::abs((ax-1)*(bx-1)) + std::abs((ay-1)*(by-1));
  MRF::CostVal d = (g_elev->operator()(by,bx) - g_elev->operator()(ay,ax))*8.;

  d = std::abs(d);
  //d = d*d;
  int CEIL = 16;
  if (d>CEIL) d = CEIL; // Very important 'discontinuity preserving'

  if (g_bad->operator()(ay,ax) or g_bad->operator()(by,bx)) d = 999;
  return d*1;
}

void EnergySurfacing2d::runWithElevationMap(RowMatrixCRef elev, RowMatrixCRef bad) {
  h = elev.rows();
  w = elev.cols();
  g_bad = &bad;
  g_elev = &elev;

  create_nrg(elev, bad);
  DataCost *data = new DataCost(dataTerm);
  //SmoothnessCost *smooth = new SmoothnessCost(smoothTerm);
  //SmoothnessCost *smooth = new SmoothnessCost(&smoothFn);
  SmoothnessCost *smooth = new SmoothnessCost(&smoothFn);
  EnergyFunction *eng    = new EnergyFunction(data,smooth);

  // Solve.
  float t;
  MRF* mrf = new Expansion(w,h,numLabels,eng);
  mrf->initialize();
  mrf->clearAnswer();
  mrf->optimize(6,t);
  MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
  MRF::EnergyVal E_data   = mrf->dataEnergy();
  printf("Total Energy = %d (Smoothness energy %d, Data Energy %d)\n", E_smooth+E_data,E_smooth,E_data);
  printf(" - took %fs\n", t);
  //for (int pix =0; pix < w*h; pix++ ) printf("Label of pixel %d is %d",pix, mrf->getLabel(pix));


  int num_moved = 0;
  output.resize(h,w);
  for (int y=0; y<h; y++)
  for (int x=0; x<w; x++) {
    int dy = mrf->getLabel(y*w+x) / 3 - 1;
    int dx = mrf->getLabel(y*w+x) % 3 - 1;
    if (dy != 0 or dx != 0) num_moved++;
    output(y,x) = elev(y+dy,x+dx);
  }
  std::cout << " - n moved: " << num_moved << "\n";


  delete mrf;
  delete smooth; delete eng; delete data;
}

#include <unordered_map>
#include <map>

void addTri(int v1, int v2, int v3,
    std::vector<Vector3>& verts,
    std::unordered_map<int32_t, std::array<int32_t,3>>& tris,
    std::unordered_map<int32_t, std::vector<int32_t>>& vert2tris,
    std::vector<float>& vertCosts) {
    //std::map<float, int32_t>& vertCosts) {
  int new_tri_id = tris.size();
  tris[new_tri_id] = { v1, v2, v3 };
  vert2tris[v1].push_back(new_tri_id);
  vert2tris[v2].push_back(new_tri_id);
  vert2tris[v3].push_back(new_tri_id);

  // NOTE: This is wrong. The cost is not the sum of area of triangles,
  // but the *resulting* sum area if we didn't have it.
  Vector3 ab = (verts[v2] - verts[v1]);
  Vector3 ac = (verts[v3] - verts[v1]);
  float triArea = .5 * ab.cross(ac).norm();
  vertCosts[v1] += triArea;
  vertCosts[v2] += triArea;
  vertCosts[v3] += triArea;
  //std::cout << " - tri with area " << triArea <<"\n";
}

#define tri_id(y,x,i) (((y)*h*4) + ((x)*4) + (i))

// Will create at most (w*h*4 verts), (w*h*2 + w*(h-1)*6 tris)
void EnergySurfacing2d::make_mesh() {
  h = output.rows();
  w = output.cols();
  // Run along entire map.
  // Create 4 vertices, upto 8 tris per pixel
  // Simplify mesh occasionally to prevent high RAM usage.
  std::vector<Vector3> verts(h*w*4);
  std::unordered_map<int32_t, std::array<int32_t,3>> tris;
  std::unordered_map<int32_t, std::vector<int32_t>> vert2tris;
  //std::map<float, int32_t> vertCosts;
  std::vector<float> vertCosts(h*w*4);

  float eps = .00001;


  float ww = w, hh = h;
  for (int y=0; y<h; y++)
  for (int x=0; x<w; x++) {
    float xx = ((float)x)/ww, yy = ((float)y)/hh;
    float xx2 = ((float)x-1)/ww, yy2 = ((float)y-1)/hh;
    Vector3 a{xx2,yy2,output(y,x)};
    Vector3 b{xx,yy2,output(y,x)};
    Vector3 c{xx,yy,output(y,x)};
    Vector3 d{xx2,yy,output(y,x)};
    verts[y*h*4+x*4+0] = a;
    verts[y*h*4+x*4+1] = b;
    verts[y*h*4+x*4+2] = c;
    verts[y*h*4+x*4+3] = d;
    verts[y*h*4+x*4+0] = Vector3 {xx2,yy2,output(y,x)};
    verts[y*h*4+x*4+1] = Vector3 {xx,yy2,output(y,x)};
    verts[y*h*4+x*4+2] = Vector3 {xx,yy,output(y,x)};
    verts[y*h*4+x*4+3] = Vector3 {xx2,yy,output(y,x)};

    // Two internal tris
    addTri( tri_id(y,x,0) , tri_id(y,x,1) , tri_id(y,x,2) , verts,tris,vert2tris,vertCosts);
    addTri( tri_id(y,x,2) , tri_id(y,x,3) , tri_id(y,x,0) , verts,tris,vert2tris,vertCosts);
    // Two left tris
    if (x>0)
      addTri( tri_id(y,x-1,1) , tri_id(y,x,0) , tri_id(y,x,3) , verts,tris,vert2tris,vertCosts),
      //addTri( tri_id(y,x,0) , tri_id(y,x,3) , tri_id(y,x-1,2) , verts,tris,vert2tris,vertCosts);
      addTri( tri_id(y,x,3) , tri_id(y,x-1,2) , tri_id(y,x-1,1) , verts,tris,vert2tris,vertCosts);
    // Two upper-left tris
    /*
    if (x>0 and y>0)
      addTri( tri_id(y-1,x-1,2) , tri_id(y-1,x,3) , tri_id(y,x,0) , verts,tris,vert2tris,vertCosts),
      addTri( tri_id(y,x,0) , tri_id(y,x-1,1) , tri_id(y-1,x-1,2) , verts,tris,vert2tris,vertCosts);
    */
    // Two upper tris
    if (y>0)
      addTri( tri_id(y-1,x,3) , tri_id(y-1,x,2) , tri_id(y,x,1) , verts,tris,vert2tris,vertCosts),
      //addTri( tri_id(y-1,x,2) , tri_id(y,x,1) , tri_id(y,x,0) , verts,tris,vert2tris,vertCosts);
      addTri( tri_id(y,x,1) , tri_id(y,x,0) , tri_id(y-1,x,3) , verts,tris,vert2tris,vertCosts);

    // TODO: push costs
    // TODO: simplify every once in a while (dequeue vertCosts, then merge)
  }

  //mesh.verts = Eigen::Map<RowMatrix>(verts[0].data(), h*w*4,3);
  mesh.verts.resize(verts.size(), 3);
  for (int i=0; i<verts.size(); i++) mesh.verts.row(i) = verts[i];
  for (auto& t : tris) { mesh.inds.push_back(t.second[0]); mesh.inds.push_back(t.second[1]); mesh.inds.push_back(t.second[2]); }

  std::cout << " - got " << verts.size() << " verts " << tris.size() << " tris.\n";
}


// Like above, but use CGAL to build a simplified mesh.
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
typedef CGAL::Simple_cartesian<double>               Kernel_;
typedef Kernel_::Point_3                              Point_3;
typedef CGAL::Surface_mesh<Point_3>                  Surface_mesh;
namespace SMS = CGAL::Surface_mesh_simplification;

void EnergySurfacing2d::make_mesh_simplified() {
  Surface_mesh smesh;

  h = output.rows();
  w = output.cols();
  float eps = .00001;

  using Vertex_index = Surface_mesh::Vertex_index;

  std::vector<Vertex_index> pred_x1(h);
  std::vector<Vertex_index> pred_x2(h);
  std::vector<Vertex_index> pred_y2(w);
  std::vector<Vertex_index> pred_yy2(w);
  std::vector<Vertex_index> pred_y3(w);

  float M = 1000;
  float ww = w, hh = h;
  for (int y=0; y<h; y++) {
    for (int x=0; x<w; x++) {
      float o = .25;
      float xx = ((float)x+o)/ww, yy = ((float)y+o)/hh;
      float xx2 = ((float)x-o)/ww, yy2 = ((float)y-o)/hh;
      auto a = smesh.add_vertex(Point_3{M*xx2,M*yy2,M*output(y,x)});
      auto b = smesh.add_vertex(Point_3{M*xx,M*yy2,M*output(y,x)});
      auto c = smesh.add_vertex(Point_3{M*xx,M*yy,M*output(y,x)});
      auto d = smesh.add_vertex(Point_3{M*xx2,M*yy,M*output(y,x)});

      smesh.add_face(a,b,c);
      smesh.add_face(c,d,a);
      if (x>0)
        smesh.add_face(pred_x1[y], a, d),
        smesh.add_face(d, pred_x2[y], pred_x1[y]);
      if (y>0)
        smesh.add_face(pred_y3[x], pred_y2[x], b),
        smesh.add_face(b, a, pred_y3[x]);
      if (x>0 and y>0)
        smesh.add_face(pred_yy2[x-1], pred_y3[x], a),
        smesh.add_face(a, pred_x1[y], pred_yy2[x-1]);

      pred_x1[y] = b;
      pred_x2[y] = c;
      pred_y2[x] = c;
      pred_y3[x] = d;
    }
    pred_yy2 = pred_y2;
  }

  int full_verts = smesh.number_of_vertices(), full_tris = smesh.number_of_faces();

  if(!CGAL::is_triangle_mesh(smesh)) {
    std::cerr << "Input geometry is not triangulated." << std::endl;
    exit(1);
  }

  /*
  SMS::Count_stop_predicate<Surface_mesh> stop(0);
  int r = SMS::edge_collapse(smesh, stop);
  */

  std::cout << " - simplifiying." << std::endl;
  //SMS::Count_stop_predicate<Surface_mesh> stop(100000);
  double stop_ratio = .05;
  SMS::Count_ratio_stop_predicate<Surface_mesh> stop(stop_ratio);
  typedef typename SMS::GarlandHeckbert_policies<Surface_mesh, Kernel_>          GH_policies;
  typedef typename GH_policies::Get_cost                                        GH_cost;
  typedef typename GH_policies::Get_placement                                   GH_placement;
  GH_policies gh_policies(smesh);
  const GH_cost& gh_cost = gh_policies.get_cost();
  SMS::Edge_length_cost<Surface_mesh> el_cost;
  const GH_placement& gh_placement = gh_policies.get_placement();
  //typedef SMS::Bounded_normal_change_placement<GH_placement>                    ThePlacement;
  //Bounded_GH_placement placement(gh_placement);
  //typedef SMS::LindstromTurk_placement<Surface_mesh>                    ThePlacement;
  typedef SMS::Midpoint_placement<Surface_mesh>                    ThePlacement;
  ThePlacement placement;
  if (1) {
  int r = SMS::edge_collapse(smesh, stop,
      CGAL::parameters::get_cost(gh_cost)
      //CGAL::parameters::get_cost(el_cost)
      //.get_placement(placement));
      .get_placement(gh_placement));
  printf(" - Deleted %d edges.\n",r);
  }

  int final_verts = smesh.number_of_vertices(), final_tris = smesh.number_of_faces();
  printf(" - Have (%d/%d verts) (%d/%d tris) (%f%% %f%% kept).\n",
      final_verts,full_verts,final_tris,full_tris,
      100.*((float)final_verts)/full_verts, 100.*((float)final_tris)/full_tris);

  mesh.verts.resize(smesh.number_of_vertices(), 3);
  std::unordered_map<Vertex_index, int32_t> v2v;
  int v_ii = 0;
  for (const auto& v : smesh.vertices()) {
    auto pt = smesh.point(v);
    v2v[v] = v_ii;
    mesh.verts.row(v_ii++) = Vector3 { pt[0], pt[1], pt[2] } * (1./M);
  }

  for (const auto& f : smesh.faces()) {
    CGAL::Vertex_around_face_circulator<Surface_mesh> vbegin(smesh.halfedge(f),smesh), done(vbegin);
    // Triangle mesh, so will always do three iters.
    int32_t tri[3];
    int ii=0;
    do {
      Vertex_index v = *vbegin;
      tri[ii++] = v2v[v];
      vbegin++;
    } while(vbegin != done);
    mesh.inds.push_back(tri[0]);
    mesh.inds.push_back(tri[1]);
    mesh.inds.push_back(tri[2]);
    if (ii != 3) std::cout << " i " << ii << "\n";
  }

}
