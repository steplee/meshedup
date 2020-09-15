#include "twod_mrf.h"


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
  MRF::CostVal d = (g_elev->operator()(by,bx) - g_elev->operator()(ay,ax))*5.;

  //d = std::abs(d);
  d = d*d;
  int CEIL = 12;
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
  mrf->optimize(5,t);  // run for 5 iterations, store time t it took
  MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
  MRF::EnergyVal E_data   = mrf->dataEnergy();
  printf("Total Energy = %d (Smoothness energy %d, Data Energy %d)\n", E_smooth+E_data,E_smooth,E_data);
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
}
