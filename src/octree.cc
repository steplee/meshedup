#include "octree.h"

#include <iostream>
#include <stack>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Geometry>


using namespace Eigen;

static inline uint64_t HASH_XYZ(float x, float y, float z) {
  uint64_t a = (x+.5) * 1048576.;
  uint64_t b = (y+.5) * 1048576.;
  uint64_t c = (z+.5) * 1048576.;
  return (a<<40) | (b<<20) | c;
}

IntegralOctree::IntegralOctree(int maxDepth, RowMatrixCRef pts0)
  : maxDepth(maxDepth), treePts(pts0) {
    leafNodes=totalNodes = 0;
  root = new Node{};
  root->loc.setZero();
  root->depth = 0;

  int noverwrote = 0;

  float s_ = static_cast<scalar>(1<<(maxDepth));
  //const RowMatrixi ipts = ((pts0 * s_).array() + .5).cast<int32_t>();
  const RowMatrixi ipts = ((pts0 * s_)).cast<int32_t>();
  //#pragma omp parallel for schedule(static)
  for (int i=0; i<ipts.rows(); i++) {
    Vector3int ipt = ipts.row(i);
    Node* node = root;
    for (int lvl=1; lvl<=maxDepth; lvl++) {
      int bit = maxDepth - lvl ;
      //int bit = lvl;
      int l = 0;
      if ((ipt(0)>>bit) & 1) l |= 1;
      if ((ipt(1)>>bit) & 1) l |= 2;
      if ((ipt(2)>>bit) & 1) l |= 4;
      if (node->children[l] == nullptr) {
        node->children[l] = new Node{};
        //node->children[l]->loc =  loc_to_int({ipt(0)>>lvl , ipt(1)>>lvl , ipt(2)>>lvl});
        //auto parentLoc = int_to_loc(node->loc);
        auto parentLoc = (node->loc);
        Vector3int childLoc = parentLoc * 2;
        if (l & 1) childLoc(0) += 1;
        if (l & 2) childLoc(1) += 1;
        if (l & 4) childLoc(2) += 1;
        //node->children[l]->loc =  loc_to_int(childLoc);
        node->children[l]->loc =  (childLoc);
        node->children[l]->depth =  (lvl);
        if (lvl == maxDepth) leafNodes++;
        totalNodes++;
        //if (lvl + 1 == 8) std::cout << "made " << node->children[l]->loc.transpose() << " " <<(lvl+1) << "\n";
      }
      node = node->children[l];
    }
    if (node->id != -1) noverwrote++;
    node->id = i;
  }
  //printf(" - note: overwrote %d pts.\n",noverwrote);
}

DistIndexPairs IntegralOctree::search(RowMatrixCRef qpts, int k) {
  int n = qpts.rows();
  DistIndexPairs out;
  out.indices.resize(n,k);
  out.dists.resize(n,k);

  struct DistIndexScalar { float d; int32_t i; };
  constexpr int POOLSIZE=32*3;
  constexpr int SEE_FACTOR=3; // see at least k*this-many points before exiting.
                              // 4 is a good safe value, but 2 is faster.
                              // Note: we must break after reaching poolsize, but will continue current
                              // level after see_factor is hit
  ENSURE(k <= POOLSIZE);

  // Stack1 only needs to hold the path to the lowest node (<maxDepth)
  // Stack2 holds a DFS and may need to be pretty large
  Node* stack[20]; // 20 is probably large enough
  int nseen = 0;
  DistIndexScalar pool[POOLSIZE];
  std::vector<Node*> stack2_container;
  stack2_container.reserve(16);

  // Did I do this thread local data right?
  // static schedule should be good: each query should be nearly same cost
  #pragma omp parallel for schedule(static) private(stack) private(nseen) private(pool) private(stack2_container)
  for (int i=0; i<n; i++) {
    const auto qpt = qpts.row(i);
    const auto qpti = (qpt * static_cast<scalar>(1<<maxDepth)).cast<int32_t>();
    Node* node = root;
    nseen = 0;

    int lvl = 0;
    for (; lvl<=maxDepth; lvl++) {
      stack[lvl] = node;
      int bit = maxDepth - lvl;
      int l = 0;
      if ((qpti(0)>>bit) & 1) l |= 1;
      if ((qpti(1)>>bit) & 1) l |= 2;
      if ((qpti(2)>>bit) & 1) l |= 4;
      node = node->children[l];
      if (node== nullptr) break;
    }

    // Note: this is only approximate nearest, we should go one level past when we see k
    Node* last = nullptr;
    std::stack<Node*, std::vector<Node*>> stack2(stack2_container);
    while (nseen<k*SEE_FACTOR and lvl>0 and nseen<POOLSIZE) {
      node = stack[--lvl];
      for (int l=0; l<8; l++) if (node->children[l] and node->children[l]!=last) stack2.push(node->children[l]);
      while (not stack2.empty()) {
        Node* node2 = stack2.top();
        stack2.pop();
        if (node2->id != -1) {
          pool[nseen++] = DistIndexScalar { (qpt-treePts.row(node2->id)).squaredNorm() , node2->id };
          if (nseen>POOLSIZE) break;
        } else
          // Note we do not check 'last' here, that is handled in the first push to stack2
          for (int l=0; l<8; l++) if (node2->children[l]) {
            stack2.push(node2->children[l]);
          }
      }
      last = node;
    }

    std::sort(pool, pool+nseen, [](const DistIndexScalar &a, const DistIndexScalar &b) { return a.d < b.d; });

    int size = nseen < k ? nseen : k;
    for (int kk=0; kk<size; kk++) {
      out.dists(i,kk) = pool[kk].d;
      out.indices(i,kk) = pool[kk].i;
    }
    for (int kk=size; kk<k; kk++) {
      out.dists(i,kk) = 9e12;
      out.indices(i,kk) = -1;
    }
  }

  return out;
}

Node* IntegralOctree::searchNode(const Vector3int& loc, int depth) {
  Node* node = root;

  while (true) {
    if (node->depth == depth) return node;
    int l = 0;
    int bit = depth - node->depth - 1;
    if ((loc(0)>>bit) & 1) l |= 1;
    if ((loc(1)>>bit) & 1) l |= 2;
    if ((loc(2)>>bit) & 1) l |= 4;
    //std::cout << " - search " << loc.transpose() << " at " << node->loc.transpose() 
    //<< " " << (int)node->depth << " going " << (l&1) << " "<< (l&2) << " " << (l&4) << "\n";
    if (node->children[l]) node = node->children[l];
    else return node;
  }
}

void IntegralOctree::render(int toDepth) {
  glEnableClientState(GL_VERTEX_ARRAY);
  static const float verts[] = { 0,0,0, 1,0,0, 1,1,0, 0,1,0,  0,0,1, 1,0,1, 1,1,1, 0,1,1 };
  static const int inds[] = { 0,1, 1,2, 2,3, 3,0, 0,4, 1,5, 2,6, 3,7,  4,5, 5,6, 6,7, 7,4};
  glVertexPointer(3, GL_FLOAT, 0, verts);
  glMatrixMode(GL_MODELVIEW);

  // This is hacky ... didn't feel like using a shader pipeline
  float m_[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, m_);
  Eigen::Map<Eigen::Matrix4f> m(m_);
  Vector3 eye = -m.topLeftCorner<3,3>().transpose() * m.topRightCorner<3,1>();

  std::stack<Node*> st;
  st.push(root);
  while (not st.empty()) {
    Node* node = st.top(); st.pop();

    glPushMatrix();
    Vector3int loc = (node->loc);
    float size = 1<<(node->depth);
    float s = 1./size;

    Vector3 off = (loc.cast<float>()) * s;

    //if (depth<5) std::cout << " render [" << depth << "] " << loc.transpose() << " (" << off.transpose() << ")\n";
    float mm[16] = {s,0,0,0, 0,s,0,0, 0,0,s,0, off(0),off(1),off(2),1};
    glMultMatrixf(mm);

    float dd = ((float)node->depth+1)/((float)toDepth);
    float alpha = std::fmax(0, .6 - ((eye.array()+s/2).matrix() - off).norm());
    alpha = 3 * alpha * alpha + std::fmax(0,-node->depth/3. + 1.);
    alpha = 1;
    glColor4f(dd*dd,0,1,alpha);

    if (node->depth < 3 or alpha>0)
      for (int i=0;i<8;i++) if (node->depth+1<toDepth and node->children[i]) st.push(node->children[i]);

    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, (void*)inds);
    glPopMatrix();
  }
  glDisableClientState(GL_VERTEX_ARRAY);
}

void IntegralOctree::render2() {
  glEnableClientState(GL_VERTEX_ARRAY);
  static const float verts[] = { 0,0,0, 1,0,0, 1,1,0, 0,1,0,  0,0,1, 1,0,1, 1,1,1, 0,1,1 };
  static const int inds[] = { 0,1, 1,2, 2,3, 3,0, 0,4, 1,5, 2,6, 3,7,  4,5, 5,6, 6,7, 7,4};
  glVertexPointer(3, GL_FLOAT, 0, verts);
  glMatrixMode(GL_MODELVIEW);

  // This is hacky ... didn't feel like writing a shader.
  float view_[16], proj_[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, view_);
  glGetFloatv(GL_PROJECTION_MATRIX, proj_);
  Eigen::Map<Eigen::Matrix4f> view__(view_);
  Eigen::Map<Eigen::Matrix4f> proj__(proj_);
  Eigen::Matrix4f mvp_ = proj__ * view__;
  Vector3 eye = -view__.topLeftCorner<3,3>().transpose() * view__.topRightCorner<3,1>();

  Vector3 z_plus = -view__.block<1,3>(2,0);
  //Vector3 z_plus = view__.block<3,1>(0,2);
  int child_mask = 0, child_bit = 0;
  //for (int i=0; i<2; i++)
  //for (int j=0; j<2; j++)
  //for (int k=0; k<2; k++)
    //if (i ^ (z_plus(0)>0) == 0) child_mask |=
  auto zp = z_plus.cwiseAbs();
  if (zp(0) > zp(1) and zp(0) > zp(2)) child_mask = 1, child_bit = (z_plus(0)>0)<<0;
  if (zp(1) > zp(0) and zp(1) > zp(2)) child_mask = 2, child_bit = (z_plus(1)>0)<<1;
  if (zp(2) > zp(0) and zp(2) > zp(1)) child_mask = 4, child_bit = (z_plus(2)>0)<<2;
  std::cout << view__ << "\n";
  std::cout << " z_plus : " << z_plus.transpose() << " mask " << (child_mask&4) << " " << (child_mask&2) << " " << (child_mask&1) <<"\n";

  Vector3 unit = view__.block<1,3>(1,0);
  //Vector3 unit = Vector3::UnitX() * .000001;
  float unit_norm = unit.norm();

  float F = .951;
  Eigen::AlignedBox<float,2> frustum(
      Vector2 { -F, -F },
      Vector2 { F, F } );

  std::stack<Node*> st;
  st.push(root);

  for (int i=0; i<2; i++)
  for (int j=0; j<2; j++)
  for (int k=0; k<2; k++) {
    Vector3 x = Vector3 { i , j , k };
    Vector4 x_ = mvp_ * x.homogeneous();
    Vector3 y = x_.head<3>() / x_(3);
    std::cout << " - Frustum projection: " << x.transpose() << " ==> " << y.transpose() << "\n";
  }

  while (not st.empty()) {
    Node* node = st.top(); st.pop();

    Vector3int loc = (node->loc);
    float size = 1<<(node->depth);
    float s = 1./size;

    Vector3 off = (loc.cast<float>()) * s;
    Vector3 off_ctr = ((loc.cast<float>()) * s).array() + s/2.;

    //if (depth<5) std::cout << " render [" << depth << "] " << loc.transpose() << " (" << off.transpose() << ")\n";

    /*
    Vector4 a = mvp_ * off_ctr.homogeneous();
    Vector4 b = mvp_ * (off_ctr+unit*s).homogeneous();
    Vector3 aa = a.head<3>() / a(3);
    Vector3 bb = b.head<3>() / b(3);
    float screen_size = (aa - bb).norm() / unit_norm;


    //if (node->depth<3) std::cout << " render [" << (int)node->depth << "] " << loc.transpose() << " (" << screen_size << ")  " << aa.transpose() << "\n";
    float check = 1 + s*1.4;
    if (aa(0) < -check or aa(0) > check or aa(1) < -check or aa(1) < -check or aa(2) < 0) screen_size = 0;
    */


    float thresh = .02;
    float screen_size = 1;

    if (screen_size > thresh) {
      float dd = ((float)node->depth+1)/((float)maxDepth);
      float alpha = std::fmax(0, .6 - ((eye.array()+s/2).matrix() - off).norm());
      alpha = 3 * alpha * alpha;
      alpha = 1;
      glColor4f(dd*dd,0,1,alpha);

      float mm[16] = {s,0,0,0, 0,s,0,0, 0,0,s,0, off(0),off(1),off(2),1};
      glPushMatrix();
      glMultMatrixf(mm);


      if (node->depth < maxDepth or alpha>0) {
        float dists_[8];
        float dists[8];
        for (int i=0;i<8;i++) dists[i] = dists_[i] = 99999;
        // TODO Check four corners, not the center.
        /*for (int i=0;i<8;i++) if (node->children[i]) {
          Vector3 off_ = (Vector3 { (i&1) , (i&2)>>1 , (i&4)>>2 }.array()-.5) * s * .5;
          if ( node->depth==0) std::cout << off_.transpose() <<"\n";
          Vector3 pp = off_ctr + off_;
          float d = 999999;
          //if (z_plus.dot(pp-eye) > 0)
            d = (eye - pp).squaredNorm();
          dists_[i] = dists[i] = d;
        }*/

        for (int i=0;i<8;i++) if (node->children[i]) {
          Vector3 tl = (Vector3 { (i&1) , (i&2)>>1 , (i&4)>>2 }.array() * s * .5) + off.array();
          Vector3 br = tl.array() + s*.5;
          Vector4 u_ = mvp_ * tl.homogeneous();
          Vector4 v_ = mvp_ * br.homogeneous();
          Vector3 u = u_.head<3>() / (u_(3));
          Vector3 v = v_.head<3>() / (v_(3));
          //u(2) = sqrt(u(2)); v(2) = sqrt(v(2));
          //u(2) = .99; v(2) = .995;
          if (u(0) > v(0)) {float t = u(0); u(0)=v(0); v(0)=t; }
          if (u(1) > v(1)) {float t = u(1); u(1)=v(1); v(1)=t; }
          if (u(2) > v(2)) {float t = u(2); u(2)=v(2); v(2)=t; }
          //if (u_(2) > u_(3) or v_(2) > v_(3)) u=v;
          //if (u_(2)/u_(3) < 0 and v_(2)/v_(3) < 0) u=v;
          Vector2 uu = u.head<2>(), vv = v.head<2>();
          Eigen::AlignedBox<float,2> bb(uu,vv);
          auto xx = bb.intersection(frustum);
          float dd = -xx.volume();
          //float dd = -bb.intersection(frustum).volume() / (v(2)-u(2));
          //float dd = -bb.intersection(frustum).volume();
          if (u_(3) < 0 or v_(3) < 0) dd = 99999;
          //if (bb.contains(Vector3::Zero())) dd = -999; // If camera in box, have good score
          if (Eigen::AlignedBox<float,3>(tl,br).contains(eye)) dd = -999; // If camera in box, have good score
          dists_[i] = dists[i] = dd;
          //dists_[i] = dists[i] = -bb.intersection(frustum).diagonal().head<2>().squaredNorm();
          //if (node->depth == 9) std::cout << u.transpose() << " " << v.transpose() << " d " << dd << "\n";
          //if (node->depth == 9) std::cout << " from " << xx.min().transpose() << " " << xx.max().transpose() << "\n";
          //if (node->depth == 9) std::cout << "from2 " << bb.min().transpose() << " " << bb.max().transpose() << "\n";

          if (dd < -.005)
          st.push(node->children[i]);
        }


        // See if camera lies in one of the children
        /*if (eye(0)>off(0) and eye(1)>off(1) and eye(2)>off(2) and eye(0)<off(0)+s and eye(1)<off(1)+s and eye(2)<off(2)+s){
          int i = eye(0)>off(0)+s*.5;
          int j = eye(1)>off(1)+s*.5;
          int k = eye(2)>off(2)+s*.5;
          dists[(i<<2)+(j<<1)+k] = 0;
          dists_[(i<<2)+(j<<1)+k] = 0;
        }*/
        //std::sort(dists, dists+8);
        //for (int i=0;i<8;i++) if (dists_[i] < dists[4] and node->children[i]) {
          //if (dists_[i] < -.0005)
          //st.push(node->children[i]);
        //}
        //for (int i=0;i<8;i++) if (node->children[i]) st.push(node->children[i]);
      }

      glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, (void*)inds);
      glPopMatrix();
    }
  }
  glDisableClientState(GL_VERTEX_ARRAY);
}
