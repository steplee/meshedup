#pragma once

#include "common.h"

#include <memory>

struct Node {
  //uint64_t loc;
  Vector3int loc;
  uint8_t depth;
  int32_t id = -1;
  Node* children[8];

  inline Vector3 offsetTopLeft() const {
    float size = 1. / (1<<depth);
    return Vector3 {
      (loc(0)) * size,
      (loc(1)) * size,
      (loc(2)) * size };
  }
  inline Vector3 offsetCenter() const {
    float size = 2. / (1<<(depth-1));
    float size2 = size / 2.;
    return Vector3 {
      (loc(0)) * size + size2,
      (loc(1)) * size + size2,
      (loc(2)) * size + size2 };
  }
};
struct IntegralOctree {
  IntegralOctree(int maxDepth, RowMatrixCRef pts);
  RowMatrix treePts;
  Node* root;
  int maxDepth;
  int totalNodes, leafNodes;

  void render(int toDepth);
  void render2();

  DistIndexPairs search(RowMatrixCRef qpts, int k);
  Node* searchNode(const Vector3int& loc, int depth);
};

