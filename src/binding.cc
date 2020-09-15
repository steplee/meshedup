#include <Eigen/StdVector>
#include <torch/extension.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "octree.h"
#include "mesh.h"
#include "dt.h"
#include "mfmc.h"
#include "vu.h"
#include "twod_mrf.h"

//https://stackoverflow.com/questions/48982143/returning-and-passing-around-raw-pod-pointers-arrays-with-python-c-and-pyb
// Just a pointer type that doesn't delete or refcnt
template <class T> class ptr_wrapper
{
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T* ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
        T& operator* () const { return *ptr; }
        T* operator->() const { return  ptr; }
        T* get() const { return ptr; }
        void destroy() { /*delete ptr*/; }
        T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};
PYBIND11_DECLARE_HOLDER_TYPE(T, ptr_wrapper<T>);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Node, ptr_wrapper<Node>>(m, "Node")
    .def_readwrite("loc", &Node::loc)
    .def_readwrite("depth", &Node::depth)
    .def_readwrite("id", &Node::id)
    .def("child", [](Node& self, int i) { return self.children[i]; })
    .def("__repr__", [](Node& s) { return "Node(" 
        + std::to_string(s.loc(0)) + " " + std::to_string(s.loc(1))
        + " " + std::to_string(s.loc(2)) + ", " + std::to_string(s.depth)
        + " id " + std::to_string(s.id) + ")"; })
    ;
  py::class_<IntegralOctree, std::shared_ptr<IntegralOctree>>(m, "IntegralOctree")
    .def(py::init<int, RowMatrixCRef>())
    .def("render", &IntegralOctree::render)
    .def("render2", &IntegralOctree::render2)
    .def("search", &IntegralOctree::search)
    .def("searchNode", &IntegralOctree::searchNode)
    .def("__repr__", [](IntegralOctree& s) { return "IntegralOctree("
          "maxDepth " + std::to_string(s.maxDepth)
        + ", totalNodes " + std::to_string(s.totalNodes)
        + ", leafNodes " + std::to_string(s.leafNodes) + ")"; });

  py::class_<IndexedMesh, std::shared_ptr<IndexedMesh>>(m, "IndexedMesh")
    .def(py::init<>())
    .def_readwrite("verts", &IndexedMesh::verts)
    .def_readwrite("inds", &IndexedMesh::inds)
    .def_readwrite("vertexNormals", &IndexedMesh::vertexNormals)
    .def_readwrite("faceNormals", &IndexedMesh::faceNormals)
    .def_readwrite("uvs", &IndexedMesh::uvs)
    .def_readwrite("tex", &IndexedMesh::tex)
    .def_readwrite("mode", &IndexedMesh::mode)
    .def("render", &IndexedMesh::render)
    .def("print", &IndexedMesh::print)
    .def("bake", &IndexedMesh::bake)
    .def("upload", &IndexedMesh::upload)
    ;

  py::class_<DTOpts, std::shared_ptr<DTOpts>>(m, "DTOpts")
    .def(py::init<>())
    .def_readwrite("createMesh", &DTOpts::createMesh)
    ;
  py::class_<DelaunayTetrahedrialization, std::shared_ptr<DelaunayTetrahedrialization>>(m, "DelaunayTetrahedrialization")
    .def(py::init<const DTOpts&>())
    .def_readwrite("mesh", &DelaunayTetrahedrialization::mesh)
    .def("run", &DelaunayTetrahedrialization::run)
    ;

  py::class_<MaxFlowMinCut, std::shared_ptr<MaxFlowMinCut>>(m, "MaxFlowMinCut")
    .def(py::init<int>())
    .def_readwrite("N", &MaxFlowMinCut::N)
    .def_readwrite("maxFlow", &MaxFlowMinCut::maxFlow)
    .def_readwrite("flow", &MaxFlowMinCut::flow)
    .def_readwrite("minCutS", &MaxFlowMinCut::minCutS)
    .def_readwrite("minCutT", &MaxFlowMinCut::minCutT)
    .def("getMinCutEdges", [](MaxFlowMinCut& mfmc) {
        // TODO: Should directly create a python dict (or just a list...)
        std::unordered_set<std::pair<int32_t,int32_t>> out;
        for (auto& e : mfmc.minCutEdges) out.insert({e.u,e.v});
        return out;
    })
    .def("printViz", &MaxFlowMinCut::printViz)
    .def("run", &MaxFlowMinCut::run)
    .def("addEdge", &MaxFlowMinCut::addEdge)
    .def("setSourceSink", &MaxFlowMinCut::setSourceSink)
    ;

  py::class_<VuMeshing, std::shared_ptr<VuMeshing>>(m, "VuMeshing")
    .def(py::init<std::shared_ptr<DelaunayTetrahedrialization>, float>())
    .def("runWithElevationMap", &VuMeshing::runWithElevationMap)
    .def_readwrite("assignmentMesh", &VuMeshing::assignmentMesh)
    .def_readwrite("mesh", &VuMeshing::mesh);

  m.attr("SHAPE_QUADRATIC") = py::int_(SHAPE_QUADRATIC);
  m.attr("SHAPE_ABSOLUTE") = py::int_(SHAPE_ABSOLUTE);
  py::class_<EnergySurfacing2d, std::shared_ptr<EnergySurfacing2d>>(m, "EnergySurfacing2d")
    .def(py::init<>())
    .def("runWithElevationMap", &EnergySurfacing2d::runWithElevationMap)
    .def_readwrite("dataShape", &EnergySurfacing2d::dataShape)
    .def_readwrite("dataBoundaryCost", &EnergySurfacing2d::dataBoundaryCost)
    .def_readwrite("smoothMult", &EnergySurfacing2d::smoothMult)
    .def_readwrite("smoothMult2", &EnergySurfacing2d::smoothMult2)
    .def_readwrite("output", &EnergySurfacing2d::output)
    .def_readwrite("mesh", &EnergySurfacing2d::mesh);
}

