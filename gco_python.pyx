import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.int32_t SiteID_dtype
ctypedef np.int32_t LabelType_dtype
ctypedef np.int32_t FuncID_dtype
ctypedef np.float32_t EnergyTermType_dtype

LabelType_ctype = np.NPY_INT32

cdef extern from "GCoptMultiSmooth.h":
    cdef cppclass GCoptMultiSmooth:
        GCoptMultiSmooth(int n_vertices, int n_labels, int n_smoothFuncs) except +
        void setDataCost(EnergyTermType_dtype *) except +
        void addEdges(SiteID_dtype *site1, SiteID_dtype *site2, int n_edges, FuncID_dtype smooth_func_id) except +
        FuncID_dtype addSmoothCost(EnergyTermType_dtype *) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        void whatLabel(SiteID_dtype start, SiteID_dtype count, LabelType_dtype *labeling) except +
        EnergyTermType_dtype giveDataEnergy() except +
        EnergyTermType_dtype giveSmoothEnergy() except +

cdef class PyGCoptMultiSmooth:

    # thisptr hold a C++ instance which we're wrapping
    cdef GCoptMultiSmooth *thisptr

    cdef int n_vertices
    cdef int n_labels

    def __cinit__(self, int n_vertices, int n_labels, int n_smoothFuncs):
        self.thisptr = new GCoptMultiSmooth(n_vertices, n_labels, n_smoothFuncs)
        self.n_vertices = n_vertices
        self.n_labels = n_labels

    def __dealloc__(self):
        del self.thisptr

    def setDataCost(self, np.ndarray[EnergyTermType_dtype, ndim=2, mode='c'] unary_cost):
        cdef int n_vertices = unary_cost.shape[0]
        cdef int n_labels = unary_cost.shape[1]
        assert n_vertices == self.n_vertices
        assert n_labels == self.n_labels
        self.thisptr.setDataCost(<EnergyTermType_dtype*>unary_cost.data)

    def addSmoothCost(self, np.ndarray[EnergyTermType_dtype, ndim=2, mode='c'] smooth_cost):
        assert smooth_cost.shape[0] == self.n_labels
        assert smooth_cost.shape[1] == self.n_labels
        cdef FuncID_dtype smooth_func_id = self.thisptr.addSmoothCost(<EnergyTermType_dtype*>smooth_cost.data)
        return smooth_func_id

    def addEdges(self,
                 np.ndarray[SiteID_dtype, ndim=1, mode='c'] site1,
                 np.ndarray[SiteID_dtype, ndim=1, mode='c'] site2,
                 FuncID_dtype smooth_func_id):
        cdef int n_edges = site1.size
        assert site2.size == n_edges
        self.thisptr.addEdges(<SiteID_dtype*>site1.data, <SiteID_dtype*>site2.data, n_edges, smooth_func_id)

    def expansion(self, int n_iterations):
        self.thisptr.expansion(n_iterations)

    def getLabeling(self):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = self.n_vertices
        cdef np.ndarray[LabelType_dtype, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, LabelType_ctype)
        self.thisptr.whatLabel(0, result.size, <LabelType_dtype*>result.data)
        return result

    def giveDataEnergy(self):
        return self.thisptr.giveDataEnergy()

    def giveSmoothEnergy(self):
        return self.thisptr.giveSmoothEnergy()
