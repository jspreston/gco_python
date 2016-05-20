import numpy as np
cimport numpy as np
from libcpp cimport bool

np.import_array()

ctypedef np.int32_t SiteID_dtype
ctypedef np.int32_t LabelID_dtype
ctypedef np.int32_t FuncID_dtype
ctypedef np.float32_t EnergyTermType_dtype

LabelID_ctype = np.NPY_INT32

cdef extern from "GCoptMultiSmooth.h":
    cdef cppclass GCoptMultiSmooth:
        GCoptMultiSmooth(int n_vertices, int n_labels, int n_smoothFuncs) except +
        void copyDataCost(EnergyTermType_dtype *) except +
        void addEdges(SiteID_dtype *site1, SiteID_dtype *site2, int n_edges, FuncID_dtype smooth_func_id,
                      EnergyTermType_dtype *weights) except +
        FuncID_dtype addSmoothCost(EnergyTermType_dtype *) except +
        bool alpha_expansion(LabelID_dtype alpha_label) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        void whatLabel(SiteID_dtype start, SiteID_dtype count, LabelID_dtype *labeling) except +
        EnergyTermType_dtype giveDataEnergy() except +
        EnergyTermType_dtype giveSmoothEnergy() except +
        EnergyTermType_dtype compute_energy() except +
        void setVerbosity(int level) except +
        void setLabel(SiteID_dtype site, LabelID_dtype label) except +
        void setLabels(SiteID_dtype *sites, LabelID_dtype *labels, int n_labels) except +
        void setAllLabels(LabelID_dtype *labels) except +
        LabelID_dtype num_labels() except +
        SiteID_dtype num_sites() except +

cdef class PyGCoptMultiSmooth:

    # thisptr hold a C++ instance which we're wrapping
    cdef GCoptMultiSmooth *thisptr

    cdef public int n_vertices
    cdef public int n_labels

    def __cinit__(self,
                  int n_vertices,
                  int n_labels,
                  int n_smoothFuncs,
                  int verbosity=0):
        self.thisptr = new GCoptMultiSmooth(n_vertices, n_labels, n_smoothFuncs)
        self.thisptr.setVerbosity(verbosity)
        self.n_vertices = n_vertices
        self.n_labels = n_labels

    def __dealloc__(self):
        del self.thisptr

    def setDataCost(self, np.ndarray[EnergyTermType_dtype, ndim=2, mode='c'] unary_cost):
        """
        Copies unary_cost to internal buffer
        """
        cdef int n_vertices = unary_cost.shape[0]
        cdef int n_labels = unary_cost.shape[1]
        assert n_vertices == self.n_vertices
        assert n_labels == self.n_labels
        self.thisptr.copyDataCost(<EnergyTermType_dtype*>unary_cost.data)

    def addSmoothCost(self, np.ndarray[EnergyTermType_dtype, ndim=2, mode='c'] smooth_cost):
        assert smooth_cost.shape[0] == self.n_labels
        assert smooth_cost.shape[1] == self.n_labels
        cdef FuncID_dtype smooth_func_id = self.thisptr.addSmoothCost(<EnergyTermType_dtype*>smooth_cost.data)
        return smooth_func_id

    def addEdges(self,
                 np.ndarray[SiteID_dtype, ndim=1, mode='c'] site1,
                 np.ndarray[SiteID_dtype, ndim=1, mode='c'] site2,
                 FuncID_dtype smooth_func_id,
                 np.ndarray[EnergyTermType_dtype, ndim=1, mode='c'] weights=None):
        cdef int n_edges = site1.size
        assert site2.size == n_edges
        cdef EnergyTermType_dtype *wptr = NULL
        if weights is not None:
            wptr = <EnergyTermType_dtype*>weights.data
        self.thisptr.addEdges(<SiteID_dtype*>site1.data, <SiteID_dtype*>site2.data, n_edges, smooth_func_id, wptr)

    def expansion(self, int n_iterations):
        self.thisptr.expansion(n_iterations)

    def alpha_expansion(self, LabelID_dtype alpha_label):
        self.thisptr.alpha_expansion(alpha_label)

    def getLabeling(self):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = self.n_vertices
        cdef np.ndarray[LabelID_dtype, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, LabelID_ctype)
        self.thisptr.whatLabel(0, result.size, <LabelID_dtype*>result.data)
        return result

    def dataEnergy(self):
        return self.thisptr.giveDataEnergy()

    def smoothEnergy(self):
        return self.thisptr.giveSmoothEnergy()

    def energy(self):
        ttl_en = self.thisptr.compute_energy()
        data_en = self.thisptr.giveDataEnergy()
        smooth_en = self.thisptr.giveSmoothEnergy()
        return (ttl_en, data_en, smooth_en)

    def setLabels(self,
                  np.ndarray[SiteID_dtype, ndim=1, mode='c'] sites,
                  np.ndarray[LabelID_dtype, ndim=1, mode='c'] labels):
        cdef int n_labels = sites.size
        assert n_labels == labels.size
        self.thisptr.setLabels(<SiteID_dtype*>sites.data,
                               <LabelID_dtype*>labels.data,
                               n_labels)

    def setAllLabels(self,
                     np.ndarray[LabelID_dtype, ndim=1, mode='c'] labels):
        assert labels.size == self.n_vertices
        self.thisptr.setAllLabels(<LabelID_dtype*>labels.data)

    def printinfo(self):
        print 'num labels: ', self.thisptr.num_labels()
        print 'num sites: ', self.thisptr.num_sites()
