import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        GCoptimizationGridGraph(int width, int height, int n_labels) except +
        void setDataCost(int *) except +
        void setSmoothCost(int *) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        void setSmoothCostVH(int* pairwise, int* V, int* H) except +
        int whatLabel(int node) except +

    cdef cppclass GCoptimizationGeneralGraph:
        GCoptimizationGeneralGraph(int n_vertices, int n_labels) except +
        void setDataCost(int *) except +
        void setSmoothCost(int *) except +
        void setNeighbors(int, int) except +
        void setNeighbors(int, int, int) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        int whatLabel(int node) except +

cdef extern from "GCoptMultiSmooth.h":
    cdef cppclass GCoptMultiSmooth:
        GCoptMultiSmooth(int n_vertices, int n_labels, int n_smoothFuncs) except +
        void setDataCost(int *) except +
        void addEdges(int *site1, int *site2, int n_edges, int smooth_func_id) except +
        int addSmoothCost(int *) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        int whatLabel(int node) except +

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

    def setDataCost(self, np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost):
        cdef int n_vertices = unary_cost.shape[0]
        assert n_vertices == self.n_vertices
        self.thisptr.setDataCost(<int*>unary_cost.data)

    def addSmoothCost(self, np.ndarray[np.int32_t, ndim=2, mode='c'] smooth_cost):
        assert smooth_cost.shape[0] == self.n_labels
        assert smooth_cost.shape[1] == self.n_labels
        smooth_func_id = self.thisptr.addSmoothCost(<int*>smooth_cost.data)
        return smooth_func_id

    def addEdges(self,
                 np.ndarray[np.int32_t, ndim=1, mode='c'] site1,
                 np.ndarray[np.int32_t, ndim=1, mode='c'] site2,
                 int smooth_func_id):
        cdef int n_edges = site1.size
        assert site2.size == n_edges
        self.thisptr.addEdges(<int*>site1.data, <int*>site2.data, n_edges, smooth_func_id)

    def expansion(self, int n_iterations):
        self.thisptr.expansion(n_iterations)

def cut_simple(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)

    del gc
    return result

def cut_simple_vh(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] costV,
        np.ndarray[np.int32_t, ndim=2, mode='c'] costH,
        n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    costV: ndarray, int32, shape=(width, height)
        Vertical edge weights
    costH: ndarray, int32, shape=(width, height)
        Horizontal edge weights
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    if costV.shape[0] != w or costH.shape[0] != w or costV.shape[1] != h or costH.shape[1] != h:
        raise ValueError("incorrect costV or costH dimensions.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCostVH(<int*>pairwise_cost.data, <int*>costV.data, <int*>costH.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    del gc
    return result


def cut_from_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    edges: ndarray, int32, shape(n_edges, 2 or 3)
        Rows correspond to edges in graph, given as vertex indices.
        if edges is n_edges x 3 then third parameter is used as edge weight
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    if unary_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for e in edges:
        if e.shape[0] == 3:
            gc.setNeighbors(e[0], e[1], e[2])
        else:
            gc.setNeighbors(e[0], e[1])
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)
    del gc
    return result
