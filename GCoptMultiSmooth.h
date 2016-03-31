#ifndef __GCOPTMULTISMOOTH_H__
#define __GCOPTMULTISMOOTH_H__
// Due to quiet bugs in function template specialization, it is not
// safe to use earlier MS compilers.
#if defined(_MSC_VER) && _MSC_VER < 1400
#error Requires Visual C++ 2005 (VC8) compiler or later.
#endif

#include "GCoptimization.h"
#include <utility>
#include <map>

#define DEFAULT_MAX_SMOOTHFUNCS 100

class GCoptMultiSmooth : public GCoptimizationGeneralGraph
{

public:

    typedef GCoptimizationGeneralGraph::EnergyTermType EnergyTermType;
    typedef GCoptimizationGeneralGraph::SiteID SiteID;
    typedef GCoptimizationGeneralGraph::LabelID LabelID;
    typedef int FuncID;
    typedef std::pair<GCoptimization::SiteID, GCoptimization::SiteID> Edge;

    // This is the constructor for non-grid graphs. Neighborhood structure must  be specified by
    // setNeighbors()  function
    GCoptMultiSmooth(SiteID num_sites,LabelID num_labels,int num_smoothFuncts=DEFAULT_MAX_SMOOTHFUNCS);
    virtual ~GCoptMultiSmooth();

    // void addEdge(SiteID site1, SiteID site2, FuncID smooth_func_id);
    void addEdges(SiteID *site1, SiteID *site2, int n_edges, FuncID smooth_func_id);

    void copyDataCost(EnergyTermType *dataArray);

    FuncID addSmoothCost(EnergyTermType *smoothArray);

    void setLabels(SiteID *sites, LabelID *labels, int n_labels);
    void setAllLabels(LabelID *labels);

    // accessors for debugging
    SiteID num_sites(){ return m_num_sites; }
    LabelID num_labels(){ return m_num_labels; }

    friend EnergyTermType multi_smooth_func(SiteID, SiteID, LabelID, LabelID, void*);

private:

    int m_nSmoothArrays;
    int m_maxSmoothFuncs;
    std::map<Edge, FuncID> m_edgeFuncID;
    EnergyTermType **m_smoothArrayList;
    EnergyTermType *m_dataCost;
};

#endif // __GCOPTMULTISMOOTH_H__
