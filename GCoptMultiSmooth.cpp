#include "GCoptMultiSmooth.h"

////////////////////////////////////////////////////////////////////////////////////////////////
// Functions for the GCoptMultiSmooth, derived from GCoptimizationGeneralGraph
////////////////////////////////////////////////////////////////////////////////////////////////////

GCoptMultiSmooth::EnergyTermType multi_smooth_func(GCoptMultiSmooth::SiteID s1,
						  GCoptMultiSmooth::SiteID s2,
						  GCoptMultiSmooth::LabelID l1,
						  GCoptMultiSmooth::LabelID l2,
						  void *data)
{
    GCoptMultiSmooth *ob = static_cast<GCoptMultiSmooth*>(data);
    if(s1 > s2){
	// swap
	GCoptMultiSmooth::SiteID tmp = s2;
	s2 = s1;
	s1 = tmp;
    }
    GCoptMultiSmooth::FuncID fid = ob->m_edgeFuncID[GCoptMultiSmooth::Edge(s1,s2)];

    return ob->m_smoothArrayList[fid][l1*ob->m_num_labels+l2];

}

GCoptMultiSmooth::
GCoptMultiSmooth(SiteID num_sites,LabelID num_labels,int num_smoothFuncs)
    : GCoptimizationGeneralGraph(num_sites,num_labels),
      m_dataCost(NULL),
      m_nSmoothArrays(0),
      m_maxSmoothFuncs(num_smoothFuncs),
      m_smoothArrayList(NULL)
{
    m_smoothArrayList = new EnergyTermType*[m_maxSmoothFuncs];
    this->setSmoothCost(multi_smooth_func, static_cast<void*>(this));
}

//------------------------------------------------------------------

GCoptMultiSmooth::
~GCoptMultiSmooth()
{
    if(m_dataCost){
	delete [] m_dataCost;
	m_dataCost = NULL;
    }

    if(m_smoothArrayList){
	for(int idx=0;idx<m_nSmoothArrays;idx++){
	    delete m_smoothArrayList[idx];
	}
	m_nSmoothArrays = 0;
	delete [] m_smoothArrayList;
	m_smoothArrayList = NULL;
    }

}

//------------------------------------------------------------------

void
GCoptMultiSmooth::
copyDataCost(EnergyTermType *dataArray)
{
    size_t size = this->m_num_sites*this->m_num_labels;
    m_dataCost = new EnergyTermType[size];
    memcpy(m_dataCost, dataArray, size*sizeof(EnergyTermType));
    this->setDataCost(m_dataCost);
}

//------------------------------------------------------------------

void
GCoptMultiSmooth::
addEdges(SiteID *site1, SiteID *site2, int n_edges, FuncID smooth_func_id, EnergyTermType *weights)
{
    for(int idx=0;idx<n_edges;++idx){
	if(smooth_func_id >= m_nSmoothArrays){
	    handleError("Undefined smooth_func_id");
	}
	SiteID siteA = site1[idx];
	SiteID siteB = site2[idx];
	if(siteA > siteB){
	    // swap
	    SiteID tmp = siteB;
	    siteB = siteA;
	    siteA = tmp;
	}
	EnergyTermType w = weights ? weights[idx] : 1.0;
	setNeighbors(siteA, siteB, w);
	m_edgeFuncID[Edge(siteA, siteB)] = smooth_func_id;
    }
}

//------------------------------------------------------------------

GCoptMultiSmooth::FuncID
GCoptMultiSmooth::
addSmoothCost(EnergyTermType *smoothArray)
{
    if(m_nSmoothArrays >= m_maxSmoothFuncs){
	handleError("exceeded maximum number of edge penalty arrays");
    }
    size_t size = this->m_num_labels*this->m_num_labels;
    EnergyTermType* table = new EnergyTermType[size];
    memcpy(table,smoothArray,size*sizeof(EnergyTermType));
    m_smoothArrayList[m_nSmoothArrays] = table;
    m_nSmoothArrays++;
    return m_nSmoothArrays-1;
}

void
GCoptMultiSmooth::
setLabels(SiteID *sites, LabelID *labels, int n_labels)
{
    for(int idx=0; idx<n_labels; ++idx){
	this->setLabel(sites[idx], labels[idx]);
    }
}

void
GCoptMultiSmooth::
setAllLabels(LabelID *labels)
{

    memcpy(m_labeling, labels, m_num_sites*sizeof(LabelID));
    m_labelingInfoDirty = true;

}
