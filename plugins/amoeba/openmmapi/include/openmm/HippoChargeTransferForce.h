#ifndef OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_H_
#define OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_H_

#include "internal/windowsExportAmoeba.h"
#include "openmm/Force.h"
#include <vector>

namespace OpenMM {

class OPENMM_EXPORT_AMOEBA HippoChargeTransferForce : public Force {
public:
    enum CovalentType { Covalent12 = 0, Covalent13 = 1, Covalent14 = 2, Covalent15 = 3, CovalentEnd = 4 };

    HippoChargeTransferForce() {
    	ct3scale = 0.0;
    	ct4scale = 0.4;
    	ct5scale = 0.8;
    }

    void setCTScales(double c3, double c4, double c5) {
    	ct3scale = c3;
    	ct4scale = c4;
    	ct5scale = c5;
    }

    void getCTScales(double& c3, double& c4, double& c5) const {
    	c3 = ct3scale;
    	c4 = ct4scale;
    	c5 = ct5scale;
    }

    int addCTSite(double alpha, double charge)
    {
        ctalpha.push_back(alpha);
        ctcharge.push_back(charge);

        std::vector<std::vector<int> > newlist;
        newlist.resize(CovalentEnd);
        covalentMaps.push_back(newlist);
        return ctalpha.size() - 1;
    }

    int getNumCTSites() const
    {
        return ctalpha.size();
    }

    double getCutoffDistance() const
    {
        return cutoffDistance;
    }

    void setCutoffDistance(double distance)
    {
        cutoffDistance = distance;
    }

    void getCTParameters(int index, double& alpha, double& charge) const
    {
        alpha = ctalpha[index];
        charge = ctcharge[index];
    }

    void setCTParameters(int index, double alpha, double charge)
    {
        ctalpha[index] = alpha;
        ctcharge[index] = charge;
    }

    void getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const
    {
        const std::vector<int>& list = covalentMaps[index][typeId];
        covalentAtoms.resize(list.size());
        for (int ii = 0; ii < covalentAtoms.size(); ++ii) {
            covalentAtoms[ii] = list[ii];
        }
    }

    void setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms)
    {
        std::vector<int>& list = covalentMaps[index][typeId];
        list.resize(covalentAtoms.size());
        for (int ii = 0; ii < covalentAtoms.size(); ++ii) {
            list[ii] = covalentAtoms[ii];
        }
    }

protected:
    ForceImpl* createImpl() const;
    double cutoffDistance;
    double ct3scale, ct4scale, ct5scale;
    std::vector<double> ctalpha, ctcharge;
    std::vector<std::vector<std::vector<int> > > covalentMaps;
};

} // namespace OpenMM

#endif
