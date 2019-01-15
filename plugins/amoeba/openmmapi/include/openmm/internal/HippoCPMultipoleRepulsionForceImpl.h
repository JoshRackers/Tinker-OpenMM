#ifndef OPENMM_HIPPO_CP_MULTIPOLE_REPULSION_FORCE_IMPL_H_
#define OPENMM_HIPPO_CP_MULTIPOLE_REPULSION_FORCE_IMPL_H_

#include "openmm/internal/ForceImpl.h"
#include "openmm/HippoCPMultipoleRepulsionForce.h"
#include "openmm/Kernel.h"

#include <string>

namespace OpenMM {

class System;

class OPENMM_EXPORT_AMOEBA HippoCPMultipoleRepulsionForceImpl : public ForceImpl {
public:
    HippoCPMultipoleRepulsionForceImpl(const HippoCPMultipoleRepulsionForce& iowner)
    : owner(iowner) {}

    ~HippoCPMultipoleRepulsionForceImpl() {}

    void initialize(ContextImpl& context);
    const HippoCPMultipoleRepulsionForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context) {}
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();

private:
    const HippoCPMultipoleRepulsionForce& owner;
    Kernel kernel;
};

}

#endif
