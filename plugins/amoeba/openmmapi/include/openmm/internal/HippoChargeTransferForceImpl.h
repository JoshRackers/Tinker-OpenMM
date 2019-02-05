#ifndef OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_IMPL_H_
#define OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_IMPL_H_

#include "openmm/internal/ForceImpl.h"
#include "openmm/HippoChargeTransferForce.h"
#include "openmm/Kernel.h"

#include <string>

namespace OpenMM {

  class System;

  class OPENMM_EXPORT_AMOEBA HippoChargeTransferForceImpl : public ForceImpl {
  public:
    HippoChargeTransferForceImpl(const HippoChargeTransferForce& iowner)
      : owner(iowner) {}
    ~HippoChargeTransferForceImpl() {}

    void initialize(ContextImpl& context) ;

    const HippoChargeTransferForce& getOwner() const {
      return owner;
    }

    void updateContextState(ContextImpl& context) {}

    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) ;

    std::map<std::string, double>  getDefaultParameters() {
      return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }

    virtual std::vector<std::string> getKernelNames() ;

  private:
    const HippoChargeTransferForce& owner;
    Kernel kernel;    

  };

}
#endif
