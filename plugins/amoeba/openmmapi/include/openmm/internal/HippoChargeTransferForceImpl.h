#ifndef OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_IMPL_H_
#define OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_IMPL_H_

#include "openmm/HippoChargeTransferForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"

#include <string>

namespace OpenMM {
class System;

class OPENMM_EXPORT_AMOEBA HippoChargeTransferForceImpl : public ForceImpl {
public:
   HippoChargeTransferForceImpl(const HippoChargeTransferForce& iowner)
      : owner(iowner) {}

   ~HippoChargeTransferForceImpl() {}

   void initialize(ContextImpl& context);

   const HippoChargeTransferForce& getOwner() const { return owner; }

   void updateContextState(ContextImpl& context) {}

   double calcForcesAndEnergy(
      ContextImpl& context, bool includeForces, bool includeEnergy, int groups);

   std::map<std::string, double> getDefaultParameters() {
      // This force field doesn't define any parameters.
      return std::map<std::string, double>();
   }

   virtual std::vector<std::string> getKernelNames();

private:
   const HippoChargeTransferForce& owner;
   Kernel                          kernel;
};
} // namespace OpenMM

#endif
