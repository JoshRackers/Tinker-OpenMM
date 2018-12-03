#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/HippoChargeTransferForce.h"
#include "openmm/internal/HippoChargeTransferForceImpl.h"

using namespace OpenMM;

ForceImpl* HippoChargeTransferForce::createImpl() const {
  return new HippoChargeTransferForceImpl(*this);
}
