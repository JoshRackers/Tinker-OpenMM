#ifdef WIN32
#define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/HippoChargeTransferForceImpl.h"
#include "openmm/amoebaKernels.h"

using namespace OpenMM;

void HippoChargeTransferForceImpl::initialize(ContextImpl& context) {
  kernel = context.getPlatform().createKernel(CalcHippoChargeTransferForceKernel::Name(), context);
  kernel.getAs<CalcHippoChargeTransferForceKernel>().initialize(context.getSystem(), owner);
}

double HippoChargeTransferForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if ((groups & (1 << owner.getForceGroup())) != 0)
    return kernel.getAs<CalcHippoChargeTransferForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}


std::vector<std::string> HippoChargeTransferForceImpl::getKernelNames() {
  std::vector<std::string> names;
  names.push_back(CalcHippoChargeTransferForceKernel::Name());
  return names;
}


