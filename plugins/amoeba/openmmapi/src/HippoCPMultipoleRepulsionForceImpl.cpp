#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/HippoCPMultipoleRepulsionForceImpl.h"
#include "openmm/amoebaKernels.h"

using namespace OpenMM;

void HippoCPMultipoleRepulsionForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcHippoCPMultipoleRepulsionForceKernel::Name(), context);
    kernel.getAs<CalcHippoCPMultipoleRepulsionForceKernel>().initialize(context.getSystem(), owner);
}

double HippoCPMultipoleRepulsionForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0)
        return kernel.getAs<CalcHippoCPMultipoleRepulsionForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> HippoCPMultipoleRepulsionForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcHippoCPMultipoleRepulsionForceKernel::Name());
    return names;
}
