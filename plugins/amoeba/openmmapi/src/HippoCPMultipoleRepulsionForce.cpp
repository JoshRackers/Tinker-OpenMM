#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/HippoCPMultipoleRepulsionForce.h"
#include "openmm/internal/HippoCPMultipoleRepulsionForceImpl.h"

using namespace OpenMM;

ForceImpl* HippoCPMultipoleRepulsionForce::createImpl() const {
	return new HippoCPMultipoleRepulsionForceImpl(*this);
}
