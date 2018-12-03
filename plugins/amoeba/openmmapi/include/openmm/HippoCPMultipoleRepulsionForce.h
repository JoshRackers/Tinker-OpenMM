#ifndef OPENMM_HIPPO_CP_MULTIPOLE_REPULSION_H_
#define OPENMM_HIPPO_CP_MULTIPOLE_REPULSION_H_

#include "openmm/Force.h"
#include "internal/windowsExportAmoeba.h"
#include <vector>

namespace OpenMM {

class OPENMM_EXPORT_AMOEBA HippoCPMultipoleRepulsionForce : public Force {
public:
    HippoCPMultipoleRepulsionForce() {}

protected:
    ForceImpl* createImpl() const;
};

}

#endif
