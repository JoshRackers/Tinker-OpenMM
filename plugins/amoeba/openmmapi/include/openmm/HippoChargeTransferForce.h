#ifndef OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_H_
#define OPENMM_HIPPO_CHARGE_TRANSFER_FORCE_H_

#include "internal/windowsExportAmoeba.h"
#include "openmm/Force.h"
#include <vector>

namespace OpenMM {

class OPENMM_EXPORT_AMOEBA HippoChargeTransferForce : public Force {
public:
   enum MultipoleAxisTypes {
      ZThenX            = 0,
      Bisector          = 1,
      ZBisect           = 2,
      ThreeFold         = 3,
      ZOnly             = 4,
      NoAxisType        = 5,
      LastAxisTypeIndex = 6
   };

   enum CovalentType {
      Covalent12  = 0,
      Covalent13  = 1,
      Covalent14  = 2,
      Covalent15  = 3,
      CovalentEnd = 4
   };

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

   // charge transfer
   int addCTSite(double alpha, double charge) {
      ctalpha.push_back(alpha);
      ctcharge.push_back(charge);

      std::vector<std::vector<int> > newlist;
      newlist.resize(CovalentEnd);
      covalentMaps.push_back(newlist);
      return ctalpha.size() - 1;
   }

   int getNumCTSites() const { return ctalpha.size(); }

   double getCTTaperDistance() const { return ctTaperDistance; }

   double getCTCutoffDistance() const { return ctCutoffDistance; }

   void setCTCutoffDistance(double taper, double distance) {
      ctTaperDistance  = taper;
      ctCutoffDistance = distance;
   }

   void getCTParameters(int index, double& alpha, double& charge) const {
      alpha  = ctalpha[index];
      charge = ctcharge[index];
   }

   void setCTParameters(int index, double alpha, double charge) {
      ctalpha[index]  = alpha;
      ctcharge[index] = charge;
   }

   // multipole
   int addMultipole(double charge, const std::vector<double>& molecularDipole,
      const std::vector<double>& molecularQuadrupole, int axisType,
      int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY) {
      multipoles.push_back(
         MultipoleInfo(charge, molecularDipole, molecularQuadrupole, axisType,
            multipoleAtomZ, multipoleAtomX, multipoleAtomY));
      return multipoles.size() - 1;
   }

   void getMultipoleParameters(int index, double& charge,
      std::vector<double>& molecularDipole,
      std::vector<double>& molecularQuadrupole, int& axisType,
      int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY) const {
      charge = multipoles[index].charge;

      molecularDipole.resize(3);
      molecularDipole[0] = multipoles[index].molecularDipole[0];
      molecularDipole[1] = multipoles[index].molecularDipole[1];
      molecularDipole[2] = multipoles[index].molecularDipole[2];

      molecularQuadrupole.resize(9);
      molecularQuadrupole[0] = multipoles[index].molecularQuadrupole[0];
      molecularQuadrupole[1] = multipoles[index].molecularQuadrupole[1];
      molecularQuadrupole[2] = multipoles[index].molecularQuadrupole[2];
      molecularQuadrupole[3] = multipoles[index].molecularQuadrupole[3];
      molecularQuadrupole[4] = multipoles[index].molecularQuadrupole[4];
      molecularQuadrupole[5] = multipoles[index].molecularQuadrupole[5];
      molecularQuadrupole[6] = multipoles[index].molecularQuadrupole[6];
      molecularQuadrupole[7] = multipoles[index].molecularQuadrupole[7];
      molecularQuadrupole[8] = multipoles[index].molecularQuadrupole[8];

      axisType       = multipoles[index].axisType;
      multipoleAtomZ = multipoles[index].multipoleAtomZ;
      multipoleAtomX = multipoles[index].multipoleAtomX;
      multipoleAtomY = multipoles[index].multipoleAtomY;
   }

   void setMultipoleParameters(int index, double charge,
      const std::vector<double>& molecularDipole,
      const std::vector<double>& molecularQuadrupole, int axisType,
      int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY) {
      multipoles[index].charge = charge;

      multipoles[index].molecularDipole[0] = molecularDipole[0];
      multipoles[index].molecularDipole[1] = molecularDipole[1];
      multipoles[index].molecularDipole[2] = molecularDipole[2];

      multipoles[index].molecularQuadrupole[0] = molecularQuadrupole[0];
      multipoles[index].molecularQuadrupole[1] = molecularQuadrupole[1];
      multipoles[index].molecularQuadrupole[2] = molecularQuadrupole[2];
      multipoles[index].molecularQuadrupole[3] = molecularQuadrupole[3];
      multipoles[index].molecularQuadrupole[4] = molecularQuadrupole[4];
      multipoles[index].molecularQuadrupole[5] = molecularQuadrupole[5];
      multipoles[index].molecularQuadrupole[6] = molecularQuadrupole[6];
      multipoles[index].molecularQuadrupole[7] = molecularQuadrupole[7];
      multipoles[index].molecularQuadrupole[8] = molecularQuadrupole[8];

      multipoles[index].axisType       = axisType;
      multipoles[index].multipoleAtomZ = multipoleAtomZ;
      multipoles[index].multipoleAtomX = multipoleAtomX;
      multipoles[index].multipoleAtomY = multipoleAtomY;
   }

   bool getUsePME() const { return usePME; }

   void setUsePME(bool ifUsePME) { usePME = ifUsePME; }

   double getPMECutoffDistance() const { return pmeCutoffDistance; }

   void setPMECutoffDistance(double distance) { pmeCutoffDistance = distance; }

   void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
      alpha = ewaldAlpha;
      nx    = pmenx;
      ny    = pmeny;
      nz    = pmenz;
   }

   void setPMEParameters(double alpha, int nx, int ny, int nz) {
      ewaldAlpha = alpha;
      pmenx      = nx;
      pmeny      = ny;
      pmenz      = nz;
   }

   double getEwaldErrorTolerance() const { return ewaldErrorTolerance; }

   void setEwaldErrorTolerance(double tol) { ewaldErrorTolerance = tol; }

   int getPMEOrder() const { return pmeorder; }

   void setPMEOrder(int order) { pmeorder = order; }

   // repulsion
   double getRepelTaperDistance() const { return repelTaperDistance; }

   double getRepelCutoffDistance() const { return repelCutoffDistance; }

   void setRepelCutoffDistance(double taper, double distance) {
      repelTaperDistance  = taper;
      repelCutoffDistance = distance;
   }

   int addRepelSite(double inputSize, double inputDmp, double inputElepr) {
      sizpr.push_back(inputSize);
      dmppr.push_back(inputDmp);
      elepr.push_back(inputElepr);

      return elepr.size() - 1;
   }

   void getRepelParameters(int index, double& outputSize, double& outputDmp,
      double& outputElepr) const {
      outputSize  = sizpr[index];
      outputDmp   = dmppr[index];
      outputElepr = elepr[index];
   }

   void setRepelParameters(
      int index, double inputSize, double inputDmp, double inputElepr) {
      sizpr[index] = inputSize;
      dmppr[index] = inputDmp;
      elepr[index] = inputElepr;
   }

   // charge penetration electrostatics
   double getCPMultipoleCutoffDistance() const {
      return multipoleCutoffDistance;
   }

   void setCPMultipoleCutoffDistance(double distance) {
      multipoleCutoffDistance = distance;
   }

   int addCPMultipoleSite(
      double inputCore, double inputVal, double inputAlpha) {
      pcore.push_back(inputCore);
      pval.push_back(inputVal);
      palpha.push_back(inputAlpha);

      return pcore.size() - 1;
   }

   void getCPMultipoleParameters(int index, double& outputCore,
      double& outputVal, double& outputAlpha) const {
      outputCore  = pcore[index];
      outputVal   = pval[index];
      outputAlpha = palpha[index];
   }

   void setCPMultipoleParameters(
      int index, double inputCore, double inputVal, double inputAlpha) {
      pcore[index]  = inputCore;
      pval[index]   = inputVal;
      palpha[index] = inputAlpha;
   }

   void getCovalentMap(
      int index, CovalentType typeId, std::vector<int>& covalentAtoms) const {
      const std::vector<int>& list = covalentMaps[index][typeId];
      covalentAtoms.resize(list.size());
      for (int ii = 0; ii < covalentAtoms.size(); ++ii) {
         covalentAtoms[ii] = list[ii];
      }
   }

   void setCovalentMap(
      int index, CovalentType typeId, const std::vector<int>& covalentAtoms) {
      std::vector<int>& list = covalentMaps[index][typeId];
      list.resize(covalentAtoms.size());
      for (int ii = 0; ii < covalentAtoms.size(); ++ii) {
         list[ii] = covalentAtoms[ii];
      }
   }

protected:
   ForceImpl* createImpl() const;

   // multipole
   class MultipoleInfo {
   public:
      int    axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
      double charge;
      std::vector<double>            molecularDipole;
      std::vector<double>            molecularQuadrupole;
      std::vector<std::vector<int> > covalentInfo;

      MultipoleInfo() {
         axisType = multipoleAtomZ = multipoleAtomX = multipoleAtomY = -1;

         charge = 0.0;
         molecularDipole.resize(3);
         molecularQuadrupole.resize(9);
      }

      MultipoleInfo(double          charge,
         const std::vector<double>& inputMolecularDipole,
         const std::vector<double>& inputMolecularQuadrupole, int axisType,
         int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY)
         : axisType(axisType)
         , multipoleAtomZ(multipoleAtomZ)
         , multipoleAtomX(multipoleAtomX)
         , multipoleAtomY(multipoleAtomY)
         , charge(charge) {
         molecularDipole.resize(3);
         molecularDipole[0] = inputMolecularDipole[0];
         molecularDipole[1] = inputMolecularDipole[1];
         molecularDipole[2] = inputMolecularDipole[2];

         molecularQuadrupole.resize(9);
         molecularQuadrupole[0] = inputMolecularQuadrupole[0];
         molecularQuadrupole[1] = inputMolecularQuadrupole[1];
         molecularQuadrupole[2] = inputMolecularQuadrupole[2];
         molecularQuadrupole[3] = inputMolecularQuadrupole[3];
         molecularQuadrupole[4] = inputMolecularQuadrupole[4];
         molecularQuadrupole[5] = inputMolecularQuadrupole[5];
         molecularQuadrupole[6] = inputMolecularQuadrupole[6];
         molecularQuadrupole[7] = inputMolecularQuadrupole[7];
         molecularQuadrupole[8] = inputMolecularQuadrupole[8];
      }
   };
   std::vector<MultipoleInfo> multipoles;
   bool                       usePME;
   double pmeCutoffDistance, ewaldAlpha, ewaldErrorTolerance;
   int    pmeorder, pmenx, pmeny, pmenz;

   // charge transfer
   double              ctTaperDistance, ctCutoffDistance;
   double              ct3scale, ct4scale, ct5scale;
   std::vector<double> ctalpha, ctcharge;

   // repulsion
   double              repelTaperDistance, repelCutoffDistance;
   std::vector<double> sizpr, dmppr, elepr;

   // charge penetration electrostatics
   double              multipoleCutoffDistance;
   std::vector<double> pcore, pval, palpha;

   std::vector<std::vector<std::vector<int> > > covalentMaps;
};

} // namespace OpenMM

#endif
