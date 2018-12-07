class CudaCalcHippoChargeTransferForceKernel::ForceInfo : public CudaForceInfo {
public:
   ForceInfo(const HippoChargeTransferForce& force)
      : force(force) {}

   bool areParticlesIdentical(int p1, int p2) {
      double a1, a2, c1, c2;
      force.getCTParameters(p1, a1, c1);
      force.getCTParameters(p2, a2, c2);
      if (a1 != a2 || c1 != c2) {
         return false;
      } else {
         return true;
      }

      // (maybe) FIXME for multipole info and repulsion info
   }

   int getNumParticleGroups() { return force.getNumCTSites(); }

   void getParticlesInGroup(int index, vector<int>& particles) {
      force.getCovalentMap(
         index, HippoChargeTransferForce::Covalent12, particles);
      particles.push_back(index);
   }

   bool areGroupsIdentical(int group1, int group2) { return group1 == group2; }

private:
   const HippoChargeTransferForce& force;
};

CudaCalcHippoChargeTransferForceKernel::CudaCalcHippoChargeTransferForceKernel(
   std::string name, const Platform& platform, CudaContext& cu,
   const System& system)
   : CalcHippoChargeTransferForceKernel(name, platform)
   , hasInitializedScaleFactors(false)
   , cu(cu)
   , system(system) {}

CudaCalcHippoChargeTransferForceKernel::
   ~CudaCalcHippoChargeTransferForceKernel() {
   cu.setAsCurrent();
}

class HippoParameterCopier {
private:
   bool           useDoublePrecision;
   CudaContext&   cu;
   CudaArray&     array;
   vector<double> arrayVec;

public:
   HippoParameterCopier(CudaArray& inarray, CudaContext& incu, int arraySize,
      std::string arrayName, bool useDouble)
      : useDoublePrecision(useDouble)
      , cu(incu)
      , array(inarray)
      , arrayVec(arraySize) {
      if (useDoublePrecision) {
         array.initialize<double>(cu, arraySize, arrayName);
      } else {
         array.initialize<float>(cu, arraySize, arrayName);
      }
   }

   void operator()(int pos, double param) {
      double* ptrd = &arrayVec[0];
      float*  ptrf = (float*)ptrd;
      if (useDoublePrecision) {
         ptrd[pos] = param;
      } else {
         ptrf[pos] = (float)param;
      }
   }

   void upload() { array.upload(&arrayVec[0]); }
};

void CudaCalcHippoChargeTransferForceKernel::initialize(
   const System& system, const HippoChargeTransferForce& force) {
   try {
      cu.setAsCurrent();
      map<string, string> defines;

      bool useDoublePrecision = cu.getUseDoublePrecision();
      int  numAtoms           = cu.getNumAtoms();
      int  paddedNumAtoms     = cu.getPaddedNumAtoms();
      int  elementSize = (useDoublePrecision ? sizeof(double) : sizeof(float));
      defines["NUM_ATOMS"]        = cu.intToString(numAtoms);
      defines["PADDED_NUM_ATOMS"] = cu.intToString(paddedNumAtoms);
      defines["USE_CUTOFF"]       = "";

      // axis, charge from posq, dipole, and quadrupole
      axisInfo.initialize<int4>(cu, paddedNumAtoms, "axisInfo");
      CudaArray& posq = cu.getPosq();
      if (useDoublePrecision) {
         localFrameDipoles.initialize<double>(
            cu, 3 * paddedNumAtoms, "localFrameDipoles");
         localFrameQuadrupoles.initialize<double>(
            cu, 5 * paddedNumAtoms, "localFrameQuadrupoles");
         globalFrameDipoles.initialize<double>(
            cu, 3 * paddedNumAtoms, "globalFrameDipoles");
         globalFrameQuadrupoles.initialize<double>(
            cu, 5 * paddedNumAtoms, "globalFrameQuadrupoles");
      } else {
         localFrameDipoles.initialize<float>(
            cu, 3 * paddedNumAtoms, "localFrameDipoles");
         localFrameQuadrupoles.initialize<float>(
            cu, 5 * paddedNumAtoms, "localFrameQuadrupoles");
         globalFrameDipoles.initialize<float>(
            cu, 3 * paddedNumAtoms, "globalFrameDipoles");
         globalFrameQuadrupoles.initialize<float>(
            cu, 5 * paddedNumAtoms, "globalFrameQuadrupoles");
      }

      vector<int4>    axisInfoVec(axisInfo.getSize());
      vector<double4> posqVec(posq.getSize());
      vector<double>  localFrameDipolesVec(localFrameDipoles.getSize());
      vector<double>  localFrameQuadrupolesVec(localFrameQuadrupoles.getSize());

      double4* posqd        = &posqVec[0];
      double*  localdipoled = &localFrameDipolesVec[0];
      double*  localquadd   = &localFrameQuadrupolesVec[0];
      float4*  posqf        = (float4*)posqd;
      float*   localdipolef = (float*)localdipoled;
      float*   localquadf   = (float*)localquadd;

      for (int ii = 0; ii < numAtoms; ++ii) {
         int            axisType, atomz, atomx, atomy;
         double         charge;
         vector<double> dipole, quadrupole;
         force.getMultipoleParameters(
            ii, charge, dipole, quadrupole, axisType, atomz, atomx, atomy);

         axisInfoVec[ii] = make_int4(atomx, atomy, atomz, axisType);
         if (useDoublePrecision) {
            posqd[ii]                = make_double4(0, 0, 0, charge);
            localdipoled[3 * ii]     = dipole[0];
            localdipoled[3 * ii + 1] = dipole[1];
            localdipoled[3 * ii + 2] = dipole[2];
            localquadd[5 * ii]       = quadrupole[0]; // xx
            localquadd[5 * ii + 1]   = quadrupole[1]; // xy
            localquadd[5 * ii + 2]   = quadrupole[2]; // xz
            localquadd[5 * ii + 3]   = quadrupole[4]; // yy
            localquadd[5 * ii + 4]   = quadrupole[5]; // yz
                                                      // zz is unnecessary
         } else {
            posqf[ii]                = make_float4(0, 0, 0, (float)charge);
            localdipolef[3 * ii]     = (float)dipole[0];
            localdipolef[3 * ii + 1] = (float)dipole[1];
            localdipolef[3 * ii + 2] = (float)dipole[2];
            localquadf[5 * ii]       = (float)quadrupole[0]; // xx
            localquadf[5 * ii + 1]   = (float)quadrupole[1]; // xy
            localquadf[5 * ii + 2]   = (float)quadrupole[2]; // xz
            localquadf[5 * ii + 3]   = (float)quadrupole[4]; // yy
            localquadf[5 * ii + 4]   = (float)quadrupole[5]; // yz
            // zz is unnecessary
         }
      }

      axisInfo.upload(axisInfoVec);
      posq.upload(&posqVec[0]);
      localFrameDipoles.upload(&localFrameDipolesVec[0]);
      localFrameQuadrupoles.upload(&localFrameQuadrupolesVec[0]);

      // charge transfer
      double chgtrntaper      = force.getCTTaperDistance();
      double chgtrncutoff     = force.getCTCutoffDistance();
      defines["CHGTRN_TAPER"] = cu.doubleToString(chgtrntaper);
      defines["CHGTRN_CUTOFF_SQUARED"]
         = cu.doubleToString(chgtrncutoff * chgtrncutoff);
      if (chgtrntaper < chgtrncutoff) {
         defines["CHGTRN_TAPER_C3"]
            = cu.doubleToString(10 / pow(chgtrntaper - chgtrncutoff, 3.0));
         defines["CHGTRN_TAPER_C4"]
            = cu.doubleToString(15 / pow(chgtrntaper - chgtrncutoff, 4.0));
         defines["CHGTRN_TAPER_C5"]
            = cu.doubleToString(6 / pow(chgtrntaper - chgtrncutoff, 5.0));
      } else {
         defines["CHGTRN_TAPER_C3"] = "0";
         defines["CHGTRN_TAPER_C4"] = "0";
         defines["CHGTRN_TAPER_C5"] = "0";
      }

      double cscales[3];
      force.getCTScales(cscales[0], cscales[1], cscales[2]);
      defines["CHARGETRANSFER13SCALE"] = cu.doubleToString(cscales[0]);
      defines["CHARGETRANSFER14SCALE"] = cu.doubleToString(cscales[1]);
      defines["CHARGETRANSFER15SCALE"] = cu.doubleToString(cscales[2]);

      if (useDoublePrecision) {
         chgct.initialize<double>(cu, paddedNumAtoms, "chgct");
         dmpct.initialize<double>(cu, paddedNumAtoms, "dmpct");
      } else {
         chgct.initialize<float>(cu, paddedNumAtoms, "chgct");
         dmpct.initialize<float>(cu, paddedNumAtoms, "dmpct");
      }
      vector<double> chgctVec(chgct.getSize());
      vector<double> dmpctVec(dmpct.getSize());

      double* chgctd = &chgctVec[0];
      double* dmpctd = &dmpctVec[0];
      float*  chgctf = (float*)chgctd;
      float*  dmpctf = (float*)dmpctd;

      for (int ii = 0; ii < numAtoms; ++ii) {
         double a, c;
         force.getCTParameters(ii, a, c);
         if (useDoublePrecision) {
            chgctd[ii] = c;
            dmpctd[ii] = a;
         } else {
            chgctf[ii] = (float)c;
            dmpctf[ii] = (float)a;
         }
      }
      chgct.upload(&chgctVec[0]);
      dmpct.upload(&dmpctVec[0]);

      // repulsion
      double repeltaper      = force.getRepelTaperDistance();
      double repelcutoff     = force.getRepelCutoffDistance();
      defines["REPEL_TAPER"] = cu.doubleToString(repeltaper);
      defines["REPEL_CUTOFF_SQUARED"]
         = cu.doubleToString(repelcutoff * repelcutoff);

      if (useDoublePrecision) {
         sizpr.initialize<double>(cu, paddedNumAtoms, "sizpr");
         dmppr.initialize<double>(cu, paddedNumAtoms, "dmppr");
         elepr.initialize<double>(cu, paddedNumAtoms, "elepr");
      } else {
         sizpr.initialize<float>(cu, paddedNumAtoms, "sizpr");
         dmppr.initialize<float>(cu, paddedNumAtoms, "dmppr");
         elepr.initialize<float>(cu, paddedNumAtoms, "elepr");
      }
      vector<double> sizprVec(sizpr.getSize());
      vector<double> dmpprVec(dmppr.getSize());
      vector<double> eleprVec(elepr.getSize());

      double* sizprd = &sizprVec[0];
      double* dmpprd = &dmpprVec[0];
      double* eleprd = &eleprVec[0];
      float*  sizprf = (float*)sizprd;
      float*  dmpprf = (float*)dmpprd;
      float*  eleprf = (float*)eleprd;

      for (int ii = 0; ii < numAtoms; ++ii) {
         double s, d, e;
         force.getRepelParameters(ii, s, d, e);
         if (useDoublePrecision) {
            sizprd[ii] = s;
            dmpprd[ii] = d;
            eleprd[ii] = e;
         } else {
            sizprf[ii] = (float)s;
            dmpprf[ii] = (float)d;
            eleprf[ii] = (float)e;
         }
      }
      sizpr.upload(&sizprVec[0]);
      dmppr.upload(&dmpprVec[0]);
      elepr.upload(&eleprVec[0]);

      // charge penetration electrostatics
      double chgpencutoff = force.getCPMultipoleCutoffDistance();
      defines["CHGPEN_CUTOFF_SQUARED"]
         = cu.doubleToString(chgpencutoff * chgpencutoff);

      HippoParameterCopier copy_pcore(
         pcore, cu, paddedNumAtoms, "pcore", useDoublePrecision);
      HippoParameterCopier copy_pval(
         pval, cu, paddedNumAtoms, "pval", useDoublePrecision);
      HippoParameterCopier copy_palpha(
         palpha, cu, paddedNumAtoms, "palpha", useDoublePrecision);
      for (int ii = 0; ii < numAtoms; ++ii) {
         double c, v, a;
         force.getCPMultipoleParameters(ii, c, v, a);
         copy_pcore(ii, c);
         copy_pval(ii, v);
         copy_palpha(ii, a);
      }
      copy_pcore.upload();
      copy_pval.upload();
      copy_palpha.upload();

      // exclusions
      vector<vector<int>> exclusions(numAtoms);
      for (int i = 0; i < numAtoms; i++) {
         vector<int> atoms;
         set<int>    allAtoms;
         allAtoms.insert(i);

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent12, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii) {}
         allAtoms.insert(atoms.begin(), atoms.end());

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent13, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii) {}
         allAtoms.insert(atoms.begin(), atoms.end());
         for (set<int>::const_iterator iter = allAtoms.begin();
              iter != allAtoms.end(); ++iter)
            covalentFlagValues.push_back(make_int3(i, *iter, 0));

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent14, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii) {}
         allAtoms.insert(atoms.begin(), atoms.end());
         for (int j = 0; j < (int)atoms.size(); j++)
            covalentFlagValues.push_back(make_int3(i, atoms[j], 1));

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent15, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii) {}
         for (int j = 0; j < (int)atoms.size(); j++)
            covalentFlagValues.push_back(make_int3(i, atoms[j], 2));
         allAtoms.insert(atoms.begin(), atoms.end());

         exclusions[i].insert(
            exclusions[i].end(), allAtoms.begin(), allAtoms.end());
      }

      for (int i = 0; i < covalentFlagValues.size(); ++i) {
         int atom1 = covalentFlagValues[i].x;
         int atom2 = covalentFlagValues[i].y;
         int value = covalentFlagValues[i].z;
         int f1    = (value == 0 || value == 1 ? 1 : 0);
         int f2    = (value == 0 || value == 2 ? 1 : 0);
      }

      set<pair<int, int>> tilesWithExclusions;
      for (int atom1 = 0; atom1 < (int)exclusions.size(); ++atom1) {
         int x = atom1 / CudaContext::TileSize;
         for (int j = 0; j < (int)exclusions[atom1].size(); ++j) {
            int atom2 = exclusions[atom1][j];
            int y     = atom2 / CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
         }
      }

      // define macros for cuda source code

      defines["TILE_SIZE"]  = cu.intToString(CudaContext::TileSize);
      int numExclusionTiles = tilesWithExclusions.size();
      defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
      int numContexts = cu.getPlatformData().contexts.size();
      int startExclusionIndex
         = cu.getContextIndex() * numExclusionTiles / numContexts;
      int endExclusionIndex
         = (cu.getContextIndex() + 1) * numExclusionTiles / numContexts;
      defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
      defines["LAST_EXCLUSION_TILE"]  = cu.intToString(endExclusionIndex);

      int    maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
      double energyAndForceMemory = 8.0 * elementSize + 2.0 * sizeof(int);
      energyAndForceThreads
         = min(maxThreads, cu.computeThreadBlockSize(energyAndForceMemory));
      defines["THREAD_BLOCK_SIZE"] = cu.intToString(energyAndForceThreads);

      printf(" start compiling CT module\n");
      CUmodule hippoModule = cu.createModule(CudaKernelSources::vectorOps
            + CudaAmoebaKernelSources::hippoRotpole
            + CudaAmoebaKernelSources::hippoCTexample,
         defines);
      printf(" compiling CT module 100 \n");
      fflush(stdout);
      energyAndForceKernel = cu.getKernel(hippoModule, "computeChargeTransfer");
      printf(" end compiling CT module\n");

      rotpoleKernel = cu.getKernel(hippoModule, "hippoRotpole");

      double cutoffDistanceForNBList = max(chgtrncutoff, repelcutoff);
      cu.getNonbondedUtilities().addInteraction(true, true, true,
         cutoffDistanceForNBList, exclusions, "", force.getForceGroup());
      printf(" before setUsePadding\n");
      cu.getNonbondedUtilities().setUsePadding(false);
      cu.addForce(new ForceInfo(force));
   } catch (std::exception& e) {
      printf(
         " exception thrown from "
         "CalcHippoChargeTransferForceKernel::initialize() -- %s\n",
         e.what());
   }
}

void CudaCalcHippoChargeTransferForceKernel::initializeScaleFactors() {
   hasInitializedScaleFactors = true;
   CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();

   // Figure out the covalent flag values to use for each atom pair.

   vector<ushort2> exclusionTiles;
   nb.getExclusionTiles().download(exclusionTiles);
   map<pair<int, int>, int> exclusionTileMap;
   printf(
      " -- total number of exclusions tiles %d\n", (int)exclusionTiles.size());
   for (int i = 0; i < (int)exclusionTiles.size(); i++) {
      ushort2 tile                                = exclusionTiles[i];
      exclusionTileMap[make_pair(tile.x, tile.y)] = i;
   }
   covalentFlags.initialize<uint2>(
      cu, nb.getExclusions().getSize(), "covalentFlags");
   vector<uint2> covalentFlagsVec(
      nb.getExclusions().getSize(), make_uint2(0, 0));
   for (int i = 0; i < (int)covalentFlagValues.size(); i++) {
      int atom1   = covalentFlagValues[i].x;
      int atom2   = covalentFlagValues[i].y;
      int value   = covalentFlagValues[i].z;
      int x       = atom1 / CudaContext::TileSize;
      int offset1 = atom1 - x * CudaContext::TileSize;
      int y       = atom2 / CudaContext::TileSize;
      int offset2 = atom2 - y * CudaContext::TileSize;
      int f1      = (value == 0 || value == 1 ? 1 : 0);
      int f2      = (value == 0 || value == 2 ? 1 : 0);
      if (x == y) {
         int index = exclusionTileMap[make_pair(x, y)] * CudaContext::TileSize;
         covalentFlagsVec[index + offset1].x |= f1 << offset2;
         covalentFlagsVec[index + offset1].y |= f2 << offset2;
         covalentFlagsVec[index + offset2].x |= f1 << offset1;
         covalentFlagsVec[index + offset2].y |= f2 << offset1;
      } else if (x > y) {
         int index = exclusionTileMap[make_pair(x, y)] * CudaContext::TileSize;
         covalentFlagsVec[index + offset1].x |= f1 << offset2;
         covalentFlagsVec[index + offset1].y |= f2 << offset2;
      } else {
         int index = exclusionTileMap[make_pair(y, x)] * CudaContext::TileSize;
         covalentFlagsVec[index + offset2].x |= f1 << offset1;
         covalentFlagsVec[index + offset2].y |= f2 << offset1;
      }
   }
   covalentFlags.upload(covalentFlagsVec);
}

double CudaCalcHippoChargeTransferForceKernel::execute(
   ContextImpl& context, bool includeForces, bool includeEnergy) {
   printf(" -- CT execute first line\n");
   if (!hasInitializedScaleFactors) {
      initializeScaleFactors();
   }

   void* rotpoleArgs[] = {&cu.getPosq().getDevicePointer(),
      &axisInfo.getDevicePointer(), &localFrameDipoles.getDevicePointer(),
      &localFrameQuadrupoles.getDevicePointer(),
      &globalFrameDipoles.getDevicePointer(),
      &globalFrameQuadrupoles.getDevicePointer()};
   cu.executeKernel(rotpoleKernel, rotpoleArgs, cu.getNumAtoms());

   printf(" -- CT execute \n");

   CudaNonbondedUtilities& nb             = cu.getNonbondedUtilities();
   int                     startTileIndex = nb.getStartTileIndex();
   int                     numTileIndices = nb.getNumTiles();
   unsigned int            maxTiles       = nb.getInteractingTiles().getSize();

   void* argsEnergyAndForce[] = {&cu.getForce().getDevicePointer(),
      &cu.getEnergyBuffer().getDevicePointer(),
      &cu.getPosq().getDevicePointer(), &covalentFlags.getDevicePointer(),
      &nb.getExclusionTiles().getDevicePointer(), &startTileIndex,
      &numTileIndices,

      &nb.getInteractingTiles().getDevicePointer(),
      &nb.getInteractionCount().getDevicePointer(),
      cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
      cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
      cu.getPeriodicBoxVecZPointer(), &maxTiles,
      &nb.getBlockCenters().getDevicePointer(),
      &nb.getInteractingAtoms().getDevicePointer(),

      &chgct.getDevicePointer(), &dmpct.getDevicePointer()

                                    ,
      &localFrameDipoles.getDevicePointer(),
      &localFrameQuadrupoles.getDevicePointer(), &sizpr.getDevicePointer(),
      &dmppr.getDevicePointer(), &elepr.getDevicePointer()};

   int numForceThreadBlocks = nb.getNumForceThreadBlocks();
   printf(" -- numForceThreadBlocks %d energyAndForceThreads = %d\n",
      numForceThreadBlocks, energyAndForceThreads);
   cu.executeKernel(energyAndForceKernel, argsEnergyAndForce,
      numForceThreadBlocks * energyAndForceThreads, energyAndForceThreads);

   return 0.0;
}

void CudaCalcHippoChargeTransferForceKernel::copyParametersToContext(
   ContextImpl& context, const HippoChargeTransferForce& force) {
   cu.setAsCurrent();
}
