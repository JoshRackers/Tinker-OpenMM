#include <numeric>

class CudaCalcHippoChargeTransferForceKernel::ForceInfo : public CudaForceInfo
{
   public:
   ForceInfo(const HippoChargeTransferForce &force)
       : force(force) {}

   bool areParticlesIdentical(int p1, int p2)
   {
      double a1, a2, c1, c2;
      force.getCTParameters(p1, a1, c1);
      force.getCTParameters(p2, a2, c2);
      if (a1 != a2 || c1 != c2)
      {
         return false;
      }
      else
      {
         return true;
      }

      // (maybe) FIXME for multipole info and repulsion info
   }

   int getNumParticleGroups() { return force.getNumCTSites(); }

   void getParticlesInGroup(int index, vector<int> &particles)
   {
      force.getCovalentMap(
          index, HippoChargeTransferForce::Covalent12, particles);
      particles.push_back(index);
   }

   bool areGroupsIdentical(int group1, int group2) { return group1 == group2; }

 private:
   const HippoChargeTransferForce &force;
};

CudaCalcHippoChargeTransferForceKernel::CudaCalcHippoChargeTransferForceKernel(
    std::string name, const Platform &platform, CudaContext &cu,
    const System &system)
    : CalcHippoChargeTransferForceKernel(name, platform), hasInitializedScaleFactors(false), hasInitializedFFT(false), cu(cu), system(system) {}

CudaCalcHippoChargeTransferForceKernel::
    ~CudaCalcHippoChargeTransferForceKernel()
{
   cu.setAsCurrent();

   if (hasInitializedFFT)
      cufftDestroy(fft);
}

class HippoParameterCopier
{
 private:
   bool useDoublePrecision;
   CudaContext &cu;
   CudaArray &array;
   vector<double> arrayVec;

 public:
   HippoParameterCopier(CudaArray &inarray, CudaContext &incu, int arraySize, std::string arrayName, bool useDouble)
       : useDoublePrecision(useDouble), cu(incu), array(inarray), arrayVec(arraySize)
   {
      if (useDoublePrecision)
      {
          array.initialize<double>(cu, arraySize, arrayName);
      }
      else
      {
          array.initialize<float>(cu, arraySize, arrayName);
      }
   }

   void operator()(int pos, double param)
   {
      double *ptrd = &arrayVec[0];
      float *ptrf = (float *)ptrd;
      if (useDoublePrecision)
      {
         ptrd[pos] = param;
      }
      else
      {
         ptrf[pos] = (float)param;
      }
   }

   void upload() { array.upload(&arrayVec[0]); }
   void download(void *ptr) { array.download(ptr); }
};

void CudaCalcHippoChargeTransferForceKernel::initialize(const System &system, const HippoChargeTransferForce &force)
{
   try
   {
      cu.setAsCurrent();
      map<string, string> defines;

      useDoublePrecision = cu.getUseDoublePrecision();
      numAtoms = cu.getNumAtoms();
      paddedNumAtoms = cu.getPaddedNumAtoms();
      int elementSize = (useDoublePrecision ? sizeof(double) : sizeof(float));
      defines["NUM_ATOMS"] = cu.intToString(numAtoms);
      defines["PADDED_NUM_ATOMS"] = cu.intToString(paddedNumAtoms);
      defines["USE_CUTOFF"] = "";
      //    defines["EPSILON_FACTOR"] = cu.doubleToString(138.9354558456);
      //    defines["EPSILON_FACTOR"] = cu.doubleToString(138.9354574);
      //    defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
      //    defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
      //    defines["EWALD_ALPHA"] = cu.doubleToString(0.000001);

      //    torque = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "torque");
      //    cu.addAutoclearBuffer(*torque);

      //    field = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "field");
      //    cu.addAutoclearBuffer(*field);

      // axis, charge from posq, dipole, and quadrupole
      axisInfo.initialize<int4>(cu, paddedNumAtoms, "axisInfo");
      CudaArray &posq = cu.getPosq();
      localFrameDipoles.initialize(
          cu, 3 * paddedNumAtoms, elementSize, "localFrameDipoles");
      localFrameQuadrupoles.initialize(
          cu, 5 * paddedNumAtoms, elementSize, "localFrameQuadrupoles");
      globalFrameDipoles.initialize(
          cu, 3 * paddedNumAtoms, elementSize, "globalFrameDipoles");
      globalFrameQuadrupoles.initialize(
          cu, 5 * paddedNumAtoms, elementSize, "globalFrameQuadrupoles");
      inducedDipole.initialize(
          cu, 3 * paddedNumAtoms, elementSize, "inducedDipole");
      inducedDipoleP.initialize(
          cu, 3 * paddedNumAtoms, elementSize, "inducedDipoleP");
      mpoleField.initialize(
          cu, 3 * paddedNumAtoms, sizeof(long long), "mpoleField");
      mpoleFieldP.initialize(
          cu, 3 * paddedNumAtoms, sizeof(long long), "mpoleFieldP");
      inducedField.initialize(
          cu, 3 * paddedNumAtoms, sizeof(long long), "inducedField");
      inducedFieldP.initialize(
          cu, 3 * paddedNumAtoms, sizeof(long long), "inducedFieldP");
      torque.initialize(cu, 3 * paddedNumAtoms, sizeof(long long), "torque");
      cu.addAutoclearBuffer(mpoleField);
      cu.addAutoclearBuffer(mpoleFieldP);
      cu.addAutoclearBuffer(torque);

      //    if (useDoublePrecision) {
      //        //torque.initialize<double>(cu, 3*paddedNumAtoms, "torque");
      //        localFrameDipoles.initialize<double>(cu, 3*paddedNumAtoms, "localFrameDipoles");
      //        localFrameQuadrupoles.initialize<double>(cu, 5*paddedNumAtoms, "localFrameQuadrupoles");
      //        globalFrameDipoles.initialize<double>(cu, 3*paddedNumAtoms, "globalFrameDipoles");
      //        globalFrameQuadrupoles.initialize<double>(cu, 5*paddedNumAtoms, "globalFrameQuadrupoles");
      //    } else {
      //        //torque.initialize<float>(cu, 3*paddedNumAtoms, "torque");
      //        localFrameDipoles.initialize<float>(cu, 3*paddedNumAtoms, "localFrameDipoles");
      //        localFrameQuadrupoles.initialize<float>(cu, 5*paddedNumAtoms, "localFrameQuadrupoles");
      //        globalFrameDipoles.initialize<float>(cu, 3*paddedNumAtoms, "globalFrameDipoles");
      //        globalFrameQuadrupoles.initialize<float>(cu, 5*paddedNumAtoms, "globalFrameQuadrupoles");
      //    }

      vector<int4> axisInfoVec(axisInfo.getSize());
      vector<double4> posqVec(posq.getSize());
      vector<double> localFrameDipolesVec(localFrameDipoles.getSize());
      vector<double> localFrameQuadrupolesVec(localFrameQuadrupoles.getSize());

      double4 *posqd = &posqVec[0];
      double *localdipoled = &localFrameDipolesVec[0];
      double *localquadd = &localFrameQuadrupolesVec[0];
      float4 *posqf = (float4 *)posqd;
      float *localdipolef = (float *)localdipoled;
      float *localquadf = (float *)localquadd;

      for (int ii = 0; ii < numAtoms; ++ii)
      {
         int axisType, atomz, atomx, atomy;
         double charge;
         vector<double> dipole, quadrupole;
         force.getMultipoleParameters(ii, charge, dipole, quadrupole, axisType, atomz, atomx, atomy);

         axisInfoVec[ii] = make_int4(atomx, atomy, atomz, axisType);
         if (useDoublePrecision)
         {
            posqd[ii] = make_double4(0, 0, 0, charge);
            localdipoled[3 * ii] = dipole[0];
            localdipoled[3 * ii + 1] = dipole[1];
            localdipoled[3 * ii + 2] = dipole[2];
            localquadd[5 * ii] = quadrupole[0];     // xx
            localquadd[5 * ii + 1] = quadrupole[1]; // xy
            localquadd[5 * ii + 2] = quadrupole[2]; // xz
            localquadd[5 * ii + 3] = quadrupole[4]; // yy
            localquadd[5 * ii + 4] = quadrupole[5]; // yz
                                                    // zz is unnecessary
         }
         else
         {
            posqf[ii] = make_float4(0, 0, 0, (float)charge);
            localdipolef[3 * ii] = (float)dipole[0];
            localdipolef[3 * ii + 1] = (float)dipole[1];
            localdipolef[3 * ii + 2] = (float)dipole[2];
            localquadf[5 * ii] = (float)quadrupole[0];     // xx
            localquadf[5 * ii + 1] = (float)quadrupole[1]; // xy
            localquadf[5 * ii + 2] = (float)quadrupole[2]; // xz
            localquadf[5 * ii + 3] = (float)quadrupole[4]; // yy
            localquadf[5 * ii + 4] = (float)quadrupole[5]; // yz
                                                           // zz is unnecessary
         }
      }

      axisInfo.upload(axisInfoVec);
      posq.upload(&posqVec[0]);
      localFrameDipoles.upload(&localFrameDipolesVec[0]);
      localFrameQuadrupoles.upload(&localFrameQuadrupolesVec[0]);

      // PME
      pmeorder = 0;
      pmeCutoffDistance = 0.0;

      if (force.getUsePME())
      {
         pmeorder = force.getPMEOrder();
         pmeCutoffDistance = force.getPMECutoffDistance();
         int nx, ny, nz;
         force.getPMEParameters(ewaldAlpha, nx, ny, nz);
         if (nx == 0 || ewaldAlpha == 0.0)
         {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(pmeCutoffDistance);
            NonbondedForceImpl::calcPMEParameters(
                system, nb, ewaldAlpha, nx, ny, nz);
         }
         nfft1 = CudaFFT3D::findLegalDimension(nx);
         nfft2 = CudaFFT3D::findLegalDimension(ny);
         nfft3 = CudaFFT3D::findLegalDimension(nz);

         defines["PME_ORDER"] = cu.intToString(pmeorder);
         defines["EWALD_ALPHA"] = cu.doubleToString(ewaldAlpha);
         defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
         defines["USE_EwALD"] = "";
         defines["USE_CUTOFF"] = "";
         defines["USE_PERIODIC"] = "";
         defines["PME_CUTOFF_SQUARED"] = cu.doubleToString(pmeCutoffDistance * pmeCutoffDistance);
         defines["GRID_SIZE_X"] = cu.intToString(nfft1);
         defines["GRID_SIZE_Y"] = cu.intToString(nfft2);
         defines["GRID_SIZE_Z"] = cu.intToString(nfft3);
         defines["EPSILON_FACTOR"] = cu.doubleToString(138.9354558456);

         printf("Ewald alpha = %12.4f\n",ewaldAlpha);

         // PME gird
         int ntot = nfft1 * nfft2 * nfft3;
         qgrid.initialize(cu, ntot, 2 * elementSize, "qgrid");
         cu.addAutoclearBuffer(qgrid);
         bsmod1.initialize(cu, nfft1, elementSize, "bsmod1");
         bsmod2.initialize(cu, nfft2, elementSize, "bsmod2");
         bsmod3.initialize(cu, nfft3, elementSize, "bsmod3");
         fracDipoles.initialize(
             cu, 3 * paddedNumAtoms, elementSize, "fracQuadrupoles");
         // Tinker cmp: xx yy zz xy xz yz
         // OpenMM: xx xy xz yy yz zz
         fracQuadrupoles.initialize(
             cu, 6 * paddedNumAtoms, elementSize, "fracQuadrupoles");
         // f0(1),  f0(2),  f0(3),  ..., f0(NATOMS)
         // f1(1),  f1(2),  f1(3),  ..., f1(NATOMS)
         // ...
         // f19(1), f19(2), f19(3), ..., f19(NATOMS)
         fphi.initialize(cu, 20 * numAtoms, elementSize, "fphi");
         cphi.initialize(cu, 10 * numAtoms, elementSize, "cphi");
         phid.initialize(cu, 10 * numAtoms, elementSize, "phid");
         phip.initialize(cu, 10 * numAtoms, elementSize, "phip");
         phidp.initialize(cu, 20 * numAtoms, elementSize, "phidp");

         cufftResult result = cufftPlan3d(&fft, nfft1, nfft2, nfft3,
                                          useDoublePrecision ? CUFFT_Z2Z : CUFFT_C2C);
         if (result != CUFFT_SUCCESS)
            throw OpenMMException(
                "Error initializing FFT: " + cu.intToString(result));
         hasInitializedFFT = true;

         // initialize bsmod (b-spline moduli)
         // reference: tinker subroutine "moduli", "bspline", "dftmod"
         vector<double> data(pmeorder);
         double x = 0.0;
         data[0] = 1.0 - x;
         data[1] = x;
         for (int i = 2; i < pmeorder; i++)
         {
            double denom = 1.0 / i;
            data[i] = x * data[i - 1] * denom;
            for (int j = 1; j < i; j++)
               data[i - j] = ((x + j) * data[i - j - 1] + ((i - j + 1) - x) * data[i - j]) * denom;
            data[0] = (1.0 - x) * data[0] * denom;
         }
         int maxSize = max(max(nfft1, nfft2), nfft3);
         vector<double> bsplines_data(maxSize + 1, 0.0);
         for (int i = 2; i <= pmeorder + 1; i++)
         {
            bsplines_data[i] = data[i - 2];
         }
         for (int dim = 0; dim < 3; dim++)
         {
            int ndata = (dim == 0 ? nfft1 : dim == 1 ? nfft2 : nfft3);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0 * M_PI / ndata;
            for (int i = 0; i < ndata; i++)
            {
               double sc = 0.0;
               double ss = 0.0;
               for (int j = 1; j <= ndata; j++)
               {
                  double arg = factor * i * (j - 1);
                  sc += bsplines_data[j] * cos(arg);
                  ss += bsplines_data[j] * sin(arg);
               }
               moduli[i] = sc * sc + ss * ss;
            }

            //////////////////////
            //////////////////////
            // fix for exponential Euler spline interpolation failure
            // Tinker uses 0.5, OpenMM uses 0.9

            double eps = 1.0e-7;
            if (moduli[0] < eps)
            {
               moduli[0] = 0.5 * moduli[1];
            }
            for (int i = 1; i < ndata - 1; ++i)
            {
               if (moduli[i] < eps)
               {
                  moduli[i] = 0.5 * (moduli[i - 1] + moduli[i + 1]);
               }
            }
            if (moduli[ndata - 1] < eps)
            {
               moduli[ndata - 1] = 0.5 * moduli[ndata - 2];
            }
            //////////////////////

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++)
            {
               int k = i - 1;
               if (i > ndata / 2)
                  k = k - ndata;
               double zeta;
               if (k == 0)
                  zeta = 1.0;
               else
               {
                  double sum1 = 1.0;
                  double sum2 = 1.0;
                  factor = M_PI * k / ndata;
                  for (int j = 1; j <= jcut; j++)
                  {
                     double arg = factor / (factor + M_PI * j);
                     sum1 += pow(arg, pmeorder);
                     sum2 += pow(arg, 2 * pmeorder);
                  }
                  for (int j = 1; j <= jcut; j++)
                  {
                     double arg = factor / (factor - M_PI * j);
                     sum1 += pow(arg, pmeorder);
                     sum2 += pow(arg, 2 * pmeorder);
                  }
                  zeta = sum2 / sum1;
               }
               moduli[i - 1] = moduli[i - 1] * zeta * zeta;
            }
            if (useDoublePrecision)
            {
               if (dim == 0)
                  bsmod1.upload(moduli);
               else if (dim == 1)
                  bsmod2.upload(moduli);
               else
                  bsmod3.upload(moduli);
            }
            else
            {
               vector<float> modulif(ndata);
               for (int i = 0; i < ndata; ++i)
                  modulif[i] = (float)moduli[i];
               if (dim == 0)
                  bsmod1.upload(modulif);
               else if (dim == 1)
                  bsmod2.upload(modulif);
               else
                  bsmod3.upload(modulif);
            }
         }
      }

      // PCG
      rsd_ll.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "rsd_ll");
      rsd_real.initialize(cu, 3*paddedNumAtoms, elementSize, "rsd_real");
      pvec.initialize(cu, 3*paddedNumAtoms, elementSize, "pvec");
      tp.initialize(cu, 3*paddedNumAtoms, elementSize, "tp");
      ptp.initialize(cu, 3*paddedNumAtoms, elementSize, "ptp");
      rmr.initialize(cu, 3*paddedNumAtoms, elementSize, "rmr");

      force.getPCGParameters(inducedEpsilon,maxIterations);

      // charge transfer
      double chgtrntaper = force.getCTTaperDistance();
      double chgtrncutoff = force.getCTCutoffDistance();
      defines["CHGTRN_TAPER"] = cu.doubleToString(chgtrntaper);
      defines["CHGTRN_CUTOFF_SQUARED"] = cu.doubleToString(chgtrncutoff * chgtrncutoff);
      if (chgtrntaper < chgtrncutoff)
      {
         defines["CHGTRN_TAPER_C3"] = cu.doubleToString(10 / pow(chgtrntaper - chgtrncutoff, 3.0));
         defines["CHGTRN_TAPER_C4"] = cu.doubleToString(15 / pow(chgtrntaper - chgtrncutoff, 4.0));
         defines["CHGTRN_TAPER_C5"] = cu.doubleToString(6 / pow(chgtrntaper - chgtrncutoff, 5.0));
      }
      else
      {
         defines["CHGTRN_TAPER_C3"] = "0";
         defines["CHGTRN_TAPER_C4"] = "0";
         defines["CHGTRN_TAPER_C5"] = "0";
      }

      double cscales[3];
      force.getCTScales(cscales[0], cscales[1], cscales[2]);
      defines["CHARGETRANSFER13SCALE"] = cu.doubleToString(cscales[0]);
      defines["CHARGETRANSFER14SCALE"] = cu.doubleToString(cscales[1]);
      defines["CHARGETRANSFER15SCALE"] = cu.doubleToString(cscales[2]);

      double wscales[3];
      force.getWScales(wscales[0], wscales[1], wscales[2]);
      printf("wscales: %12.4f%12.4f%12.4f\n",wscales[0],wscales[1],wscales[2]);
      defines["INDUCEDDIPOLE12SCALE"] = cu.doubleToString(wscales[0]);
      defines["INDUCEDDIPOLE13SCALE"] = cu.doubleToString(wscales[1]);
      defines["INDUCEDDIPOLE14SCALE"] = cu.doubleToString(wscales[2]);

      if (useDoublePrecision)
      {
         chgct.initialize<double>(cu, paddedNumAtoms, "chgct");
         dmpct.initialize<double>(cu, paddedNumAtoms, "dmpct");
      }
      else
      {
         chgct.initialize<float>(cu, paddedNumAtoms, "chgct");
         dmpct.initialize<float>(cu, paddedNumAtoms, "dmpct");
      }
      vector<double> chgctVec(chgct.getSize());
      vector<double> dmpctVec(dmpct.getSize());

      double *chgctd = &chgctVec[0];
      double *dmpctd = &dmpctVec[0];
      float *chgctf = (float *)chgctd;
      float *dmpctf = (float *)dmpctd;

      for (int ii = 0; ii < numAtoms; ++ii)
      {
         double a, c;
         force.getCTParameters(ii, a, c);
         if (useDoublePrecision)
         {
            chgctd[ii] = c;
            dmpctd[ii] = a;
         }
         else
         {
            chgctf[ii] = (float)c;
            dmpctf[ii] = (float)a;
         }
      }
      chgct.upload(&chgctVec[0]);
      dmpct.upload(&dmpctVec[0]);

      // dispersion
      double disptaper = force.getDispTaperDistance();
      double dispcutoff = force.getDispCutoffDistance();
      defines["DISP_TAPER"] = cu.doubleToString(disptaper);
      defines["DISP_CUTOFF_SQUARED"] = cu.doubleToString(dispcutoff * dispcutoff);
      if (disptaper < dispcutoff)
      {
         defines["DISP_TAPER_C3"] = cu.doubleToString(10 / pow(disptaper - dispcutoff, 3.0));
         defines["DISP_TAPER_C4"] = cu.doubleToString(15 / pow(disptaper - dispcutoff, 4.0));
         defines["DISP_TAPER_C5"] = cu.doubleToString(6 / pow(disptaper - dispcutoff, 5.0));
      }
      else
      {
         defines["DISP_TAPER_C3"] = "0";
         defines["DISP_TAPER_C4"] = "0";
         defines["DISP_TAPER_C5"] = "0";
      }

      if (useDoublePrecision)
      {
         csix.initialize<double>(cu, paddedNumAtoms, "csix");
      }
      else
      {
         csix.initialize<float>(cu, paddedNumAtoms, "csix");
      }
      vector<double> csixVec(csix.getSize());

      double *csixd = &csixVec[0];
      float *csixf = (float *)csixd;

      for (int ii = 0; ii < numAtoms; ++ii)
      {
         double c;
         force.getDispParameters(ii, c);
         if (useDoublePrecision)
         {
            csixd[ii] = c;
         }
         else
         {
            csixf[ii] = (float)c;
         }
      }
      csix.upload(&csixVec[0]);

      // repulsion
      double repeltaper = force.getRepelTaperDistance();
      double repelcutoff = force.getRepelCutoffDistance();
      defines["REPEL_TAPER"] = cu.doubleToString(repeltaper);
      defines["REPEL_CUTOFF_SQUARED"] = cu.doubleToString(repelcutoff * repelcutoff);
      if (repeltaper < repelcutoff)
      {
         defines["REPEL_TAPER_C3"] = cu.doubleToString(10 / pow(repeltaper - repelcutoff, 3.0));
         defines["REPEL_TAPER_C4"] = cu.doubleToString(15 / pow(repeltaper - repelcutoff, 4.0));
         defines["REPEL_TAPER_C5"] = cu.doubleToString(6 / pow(repeltaper - repelcutoff, 5.0));
      }
      else
      {
         defines["REPEL_TAPER_C3"] = "0";
         defines["REPEL_TAPER_C4"] = "0";
         defines["REPEL_TAPER_C5"] = "0";
      }

      if (useDoublePrecision)
      {
         sizpr.initialize<double>(cu, paddedNumAtoms, "sizpr");
         dmppr.initialize<double>(cu, paddedNumAtoms, "dmppr");
         elepr.initialize<double>(cu, paddedNumAtoms, "elepr");
      }
      else
      {
         sizpr.initialize<float>(cu, paddedNumAtoms, "sizpr");
         dmppr.initialize<float>(cu, paddedNumAtoms, "dmppr");
         elepr.initialize<float>(cu, paddedNumAtoms, "elepr");
      }
      vector<double> sizprVec(sizpr.getSize());
      vector<double> dmpprVec(dmppr.getSize());
      vector<double> eleprVec(elepr.getSize());

      double *sizprd = &sizprVec[0];
      double *dmpprd = &dmpprVec[0];
      double *eleprd = &eleprVec[0];
      float *sizprf = (float *)sizprd;
      float *dmpprf = (float *)dmpprd;
      float *eleprf = (float *)eleprd;

      for (int ii = 0; ii < numAtoms; ++ii)
      {
         double s, d, e;
         force.getRepelParameters(ii, s, d, e);
         if (useDoublePrecision)
         {
            sizprd[ii] = s;
            dmpprd[ii] = d;
            eleprd[ii] = e;
         }
         else
         {
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
      defines["CHGPEN_CUTOFF_SQUARED"] = cu.doubleToString(chgpencutoff * chgpencutoff);

      HippoParameterCopier copy_pcore(pcore, cu, paddedNumAtoms, "pcore", useDoublePrecision);
      HippoParameterCopier copy_pval(pval, cu, paddedNumAtoms, "pval", useDoublePrecision);
      HippoParameterCopier copy_palpha(palpha, cu, paddedNumAtoms, "palpha", useDoublePrecision);
      HippoParameterCopier copy_polarity(polarity, cu, paddedNumAtoms, "polarity", useDoublePrecision);
      for (int ii = 0; ii < numAtoms; ++ii)
      {
         double c, v, a, p;
         force.getCPMultipoleParameters(ii, c, v, a, p);
         copy_pcore(ii, c);
         copy_pval(ii, v);
         copy_palpha(ii, a);
         copy_polarity(ii, p);
      }
      copy_pcore.upload();
      copy_pval.upload();
      copy_palpha.upload();
      copy_polarity.upload();

      // exclusions
      vector<vector <int> > exclusions(numAtoms);
      for (int i = 0; i < numAtoms; i++)
      {
         vector<int> atoms;
         set<int> allAtoms;
         allAtoms.insert(i);

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent12, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii)
         {
         }
         allAtoms.insert(atoms.begin(), atoms.end());
         for (set<int>::const_iterator iter = allAtoms.begin(); iter != allAtoms.end(); ++iter)
            covalentFlag24Values.push_back(make_int3(i, *iter, 0));

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent13, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii)
         {
         }
         allAtoms.insert(atoms.begin(), atoms.end());
         for (set<int>::const_iterator iter = allAtoms.begin(); iter != allAtoms.end(); ++iter)
            covalentFlagValues.push_back(make_int3(i, *iter, 0));
         for (set<int>::const_iterator iter = allAtoms.begin(); iter != allAtoms.end(); ++iter)
            covalentFlag24Values.push_back(make_int3(i, *iter, 1));

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent14, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii)
         {
         }
         allAtoms.insert(atoms.begin(), atoms.end());
         for (int j = 0; j < (int)atoms.size(); j++)
            covalentFlagValues.push_back(make_int3(i, atoms[j], 1));
         for (int j = 0; j < (int)atoms.size(); j++)
            covalentFlag24Values.push_back(make_int3(i, atoms[j], 2));

         force.getCovalentMap(i, HippoChargeTransferForce::Covalent15, atoms);
         for (int ii = 0; ii < atoms.size(); ++ii)
         {
         }
         for (int j = 0; j < (int)atoms.size(); j++)
            covalentFlagValues.push_back(make_int3(i, atoms[j], 2));
         allAtoms.insert(atoms.begin(), atoms.end());

         exclusions[i].insert(exclusions[i].end(), allAtoms.begin(), allAtoms.end());
      }

      for (int i = 0; i < covalentFlagValues.size(); ++i)
      {
         int atom1 = covalentFlagValues[i].x;
         int atom2 = covalentFlagValues[i].y;
         int value = covalentFlagValues[i].z;
         int f1 = (value == 0 || value == 1 ? 1 : 0);
         int f2 = (value == 0 || value == 2 ? 1 : 0);
      }

      set<pair<int, int> > tilesWithExclusions;
      for (int atom1 = 0; atom1 < (int)exclusions.size(); ++atom1)
      {
         int x = atom1 / CudaContext::TileSize;
         for (int j = 0; j < (int)exclusions[atom1].size(); ++j)
         {
            int atom2 = exclusions[atom1][j];
            int y = atom2 / CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
         }
      }

      for (int i = 0; i < covalentFlag24Values.size(); ++i)
      {
         int atom1 = covalentFlag24Values[i].x;
         int atom2 = covalentFlag24Values[i].y;
         int value = covalentFlag24Values[i].z;
         int f1 = (value == 0 || value == 1 ? 1 : 0);
         int f2 = (value == 0 || value == 2 ? 1 : 0);
      }

      // define macros for cuda source code

      defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
      int numExclusionTiles = tilesWithExclusions.size();
      defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
      int numContexts = cu.getPlatformData().contexts.size();
      int startExclusionIndex = cu.getContextIndex() * numExclusionTiles / numContexts;
      int endExclusionIndex = (cu.getContextIndex() + 1) * numExclusionTiles / numContexts;
      defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
      defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);

      int maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
      double energyAndForceMemory = 30.0 * elementSize + 2.0 * sizeof(int);
      energyAndForceThreads = min(maxThreads, cu.computeThreadBlockSize(energyAndForceMemory));
      defines["THREAD_BLOCK_SIZE"] = cu.intToString(energyAndForceThreads);

      //    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaAmoebaKernelSources::multipoles, defines);
      //    mapTorqueKernel = cu.getKernel(module, "mapTorqueToForce");

      printf(" start compiling CT module\n");
      CUmodule hippoModule = cu.createModule(CudaKernelSources::vectorOps + CudaAmoebaKernelSources::hippoRotpole + 
         CudaAmoebaKernelSources::hippoCTexample + CudaAmoebaKernelSources::hippoPolar, defines);
      printf(" compiling CT module 100 \n");
      fflush(stdout);
      energyAndForceKernel = cu.getKernel(hippoModule, "computeChargeTransfer");
      printf(" end compiling CT module\n");

      printf(" start compiling Polar module");
      epolarKernel = cu.getKernel(hippoModule, "computePolar");
      printf(" end compiling Polar module");

      dipoleDotPolarityKernel = cu.getKernel(hippoModule, "dipoleDotPolarity");
      ufieldKernel = cu.getKernel(hippoModule, "ufieldReal");
      rotpoleKernel = cu.getKernel(hippoModule, "hippoRotpole");
      mapTorqueKernel = cu.getKernel(hippoModule, "mapTorqueToForce");

      if (force.getUsePME())
      {
         // set up PME
         CUmodule pmemodule = cu.createModule(
             CudaKernelSources::vectorOps + CudaAmoebaKernelSources::hippoPME,
             defines);

         // kernels
         cmp_to_fmp_kernel = cu.getKernel(pmemodule, "cmp_to_fmp");
         grid_mpole_kernel = cu.getKernel(pmemodule, "grid_mpole");
         grid_uind_kernel = cu.getKernel(pmemodule, "grid_uind");
         grid_convert_to_double_kernel = cu.getKernel(pmemodule, "grid_convert_to_double");
         pme_convolution_kernel = cu.getKernel(pmemodule, "pme_convolution");
         fphi_mpole_kernel = cu.getKernel(pmemodule, "fphi_mpole");
         fphi_uind_kernel = cu.getKernel(pmemodule, "fphi_uind");
         fphi_to_cphi_kernel = cu.getKernel(pmemodule, "fphi_to_cphi");
         ufield_recip_self_kernel = cu.getKernel(pmemodule, "ufield_recip_self");
         cuFuncSetCacheConfig(grid_mpole_kernel, CU_FUNC_CACHE_PREFER_L1);
         cuFuncSetCacheConfig(grid_uind_kernel, CU_FUNC_CACHE_PREFER_L1);
         cuFuncSetCacheConfig(fphi_mpole_kernel, CU_FUNC_CACHE_PREFER_L1);
         cuFuncSetCacheConfig(fphi_uind_kernel, CU_FUNC_CACHE_PREFER_L1);

         recip_mpole_energy_force_torque_kernel = cu.getKernel(pmemodule, "recip_mpole_energy_force_torque");
         //torque_to_force_kernel = cu.getKernel(pmemodule, "torque_to_force");
 
         muDotMpoleKernel = cu.getKernel(hippoModule, "muDotMpoleBuffer");
 
         fphi_to_cphi_ind_kernel = cu.getKernel(pmemodule, "fphi_to_cphi_ind");
         eprecip_kernel = cu.getKernel(pmemodule, "eprecip");
      }

      // PCG
      convertLongLongToReal3NKernel = cu.getKernel(hippoModule, "convert_long_long_to_real_3N");
      invAlphaPlusTUKernel = cu.getKernel(hippoModule, "invAlphaPlusTU");
      dotProductBufferKernel = cu.getKernel(hippoModule, "dotProductBuffer");
      quadraticBufferKernel = cu.getKernel(hippoModule, "quadraticBuffer");
      updateMuRsdKernel = cu.getKernel(hippoModule, "updateMuRsd");
      updatePVecKernel = cu.getKernel(hippoModule, "updatePVec");

      double cutoffDistanceForNBList = max(chgtrncutoff, repelcutoff);
      cutoffDistanceForNBList = max(cutoffDistanceForNBList, pmeCutoffDistance);
      cu.getNonbondedUtilities().addInteraction(true, true, true, cutoffDistanceForNBList, exclusions, "", force.getForceGroup());
      printf(" before setUsePadding\n");
      cu.getNonbondedUtilities().setUsePadding(false);
      cu.addForce(new ForceInfo(force));
   }
   catch (std::exception &e)
   {
      printf(" exception thrown from CalcHippoChargeTransferForceKernel::initialize() -- %s\n", e.what());
   }
}

void CudaCalcHippoChargeTransferForceKernel::initializeScaleFactors()
{
   hasInitializedScaleFactors = true;
   CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();

   // Figure out the covalent flag values to use for each atom pair.

   vector<ushort2> exclusionTiles;
   nb.getExclusionTiles().download(exclusionTiles);
   map<pair<int, int>, int> exclusionTileMap;
   printf(" -- total number of exclusions tiles %d\n", (int)exclusionTiles.size());
   for (int i = 0; i < (int)exclusionTiles.size(); i++)
   {
      ushort2 tile = exclusionTiles[i];
      exclusionTileMap[make_pair(tile.x, tile.y)] = i;
   }

   covalentFlags.initialize<uint2>(cu, nb.getExclusions().getSize(), "covalentFlags");
   vector<uint2> covalentFlagsVec(nb.getExclusions().getSize(), make_uint2(0, 0));
   for (int i = 0; i < (int)covalentFlagValues.size(); i++)
   {
      int atom1 = covalentFlagValues[i].x;
      int atom2 = covalentFlagValues[i].y;
      int value = covalentFlagValues[i].z;
      int x = atom1 / CudaContext::TileSize;
      int offset1 = atom1 - x * CudaContext::TileSize;
      int y = atom2 / CudaContext::TileSize;
      int offset2 = atom2 - y * CudaContext::TileSize;
      int f1 = (value == 0 || value == 1 ? 1 : 0);
      int f2 = (value == 0 || value == 2 ? 1 : 0);
      if (x == y)
      {
         int index = exclusionTileMap[make_pair(x, y)] * CudaContext::TileSize;
         covalentFlagsVec[index + offset1].x |= f1 << offset2;
         covalentFlagsVec[index + offset1].y |= f2 << offset2;
         covalentFlagsVec[index + offset2].x |= f1 << offset1;
         covalentFlagsVec[index + offset2].y |= f2 << offset1;
      }
      else if (x > y)
      {
         int index = exclusionTileMap[make_pair(x, y)] * CudaContext::TileSize;
         covalentFlagsVec[index + offset1].x |= f1 << offset2;
         covalentFlagsVec[index + offset1].y |= f2 << offset2;
      }
      else
      {
         int index = exclusionTileMap[make_pair(y, x)] * CudaContext::TileSize;
         covalentFlagsVec[index + offset2].x |= f1 << offset1;
         covalentFlagsVec[index + offset2].y |= f2 << offset1;
      }
   }
   covalentFlags.upload(covalentFlagsVec);


   covalentFlags24.initialize<uint2>(cu, nb.getExclusions().getSize(), "covalentFlags24");
   vector<uint2> covalentFlags24Vec(nb.getExclusions().getSize(), make_uint2(0, 0));
   for (int i = 0; i < (int)covalentFlag24Values.size(); i++)
   {
      int atom1 = covalentFlag24Values[i].x;
      int atom2 = covalentFlag24Values[i].y;
      int value = covalentFlag24Values[i].z;
      int x = atom1 / CudaContext::TileSize;
      int offset1 = atom1 - x * CudaContext::TileSize;
      int y = atom2 / CudaContext::TileSize;
      int offset2 = atom2 - y * CudaContext::TileSize;
      int f1 = (value == 0 || value == 1 ? 1 : 0);
      int f2 = (value == 0 || value == 2 ? 1 : 0);
      if (x == y)
      {
         int index = exclusionTileMap[make_pair(x, y)] * CudaContext::TileSize;
         covalentFlags24Vec[index + offset1].x |= f1 << offset2;
         covalentFlags24Vec[index + offset1].y |= f2 << offset2;
         covalentFlags24Vec[index + offset2].x |= f1 << offset1;
         covalentFlags24Vec[index + offset2].y |= f2 << offset1;
      }
      else if (x > y)
      {
         int index = exclusionTileMap[make_pair(x, y)] * CudaContext::TileSize;
         covalentFlags24Vec[index + offset1].x |= f1 << offset2;
         covalentFlags24Vec[index + offset1].y |= f2 << offset2;
      }
      else
      {
         int index = exclusionTileMap[make_pair(y, x)] * CudaContext::TileSize;
         covalentFlags24Vec[index + offset2].x |= f1 << offset1;
         covalentFlags24Vec[index + offset2].y |= f2 << offset1;
      }
   }
   covalentFlags24.upload(covalentFlags24Vec);
}

// PME utilities

void CudaCalcHippoChargeTransferForceKernel::cmp_to_fmp() {
   void* cmp_to_fmp_args[] = {&globalFrameDipoles.getDevicePointer(),
      &globalFrameQuadrupoles.getDevicePointer(),
      &fracDipoles.getDevicePointer(), &fracQuadrupoles.getDevicePointer(),
      recipBoxVectorPointer[0], recipBoxVectorPointer[1],
      recipBoxVectorPointer[2]};
   cu.executeKernel(cmp_to_fmp_kernel, cmp_to_fmp_args, numAtoms);
}

void CudaCalcHippoChargeTransferForceKernel::grid_mpole() {
   void* grid_mpole_args[]
      = {&cu.getPosq().getDevicePointer(), &fracDipoles.getDevicePointer(),
         &fracQuadrupoles.getDevicePointer(), &qgrid.getDevicePointer(),
         cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
         cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0],
         recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
   cu.executeKernel(grid_mpole_kernel, grid_mpole_args, numAtoms);
   if (useDoublePrecision) {
      void* grid_convert_to_double_args[] = {&qgrid.getDevicePointer()};
      cu.executeKernel(
         grid_convert_to_double_kernel, grid_convert_to_double_args, numAtoms);
   }
}

void CudaCalcHippoChargeTransferForceKernel::grid_uind(CudaArray& inp) {
   void* grid_uind_args[]
      = {&cu.getPosq().getDevicePointer(), &inp.getDevicePointer(),
         &inducedDipoleP.getDevicePointer(), &qgrid.getDevicePointer(),
         cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
         cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0],
         recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
   cu.executeKernel(grid_uind_kernel, grid_uind_args, numAtoms);
   if (useDoublePrecision) {
      void* grid_convert_to_double_args[] = {&qgrid.getDevicePointer()};
      cu.executeKernel(
         grid_convert_to_double_kernel, grid_convert_to_double_args, numAtoms);
   }
}

void CudaCalcHippoChargeTransferForceKernel::fftfront() {
   if (useDoublePrecision) {
      cufftExecZ2Z(fft, (double2*)qgrid.getDevicePointer(),
         (double2*)qgrid.getDevicePointer(), CUFFT_FORWARD);
   } else {
      cufftExecC2C(fft, (float2*)qgrid.getDevicePointer(),
         (float2*)qgrid.getDevicePointer(), CUFFT_FORWARD);
   }
}

void CudaCalcHippoChargeTransferForceKernel::pme_convolution() {
   void* pme_convolution_args[]
      = {&qgrid.getDevicePointer(), &bsmod1.getDevicePointer(),
         &bsmod2.getDevicePointer(), &bsmod3.getDevicePointer(),
         cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0],
         recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
   cu.executeKernel(
      pme_convolution_kernel, pme_convolution_args, nfft1 * nfft2 * nfft3, 256);
}

void CudaCalcHippoChargeTransferForceKernel::fftback() {
   if (useDoublePrecision) {
      cufftExecZ2Z(fft, (double2*)qgrid.getDevicePointer(),
         (double2*)qgrid.getDevicePointer(), CUFFT_INVERSE);
   } else {
      cufftExecC2C(fft, (float2*)qgrid.getDevicePointer(),
         (float2*)qgrid.getDevicePointer(), CUFFT_INVERSE);
   }
}

void CudaCalcHippoChargeTransferForceKernel::fphi_mpole() {
   void* fphi_mpole_args[] = {&qgrid.getDevicePointer(),
      &fphi.getDevicePointer(), &mpoleField.getDevicePointer(),
      &mpoleFieldP.getDevicePointer(), &cu.getPosq().getDevicePointer(),
      &globalFrameDipoles.getDevicePointer(), cu.getPeriodicBoxVecXPointer(),
      cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
      recipBoxVectorPointer[0], recipBoxVectorPointer[1],
      recipBoxVectorPointer[2]};
   cu.executeKernel(fphi_mpole_kernel, fphi_mpole_args, numAtoms);
}

void CudaCalcHippoChargeTransferForceKernel::fphi_uind() {
   void* fphi_uind_args[] = {&qgrid.getDevicePointer(),
      &phid.getDevicePointer(), &phip.getDevicePointer(),
      &phidp.getDevicePointer(), &cu.getPosq().getDevicePointer(),
      cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
      cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0],
      recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
   cu.executeKernel(fphi_uind_kernel, fphi_uind_args, numAtoms);
}

void CudaCalcHippoChargeTransferForceKernel::fphi_to_cphi(CudaArray& phiarray) {
   void* fphi_to_cphi_args[] = {&phiarray.getDevicePointer(),
      &cphi.getDevicePointer(), recipBoxVectorPointer[0],
      recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
   cu.executeKernel(fphi_to_cphi_kernel, fphi_to_cphi_args, numAtoms);
}

void CudaCalcHippoChargeTransferForceKernel::ufield_recip_self(CudaArray& inp, CudaArray& outp) {
   void* args[] = {&phid.getDevicePointer(), &phip.getDevicePointer(),
      &outp.getDevicePointer(), &inducedFieldP.getDevicePointer(),
      &inp.getDevicePointer(), &inducedDipoleP.getDevicePointer(),
      recipBoxVectorPointer[0], recipBoxVectorPointer[1],
      recipBoxVectorPointer[2]};
   cu.executeKernel(ufield_recip_self_kernel, args, numAtoms);
}

void CudaCalcHippoChargeTransferForceKernel::recip_mpole_energy_force_torque() {
   void* recip_mpole_energy_force_torque_args[]
      = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(),
         &torque.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
         &globalFrameDipoles.getDevicePointer(),
         &globalFrameQuadrupoles.getDevicePointer(),
         &fracDipoles.getDevicePointer(), &fracQuadrupoles.getDevicePointer(),
         &fphi.getDevicePointer(), &cphi.getDevicePointer(),
         recipBoxVectorPointer[0], recipBoxVectorPointer[1],
         recipBoxVectorPointer[2]};
   cu.executeKernel(recip_mpole_energy_force_torque_kernel,
      recip_mpole_energy_force_torque_args, numAtoms);
}

//void CudaCalcHippoChargeTransferForceKernel::torque_to_force() {
//   void* torque_to_force_args[]
//      = {&cu.getForce().getDevicePointer(), &torque.getDevicePointer(),
//         &cu.getPosq().getDevicePointer(), &axisInfo.getDevicePointer()};
//   cu.executeKernel(torque_to_force_kernel, torque_to_force_args, numAtoms);
//}

void CudaCalcHippoChargeTransferForceKernel::ufield(CudaArray& inp, CudaArray& outp) {
   // cu.clearBuffer(inducedField);
   cu.clearBuffer(outp);
   cu.clearBuffer(inducedFieldP);

   // real space mutual field

   CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
   int startTileIndex = nb.getStartTileIndex();
   int numTileIndices = nb.getNumTiles();
   unsigned int maxTiles = nb.getInteractingTiles().getSize();

   void *argsUfield_real[] = {
       &outp.getDevicePointer(),
       &cu.getPosq().getDevicePointer(),
       &covalentFlags24.getDevicePointer(),
       &nb.getExclusionTiles().getDevicePointer(),
       &startTileIndex, &numTileIndices,

       &nb.getInteractingTiles().getDevicePointer(),
       &nb.getInteractionCount().getDevicePointer(),
       cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
       cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
       &maxTiles, &nb.getBlockCenters().getDevicePointer(),
       &nb.getInteractingAtoms().getDevicePointer(),

       &inp.getDevicePointer(),
       &palpha.getDevicePointer()};

   int numForceThreadBlocks = nb.getNumForceThreadBlocks();

   cu.executeKernel(ufieldKernel, argsUfield_real, numForceThreadBlocks * energyAndForceThreads, energyAndForceThreads);

   // recip+self mutual field
   cu.clearBuffer(qgrid);

   grid_uind(inp);
   fftfront();
   pme_convolution();
   fftback();
   fphi_uind();
   ufield_recip_self(inp, outp);

   std::vector<long long> inducedFieldVec;
   inducedFieldVec.reserve(outp.getSize());
   outp.download(inducedFieldVec);
   for (int i = 0; i < paddedNumAtoms; ++i) {
      // x1, x2, x3, ..., xN; y1, y2, y3, ..., yN; z1, z2, z3, ..., zN.
      long long fx, fy, fz;
      fx = inducedFieldVec[i];
      fy = inducedFieldVec[i+paddedNumAtoms];
      fz = inducedFieldVec[i+2*paddedNumAtoms];
      double dx, dy, dz;
      const double long_long_to_double = 1.0 / 0x100000000;
      dx = long_long_to_double * fx;
      dy = long_long_to_double * fy;
      dz = long_long_to_double * fz;
      printf("total induced field for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
   }
}

void CudaCalcHippoChargeTransferForceKernel::tmat(CudaArray& inp, CudaArray& outp) {
   // T = 1/alpha + Tu
   // -Tu == ufield
   
   // first, store -Tu.inp into rsd_ll;
   ufield(inp, rsd_ll);

   // calculate 1/alpha.inp and subtract rsd_ll
   void* inv_args[] = {
      &inp.getDevicePointer(),
      &polarity.getDevicePointer(),
      &rsd_ll.getDevicePointer(),
      &outp.getDevicePointer()
   };
   cu.executeKernel(invAlphaPlusTUKernel, inv_args, numAtoms);
}

template <class T> void CudaCalcHippoChargeTransferForceKernel::diagPCGInit() {
  // T matrix = (1/alpha + Tu), where (-Tu).mu gives mutual field.
  // M -- diag. preconditioner


      // print induced dipoles
      std::vector<double> inducedDipoleVec;
      inducedDipoleVec.reserve(inducedDipole.getSize());
      inducedDipole.download(&inducedDipoleVec[0]);
      for (int i = 0; i < numAtoms; ++i) {
         double dx, dy, dz;
         if (useDoublePrecision) {
            dx = inducedDipoleVec[3*i];
            dy = inducedDipoleVec[3*i+1];
            dz = inducedDipoleVec[3*i+2];
         }
         else {
            float* ptr = (float*)(&inducedDipoleVec[0]);
            dx = ptr[3*i];
            dy = ptr[3*i+1];
            dz = ptr[3*i+2];
         }
         printf("init induced dipole for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
      }



  // CudaArray mu -- induced dipole
  // CudaArray rsd -- perm. field residual, rsd = E - T.mu

  ufield(inducedDipole, rsd_ll);
  void* convert_args[] = {&rsd_ll.getDevicePointer(), &rsd_real.getDevicePointer()};
  cu.executeKernel(convertLongLongToReal3NKernel, convert_args, numAtoms);

      std::vector<double> rsdVec;
      rsdVec.reserve(rsd_real.getSize());
      rsd_real.download(&rsdVec[0]);
      for (int i = 0; i < numAtoms; ++i) {
         double dx, dy, dz;
         if (useDoublePrecision) {
            dx = rsdVec[3*i];
            dy = rsdVec[3*i+1];
            dz = rsdVec[3*i+2];
         }
         else {
            float* ptr = (float*)(&rsdVec[0]);
            dx = ptr[3*i];
            dy = ptr[3*i+1];
            dz = ptr[3*i+2];
         }
         printf("rsd for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
      }



  
  // CudaArray pvec -- search direction, pvec0 = diag. rsd0
  void* pvec_args[] = {
     &rsd_ll.getDevicePointer(), &polarity.getDevicePointer(), &pvec.getDevicePointer()
  };
  cu.executeKernel(dipoleDotPolarityKernel, pvec_args, numAtoms);
  
  // CudaArray tp -- T.pvec
  tmat(pvec, tp);

  // CudaArray ptp -- pvec.T.pvec
  void* ptp_args[] = {
     &pvec.getDevicePointer(), &tp.getDevicePointer(), &ptp.getDevicePointer()
  };
  cu.executeKernel(dotProductBufferKernel, ptp_args, numAtoms);
  
  // CudaArray rmr -- rsd.M.rsd
  void* rmr_args[] = {
     &rsd_real.getDevicePointer(), &rsd_real.getDevicePointer(), &polarity.getDevicePointer(),
     &rmr.getDevicePointer()
  };
  cu.executeKernel(quadraticBufferKernel, rmr_args, numAtoms);

  // fcb -- fast cpu buffer
  T* fcb = (T*) cu.getPinnedBuffer();
  rmr.download(fcb);
  for (int i = 0; i < 3 * numAtoms; ++i) {
     T val = fcb[i];
     printf("pcg init rmr val for pos %d = %12.4lf\n", i+1, val);
  }
  rmrSum = std::accumulate(fcb, fcb + 3 * numAtoms, (T)0);
  printf("rmrSum value = %12.4lf\n", rmrSum);
}

// iteration
template <class T> bool  CudaCalcHippoChargeTransferForceKernel::diagPCGIter(int niter) {
  // fcb -- fast cpu buffer
  T* fcb = (T*)cu.getPinnedBuffer();

  //////// gamma
  ptp.download(fcb);
  T ptp_sum = std::accumulate(fcb, fcb + 3 * numAtoms, (T)0);
  T rmr_sum_0 = rmrSum;
  T gamma = rmr_sum_0 / ptp_sum;
  printf("%6d iteration , rmr_sum_0 = %12.4lf, ptp_sum = %12.4lf\n", niter, rmr_sum_0, ptp_sum);

  //////// mu <--updated by-- mu + g0 * pvec;
  //////// rsd <--updated by-- rsd - g0 * T.pvec;
  void* mursd_args[] = {&inducedDipole.getDevicePointer(), &rsd_real.getDevicePointer(),
  &pvec.getDevicePointer(), &tp.getDevicePointer(), &gamma};
  cu.executeKernel(updateMuRsdKernel, mursd_args, numAtoms);

      // print new rsd
      std::vector<double> inducedDipoleVec;
      inducedDipoleVec.reserve(inducedDipole.getSize());
      rsd_real.download(&inducedDipoleVec[0]);
      for (int i = 0; i < numAtoms; ++i) {
         double dx, dy, dz;
         if (useDoublePrecision) {
            dx = inducedDipoleVec[3*i];
            dy = inducedDipoleVec[3*i+1];
            dz = inducedDipoleVec[3*i+2];
         }
         else {
            float* ptr = (float*)(&inducedDipoleVec[0]);
            dx = ptr[3*i];
            dy = ptr[3*i+1];
            dz = ptr[3*i+2];
         }
         printf("100 rsd for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
      }


  //////// beta
  // CudaArray rmr -- rsd.M.rsd
  void* rmr_args[] = {
     &rsd_real.getDevicePointer(), &rsd_real.getDevicePointer(), &polarity.getDevicePointer(),
     &rmr.getDevicePointer()
  };
  cu.executeKernel(quadraticBufferKernel, rmr_args, numAtoms);
  
  rmr.download(fcb);
  T rmr_sum_1 = std::accumulate(fcb, fcb + 3 * numAtoms, (T)0);

  // check convergence -- rsd.rsd
  rsd_real.download(fcb);
  for (int i = 0; i < 3 * numAtoms; ++i) {
     fcb[i] = fcb[i] * fcb[i];
  }
  T conv = inducedEpsilon;
  conv = conv * conv * numAtoms * numAtoms;
  if (std::accumulate(fcb, fcb + 3 * numAtoms, (T)0) < conv) {
     printf("we're going to return true at %dth iteration \n",niter);
    return true;
  }

  T beta = rmr_sum_1 / rmr_sum_0;
  rmrSum = rmr_sum_1;

  printf("beta = %12.4lf, gamma = %12.4lf \n",beta,gamma);

  //////// pvec <--updated by-- M.rsd + beta*pvec
  void* pvec_args[] = {&pvec.getDevicePointer(), &rsd_real.getDevicePointer(),
  &polarity.getDevicePointer(), &beta};
  cu.executeKernel(updatePVecKernel, pvec_args, numAtoms);

      inducedDipoleVec.reserve(inducedDipole.getSize());
      pvec.download(&inducedDipoleVec[0]);
      for (int i = 0; i < numAtoms; ++i) {
         double dx, dy, dz;
         if (useDoublePrecision) {
            dx = inducedDipoleVec[3*i];
            dy = inducedDipoleVec[3*i+1];
            dz = inducedDipoleVec[3*i+2];
         }
         else {
            float* ptr = (float*)(&inducedDipoleVec[0]);
            dx = ptr[3*i];
            dy = ptr[3*i+1];
            dz = ptr[3*i+2];
         }
         printf("100 pvec for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
      }


  //////// update tp
  tmat(pvec, tp);

  //////// ptp
  void* ptp_args[] = {
     &pvec.getDevicePointer(), &tp.getDevicePointer(), &ptp.getDevicePointer()
  };
  cu.executeKernel(dotProductBufferKernel, ptp_args, numAtoms);

   printf("still not converged at %dth iteration \n",niter);
   return false;
}


template <class T> void CudaCalcHippoChargeTransferForceKernel::diagPCG(int maxIterations) {
  diagPCGInit<T>();
  printf("max iteration = %d \n",maxIterations);
  for (int i = 0; i < maxIterations; ++i) {
     printf("starting iteration %d\n",i);
    if (diagPCGIter<T>(i))
      break;
  }
}











double CudaCalcHippoChargeTransferForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy)
{
   printf(" -- CT execute first line\n");
   if (!hasInitializedScaleFactors)
   {
      initializeScaleFactors();
   }

   void *rotpoleArgs[] = {&cu.getPosq().getDevicePointer(), &axisInfo.getDevicePointer(),
                          &localFrameDipoles.getDevicePointer(), &localFrameQuadrupoles.getDevicePointer(),
                          &globalFrameDipoles.getDevicePointer(), &globalFrameQuadrupoles.getDevicePointer()};
   cu.executeKernel(rotpoleKernel, rotpoleArgs, cu.getNumAtoms());

   printf(" -- CT execute \n");
   printf(" RACKZILLA - attack \n");

   CudaNonbondedUtilities &nb = cu.getNonbondedUtilities();
   int startTileIndex = nb.getStartTileIndex();
   int numTileIndices = nb.getNumTiles();
   unsigned int maxTiles = nb.getInteractingTiles().getSize();

   // execute energy and force kernel

   void *argsEnergyAndForce[] = {
       &cu.getForce().getDevicePointer(),
        //&torque->getDevicePointer(),
       &torque.getDevicePointer(),
       &cu.getEnergyBuffer().getDevicePointer(),
       //&field->getDevicePointer(),
       &mpoleField.getDevicePointer(),
       &cu.getPosq().getDevicePointer(),
       &covalentFlags.getDevicePointer(),
       &nb.getExclusionTiles().getDevicePointer(),
       &startTileIndex, &numTileIndices,

       &nb.getInteractingTiles().getDevicePointer(),
       &nb.getInteractionCount().getDevicePointer(),
       cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
       cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
       &maxTiles, &nb.getBlockCenters().getDevicePointer(),
       &nb.getInteractingAtoms().getDevicePointer(),

       &chgct.getDevicePointer(), &dmpct.getDevicePointer(),

       &globalFrameDipoles.getDevicePointer(), &globalFrameQuadrupoles.getDevicePointer(), &sizpr.getDevicePointer(), &dmppr.getDevicePointer(), &elepr.getDevicePointer(), &pcore.getDevicePointer(), &pval.getDevicePointer(), &palpha.getDevicePointer(), &csix.getDevicePointer()};

   int numForceThreadBlocks = nb.getNumForceThreadBlocks();
   printf(" -- numForceThreadBlocks %d energyAndForceThreads = %d\n", numForceThreadBlocks, energyAndForceThreads);

   cu.executeKernel(energyAndForceKernel, argsEnergyAndForce, numForceThreadBlocks * energyAndForceThreads, energyAndForceThreads);

   // print real part of mpolefield
   std::vector<long long> mpoleFieldVec;
   mpoleFieldVec.reserve(mpoleField.getSize());
   mpoleField.download(mpoleFieldVec);
   for (int i = 0; i < paddedNumAtoms; ++i) {
      // x1, x2, x3, ..., xN; y1, y2, y3, ..., yN; z1, z2, z3, ..., zN.
      long long fx, fy, fz;
      fx = mpoleFieldVec[i];
      fy = mpoleFieldVec[i+paddedNumAtoms];
      fz = mpoleFieldVec[i+2*paddedNumAtoms];
      double dx, dy, dz;
      const double long_long_to_double = 1.0 / 0x100000000;
      dx = long_long_to_double * fx;
      dy = long_long_to_double * fy;
      dz = long_long_to_double * fz;
      printf("real field for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
   }

   printf(" Finished Real Space \n");

   // if use PME

   printf("Initialized PME? %d \n",hasInitializedFFT);

   if (hasInitializedFFT)
   {
      // reciporcal box vectors
      Vec3 boxVectors[3];
      cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
      double determinant = boxVectors[0][0] * boxVectors[1][1] * boxVectors[2][2];
      double scale = 1.0 / determinant;
      double3 recipBoxVectors[3];
      recipBoxVectors[0] = make_double3(boxVectors[1][1] * boxVectors[2][2] * scale, 0, 0);
      recipBoxVectors[1] = make_double3(-boxVectors[1][0] * boxVectors[2][2] * scale,
                                        boxVectors[0][0] * boxVectors[2][2] * scale, 0);
      recipBoxVectors[2] = make_double3((boxVectors[1][0] * boxVectors[2][1] - boxVectors[1][1] * boxVectors[2][0]) * scale,
                                        -boxVectors[0][0] * boxVectors[2][1] * scale,
                                        boxVectors[0][0] * boxVectors[1][1] * scale);
      float3 recipBoxVectorsFloat[3];
      if (useDoublePrecision)
      {
         recipBoxVectorPointer[0] = &recipBoxVectors[0];
         recipBoxVectorPointer[1] = &recipBoxVectors[1];
         recipBoxVectorPointer[2] = &recipBoxVectors[2];
      }
      else
      {
         recipBoxVectorsFloat[0] = make_float3((float)recipBoxVectors[0].x, 0, 0);
         recipBoxVectorsFloat[1] = make_float3(
             (float)recipBoxVectors[1].x, (float)recipBoxVectors[1].y, 0);
         recipBoxVectorsFloat[2] = make_float3((float)recipBoxVectors[2].x,
                                               (float)recipBoxVectors[2].y, (float)recipBoxVectors[2].z);
         recipBoxVectorPointer[0] = &recipBoxVectorsFloat[0];
         recipBoxVectorPointer[1] = &recipBoxVectorsFloat[1];
         recipBoxVectorPointer[2] = &recipBoxVectorsFloat[2];
      }

      // cartesian mpole to fractional mpole
      cmp_to_fmp();
      // put fractrional mpole on the grid
      grid_mpole();
      fftfront();
      pme_convolution();
      fftback();
      // get fractional potential/field/field gradient due to mpole;
      // this function also computes the (recip+self) d/p mpole fields.
      fphi_mpole();
      // convert fractional phi to cartesian phi
      fphi_to_cphi(fphi);

      // print mpoleField

      std::vector<long long> mpoleFieldVec;
      mpoleFieldVec.reserve(mpoleField.getSize());
      mpoleField.download(mpoleFieldVec);
      for (int i = 0; i < paddedNumAtoms; ++i) {
         // x1, x2, x3, ..., xN; y1, y2, y3, ..., yN; z1, z2, z3, ..., zN.
         long long fx, fy, fz;
         fx = mpoleFieldVec[i];
         fy = mpoleFieldVec[i+paddedNumAtoms];
         fz = mpoleFieldVec[i+2*paddedNumAtoms];
         double dx, dy, dz;
         const double long_long_to_double = 1.0 / 0x100000000;
         dx = long_long_to_double * fx;
         dy = long_long_to_double * fy;
         dz = long_long_to_double * fz;
         printf("field for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
      }

      // recip force and torque
      recip_mpole_energy_force_torque();

      // add real space d/p mpole field -> total d/p field
      // d field is correct!

      // use mu_direct (alpha.field) as mu_0

      void* dotArgs[] = {
         &mpoleField.getDevicePointer(),
         &polarity.getDevicePointer(),
         &inducedDipole.getDevicePointer()};
      cu.executeKernel(dipoleDotPolarityKernel, dotArgs, numAtoms);
      cu.clearBuffer(inducedDipoleP);


      // print induced dipoles
      //std::vector<double> inducedDipoleVec;
      //inducedDipoleVec.reserve(inducedDipole.getSize());
      //inducedDipole.download(&inducedDipoleVec[0]);
      //for (int i = 0; i < numAtoms; ++i) {
      //   double dx, dy, dz;
      //   if (useDoublePrecision) {
      //      dx = inducedDipoleVec[3*i];
      //      dy = inducedDipoleVec[3*i+1];
      //      dz = inducedDipoleVec[3*i+2];
      //   }
      //   else {
      //      float* ptr = (float*)(&inducedDipoleVec[0]);
      //      dx = ptr[3*i];
      //      dy = ptr[3*i+1];
      //      dz = ptr[3*i+2];
      //   }
      //   printf("dipole for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx, dy, dz);
      //}

      //ufield();




      // subtract T.mu_0 from E -> residual = E - T.mu_0

      // reduce the residual via iterations

      if (useDoublePrecision) {
          diagPCG<double>(maxIterations);
      } else {
          diagPCG<float>(maxIterations);
      }

      // print induced dipoles
      std::vector<double> inducedDipoleVec;
      inducedDipoleVec.reserve(inducedDipole.getSize());
      inducedDipole.download(&inducedDipoleVec[0]);
      for (int i = 0; i < numAtoms; ++i) {
         double dx, dy, dz;
         if (useDoublePrecision) {
            dx = inducedDipoleVec[3*i];
            dy = inducedDipoleVec[3*i+1];
            dz = inducedDipoleVec[3*i+2];
         }
         else {
            float* ptr = (float*)(&inducedDipoleVec[0]);
            dx = ptr[3*i];
            dy = ptr[3*i+1];
            dz = ptr[3*i+2];
         }
         printf("final induced dipole (debye) for atom %5d %12.5e%12.5e%12.5e\n", i+1, dx/0.020819434, dy/0.020819434, dz/0.020819434);
      }




      // assume the induced dipoles are converged
      // (real space + self) * (energy/force/torque)

      // compute real space polarization energy and force

      // induced dipole energy
      void* muDotMpoleArgs[] = {
         &inducedDipole.getDevicePointer(),
         &mpoleField.getDevicePointer(),
         &cu.getEnergyBuffer().getDevicePointer()};
      cu.executeKernel(muDotMpoleKernel, muDotMpoleArgs, numAtoms);


      void *argsPolar[] = {
          &cu.getForce().getDevicePointer(),
          &torque.getDevicePointer(),
          &cu.getEnergyBuffer().getDevicePointer(),
          &cu.getPosq().getDevicePointer(),
          &covalentFlags.getDevicePointer(),
          &covalentFlags24.getDevicePointer(),
          &nb.getExclusionTiles().getDevicePointer(),
          &startTileIndex, &numTileIndices,
   
          &nb.getInteractingTiles().getDevicePointer(),
          &nb.getInteractionCount().getDevicePointer(),
          cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
          cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
          &maxTiles, &nb.getBlockCenters().getDevicePointer(),
          &nb.getInteractingAtoms().getDevicePointer(),
      
          &globalFrameDipoles.getDevicePointer(), &globalFrameQuadrupoles.getDevicePointer(), 
          &pcore.getDevicePointer(), &pval.getDevicePointer(), &palpha.getDevicePointer(), 
          
          &inducedDipole.getDevicePointer()};
   
      int numForceThreadBlocks = nb.getNumForceThreadBlocks();
   
      cu.executeKernel(epolarKernel, argsPolar, numForceThreadBlocks * energyAndForceThreads, energyAndForceThreads);


      // recip energy/force/torque

      void* fphi_to_cphi_ind_args[] = {&fphi.getDevicePointer(), &cphi.getDevicePointer(),
                             recipBoxVectorPointer[0], recipBoxVectorPointer[1],
                             recipBoxVectorPointer[2]};
      cu.executeKernel(fphi_to_cphi_ind_kernel, fphi_to_cphi_ind_args, numAtoms);

      void* eprecip_args[] = {&cu.getPosq(),
                        &cu.getForce(),
                        &torque.getDevicePointer(),
                        &cu.getEnergyBuffer(),
                        &globalFrameDipoles.getDevicePointer(),
                        &globalFrameQuadrupoles.getDevicePointer(),
                        &fracDipoles.getDevicePointer(),
                        &fracQuadrupoles.getDevicePointer(),
                        &inducedDipole.getDevicePointer(),
                        &inducedDipoleP.getDevicePointer(),
                        &fphi.getDevicePointer(),
                        &phid.getDevicePointer(),
                        &phid.getDevicePointer(),
                        &phidp.getDevicePointer(),
                        &cphi.getDevicePointer(),
                        recipBoxVectorPointer[0],
                        recipBoxVectorPointer[1],
                        recipBoxVectorPointer[2]};
      cu.executeKernel(eprecip_kernel, eprecip_args, numAtoms);






   }

   // Map torques to force.

   void *mapTorqueArgs[] = {&cu.getForce().getDevicePointer(),
                            //&torque->getDevicePointer(),
                            &torque.getDevicePointer(),
                            &cu.getPosq().getDevicePointer(),
                            //&multipoleParticles->getDevicePointer(),
                            &axisInfo.getDevicePointer()};
   
   //printf("WARNING! not resolving torques\n");
   cu.executeKernel(mapTorqueKernel, mapTorqueArgs, cu.getNumAtoms());

   return 0.0;
}

void CudaCalcHippoChargeTransferForceKernel::copyParametersToContext(ContextImpl &context, const HippoChargeTransferForce &force)
{
   cu.setAsCurrent();
}
