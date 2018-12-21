class CudaCalcHippoChargeTransferForceKernel
   : public CalcHippoChargeTransferForceKernel {
public:
   CudaCalcHippoChargeTransferForceKernel(std::string name,
      const Platform& platform, CudaContext& cu, const System& system);

   ~CudaCalcHippoChargeTransferForceKernel();

   void initialize(const System& system, const HippoChargeTransferForce& force);

   double execute(ContextImpl& context, bool includeForces, bool includeEnergy);

   void copyParametersToContext(
      ContextImpl& context, const HippoChargeTransferForce& force);

private:
   class ForceInfo;
   void initializeScaleFactors();

   bool useDoublePrecision;
   int  numAtoms, paddedNumAtoms;
   int  energyAndForceThreads;
   bool hasInitializedScaleFactors;
   //
   CudaContext&  cu;
   const System& system;
   // rotpole
   CUfunction rotpole_kernel;
   void       rotpole();
   // dipole and quadrupole
   CudaArray axisInfo; // x, y, z, and axis type
   CudaArray localFrameDipoles, localFrameQuadrupoles;
   CudaArray globalFrameDipoles, globalFrameQuadrupoles;
   CudaArray mpoleField, mpoleFieldP, torque;
   // PME
   bool        hasInitializedFFT;
   cufftHandle fft;
   double      pmeCutoffDistance, ewaldAlpha, ewaldErrorTolerance;
   int         pmeorder, nfft1, nfft2, nfft3;
   void*       recipBoxVectorPointer[3];
   CudaArray   qgrid, bsmod1, bsmod2, bsmod3;
   CudaArray   fracDipoles, fracQuadrupoles;
   CudaArray   fphi, cphi;
   CudaArray   phid, phip, phidp;
   //
   CUfunction cmp_to_fmp_kernel, grid_mpole_kernel,
      grid_convert_to_double_kernel;
   CUfunction pme_convolution_kernel;
   CUfunction fphi_mpole_kernel, fphi_to_cphi_kernel;
   CUfunction recip_mpole_energy_force_torque_kernel;
   CUfunction torque_to_force_kernel;
   void       cmp_to_fmp();
   void       grid_mpole();
   void       fftfront();
   void       pme_convolution();
   void       fftback();
   void       fphi_mpole();
   void       fphi_to_cphi(CudaArray& phiarray);
   void       recip_mpole_energy_force_torque();
   void       torque_to_force();
   // charge transfer
   CudaArray chgct, dmpct;
   // replusion
   CudaArray sizpr, dmppr, elepr;
   // charge penetration electrostatics
   CudaArray pcore, pval, palpha;

   std::vector<int3> covalentFlagValues;
   CudaArray         covalentFlags;

   CUfunction energyAndForceKernel;
};
