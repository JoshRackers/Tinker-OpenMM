class CudaCalcHippoChargeTransferForceKernel : public CalcHippoChargeTransferForceKernel
{
  public:
    CudaCalcHippoChargeTransferForceKernel(std::string name, const Platform &platform, CudaContext &cu, const System &system);

    ~CudaCalcHippoChargeTransferForceKernel();

    void initialize(const System &system, const HippoChargeTransferForce &force);

    double execute(ContextImpl &context, bool includeForces, bool includeEnergy);

    void copyParametersToContext(ContextImpl &context, const HippoChargeTransferForce &force);

  private:
    class ForceInfo;
    void initializeScaleFactors();

   bool useDoublePrecision;
   int numAtoms, paddedNumAtoms;

    int energyAndForceThreads;
    bool hasInitializedScaleFactors;
    CudaContext &cu;
    const System &system;
    // dipole and quadrupole
    CudaArray axisInfo; // x, y, z, and axis type
    CudaArray localFrameDipoles;
    CudaArray localFrameQuadrupoles;
    CudaArray globalFrameDipoles;
    CudaArray globalFrameQuadrupoles;
    CudaArray inducedDipole, inducedDipoleP;
    CudaArray mpoleField, mpoleFieldP;
    CudaArray inducedField, inducedFieldP;
    CudaArray torque;

    // charge transfer
    CudaArray chgct, dmpct;
    // dispersion
    CudaArray csix;
    // replusion
    CudaArray sizpr, dmppr, elepr;
    // charge penetration electrostatics
    CudaArray pcore, pval, palpha, polarity;
    // torque
    //CudaArray *torque;

    // permanent field
    CudaArray *field;

    // PME
    bool hasInitializedFFT;
    cufftHandle fft;
    double pmeCutoffDistance, ewaldAlpha, ewaldErrorTolerance;
    int pmeorder, nfft1, nfft2, nfft3;
    void *recipBoxVectorPointer[3];
    CudaArray qgrid, bsmod1, bsmod2, bsmod3;
    CudaArray fracDipoles, fracQuadrupoles;
    CudaArray fphi, cphi;
    CudaArray phid, phip, phidp;

    // PCG
    int maxIterations;
    double inducedEpsilon, rmrSum;
    CudaArray rsd_ll, rsd_real, pvec, tp, ptp, rmr;
    template <class T> void diagPCGInit();
    template <class T> bool diagPCGIter(int niter);
    template <class T> void diagPCG(int maxIterations);
    CUfunction convertLongLongToReal3NKernel;
    CUfunction invAlphaPlusTUKernel;
    CUfunction dotProductBufferKernel, quadraticBufferKernel;
    CUfunction updateMuRsdKernel, updatePVecKernel;
    void tmat(CudaArray& inp, CudaArray& outp);

    //
    CUfunction cmp_to_fmp_kernel, grid_mpole_kernel, grid_uind_kernel,
        grid_convert_to_double_kernel;
    CUfunction pme_convolution_kernel;
    CUfunction fphi_mpole_kernel, fphi_uind_kernel, fphi_to_cphi_kernel,
        ufield_recip_self_kernel;
    CUfunction recip_mpole_energy_force_torque_kernel;
    CUfunction torque_to_force_kernel;
    void cmp_to_fmp();
    void grid_mpole();
    void grid_uind(CudaArray&);
    void fftfront();
    void pme_convolution();
    void fftback();
    void fphi_mpole();
    void fphi_uind();
    void fphi_to_cphi(CudaArray &phiarray);
    void ufield_recip_self(CudaArray&, CudaArray&);
    void recip_mpole_energy_force_torque();
    void torque_to_force();
    // Expects a "real" array as input, and a "long long" array as output.
    void ufield(CudaArray&, CudaArray&);

    // kernels
    std::vector<int3> covalentFlagValues;
    CudaArray covalentFlags;
    
    std::vector<int3> covalentFlag24Values;
    CudaArray covalentFlags24;

    CUfunction energyAndForceKernel;
    CUfunction ufieldKernel;
    CUfunction epolarKernel;

    // induced dipole recip
    CUfunction fphi_to_cphi_ind_kernel;
    CUfunction eprecip_kernel;

    //
    CUfunction muDotMpoleKernel; // dipoles dot perm field
    CUfunction dipoleDotPolarityKernel; // dot product routine
    CUfunction rotpoleKernel;   // Tinker rotpole subroutine
    CUfunction mapTorqueKernel; // OpenMM torque mapping routine
};
