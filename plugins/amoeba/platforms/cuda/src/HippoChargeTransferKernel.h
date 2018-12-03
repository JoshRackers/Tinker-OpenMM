class CudaCalcHippoChargeTransferForceKernel : public CalcHippoChargeTransferForceKernel {
public:
    CudaCalcHippoChargeTransferForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);

    ~CudaCalcHippoChargeTransferForceKernel();

    void initialize(const System& system, const HippoChargeTransferForce& force);

    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);

    void copyParametersToContext(ContextImpl& context, const HippoChargeTransferForce& force);

private:
	class ForceInfo;
    void initializeScaleFactors();

    int numCTSites;
    int energyAndForceThreads;
    bool hasInitializedScaleFactors;
    CudaContext& cu;
    const System& system;
    std::vector<int3> covalentFlagValues;
    CudaArray covalentFlags;
    CudaArray alpha, chgct;
    CUfunction exampleKernel;
    CUfunction energyAndForceKernel;
};
