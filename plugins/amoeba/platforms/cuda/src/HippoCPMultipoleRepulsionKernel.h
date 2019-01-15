class CudaCalcHippoCPMultipoleRepulsionForceKernel : public CalcHippoCPMultipoleRepulsionForceKernel {
public:
    CudaCalcHippoCPMultipoleRepulsionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcHippoCPMultipoleRepulsionForceKernel();
    void initialize(const System& system, const HippoCPMultipoleRepulsionForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const HippoCPMultipoleRepulsionForce& force);

private:
    class ForceInfo;
    CudaContext& cu;
    const System& system;
    CUfunction exampleKernel;
};
