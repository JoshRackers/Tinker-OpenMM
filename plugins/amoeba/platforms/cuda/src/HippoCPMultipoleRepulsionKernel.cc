class CudaCalcHippoCPMultipoleRepulsionForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const HippoCPMultipoleRepulsionForce& force)
        : force(force)
    {}

private:
    const HippoCPMultipoleRepulsionForce& force;
};

CudaCalcHippoCPMultipoleRepulsionForceKernel::CudaCalcHippoCPMultipoleRepulsionForceKernel(
    std::string name, const Platform& platform, CudaContext& cu, const System& system)
    : CalcHippoCPMultipoleRepulsionForceKernel(name, platform)
    , cu(cu)
    , system(system)
{}

CudaCalcHippoCPMultipoleRepulsionForceKernel::~CudaCalcHippoCPMultipoleRepulsionForceKernel()
{
    cu.setAsCurrent();
}

void CudaCalcHippoCPMultipoleRepulsionForceKernel::initialize(const System& system, const HippoCPMultipoleRepulsionForce& force)
{
    cu.setAsCurrent();

    map<string, string> defines;
    CUmodule exampleModule = cu.createModule(CudaAmoebaKernelSources::hippoCPExample, defines);
    exampleKernel = cu.getKernel(exampleModule, "hippoCPExample");

    cu.addForce(new ForceInfo(force));
}

double CudaCalcHippoCPMultipoleRepulsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy)
{
    void* args[] = {};
    cu.executeKernel(exampleKernel, args, 100);

    return 0.0;
}

void CudaCalcHippoCPMultipoleRepulsionForceKernel::copyParametersToContext(ContextImpl& context, const HippoCPMultipoleRepulsionForce& force)
{
    cu.setAsCurrent();
}
