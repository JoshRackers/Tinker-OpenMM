class CudaCalcHippoChargeTransferForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const HippoChargeTransferForce& force)
        : force(force)
    {
        printf(" Cuda CT force info\n");
    }

    bool areParticlesIdentical(int p1, int p2) {
        double a1, a2, c1, c2;
        force.getCTParameters(p1, a1, c1);
        force.getCTParameters(p2, a2, c2);
        if (a1 != a2 || c1 != c2) {
            return false;
        } else {
            return true;
        }
    }

    int getNumParticleGroups() {
        return force.getNumCTSites();
    }

    void getParticlesInGroup(int index, vector<int>& particles) {
        // int particle = index/4;
        // int type = index - 4*particle;
        // force.getCovalentMap(particle, HippoChargeTransferForce::CovalentType(type), particles);
        // if (type == HippoChargeTransferForce::Covalent12) {
        //     particles.push_back(particle);
        // }
        // for (int i = 0; i < particles.size(); ++i) {
        //     printf(" CTForceInfo particle#%d type#%d : %d\n", particle, type, particles[i]);
        // }
        force.getCovalentMap(index, HippoChargeTransferForce::Covalent12, particles);
        particles.push_back(index);
    }

    bool areGroupsIdentical(int group1, int group2) {
        // return (group1%4) == (group2%4);
        return group1 == group2;
    }

private:
    const HippoChargeTransferForce& force;
};

CudaCalcHippoChargeTransferForceKernel::CudaCalcHippoChargeTransferForceKernel(
    std::string name, const Platform& platform, CudaContext& cu, const System& system)
    : CalcHippoChargeTransferForceKernel(name, platform)
    , numCTSites(0)
    , hasInitializedScaleFactors(false)
    , cu(cu)
    , system(system)
{}

CudaCalcHippoChargeTransferForceKernel::~CudaCalcHippoChargeTransferForceKernel()
{
    cu.setAsCurrent();
}

void CudaCalcHippoChargeTransferForceKernel::initialize(const System& system, const HippoChargeTransferForce& force)
{
    cu.setAsCurrent();

    printf(" -- CT initialize \n");

    numCTSites = force.getNumCTSites();
    int paddedNumAtoms = cu.getPaddedNumAtoms();

    printf(" num CT sites = %d\n", numCTSites);

    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));

    vector<vector<int> > exclusions(numCTSites);
    for (int i = 0; i < numCTSites; i++) {
        vector<int> atoms;
        set<int> allAtoms;
        allAtoms.insert(i);
        
        force.getCovalentMap(i, HippoChargeTransferForce::Covalent12, atoms);
        for (int ii = 0; ii < atoms.size(); ++ii) {
            printf(" covalent 12 for atom %d : %d\n", i, atoms[ii]);
        }
        allAtoms.insert(atoms.begin(), atoms.end());
        
        force.getCovalentMap(i, HippoChargeTransferForce::Covalent13, atoms);
        for (int ii = 0; ii < atoms.size(); ++ii) {
            printf(" covalent 13 for atom %d : %d\n", i, atoms[ii]);
        }
        allAtoms.insert(atoms.begin(), atoms.end());
        for (set<int>::const_iterator iter = allAtoms.begin(); iter != allAtoms.end(); ++iter)
            covalentFlagValues.push_back(make_int3(i, *iter, 0));
        
        force.getCovalentMap(i, HippoChargeTransferForce::Covalent14, atoms);
        for (int ii = 0; ii < atoms.size(); ++ii) {
            printf(" covalent 14 for atom %d : %d\n", i, atoms[ii]);
        }
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int j = 0; j < (int) atoms.size(); j++)
            covalentFlagValues.push_back(make_int3(i, atoms[j], 1));
        
        force.getCovalentMap(i, HippoChargeTransferForce::Covalent15, atoms);
        for (int ii = 0; ii < atoms.size(); ++ii) {
            printf(" covalent 15 for atom %d : %d\n", i, atoms[ii]);
        }
        for (int j = 0; j < (int) atoms.size(); j++)
            covalentFlagValues.push_back(make_int3(i, atoms[j], 2));
        allAtoms.insert(atoms.begin(), atoms.end());

        exclusions[i].insert(exclusions[i].end(), allAtoms.begin(), allAtoms.end());
    }

    for (int i = 0; i < covalentFlagValues.size(); ++i) {
        int atom1 = covalentFlagValues[i].x;
        int atom2 = covalentFlagValues[i].y;
        int value = covalentFlagValues[i].z;
        int f1 = (value == 0 || value == 1 ? 1 : 0);
        int f2 = (value == 0 || value == 2 ? 1 : 0);
        printf(" ## covalentFlagValues atom1 atom2 value f1 f2 %d %d %d %d %d\n", atom1, atom2, value, f1, f2);
    }

    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int j = 0; j < (int) exclusions[atom1].size(); ++j) {
            int atom2 = exclusions[atom1][j];
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }

    // set up alpha and chgct
    if (cu.getUseDoublePrecision()) {
        vector<double> alphaVec(numCTSites), chgctVec(numCTSites);
        double a, c;
        for (int ii = 0; ii < numCTSites; ++ii) {
            force.getCTParameters(ii, a, c);
            alphaVec[ii] = a;
            chgctVec[ii] = c;
        }
        alpha.initialize<double>(cu, numCTSites, "CTalpha");
        chgct.initialize<double>(cu, numCTSites, "CTchgct");
        alpha.upload(alphaVec);
        chgct.upload(chgctVec);
    } else {
        vector<float> alphaVec(numCTSites), chgctVec(numCTSites);
        double a, c;
        for (int ii = 0; ii < numCTSites; ++ii) {
            force.getCTParameters(ii, a, c);
            alphaVec[ii] = a;
            chgctVec[ii] = c;
        }
        alpha.initialize<float>(cu, numCTSites, "CTalpha");
        chgct.initialize<float>(cu, numCTSites, "CTchgct");
        alpha.upload(alphaVec);
        chgct.upload(chgctVec);
    }

    printf(" num ct sites: %d cutoff = %12.6lf forcegroup = %d\n", numCTSites, force.getCutoffDistance(), force.getForceGroup());
    for (int i = 0; i < exclusions.size(); ++i) {
        for (int j = 0; j < exclusions[i].size(); ++j) {
            printf(" exclusions[%d][%d] = %d\n", i,j, exclusions[i][j]);
        }
    }

    map<string, string> defines;

    // define macros for cuda source code

    defines["NUM_ATOMS"] = cu.intToString(numCTSites);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(paddedNumAtoms);

    double cscales[3];
    force.getCTScales(cscales[0], cscales[1], cscales[2]);
    defines["CHARGETRANSFER13SCALE"] = cu.doubleToString(cscales[0]);
    defines["CHARGETRANSFER14SCALE"] = cu.doubleToString(cscales[1]);
    defines["CHARGETRANSFER15SCALE"] = cu.doubleToString(cscales[2]);

    defines["USE_CUTOFF"] = "";
    defines["CUTOFF_SQUARED"] = cu.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());

    defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
    int numContexts = cu.getPlatformData().contexts.size();
    int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);

    int maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
    double energyAndForceMemory = 8.0*elementSize + 2.0*sizeof(int);
    energyAndForceThreads = min(maxThreads, cu.computeThreadBlockSize(energyAndForceMemory));
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(energyAndForceThreads);

    printf(" start compiling CT module\n");
    CUmodule chargeTransferModule = cu.createModule(CudaKernelSources::vectorOps
        +CudaAmoebaKernelSources::hippoCTexample, defines);
    printf(" compiling CT module 100 \n");
    fflush (stdout);
    exampleKernel = cu.getKernel(chargeTransferModule, "hippoCTexample");
    printf(" compiling CT module 200 \n");
    energyAndForceKernel = cu.getKernel(chargeTransferModule, "computeChargeTransfer");
    printf(" end compiling CT module\n");

    cu.getNonbondedUtilities().addInteraction(true, true, true, force.getCutoffDistance(), exclusions,  "", force.getForceGroup());
    printf(" before setUsePadding\n");
    cu.getNonbondedUtilities().setUsePadding(false);
    cu.addForce(new ForceInfo(force));
}


void CudaCalcHippoChargeTransferForceKernel::initializeScaleFactors() {
    hasInitializedScaleFactors = true;
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();

    // Figure out the covalent flag values to use for each atom pair.

    vector<ushort2> exclusionTiles;
    nb.getExclusionTiles().download(exclusionTiles);
    map<pair<int, int>, int> exclusionTileMap;
    printf(" -- total number of exclusions tiles %d\n", (int)exclusionTiles.size());
    for (int i = 0; i < (int) exclusionTiles.size(); i++) {
        ushort2 tile = exclusionTiles[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
        printf(" -- tile.x %d , tile.y %d , i %d\n", tile.x, tile.y, i);
    }
    covalentFlags.initialize<uint2>(cu, nb.getExclusions().getSize(), "covalentFlags");
    vector<uint2> covalentFlagsVec(nb.getExclusions().getSize(), make_uint2(0, 0));
    for (int i = 0; i < (int) covalentFlagValues.size(); i++) {
        int atom1 = covalentFlagValues[i].x;
        int atom2 = covalentFlagValues[i].y;
        int value = covalentFlagValues[i].z;
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        int y = atom2/CudaContext::TileSize;
        int offset2 = atom2-y*CudaContext::TileSize;
        int f1 = (value == 0 || value == 1 ? 1 : 0);
        int f2 = (value == 0 || value == 2 ? 1 : 0);
        printf(" -- covalentFlagValues atom1 atom2 value f1 f2 %d %d %d %d %d\n", atom1, atom2, value, f1, f2);
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
    }
    covalentFlags.upload(covalentFlagsVec);
}

double CudaCalcHippoChargeTransferForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy)
{
    printf(" -- CT execute first line\n");
    if (!hasInitializedScaleFactors) {
        initializeScaleFactors();
    }

    printf(" -- CT execute \n");

    //

    std::vector<float4> posqvec(cu.getPaddedNumAtoms());
    cu.getPosq().download(posqvec);

    for (int i = 0; i < 6; ++i) {
        printf(" -- downloaded posq values atom %d, %12.6f%12.6f%12.6f\n", i+1, posqvec[i].x, posqvec[i].y, posqvec[i].z);
    }

    //

    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    unsigned int maxTiles = nb.getInteractingTiles().getSize();

    void* argsEnergyAndForce[] = {
        &cu.getForce().getDevicePointer(),
        &cu.getEnergyBuffer().getDevicePointer(),
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

        &alpha.getDevicePointer(), &chgct.getDevicePointer()
    };

    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    printf(" -- numForceThreadBlocks %d energyAndForceThreads = %d\n", numForceThreadBlocks, energyAndForceThreads);
    cu.executeKernel(energyAndForceKernel, argsEnergyAndForce, numForceThreadBlocks*energyAndForceThreads, energyAndForceThreads);

    return 0.0;
}

void CudaCalcHippoChargeTransferForceKernel::copyParametersToContext(ContextImpl& context, const HippoChargeTransferForce& force)
{
    cu.setAsCurrent();
}
