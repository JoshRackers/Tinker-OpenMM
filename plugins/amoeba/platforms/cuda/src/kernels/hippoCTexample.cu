extern "C" __global__ void hippoCTexample()
{
    printf("Wazzzupp from GPU Thread = %2d Block = %2d \n", threadIdx.x, blockIdx.x);
}

typedef struct {
    real3 pos, force;
    real alpha, chgct;
} AtomData;

inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ alpha, const real* __restrict__ chgct)
{
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.alpha = alpha[atom];
    data.chgct = chgct[atom];
}

__device__ real computeCScaleFactor(uint2 covalent, int index)
{
    int mask = 1 << index;
    bool x = (covalent.x & mask);
    bool y = (covalent.y & mask);
    return (x ? (y ? (real)CHARGETRANSFER13SCALE : (real)CHARGETRANSFER14SCALE) : (y ? (real)CHARGETRANSFER15SCALE : (real)1.0));
}

__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real cscale, real doubleCountingFactor, 

/*
    real3& pos1, real3& pos2, real3& deltapos, real& distance, mixed& thisPairEnergy, 
// */

    mixed& energyToBeAccumulated, real4 periodicBoxSize, real4 invPeriodicBoxSize,
    real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ)
{
    // Compute the displacement.

    real3 delta;
    delta.x = atom1.pos.x - atom2.pos.x;
    delta.y = atom1.pos.y - atom2.pos.y;
    delta.z = atom1.pos.z - atom2.pos.z;
    APPLY_PERIODIC_TO_DELTA(delta)
    real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    /*
    pos1.x = atom1.pos.x;
    pos1.y = atom1.pos.y;
    pos1.z = atom1.pos.z;
    pos2.x = atom2.pos.x;
    pos2.y = atom2.pos.y;
    pos2.z = atom2.pos.z;
    deltapos.x = delta.x;
    deltapos.y = delta.y;
    deltapos.z = delta.z;
    distance = SQRT(r2);
    printf(" device %12.4f%12.4f%12.4f%12.4f%12.4f\n", 10*pos1.x, 10*pos2.x, 10*deltapos.x,10*distance,100*CUTOFF_SQUARED);
    // */
    if (r2 > CUTOFF_SQUARED)
        return;

    real rInv = RSQRT(r2);
    real r = r2 * rInv;

    real alphai = atom1.alpha;
    real alphak = atom2.alpha;
    real transferi = atom1.chgct;
    real transferk = atom2.chgct;
    real exptermi = EXP(-alphai * r); 
    real exptermk = EXP(-alphak * r);
    real energy = -transferi * exptermk - transferk * exptermi;
    energyToBeAccumulated += (mixed)doubleCountingFactor * cscale * energy;

    /*
    distance = r;
    thisPairEnergy = (mixed)doubleCountingFactor * cscale * energy;
    // */

    real de = transferi * alphak * exptermk + transferk * alphai * exptermi;
    real frcx = de * delta.x * rInv * cscale;
    real frcy = de * delta.y * rInv * cscale;
    real frcz = de * delta.z * rInv * cscale;

    atom1.force -= make_real3(frcx, frcy, frcz);
    if (doubleCountingFactor == 1) {
        atom2.force += make_real3(frcx, frcy, frcz);
    }

    // printf(" -- OneInteraction scale %.2f, energy %.6f, distance %.6f, force %12.6f%12.6f%12.6f%12.6f%12.6f \n", (float) cscale, (float) energy/4.184, (float) r,
    //     frcx/41.84, frcy/41.84, frcz/41.84, de/4.184, delta.x*10.0);
}

extern "C" __global__ void computeChargeTransfer(unsigned long long* __restrict__ forceBuffers, mixed* __restrict__ energyBuffer,
    const real4* __restrict__ posq, const uint2* __restrict__ covalentFlags, const ushort2* __restrict__ exclusionTiles, unsigned int startTileIndex,
    unsigned int numTileIndices,
#ifdef USE_CUTOFF
    const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX,
    real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
    const unsigned int* __restrict__ interactingAtoms,
#endif
    const real* __restrict__ alpha, const real* __restrict__ chgct)
{
    const unsigned int totalWarps = (blockDim.x * gridDim.x) / TILE_SIZE;
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE - 1);
    const unsigned int tbx = threadIdx.x - tgx;
    mixed energy = 0;
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];    

    // First loop: process tiles that contain exclusions.

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE + warp * (LAST_EXCLUSION_TILE - FIRST_EXCLUSION_TILE) / totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE + (warp + 1) * (LAST_EXCLUSION_TILE - FIRST_EXCLUSION_TILE) / totalWarps;

    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        AtomData data;
        unsigned int atom1 = x * TILE_SIZE + tgx;
        loadAtomData(data, atom1, posq, alpha, chgct);
        data.force = make_real3(0);
        uint2 covalent = covalentFlags[pos * TILE_SIZE + tgx];
        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].pos = data.pos;
            localData[threadIdx.x].alpha = data.alpha;
            localData[threadIdx.x].chgct = data.chgct;

            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real c = computeCScaleFactor(covalent, j);
                    mixed thisPairEnergy = 0;
                    real distance = -1.0;
                    real3 pos1, pos2, deltapos;
                    computeOneInteraction(
                        data, localData[tbx + j], c, (real)0.5, 

                        /*
                        pos1, pos2, deltapos, distance, thisPairEnergy,
                        // */


                        energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    /*
                    printf(" x == y atom = %d atom = %d and cscale = %.4f, # %12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f # , distance = %.4f, energy = %.4f box-xyz %12.4f%12.4f%12.4f\n", atom1+1, 1+atom2, (float)c,
                        pos1.x*10, pos1.y*10, pos1.z*10, pos2.x*10, pos2.y*10, pos2.z*10, deltapos.x*10, deltapos.y*10, deltapos.z*10,
                        (float) distance * 10, (float) thisPairEnergy / 4.184f,
                     10*periodicBoxSize.x, 10*periodicBoxSize.y, 10*periodicBoxSize.z);
                     // */
                }
            }

            // In this block we are double counting, so we only accumulate force
            // on atom1

            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
            atomicAdd(&forceBuffers[atom1 + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
            atomicAdd(&forceBuffers[atom1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));
        } else {
            // This is an off-diagonal tile.
//*
            unsigned int j = y * TILE_SIZE + tgx;
            loadAtomData(localData[threadIdx.x], j, posq, alpha, chgct);
            localData[threadIdx.x].force = make_real3(0);

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    float c = computeCScaleFactor(covalent, tj);
                    // printf(" x != y atom 1 = %d atom 2 = %d and cscale = %.4f\n", atom1, atom2, (float)c );
                    computeOneInteraction(
                        data, localData[tbx + tj], c, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            unsigned int offset = x * TILE_SIZE + tgx;

            // In this block we are not double counting, so we accumulate on
            // both atom1 and atom2

            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));
            offset = y * TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.z * 0x100000000)));
// */
        }
    }

        // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
        // of them (no cutoff).

//*
#ifdef USE_CUTOFF
        const unsigned int numTiles = interactionCount[0];
        if (numTiles > maxTiles)
            return; // There wasn't enough memory for the neighbor list.
        int pos = (int)(numTiles > maxTiles ? startTileIndex + warp * (long long)numTileIndices / totalWarps : warp * (long long)numTiles / totalWarps);
        int end
            = (int)(numTiles > maxTiles ? startTileIndex + (warp + 1) * (long long)numTileIndices / totalWarps : (warp + 1) * (long long)numTiles / totalWarps);
#else
        const unsigned int numTiles = numTileIndices;
        int pos = (int)(startTileIndex + warp * (long long)numTiles / totalWarps);
        int end = (int)(startTileIndex + (warp + 1) * (long long)numTiles / totalWarps);
#endif
        int skipBase = 0;
        int currentSkipIndex = tbx;
        skipTiles[threadIdx.x] = -1;

        while (pos < end) {
            bool includeTile = true;

            // Extract the coordinates of this tile.

            int x, y;
#ifdef USE_CUTOFF
            x = tiles[pos];
#else
            y = (int)floor(NUM_BLOCKS + 0.5f - SQRT((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2 * pos));
            x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
            if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
                y += (x < y ? -1 : 1);
                x = (pos - y * NUM_BLOCKS + y * (y + 1) / 2);
            }

            // Skip over tiles that have exclusions, since they were already processed.

            while (skipTiles[tbx + TILE_SIZE - 1] < pos) {
                if (skipBase + tgx < NUM_TILES_WITH_EXCLUSIONS) {
                    ushort2 tile = exclusionTiles[skipBase + tgx];
                    skipTiles[threadIdx.x] = tile.x + tile.y * NUM_BLOCKS - tile.y * (tile.y + 1) / 2;
                } else
                    skipTiles[threadIdx.x] = end;
                skipBase += TILE_SIZE;
                currentSkipIndex = tbx;
            }
            while (skipTiles[currentSkipIndex] < pos)
                currentSkipIndex++;
            includeTile = (skipTiles[currentSkipIndex] != pos);
#endif

            if (includeTile) {
                unsigned int atom1 = x * TILE_SIZE + tgx;

                // Load atom data for this tile.

                AtomData data;
                loadAtomData(data, atom1, posq, alpha, chgct);
                data.force = make_real3(0);

#ifdef USE_CUTOFF
                unsigned int j = interactingAtoms[pos * TILE_SIZE + tgx];
#else
                unsigned int j = y * TILE_SIZE + tgx;
#endif

                atomIndices[threadIdx.x] = j;
                loadAtomData(localData[threadIdx.x], j, posq, alpha, chgct);
                localData[threadIdx.x].force = make_real3(0);

                // Compute forces.

                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = atomIndices[tbx + tj];
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                        // printf(" NOEX atom 1 = %d atom 2 = %d and cscale = %.4f\n", atom1, atom2, 1.0f );
                        computeOneInteraction(
                            data, localData[tbx + tj], 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }

                // Write results.

                unsigned int offset = x * TILE_SIZE + tgx;
                atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
                atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
                atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));

#ifdef USE_CUTOFF
                offset = atomIndices[threadIdx.x];
#else
                offset = y * TILE_SIZE + tgx;
#endif

                atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.x * 0x100000000)));
                atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.y * 0x100000000)));
                atomicAdd(
                    &forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.z * 0x100000000)));
            }
            pos++;
        }
// */
        energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] += energy;
}
