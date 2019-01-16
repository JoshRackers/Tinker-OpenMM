extern "C" __global__ void hippoRotpole(const real4* __restrict__ posq, 
   const int4* __restrict__ axisInfo, 
   const real* __restrict__ localFrameDipoles,
   const real* __restrict__ localFrameQuadrupoles, 
   real* __restrict__ globalFrameDipoles, 
   real* __restrict__ globalFrameQuadrupoles)
{
    /*
        Integers/Enums defined in OpenMM API:
        0 -- 'Z-then-X'
        1 -- 'Bisector'
        2 -- 'Z-Bisect'
        3 -- '3-Fold'
        4 -- 'Z-Only'
        5 -- 'None'
    */

    /*
        z-only
           (1) norm z
           (2) select random x
           (3) x = x - (x.z)z
           (4) norm x

        z-then-x
           (1) norm z
           (2) norm x (not needed)
           (3) x = x - (x.z)z
           (4) norm x

        bisector
           (1) norm z
           (2) norm x
           (3) z = x + z
           (4) norm z
           (5) x = x - (x.z)z
           (6) norm x

        z-bisect
           (1) norm z
           (2) norm x
           (3) norm y
           (3) x = x + y
           (4) norm x
           (5) x = x - (x.z)z
           (6) norm x

        3-fold
           (1) norm z
           (2) norm x
           (3) norm y
           (4) z = x + y + z
           (5) norm z
           (6) x = x - (x.z)z
           (7) norm x

    */
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < NUM_ATOMS; atom += gridDim.x * blockDim.x) {
        int4 atomAxisInfo = axisInfo[atom]; // x, y, z, and axis type

        if (atomAxisInfo.x >= 0 && atomAxisInfo.z >= 0) {
            real4 thisParticlePos = posq[atom];
            real4 posZ = posq[atomAxisInfo.z];
            real3 vectorZ = make_real3(posZ.x - thisParticlePos.x, posZ.y - thisParticlePos.y, posZ.z - thisParticlePos.z);
            real4 posX = posq[atomAxisInfo.x];
            real3 vectorX = make_real3(posX.x - thisParticlePos.x, posX.y - thisParticlePos.y, posX.z - thisParticlePos.z);
            int axisType = atomAxisInfo.w;

            // branch based on axis type

            vectorZ = normalize(vectorZ);

            if (axisType == 1) {
                // bisector

                vectorX = normalize(vectorX);
                vectorZ += vectorX;
                vectorZ = normalize(vectorZ);
            } else if (axisType == 2 || axisType == 3) {
                // z-bisect

                if (atomAxisInfo.y >= 0 && atomAxisInfo.y < NUM_ATOMS) {
                    real4 posY = posq[atomAxisInfo.y];
                    real3 vectorY = make_real3(posY.x - thisParticlePos.x, posY.y - thisParticlePos.y, posY.z - thisParticlePos.z);
                    vectorY = normalize(vectorY);
                    vectorX = normalize(vectorX);
                    if (axisType == 2) {
                        vectorX += vectorY;
                        vectorX = normalize(vectorX);
                    } else {
                        // 3-fold

                        vectorZ += vectorX + vectorY;
                        vectorZ = normalize(vectorZ);
                    }
                }

            } else if (axisType >= 4) {
                vectorX = make_real3((real)0.1f);
            }

            // x = x - (x.z)z

            vectorX -= dot(vectorZ, vectorX) * vectorZ;
            vectorX = normalize(vectorX);
            real3 vectorY = cross(vectorZ, vectorX);

            // use identity rotation matrix for unrecognized axis types

            if (axisType < 0 || axisType > 4) {
                vectorX.x = 1;
                vectorX.y = 0;
                vectorX.z = 0;

                vectorY.x = 0;
                vectorY.y = 1;
                vectorY.z = 0;

                vectorZ.x = 0;
                vectorZ.y = 0;
                vectorZ.z = 1;
            }

            // Check the chirality and see whether it needs to be reversed

            bool reverse = false;
            if (axisType != 0 && atomAxisInfo.x >= 0 && atomAxisInfo.y >= 0 && atomAxisInfo.z >= 0) {
                real4 posY = posq[atomAxisInfo.y];
                real delta[4][3];

                delta[0][0] = thisParticlePos.x - posY.x;
                delta[0][1] = thisParticlePos.y - posY.y;
                delta[0][2] = thisParticlePos.z - posY.z;

                delta[1][0] = posZ.x - posY.x;
                delta[1][1] = posZ.y - posY.y;
                delta[1][2] = posZ.z - posY.z;

                delta[2][0] = posX.x - posY.x;
                delta[2][1] = posX.y - posY.y;
                delta[2][2] = posX.z - posY.z;

                delta[3][0] = delta[1][1] * delta[2][2] - delta[1][2] * delta[2][1];
                delta[3][1] = delta[2][1] * delta[0][2] - delta[2][2] * delta[0][1];
                delta[3][2] = delta[0][1] * delta[1][2] - delta[0][2] * delta[1][1];

                real volume = delta[3][0] * delta[0][0] + delta[3][1] * delta[1][0] + delta[3][2] * delta[2][0];
                reverse = (volume < 0);
            }

            // Transform the dipole

            unsigned int offset = 3 * atom;
            real molDipole[3];
            molDipole[0] = localFrameDipoles[offset];
            molDipole[1] = localFrameDipoles[offset + 1];
            molDipole[2] = localFrameDipoles[offset + 2];
            if (reverse)
                molDipole[1] *= -1;
            globalFrameDipoles[offset] = molDipole[0] * vectorX.x + molDipole[1] * vectorY.x + molDipole[2] * vectorZ.x;
            globalFrameDipoles[offset + 1] = molDipole[0] * vectorX.y + molDipole[1] * vectorY.y + molDipole[2] * vectorZ.y;
            globalFrameDipoles[offset + 2] = molDipole[0] * vectorX.z + molDipole[1] * vectorY.z + molDipole[2] * vectorZ.z;

            // Transform the quadrupole

            offset = 5 * atom;
            real mPoleXX = localFrameQuadrupoles[offset];
            real mPoleXY = localFrameQuadrupoles[offset + 1];
            real mPoleXZ = localFrameQuadrupoles[offset + 2];
            real mPoleYY = localFrameQuadrupoles[offset + 3];
            real mPoleYZ = localFrameQuadrupoles[offset + 4];
            real mPoleZZ = -(mPoleXX + mPoleYY);

            if (reverse) {
                mPoleXY *= -1;
                mPoleYZ *= -1;
            }

            globalFrameQuadrupoles[offset] = vectorX.x * (vectorX.x * mPoleXX + vectorY.x * mPoleXY + vectorZ.x * mPoleXZ)
                + vectorY.x * (vectorX.x * mPoleXY + vectorY.x * mPoleYY + vectorZ.x * mPoleYZ)
                + vectorZ.x * (vectorX.x * mPoleXZ + vectorY.x * mPoleYZ + vectorZ.x * mPoleZZ);
            globalFrameQuadrupoles[offset + 1] = vectorX.x * (vectorX.y * mPoleXX + vectorY.y * mPoleXY + vectorZ.y * mPoleXZ)
                + vectorY.x * (vectorX.y * mPoleXY + vectorY.y * mPoleYY + vectorZ.y * mPoleYZ)
                + vectorZ.x * (vectorX.y * mPoleXZ + vectorY.y * mPoleYZ + vectorZ.y * mPoleZZ);
            globalFrameQuadrupoles[offset + 2] = vectorX.x * (vectorX.z * mPoleXX + vectorY.z * mPoleXY + vectorZ.z * mPoleXZ)
                + vectorY.x * (vectorX.z * mPoleXY + vectorY.z * mPoleYY + vectorZ.z * mPoleYZ)
                + vectorZ.x * (vectorX.z * mPoleXZ + vectorY.z * mPoleYZ + vectorZ.z * mPoleZZ);
            globalFrameQuadrupoles[offset + 3] = vectorX.y * (vectorX.y * mPoleXX + vectorY.y * mPoleXY + vectorZ.y * mPoleXZ)
                + vectorY.y * (vectorX.y * mPoleXY + vectorY.y * mPoleYY + vectorZ.y * mPoleYZ)
                + vectorZ.y * (vectorX.y * mPoleXZ + vectorY.y * mPoleYZ + vectorZ.y * mPoleZZ);
            globalFrameQuadrupoles[offset + 4] = vectorX.y * (vectorX.z * mPoleXX + vectorY.z * mPoleXY + vectorZ.z * mPoleXZ)
                + vectorY.y * (vectorX.z * mPoleXY + vectorY.z * mPoleYY + vectorZ.z * mPoleYZ)
                + vectorZ.y * (vectorX.z * mPoleXZ + vectorY.z * mPoleYZ + vectorZ.z * mPoleZZ);
        } else {
            globalFrameDipoles[3 * atom] = localFrameDipoles[3 * atom];
            globalFrameDipoles[3 * atom + 1] = localFrameDipoles[3 * atom + 1];
            globalFrameDipoles[3 * atom + 2] = localFrameDipoles[3 * atom + 2];
            globalFrameQuadrupoles[5 * atom] = localFrameQuadrupoles[5 * atom];
            globalFrameQuadrupoles[5 * atom + 1] = localFrameQuadrupoles[5 * atom + 1];
            globalFrameQuadrupoles[5 * atom + 2] = localFrameQuadrupoles[5 * atom + 2];
            globalFrameQuadrupoles[5 * atom + 3] = localFrameQuadrupoles[5 * atom + 3];
            globalFrameQuadrupoles[5 * atom + 4] = localFrameQuadrupoles[5 * atom + 4];
        }
    }
}
