__device__ void tinker_subroutine_bsplgen(
   real4* __restrict__ thetai, real w, real* __restrict__ array) {
#define ARRAY(x, y) array[(x)-1 + ((y)-1) * PME_ORDER]
   // array <=> bsbuild in tinker module "pme"

   // initialization to get to 2nd order recursion
   ARRAY(2, 2) = w;
   ARRAY(2, 1) = 1 - w;

   // perform one pass to get to 3rd order recursion
   ARRAY(3, 3) = 0.5f * w * ARRAY(2, 2);
   ARRAY(3, 2) = 0.5f * ((1 + w) * ARRAY(2, 1) + (2 - w) * ARRAY(2, 2));
   ARRAY(3, 1) = 0.5f * (1 - w) * ARRAY(2, 1);

   for (int i = 4; i <= PME_ORDER; ++i) {
      int k = i - 1;
      real denom = RECIP(k);
      ARRAY(i, i) = denom * w * ARRAY(k, k);
      for (int j = 1; j <= i - 2; j++)
         ARRAY(i, i - j) = denom
            * ((w + j) * ARRAY(k, i - j - 1) + (i - j - w) * ARRAY(k, i - j));
      ARRAY(i, 1) = denom * (1 - w) * ARRAY(k, 1);
   }

   // get coefficients for the B-spline first derivative
   int k = PME_ORDER - 1;
   ARRAY(k, PME_ORDER) = ARRAY(k, PME_ORDER - 1);
   for (int i = PME_ORDER - 1; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);

   // get coefficients for the B-spline second derivative
   k = PME_ORDER - 2;
   ARRAY(k, PME_ORDER - 1) = ARRAY(k, PME_ORDER - 2);
   for (int i = PME_ORDER - 2; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);
   ARRAY(k, PME_ORDER) = ARRAY(k, PME_ORDER - 1);
   for (int i = PME_ORDER - 1; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);

   // get coefficients for the B-spline third derivative
   k = PME_ORDER - 3;
   ARRAY(k, PME_ORDER - 2) = ARRAY(k, PME_ORDER - 3);
   for (int i = PME_ORDER - 3; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);
   ARRAY(k, PME_ORDER - 1) = ARRAY(k, PME_ORDER - 2);
   for (int i = PME_ORDER - 2; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);
   ARRAY(k, PME_ORDER) = ARRAY(k, PME_ORDER - 1);
   for (int i = PME_ORDER - 1; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);

   // copy coefficients from temporary to permanent storage
   for (int i = 1; i <= PME_ORDER; i++)
      thetai[i - 1] = make_real4(ARRAY(PME_ORDER, i), ARRAY(PME_ORDER - 1, i),
         ARRAY(PME_ORDER - 2, i), ARRAY(PME_ORDER - 3, i));

#undef ARRAY
}

extern "C" __global__ void cmp_to_fmp(const real* __restrict__ labFrameDipole,
   const real* labFrameQuadrupole, real* __restrict__ fracDipole,
   real* __restrict__ fracQuadrupole, real3 recipBoxVecX, real3 recipBoxVecY,
   real3 recipBoxVecZ) {
   // build matrices for transforming the dipoles and quadrupoles
   __shared__ real a[3][3];
   if (threadIdx.x == 0) {
      a[0][0] = GRID_SIZE_X * recipBoxVecX.x;
      a[0][1] = GRID_SIZE_X * recipBoxVecY.x;
      a[0][2] = GRID_SIZE_X * recipBoxVecZ.x;
      a[1][0] = GRID_SIZE_Y * recipBoxVecX.y;
      a[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
      a[1][2] = GRID_SIZE_Y * recipBoxVecZ.y;
      a[2][0] = GRID_SIZE_Z * recipBoxVecX.z;
      a[2][1] = GRID_SIZE_Z * recipBoxVecY.z;
      a[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
   }
   __syncthreads();
   int index1[] = {0, 0, 0, 1, 1, 2};
   int index2[] = {0, 1, 2, 1, 2, 2};
   __shared__ real b[6][6];
   if (threadIdx.x < 36) {
      int i = threadIdx.x / 6;
      int j = threadIdx.x - 6 * i;
      b[i][j] = a[index1[i]][index1[j]] * a[index2[i]][index2[j]];
      if (index1[i] != index2[i])
         b[i][j] += a[index1[i]][index2[j]] * a[index2[i]][index1[j]];
   }
   __syncthreads();

   // transform the multipoles
   real quadScale[] = {1, 2, 2, 1, 2, 1};
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
        i += blockDim.x * gridDim.x) {
      for (int j = 0; j < 3; j++) {
         real dipole = 0;
         for (int k = 0; k < 3; k++)
            dipole += a[j][k] * labFrameDipole[3 * i + k];
         fracDipole[3 * i + j] = dipole;
      }
      for (int j = 0; j < 6; j++) {
         real quadrupole = 0;
         for (int k = 0; k < 5; k++)
            quadrupole
               += quadScale[k] * b[j][k] * labFrameQuadrupole[5 * i + k];
         quadrupole -= quadScale[5] * b[j][5]
            * (labFrameQuadrupole[5 * i] + labFrameQuadrupole[5 * i + 3]);
         fracQuadrupole[6 * i + j] = quadrupole;
      }
   }
}

extern "C" __global__ void grid_mpole(const real4* __restrict__ posq,
   const real* __restrict__ fracDipole, const real* __restrict__ fracQuadrupole,
   real2* __restrict__ pmeGrid, real4 periodicBoxVecX, real4 periodicBoxVecY,
   real4 periodicBoxVecZ, real3 recipBoxVecX, real3 recipBoxVecY,
   real3 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
   real array[PME_ORDER * PME_ORDER];
#else
   // We have shared memory to spare, and putting the workspace array there
   // reduces the load on L2 cache.
   __shared__ real sharedArray[PME_ORDER * PME_ORDER * 64];
   real* array = &sharedArray[PME_ORDER * PME_ORDER * threadIdx.x];
#endif
   real4 theta1[PME_ORDER];
   real4 theta2[PME_ORDER];
   real4 theta3[PME_ORDER];

   for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < NUM_ATOMS;
        m += blockDim.x * gridDim.x) {
      real4 pos = posq[m];
      pos -= periodicBoxVecZ * floor(pos.z * recipBoxVecZ.z + 0.5f);
      pos -= periodicBoxVecY * floor(pos.y * recipBoxVecY.z + 0.5f);
      pos -= periodicBoxVecX * floor(pos.x * recipBoxVecX.z + 0.5f);
      real atomCharge = pos.w;
      real atomDipoleX = fracDipole[m * 3];
      real atomDipoleY = fracDipole[m * 3 + 1];
      real atomDipoleZ = fracDipole[m * 3 + 2];
      real atomQuadrupoleXX = fracQuadrupole[m * 6];
      real atomQuadrupoleXY = fracQuadrupole[m * 6 + 1];
      real atomQuadrupoleXZ = fracQuadrupole[m * 6 + 2];
      real atomQuadrupoleYY = fracQuadrupole[m * 6 + 3];
      real atomQuadrupoleYZ = fracQuadrupole[m * 6 + 4];
      real atomQuadrupoleZZ = fracQuadrupole[m * 6 + 5];

      // Since we need the full set of thetas, it's faster to compute them here
      // than load them from global memory.

      real w = pos.x * recipBoxVecX.x + pos.y * recipBoxVecY.x
         + pos.z * recipBoxVecZ.x;
      real fr = GRID_SIZE_X * (w - (int)(w + 0.5f) + 0.5f);
      int ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid1 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta1, w, array);
      w = pos.y * recipBoxVecY.y + pos.z * recipBoxVecZ.y;
      fr = GRID_SIZE_Y * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid2 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta2, w, array);
      w = pos.z * recipBoxVecZ.z;
      fr = GRID_SIZE_Z * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid3 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta3, w, array);
      igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
      igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
      igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

      // Spread the charge from this atom onto each grid point.

      for (int ix = 0; ix < PME_ORDER; ix++) {
         int xbase = igrid1 + ix;
         xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
         xbase = xbase * GRID_SIZE_Y * GRID_SIZE_Z;
         real4 t = theta1[ix];

         for (int iy = 0; iy < PME_ORDER; iy++) {
            int ybase = igrid2 + iy;
            ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
            ybase = xbase + ybase * GRID_SIZE_Z;
            real4 u = theta2[iy];

            for (int iz = 0; iz < PME_ORDER; iz++) {
               int zindex = igrid3 + iz;
               zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
               int index = ybase + zindex;
               real4 v = theta3[iz];

               real term0 = atomCharge * u.x * v.x + atomDipoleY * u.y * v.x
                  + atomDipoleZ * u.x * v.y + atomQuadrupoleYY * u.z * v.x
                  + atomQuadrupoleZZ * u.x * v.z + atomQuadrupoleYZ * u.y * v.y;
               real term1 = atomDipoleX * u.x * v.x
                  + atomQuadrupoleXY * u.y * v.x + atomQuadrupoleXZ * u.x * v.y;
               real term2 = atomQuadrupoleXX * u.x * v.x;
               real add = term0 * t.x + term1 * t.y + term2 * t.z;
#ifdef USE_DOUBLE_PRECISION
               unsigned long long* ulonglong_p = (unsigned long long*)pmeGrid;
               atomicAdd(&ulonglong_p[2 * index],
                  static_cast<unsigned long long>(
                     (long long)(add * 0x100000000)));
#else
               atomicAdd(&pmeGrid[index].x, add);
#endif
            }
         }
      }
   }
}

extern "C" __global__ void grid_uind(const real4* __restrict__ posq,
   const real* __restrict__ inducedDipole,
   const real* __restrict__ inducedDipolePolar, real2* __restrict__ pmeGrid,
   real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
   real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
   real array[PME_ORDER * PME_ORDER];
#else
   // We have shared memory to spare, and putting the workspace array there
   // reduces the load on L2 cache.
   __shared__ real sharedArray[PME_ORDER * PME_ORDER * 64];
   real* array = &sharedArray[PME_ORDER * PME_ORDER * threadIdx.x];
#endif
   real4 theta1[PME_ORDER];
   real4 theta2[PME_ORDER];
   real4 theta3[PME_ORDER];
   __shared__ real cartToFrac[3][3];
   if (threadIdx.x == 0) {
      cartToFrac[0][0] = GRID_SIZE_X * recipBoxVecX.x;
      cartToFrac[0][1] = GRID_SIZE_X * recipBoxVecY.x;
      cartToFrac[0][2] = GRID_SIZE_X * recipBoxVecZ.x;
      cartToFrac[1][0] = GRID_SIZE_Y * recipBoxVecX.y;
      cartToFrac[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
      cartToFrac[1][2] = GRID_SIZE_Y * recipBoxVecZ.y;
      cartToFrac[2][0] = GRID_SIZE_Z * recipBoxVecX.z;
      cartToFrac[2][1] = GRID_SIZE_Z * recipBoxVecY.z;
      cartToFrac[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
   }
   __syncthreads();

   for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < NUM_ATOMS;
        m += blockDim.x * gridDim.x) {
      real4 pos = posq[m];
      pos -= periodicBoxVecZ * floor(pos.z * recipBoxVecZ.z + 0.5f);
      pos -= periodicBoxVecY * floor(pos.y * recipBoxVecY.z + 0.5f);
      pos -= periodicBoxVecX * floor(pos.x * recipBoxVecX.z + 0.5f);
      real3 cinducedDipole = ((const real3*)inducedDipole)[m];
      real3 cinducedDipolePolar = ((const real3*)inducedDipolePolar)[m];
      real3 finducedDipole = make_real3(cinducedDipole.x * cartToFrac[0][0]
            + cinducedDipole.y * cartToFrac[0][1]
            + cinducedDipole.z * cartToFrac[0][2],
         cinducedDipole.x * cartToFrac[1][0]
            + cinducedDipole.y * cartToFrac[1][1]
            + cinducedDipole.z * cartToFrac[1][2],
         cinducedDipole.x * cartToFrac[2][0]
            + cinducedDipole.y * cartToFrac[2][1]
            + cinducedDipole.z * cartToFrac[2][2]);
      real3 finducedDipolePolar
         = make_real3(cinducedDipolePolar.x * cartToFrac[0][0]
               + cinducedDipolePolar.y * cartToFrac[0][1]
               + cinducedDipolePolar.z * cartToFrac[0][2],
            cinducedDipolePolar.x * cartToFrac[1][0]
               + cinducedDipolePolar.y * cartToFrac[1][1]
               + cinducedDipolePolar.z * cartToFrac[1][2],
            cinducedDipolePolar.x * cartToFrac[2][0]
               + cinducedDipolePolar.y * cartToFrac[2][1]
               + cinducedDipolePolar.z * cartToFrac[2][2]);

      // Since we need the full set of thetas, it's faster to compute them here
      // than load them from global memory.

      real w = pos.x * recipBoxVecX.x + pos.y * recipBoxVecY.x
         + pos.z * recipBoxVecZ.x;
      real fr = GRID_SIZE_X * (w - (int)(w + 0.5f) + 0.5f);
      int ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid1 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta1, w, array);
      w = pos.y * recipBoxVecY.y + pos.z * recipBoxVecZ.y;
      fr = GRID_SIZE_Y * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid2 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta2, w, array);
      w = pos.z * recipBoxVecZ.z;
      fr = GRID_SIZE_Z * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid3 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta3, w, array);
      igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
      igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
      igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

      // Spread the charge from this atom onto each grid point.

      for (int ix = 0; ix < PME_ORDER; ix++) {
         int xbase = igrid1 + ix;
         xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
         xbase = xbase * GRID_SIZE_Y * GRID_SIZE_Z;
         real4 t = theta1[ix];

         for (int iy = 0; iy < PME_ORDER; iy++) {
            int ybase = igrid2 + iy;
            ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
            ybase = xbase + ybase * GRID_SIZE_Z;
            real4 u = theta2[iy];

            for (int iz = 0; iz < PME_ORDER; iz++) {
               int zindex = igrid3 + iz;
               zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
               int index = ybase + zindex;
               real4 v = theta3[iz];

               real term01
                  = finducedDipole.y * u.y * v.x + finducedDipole.z * u.x * v.y;
               real term11 = finducedDipole.x * u.x * v.x;
               real term02 = finducedDipolePolar.y * u.y * v.x
                  + finducedDipolePolar.z * u.x * v.y;
               real term12 = finducedDipolePolar.x * u.x * v.x;
               real add1 = term01 * t.x + term11 * t.y;
               real add2 = term02 * t.x + term12 * t.y;
#ifdef USE_DOUBLE_PRECISION
               unsigned long long* ulonglong_p = (unsigned long long*)pmeGrid;
               atomicAdd(&ulonglong_p[2 * index],
                  static_cast<unsigned long long>(
                     (long long)(add1 * 0x100000000)));
               atomicAdd(&ulonglong_p[2 * index + 1],
                  static_cast<unsigned long long>(
                     (long long)(add2 * 0x100000000)));
#else
               atomicAdd(&pmeGrid[index].x, add1);
               atomicAdd(&pmeGrid[index].y, add2);
#endif
            }
         }
      }
   }
}

extern "C" __global__ void grid_convert_to_double(
   long long* __restrict__ pmeGrid) {
   real* floatGrid = (real*)pmeGrid;
   const unsigned int gridSize = 2 * GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
   real scale = 1 / (real)0x100000000;
   for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < gridSize;
        index += blockDim.x * gridDim.x) {
      floatGrid[index] = scale * pmeGrid[index];
   }
}

extern "C" __global__ void pme_convolution(real2* __restrict__ pmeGrid,
   const real* __restrict__ pmeBsplineModuliX,
   const real* __restrict__ pmeBsplineModuliY,
   const real* __restrict__ pmeBsplineModuliZ, real4 periodicBoxSize,
   real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ) {
   const unsigned int gridSize = GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
   real expFactor = M_PI * M_PI / (EWALD_ALPHA * EWALD_ALPHA);
   real scaleFactor
      = RECIP(M_PI * periodicBoxSize.x * periodicBoxSize.y * periodicBoxSize.z);
   for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < gridSize;
        index += blockDim.x * gridDim.x) {
      int kx = index / (GRID_SIZE_Y * GRID_SIZE_Z);
      int remainder = index - kx * GRID_SIZE_Y * GRID_SIZE_Z;
      int ky = remainder / GRID_SIZE_Z;
      int kz = remainder - ky * GRID_SIZE_Z;
      if (kx == 0 && ky == 0 && kz == 0) {
         pmeGrid[index] = make_real2(0, 0);
         continue;
      }
      int mx = (kx < (GRID_SIZE_X + 1) / 2) ? kx : (kx - GRID_SIZE_X);
      int my = (ky < (GRID_SIZE_Y + 1) / 2) ? ky : (ky - GRID_SIZE_Y);
      int mz = (kz < (GRID_SIZE_Z + 1) / 2) ? kz : (kz - GRID_SIZE_Z);
      real mhx = mx * recipBoxVecX.x;
      real mhy = mx * recipBoxVecY.x + my * recipBoxVecY.y;
      real mhz
         = mx * recipBoxVecZ.x + my * recipBoxVecZ.y + mz * recipBoxVecZ.z;
      real bx = pmeBsplineModuliX[kx];
      real by = pmeBsplineModuliY[ky];
      real bz = pmeBsplineModuliZ[kz];
      real2 grid = pmeGrid[index];
      real m2 = mhx * mhx + mhy * mhy + mhz * mhz;
      real denom = m2 * bx * by * bz;
      real eterm = scaleFactor * EXP(-expFactor * m2) / denom;
      pmeGrid[index] = make_real2(grid.x * eterm, grid.y * eterm);
   }
}

extern "C" __global__ void fphi_mpole(const real2* __restrict__ pmeGrid,
   real* __restrict__ phi /* fphi */, long long* __restrict__ fieldBuffers,
   long long* __restrict__ fieldPolarBuffers, const real4* __restrict__ posq,
   const real* __restrict__ labFrameDipole, real4 periodicBoxVecX,
   real4 periodicBoxVecY, real4 periodicBoxVecZ, real3 recipBoxVecX,
   real3 recipBoxVecY, real3 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
   real array[PME_ORDER * PME_ORDER];
#else
   // We have shared memory to spare, and putting the workspace array there
   // reduces the load on L2 cache.
   __shared__ real sharedArray[PME_ORDER * PME_ORDER * 64];
   real* array = &sharedArray[PME_ORDER * PME_ORDER * threadIdx.x];
#endif
   real4 theta1[PME_ORDER];
   real4 theta2[PME_ORDER];
   real4 theta3[PME_ORDER];
   __shared__ real fracToCart[3][3];
   if (threadIdx.x == 0) {
      fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
      fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
      fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
      fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
      fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
      fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
      fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
      fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
      fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
   }
   __syncthreads();

   for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < NUM_ATOMS;
        m += blockDim.x * gridDim.x) {
      real4 pos = posq[m];
      pos -= periodicBoxVecZ * floor(pos.z * recipBoxVecZ.z + 0.5f);
      pos -= periodicBoxVecY * floor(pos.y * recipBoxVecY.z + 0.5f);
      pos -= periodicBoxVecX * floor(pos.x * recipBoxVecX.z + 0.5f);

      // Since we need the full set of thetas, it's faster to compute them here
      // than load them from global memory.

      real w = pos.x * recipBoxVecX.x + pos.y * recipBoxVecY.x
         + pos.z * recipBoxVecZ.x;
      real fr = GRID_SIZE_X * (w - (int)(w + 0.5f) + 0.5f);
      int ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid1 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta1, w, array);
      w = pos.y * recipBoxVecY.y + pos.z * recipBoxVecZ.y;
      fr = GRID_SIZE_Y * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid2 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta2, w, array);
      w = pos.z * recipBoxVecZ.z;
      fr = GRID_SIZE_Z * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid3 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta3, w, array);
      igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
      igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
      igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

      // Compute the potential from this grid point.

      real tuv000 = 0;
      real tuv001 = 0;
      real tuv010 = 0;
      real tuv100 = 0;
      real tuv200 = 0;
      real tuv020 = 0;
      real tuv002 = 0;
      real tuv110 = 0;
      real tuv101 = 0;
      real tuv011 = 0;
      real tuv300 = 0;
      real tuv030 = 0;
      real tuv003 = 0;
      real tuv210 = 0;
      real tuv201 = 0;
      real tuv120 = 0;
      real tuv021 = 0;
      real tuv102 = 0;
      real tuv012 = 0;
      real tuv111 = 0;
      for (int ix = 0; ix < PME_ORDER; ix++) {
         int i = igrid1 + ix - (igrid1 + ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
         real4 v = theta1[ix];
         real tu00 = 0;
         real tu10 = 0;
         real tu01 = 0;
         real tu20 = 0;
         real tu11 = 0;
         real tu02 = 0;
         real tu30 = 0;
         real tu21 = 0;
         real tu12 = 0;
         real tu03 = 0;
         for (int iy = 0; iy < PME_ORDER; iy++) {
            int j
               = igrid2 + iy - (igrid2 + iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
            real4 u = theta2[iy];
            real4 t = make_real4(0, 0, 0, 0);
            for (int iz = 0; iz < PME_ORDER; iz++) {
               int k = igrid3 + iz
                  - (igrid3 + iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
               int gridIndex
                  = i * GRID_SIZE_Y * GRID_SIZE_Z + j * GRID_SIZE_Z + k;
               real tq = pmeGrid[gridIndex].x;
               real4 tadd = theta3[iz];
               t.x += tq * tadd.x;
               t.y += tq * tadd.y;
               t.z += tq * tadd.z;
               t.w += tq * tadd.w;
            }
            tu00 += u.x * t.x;
            tu10 += u.y * t.x;
            tu01 += u.x * t.y;
            tu20 += u.z * t.x;
            tu11 += u.y * t.y;
            tu02 += u.x * t.z;
            tu30 += u.w * t.x;
            tu21 += u.z * t.y;
            tu12 += u.y * t.z;
            tu03 += u.x * t.w;
         }
         tuv000 += v.x * tu00;
         tuv100 += v.y * tu00;
         tuv010 += v.x * tu10;
         tuv001 += v.x * tu01;
         tuv200 += v.z * tu00;
         tuv020 += v.x * tu20;
         tuv002 += v.x * tu02;
         tuv110 += v.y * tu10;
         tuv101 += v.y * tu01;
         tuv011 += v.x * tu11;
         tuv300 += v.w * tu00;
         tuv030 += v.x * tu30;
         tuv003 += v.x * tu03;
         tuv210 += v.z * tu10;
         tuv201 += v.z * tu01;
         tuv120 += v.y * tu20;
         tuv021 += v.x * tu21;
         tuv102 += v.y * tu02;
         tuv012 += v.x * tu12;
         tuv111 += v.y * tu11;
      }
      phi[m] = tuv000;
      phi[m + NUM_ATOMS] = tuv100;
      phi[m + NUM_ATOMS * 2] = tuv010;
      phi[m + NUM_ATOMS * 3] = tuv001;
      phi[m + NUM_ATOMS * 4] = tuv200;
      phi[m + NUM_ATOMS * 5] = tuv020;
      phi[m + NUM_ATOMS * 6] = tuv002;
      phi[m + NUM_ATOMS * 7] = tuv110;
      phi[m + NUM_ATOMS * 8] = tuv101;
      phi[m + NUM_ATOMS * 9] = tuv011;
      phi[m + NUM_ATOMS * 10] = tuv300;
      phi[m + NUM_ATOMS * 11] = tuv030;
      phi[m + NUM_ATOMS * 12] = tuv003;
      phi[m + NUM_ATOMS * 13] = tuv210;
      phi[m + NUM_ATOMS * 14] = tuv201;
      phi[m + NUM_ATOMS * 15] = tuv120;
      phi[m + NUM_ATOMS * 16] = tuv021;
      phi[m + NUM_ATOMS * 17] = tuv102;
      phi[m + NUM_ATOMS * 18] = tuv012;
      phi[m + NUM_ATOMS * 19] = tuv111;
      real dipoleScale
         = (4 / (real)3) * (EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA) / SQRT_PI;
      long long fieldx = (long long)((dipoleScale * labFrameDipole[m * 3]
                                        - tuv100 * fracToCart[0][0]
                                        - tuv010 * fracToCart[0][1]
                                        - tuv001 * fracToCart[0][2])
         * 0x100000000);
      fieldBuffers[m] += fieldx;
      fieldPolarBuffers[m] = fieldx;
      long long fieldy = (long long)((dipoleScale * labFrameDipole[m * 3 + 1]
                                        - tuv100 * fracToCart[1][0]
                                        - tuv010 * fracToCart[1][1]
                                        - tuv001 * fracToCart[1][2])
         * 0x100000000);
      fieldBuffers[m + PADDED_NUM_ATOMS] += fieldy;
      fieldPolarBuffers[m + PADDED_NUM_ATOMS] = fieldy;
      long long fieldz = (long long)((dipoleScale * labFrameDipole[m * 3 + 2]
                                        - tuv100 * fracToCart[2][0]
                                        - tuv010 * fracToCart[2][1]
                                        - tuv001 * fracToCart[2][2])
         * 0x100000000);
      fieldBuffers[m + 2 * PADDED_NUM_ATOMS] += fieldz;
      fieldPolarBuffers[m + 2 * PADDED_NUM_ATOMS] = fieldz;
   }
}

extern "C" __global__ void fphi_uind(const real2* __restrict__ pmeGrid,
   real* __restrict__ phid, real* __restrict__ phip, real* __restrict__ phidp,
   const real4* __restrict__ posq, real4 periodicBoxVecX, real4 periodicBoxVecY,
   real4 periodicBoxVecZ, real3 recipBoxVecX, real3 recipBoxVecY,
   real3 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
   real array[PME_ORDER * PME_ORDER];
#else
   // We have shared memory to spare, and putting the workspace array there
   // reduces the load on L2 cache.
   __shared__ real sharedArray[PME_ORDER * PME_ORDER * 64];
   real* array = &sharedArray[PME_ORDER * PME_ORDER * threadIdx.x];
#endif
   real4 theta1[PME_ORDER];
   real4 theta2[PME_ORDER];
   real4 theta3[PME_ORDER];

   for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < NUM_ATOMS;
        m += blockDim.x * gridDim.x) {
      real4 pos = posq[m];
      pos -= periodicBoxVecZ * floor(pos.z * recipBoxVecZ.z + 0.5f);
      pos -= periodicBoxVecY * floor(pos.y * recipBoxVecY.z + 0.5f);
      pos -= periodicBoxVecX * floor(pos.x * recipBoxVecX.z + 0.5f);

      // Since we need the full set of thetas, it's faster to compute them here
      // than load them from global memory.

      real w = pos.x * recipBoxVecX.x + pos.y * recipBoxVecY.x
         + pos.z * recipBoxVecZ.x;
      real fr = GRID_SIZE_X * (w - (int)(w + 0.5f) + 0.5f);
      int ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid1 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta1, w, array);
      w = pos.y * recipBoxVecY.y + pos.z * recipBoxVecZ.y;
      fr = GRID_SIZE_Y * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid2 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta2, w, array);
      w = pos.z * recipBoxVecZ.z;
      fr = GRID_SIZE_Z * (w - (int)(w + 0.5f) + 0.5f);
      ifr = (int)floor(fr);
      w = fr - ifr;
      int igrid3 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta3, w, array);
      igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
      igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
      igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

      // Compute the potential from this grid point.

      real tuv100_1 = 0;
      real tuv010_1 = 0;
      real tuv001_1 = 0;
      real tuv200_1 = 0;
      real tuv020_1 = 0;
      real tuv002_1 = 0;
      real tuv110_1 = 0;
      real tuv101_1 = 0;
      real tuv011_1 = 0;
      real tuv100_2 = 0;
      real tuv010_2 = 0;
      real tuv001_2 = 0;
      real tuv200_2 = 0;
      real tuv020_2 = 0;
      real tuv002_2 = 0;
      real tuv110_2 = 0;
      real tuv101_2 = 0;
      real tuv011_2 = 0;
      real tuv000 = 0;
      real tuv001 = 0;
      real tuv010 = 0;
      real tuv100 = 0;
      real tuv200 = 0;
      real tuv020 = 0;
      real tuv002 = 0;
      real tuv110 = 0;
      real tuv101 = 0;
      real tuv011 = 0;
      real tuv300 = 0;
      real tuv030 = 0;
      real tuv003 = 0;
      real tuv210 = 0;
      real tuv201 = 0;
      real tuv120 = 0;
      real tuv021 = 0;
      real tuv102 = 0;
      real tuv012 = 0;
      real tuv111 = 0;
      for (int ix = 0; ix < PME_ORDER; ix++) {
         int i = igrid1 + ix - (igrid1 + ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
         real4 v = theta1[ix];
         real tu00_1 = 0;
         real tu01_1 = 0;
         real tu10_1 = 0;
         real tu20_1 = 0;
         real tu11_1 = 0;
         real tu02_1 = 0;
         real tu00_2 = 0;
         real tu01_2 = 0;
         real tu10_2 = 0;
         real tu20_2 = 0;
         real tu11_2 = 0;
         real tu02_2 = 0;
         real tu00 = 0;
         real tu10 = 0;
         real tu01 = 0;
         real tu20 = 0;
         real tu11 = 0;
         real tu02 = 0;
         real tu30 = 0;
         real tu21 = 0;
         real tu12 = 0;
         real tu03 = 0;
         for (int iy = 0; iy < PME_ORDER; iy++) {
            int j
               = igrid2 + iy - (igrid2 + iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
            real4 u = theta2[iy];
            real t0_1 = 0;
            real t1_1 = 0;
            real t2_1 = 0;
            real t0_2 = 0;
            real t1_2 = 0;
            real t2_2 = 0;
            real t3 = 0;
            for (int iz = 0; iz < PME_ORDER; iz++) {
               int k = igrid3 + iz
                  - (igrid3 + iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
               int gridIndex
                  = i * GRID_SIZE_Y * GRID_SIZE_Z + j * GRID_SIZE_Z + k;
               real2 tq = pmeGrid[gridIndex];
               real4 tadd = theta3[iz];
               t0_1 += tq.x * tadd.x;
               t1_1 += tq.x * tadd.y;
               t2_1 += tq.x * tadd.z;
               t0_2 += tq.y * tadd.x;
               t1_2 += tq.y * tadd.y;
               t2_2 += tq.y * tadd.z;
               t3 += (tq.x + tq.y) * tadd.w;
            }
            tu00_1 += u.x * t0_1;
            tu10_1 += u.y * t0_1;
            tu01_1 += u.x * t1_1;
            tu20_1 += u.z * t0_1;
            tu11_1 += u.y * t1_1;
            tu02_1 += u.x * t2_1;
            tu00_2 += u.x * t0_2;
            tu10_2 += u.y * t0_2;
            tu01_2 += u.x * t1_2;
            tu20_2 += u.z * t0_2;
            tu11_2 += u.y * t1_2;
            tu02_2 += u.x * t2_2;
            real t0 = t0_1 + t0_2;
            real t1 = t1_1 + t1_2;
            real t2 = t2_1 + t2_2;
            tu00 += u.x * t0;
            tu10 += u.y * t0;
            tu01 += u.x * t1;
            tu20 += u.z * t0;
            tu11 += u.y * t1;
            tu02 += u.x * t2;
            tu30 += u.w * t0;
            tu21 += u.z * t1;
            tu12 += u.y * t2;
            tu03 += u.x * t3;
         }
         tuv100_1 += v.y * tu00_1;
         tuv010_1 += v.x * tu10_1;
         tuv001_1 += v.x * tu01_1;
         tuv200_1 += v.z * tu00_1;
         tuv020_1 += v.x * tu20_1;
         tuv002_1 += v.x * tu02_1;
         tuv110_1 += v.y * tu10_1;
         tuv101_1 += v.y * tu01_1;
         tuv011_1 += v.x * tu11_1;
         tuv100_2 += v.y * tu00_2;
         tuv010_2 += v.x * tu10_2;
         tuv001_2 += v.x * tu01_2;
         tuv200_2 += v.z * tu00_2;
         tuv020_2 += v.x * tu20_2;
         tuv002_2 += v.x * tu02_2;
         tuv110_2 += v.y * tu10_2;
         tuv101_2 += v.y * tu01_2;
         tuv011_2 += v.x * tu11_2;
         tuv000 += v.x * tu00;
         tuv100 += v.y * tu00;
         tuv010 += v.x * tu10;
         tuv001 += v.x * tu01;
         tuv200 += v.z * tu00;
         tuv020 += v.x * tu20;
         tuv002 += v.x * tu02;
         tuv110 += v.y * tu10;
         tuv101 += v.y * tu01;
         tuv011 += v.x * tu11;
         tuv300 += v.w * tu00;
         tuv030 += v.x * tu30;
         tuv003 += v.x * tu03;
         tuv210 += v.z * tu10;
         tuv201 += v.z * tu01;
         tuv120 += v.y * tu20;
         tuv021 += v.x * tu21;
         tuv102 += v.y * tu02;
         tuv012 += v.x * tu12;
         tuv111 += v.y * tu11;
      }
      phid[m] = 0;
      phid[m + NUM_ATOMS] = tuv100_1;
      phid[m + NUM_ATOMS * 2] = tuv010_1;
      phid[m + NUM_ATOMS * 3] = tuv001_1;
      phid[m + NUM_ATOMS * 4] = tuv200_1;
      phid[m + NUM_ATOMS * 5] = tuv020_1;
      phid[m + NUM_ATOMS * 6] = tuv002_1;
      phid[m + NUM_ATOMS * 7] = tuv110_1;
      phid[m + NUM_ATOMS * 8] = tuv101_1;
      phid[m + NUM_ATOMS * 9] = tuv011_1;

      phip[m] = 0;
      phip[m + NUM_ATOMS] = tuv100_2;
      phip[m + NUM_ATOMS * 2] = tuv010_2;
      phip[m + NUM_ATOMS * 3] = tuv001_2;
      phip[m + NUM_ATOMS * 4] = tuv200_2;
      phip[m + NUM_ATOMS * 5] = tuv020_2;
      phip[m + NUM_ATOMS * 6] = tuv002_2;
      phip[m + NUM_ATOMS * 7] = tuv110_2;
      phip[m + NUM_ATOMS * 8] = tuv101_2;
      phip[m + NUM_ATOMS * 9] = tuv011_2;

      phidp[m] = tuv000;
      phidp[m + NUM_ATOMS * 1] = tuv100;
      phidp[m + NUM_ATOMS * 2] = tuv010;
      phidp[m + NUM_ATOMS * 3] = tuv001;
      phidp[m + NUM_ATOMS * 4] = tuv200;
      phidp[m + NUM_ATOMS * 5] = tuv020;
      phidp[m + NUM_ATOMS * 6] = tuv002;
      phidp[m + NUM_ATOMS * 7] = tuv110;
      phidp[m + NUM_ATOMS * 8] = tuv101;
      phidp[m + NUM_ATOMS * 9] = tuv011;
      phidp[m + NUM_ATOMS * 10] = tuv300;
      phidp[m + NUM_ATOMS * 11] = tuv030;
      phidp[m + NUM_ATOMS * 12] = tuv003;
      phidp[m + NUM_ATOMS * 13] = tuv210;
      phidp[m + NUM_ATOMS * 14] = tuv201;
      phidp[m + NUM_ATOMS * 15] = tuv120;
      phidp[m + NUM_ATOMS * 16] = tuv021;
      phidp[m + NUM_ATOMS * 17] = tuv102;
      phidp[m + NUM_ATOMS * 18] = tuv012;
      phidp[m + NUM_ATOMS * 19] = tuv111;
   }
}

extern "C" __global__ void fphi_to_cphi(const real* __restrict__ fphi,
   real* __restrict__ cphi, real3 recipBoxVecX, real3 recipBoxVecY,
   real3 recipBoxVecZ) {
   // build matrices for transforming the potential
   __shared__ real a[3][3];
   if (threadIdx.x == 0) {
      a[0][0] = GRID_SIZE_X * recipBoxVecX.x;
      a[1][0] = GRID_SIZE_X * recipBoxVecY.x;
      a[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
      a[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
      a[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
      a[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
      a[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
      a[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
      a[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
   }
   __syncthreads();
   int index1[] = {0, 1, 2, 0, 0, 1};
   int index2[] = {0, 1, 2, 1, 2, 2};
   __shared__ real b[6][6];
   if (threadIdx.x < 36) {
      int i = threadIdx.x / 6;
      int j = threadIdx.x - 6 * i;
      b[i][j] = a[index1[i]][index1[j]] * a[index2[i]][index2[j]];
      if (index1[j] != index2[j])
         b[i][j] += (i < 3 ? b[i][j]
                           : a[index1[i]][index2[j]] * a[index2[i]][index1[j]]);
   }
   __syncthreads();

   // transform the potential
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
        i += blockDim.x * gridDim.x) {
      cphi[10 * i] = fphi[i];
      cphi[10 * i + 1] = a[0][0] * fphi[i + NUM_ATOMS * 1]
         + a[0][1] * fphi[i + NUM_ATOMS * 2]
         + a[0][2] * fphi[i + NUM_ATOMS * 3];
      cphi[10 * i + 2] = a[1][0] * fphi[i + NUM_ATOMS * 1]
         + a[1][1] * fphi[i + NUM_ATOMS * 2]
         + a[1][2] * fphi[i + NUM_ATOMS * 3];
      cphi[10 * i + 3] = a[2][0] * fphi[i + NUM_ATOMS * 1]
         + a[2][1] * fphi[i + NUM_ATOMS * 2]
         + a[2][2] * fphi[i + NUM_ATOMS * 3];
      for (int j = 0; j < 6; j++) {
         cphi[10 * i + 4 + j] = 0;
         for (int k = 0; k < 6; k++)
            cphi[10 * i + 4 + j] += b[j][k] * fphi[i + NUM_ATOMS * (4 + k)];
      }
   }
}

extern "C" __global__ void ufield_recip_self(const real* __restrict__ phid,
   real* const __restrict__ phip, long long* __restrict__ inducedField,
   long long* __restrict__ inducedFieldPolar,
   const real* __restrict__ inducedDipole,
   const real* __restrict__ inducedDipolePolar, real3 recipBoxVecX,
   real3 recipBoxVecY, real3 recipBoxVecZ) {
   __shared__ real fracToCart[3][3];
   if (threadIdx.x == 0) {
      fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
      fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
      fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
      fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
      fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
      fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
      fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
      fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
      fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
   }
   __syncthreads();
   real selfDipoleScale
      = (4 / (real)3) * (EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA) / SQRT_PI;
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
        i += blockDim.x * gridDim.x) {
      inducedField[i] -= (long long)(0x100000000
         * (phid[i + NUM_ATOMS] * fracToCart[0][0]
              + phid[i + NUM_ATOMS * 2] * fracToCart[0][1]
              + phid[i + NUM_ATOMS * 3] * fracToCart[0][2]
              - selfDipoleScale * inducedDipole[3 * i]));
      inducedField[i + PADDED_NUM_ATOMS] -= (long long)(0x100000000
         * (phid[i + NUM_ATOMS] * fracToCart[1][0]
              + phid[i + NUM_ATOMS * 2] * fracToCart[1][1]
              + phid[i + NUM_ATOMS * 3] * fracToCart[1][2]
              - selfDipoleScale * inducedDipole[3 * i + 1]));
      inducedField[i + PADDED_NUM_ATOMS * 2] -= (long long)(0x100000000
         * (phid[i + NUM_ATOMS] * fracToCart[2][0]
              + phid[i + NUM_ATOMS * 2] * fracToCart[2][1]
              + phid[i + NUM_ATOMS * 3] * fracToCart[2][2]
              - selfDipoleScale * inducedDipole[3 * i + 2]));
      inducedFieldPolar[i] -= (long long)(0x100000000
         * (phip[i + NUM_ATOMS] * fracToCart[0][0]
              + phip[i + NUM_ATOMS * 2] * fracToCart[0][1]
              + phip[i + NUM_ATOMS * 3] * fracToCart[0][2]
              - selfDipoleScale * inducedDipolePolar[3 * i]));
      inducedFieldPolar[i + PADDED_NUM_ATOMS] -= (long long)(0x100000000
         * (phip[i + NUM_ATOMS] * fracToCart[1][0]
              + phip[i + NUM_ATOMS * 2] * fracToCart[1][1]
              + phip[i + NUM_ATOMS * 3] * fracToCart[1][2]
              - selfDipoleScale * inducedDipolePolar[3 * i + 1]));
      inducedFieldPolar[i + PADDED_NUM_ATOMS * 2] -= (long long)(0x100000000
         * (phip[i + NUM_ATOMS] * fracToCart[2][0]
              + phip[i + NUM_ATOMS * 2] * fracToCart[2][1]
              + phip[i + NUM_ATOMS * 3] * fracToCart[2][2]
              - selfDipoleScale * inducedDipolePolar[3 * i + 2]));
   }
}

extern "C" __global__ void recip_mpole_energy_force_torque(
   real4* __restrict__ posq, unsigned long long* __restrict__ forceBuffers,
   long long* __restrict__ torqueBuffers, mixed* __restrict__ energyBuffer,
   const real* __restrict__ labFrameDipole,
   const real* __restrict__ labFrameQuadrupole,
   const real* __restrict__ fracDipole, const real* __restrict__ fracQuadrupole,
   const real* __restrict__ phi, const real* __restrict__ cphi_global,
   real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ) {
   real multipole[10];
   const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
   const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
   const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
   mixed energy = 0;
   __shared__ real fracToCart[3][3];
   if (threadIdx.x == 0) {
      fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
      fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
      fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
      fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
      fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
      fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
      fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
      fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
      fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
   }
   __syncthreads();
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
        i += blockDim.x * gridDim.x) {
      // Compute the torque.

      multipole[0] = posq[i].w;
      multipole[1] = labFrameDipole[i * 3];
      multipole[2] = labFrameDipole[i * 3 + 1];
      multipole[3] = labFrameDipole[i * 3 + 2];
      multipole[4] = labFrameQuadrupole[i * 5];
      multipole[5] = labFrameQuadrupole[i * 5 + 3];
      multipole[6] = -(multipole[4] + multipole[5]);
      multipole[7] = 2 * labFrameQuadrupole[i * 5 + 1];
      multipole[8] = 2 * labFrameQuadrupole[i * 5 + 2];
      multipole[9] = 2 * labFrameQuadrupole[i * 5 + 4];

      // compute self-energy first

      real term = 2 * EWALD_ALPHA * EWALD_ALPHA;
      real fterm = -EWALD_ALPHA / SQRT_PI;
      real cii = multipole[0]*multipole[0];
      real dii = multipole[1]*multipole[1] + multipole[2]*multipole[2] + multipole[3]*multipole[3];
      real qii = 0.5*(multipole[7]*multipole[7]+multipole[8]*multipole[8]+multipole[9]*multipole[9]) 
         + multipole[4]*multipole[4] + multipole[5]*multipole[5] + multipole[6]*multipole[6];
      
      real e = fterm * (cii + term*(dii/3.0f + 2*term*qii/5.0f));

      printf("self energy of %d is %12.4f\n",i,e*EPSILON_FACTOR);

      energy += e;

      const real* cphi = &cphi_global[10 * i];

      torqueBuffers[i] += (long long)(EPSILON_FACTOR
         * (multipole[3] * cphi[2] - multipole[2] * cphi[3]
              + 2 * (multipole[6] - multipole[5]) * cphi[9]
              + multipole[8] * cphi[7] + multipole[9] * cphi[5]
              - multipole[7] * cphi[8] - multipole[9] * cphi[6])
         * 0x100000000);

      torqueBuffers[i + PADDED_NUM_ATOMS] += (long long)(EPSILON_FACTOR
         * (multipole[1] * cphi[3] - multipole[3] * cphi[1]
              + 2 * (multipole[4] - multipole[6]) * cphi[8]
              + multipole[7] * cphi[9] + multipole[8] * cphi[6]
              - multipole[8] * cphi[4] - multipole[9] * cphi[7])
         * 0x100000000);

      torqueBuffers[i + PADDED_NUM_ATOMS * 2] += (long long)(EPSILON_FACTOR
         * (multipole[2] * cphi[1] - multipole[1] * cphi[2]
              + 2 * (multipole[5] - multipole[4]) * cphi[7]
              + multipole[7] * cphi[4] + multipole[9] * cphi[8]
              - multipole[7] * cphi[5] - multipole[8] * cphi[9])
         * 0x100000000);



      // Compute the force and energy.

      multipole[1] = fracDipole[i * 3];
      multipole[2] = fracDipole[i * 3 + 1];
      multipole[3] = fracDipole[i * 3 + 2];
      multipole[4] = fracQuadrupole[i * 6];
      multipole[5] = fracQuadrupole[i * 6 + 3];
      multipole[6] = fracQuadrupole[i * 6 + 5];
      multipole[7] = fracQuadrupole[i * 6 + 1];
      multipole[8] = fracQuadrupole[i * 6 + 2];
      multipole[9] = fracQuadrupole[i * 6 + 4];

      real4 f = make_real4(0, 0, 0, 0);
      for (int k = 0; k < 10; k++) {
         energy += 0.5f * multipole[k] * phi[i + NUM_ATOMS * k];
         f.x += multipole[k] * phi[i + NUM_ATOMS * deriv1[k]];
         f.y += multipole[k] * phi[i + NUM_ATOMS * deriv2[k]];
         f.z += multipole[k] * phi[i + NUM_ATOMS * deriv3[k]];
      }
      f = make_real4(EPSILON_FACTOR
            * (f.x * fracToCart[0][0] + f.y * fracToCart[0][1]
                 + f.z * fracToCart[0][2]),
         EPSILON_FACTOR
            * (f.x * fracToCart[1][0] + f.y * fracToCart[1][1]
                 + f.z * fracToCart[1][2]),
         EPSILON_FACTOR
            * (f.x * fracToCart[2][0] + f.y * fracToCart[2][1]
                 + f.z * fracToCart[2][2]),
         0);
      forceBuffers[i]
         -= static_cast<unsigned long long>((long long)(f.x * 0x100000000));
      forceBuffers[i + PADDED_NUM_ATOMS]
         -= static_cast<unsigned long long>((long long)(f.y * 0x100000000));
      forceBuffers[i + PADDED_NUM_ATOMS * 2]
         -= static_cast<unsigned long long>((long long)(f.z * 0x100000000));

      printf ("PME forces on %d : %12.4f%12.4f%12.4f\n",i,f.x,f.y,f.z);

      //printf ("forceBuffers for %d : %12.4f%12.4f%12.4f\n",i,forceBuffers[i],forceBuffers[i + PADDED_NUM_ATOMS],forceBuffers[i+ 2*PADDED_NUM_ATOMS]);

      //atomicAdd(&forceBuffers[i], static_cast<unsigned long long>((long long)(f.x * 0x100000000)));
      //atomicAdd(&forceBuffers[i + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(f.y * 0x100000000)));
      //atomicAdd(&forceBuffers[i + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(f.z * 0x100000000)));
   }
   energyBuffer[blockIdx.x * blockDim.x + threadIdx.x]
      += EPSILON_FACTOR * energy;
}

inline __device__ real normVector(real3& v) {
   real n = SQRT(dot(v, v));
   v *= (n > 0 ? RECIP(n) : 0);
   return n;
}

extern "C" __global__ void torque_to_force(
   unsigned long long* __restrict__ forceBuffers,
   const long long* __restrict__ torqueBuffers, const real4* __restrict__ posq,
   const int4* __restrict__ multipoleParticles) {
   const int U = 0;
   const int V = 1;
   const int W = 2;
   const int R = 3;
   const int S = 4;
   const int UV = 5;
   const int UW = 6;
   const int VW = 7;
   const int UR = 8;
   const int US = 9;
   const int VS = 10;
   const int WS = 11;
   const int LastVectorIndex = 12;

   const int X = 0;
   const int Y = 1;
   const int Z = 2;
   const int I = 3;

   const real torqueScale = RECIP((double)0x100000000);

   real3 forces[4];
   real norms[LastVectorIndex];
   real3 vector[LastVectorIndex];
   real angles[LastVectorIndex][2];

   for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < NUM_ATOMS;
        atom += gridDim.x * blockDim.x) {
      int4 particles = multipoleParticles[atom];
      int axisAtom = particles.z;
      int axisType = particles.w;

      // NoAxisType

      if (axisType < 5 && particles.z >= 0) {
         real3 atomPos = trimTo3(posq[atom]);
         vector[U] = atomPos - trimTo3(posq[axisAtom]);
         norms[U] = normVector(vector[U]);

         // V is 2nd bond, or "random" vector not parallel to U

         if (axisType != 4 && particles.x >= 0) {
            vector[V] = atomPos - trimTo3(posq[particles.x]);
         } else {
            vector[V].x = 1;
            vector[V].y = 0;
            vector[V].z = 0;
            if (abs(vector[U].x / norms[U]) > 0.866) {
               vector[V].x = 0;
               vector[V].y = 1;
            }
         }
         norms[V] = normVector(vector[V]);

         // W = UxV

         if (axisType < 2 || axisType > 3)
            vector[W] = cross(vector[U], vector[V]);
         else
            vector[W] = atomPos - trimTo3(posq[particles.y]);
         norms[W] = normVector(vector[W]);

         vector[UV] = cross(vector[V], vector[U]);
         vector[UW] = cross(vector[W], vector[U]);
         vector[VW] = cross(vector[W], vector[V]);

         norms[UV] = normVector(vector[UV]);
         norms[UW] = normVector(vector[UW]);
         norms[VW] = normVector(vector[VW]);

         angles[UV][0] = dot(vector[U], vector[V]);
         angles[UV][1] = SQRT(1 - angles[UV][0] * angles[UV][0]);

         angles[UW][0] = dot(vector[U], vector[W]);
         angles[UW][1] = SQRT(1 - angles[UW][0] * angles[UW][0]);

         angles[VW][0] = dot(vector[V], vector[W]);
         angles[VW][1] = SQRT(1 - angles[VW][0] * angles[VW][0]);

         real dphi[3];
         real3 torque = make_real3(torqueScale * torqueBuffers[atom],
            torqueScale * torqueBuffers[atom + PADDED_NUM_ATOMS],
            torqueScale * torqueBuffers[atom + PADDED_NUM_ATOMS * 2]);
         dphi[U] = -dot(vector[U], torque);
         dphi[V] = -dot(vector[V], torque);
         dphi[W] = -dot(vector[W], torque);

         // z-then-x and bisector

         if (axisType == 0 || axisType == 1) {
            real factor1 = dphi[V] / (norms[U] * angles[UV][1]);
            real factor2 = dphi[W] / (norms[U]);
            real factor3 = -dphi[U] / (norms[V] * angles[UV][1]);
            real factor4 = 0;
            if (axisType == 1) {
               factor2 *= 0.5f;
               factor4 = 0.5f * dphi[W] / (norms[V]);
            }
            forces[Z] = vector[UV] * factor1 + factor2 * vector[UW];
            forces[X] = vector[UV] * factor3 + factor4 * vector[VW];
            forces[I] = -(forces[X] + forces[Z]);
            forces[Y] = make_real3(0);
         } else if (axisType == 2) {
            // z-bisect

            vector[R] = vector[V] + vector[W];

            vector[S] = cross(vector[U], vector[R]);

            norms[R] = normVector(vector[R]);
            norms[S] = normVector(vector[S]);

            vector[UR] = cross(vector[R], vector[U]);
            vector[US] = cross(vector[S], vector[U]);
            vector[VS] = cross(vector[S], vector[V]);
            vector[WS] = cross(vector[S], vector[W]);

            norms[UR] = normVector(vector[UR]);
            norms[US] = normVector(vector[US]);
            norms[VS] = normVector(vector[VS]);
            norms[WS] = normVector(vector[WS]);

            angles[UR][0] = dot(vector[U], vector[R]);
            angles[UR][1] = SQRT(1 - angles[UR][0] * angles[UR][0]);

            angles[US][0] = dot(vector[U], vector[S]);
            angles[US][1] = SQRT(1 - angles[US][0] * angles[US][0]);

            angles[VS][0] = dot(vector[V], vector[S]);
            angles[VS][1] = SQRT(1 - angles[VS][0] * angles[VS][0]);

            angles[WS][0] = dot(vector[W], vector[S]);
            angles[WS][1] = SQRT(1 - angles[WS][0] * angles[WS][0]);

            real3 t1 = vector[V] - vector[S] * angles[VS][0];
            real3 t2 = vector[W] - vector[S] * angles[WS][0];
            normVector(t1);
            normVector(t2);
            real ut1cos = dot(vector[U], t1);
            real ut1sin = SQRT(1 - ut1cos * ut1cos);
            real ut2cos = dot(vector[U], t2);
            real ut2sin = SQRT(1 - ut2cos * ut2cos);

            real dphiR = -dot(vector[R], torque);
            real dphiS = -dot(vector[S], torque);

            real factor1 = dphiR / (norms[U] * angles[UR][1]);
            real factor2 = dphiS / (norms[U]);
            real factor3 = dphi[U] / (norms[V] * (ut1sin + ut2sin));
            real factor4 = dphi[U] / (norms[W] * (ut1sin + ut2sin));
            forces[Z] = vector[UR] * factor1 + factor2 * vector[US];
            forces[X]
               = (angles[VS][1] * vector[S] - angles[VS][0] * t1) * factor3;
            forces[Y]
               = (angles[WS][1] * vector[S] - angles[WS][0] * t2) * factor4;
            forces[I] = -(forces[X] + forces[Y] + forces[Z]);
         } else if (axisType == 3) {
            // 3-fold

            forces[Z] = (vector[UW] * dphi[W] / (norms[U] * angles[UW][1])
                           + vector[UV] * dphi[V] / (norms[U] * angles[UV][1])
                           - vector[UW] * dphi[U] / (norms[U] * angles[UW][1])
                           - vector[UV] * dphi[U] / (norms[U] * angles[UV][1]))
               / 3;

            forces[X] = (vector[VW] * dphi[W] / (norms[V] * angles[VW][1])
                           - vector[UV] * dphi[U] / (norms[V] * angles[UV][1])
                           - vector[VW] * dphi[V] / (norms[V] * angles[VW][1])
                           + vector[UV] * dphi[V] / (norms[V] * angles[UV][1]))
               / 3;

            forces[Y] = (-vector[UW] * dphi[U] / (norms[W] * angles[UW][1])
                           - vector[VW] * dphi[V] / (norms[W] * angles[VW][1])
                           + vector[UW] * dphi[W] / (norms[W] * angles[UW][1])
                           + vector[VW] * dphi[W] / (norms[W] * angles[VW][1]))
               / 3;
            forces[I] = -(forces[X] + forces[Y] + forces[Z]);
         } else if (axisType == 4) {
            // z-only

            forces[Z] = vector[UV] * dphi[V] / (norms[U] * angles[UV][1])
               + vector[UW] * dphi[W] / norms[U];
            forces[X] = make_real3(0);
            forces[Y] = make_real3(0);
            forces[I] = -forces[Z];
         } else {
            forces[Z] = make_real3(0);
            forces[X] = make_real3(0);
            forces[Y] = make_real3(0);
            forces[I] = make_real3(0);
         }

         // Store results

         atomicAdd(&forceBuffers[particles.z],
            static_cast<unsigned long long>(
               (long long)(forces[Z].x * 0x100000000)));
         atomicAdd(&forceBuffers[particles.z + PADDED_NUM_ATOMS],
            static_cast<unsigned long long>(
               (long long)(forces[Z].y * 0x100000000)));
         atomicAdd(&forceBuffers[particles.z + 2 * PADDED_NUM_ATOMS],
            static_cast<unsigned long long>(
               (long long)(forces[Z].z * 0x100000000)));
         if (axisType != 4) {
            atomicAdd(&forceBuffers[particles.x],
               static_cast<unsigned long long>(
                  (long long)(forces[X].x * 0x100000000)));
            atomicAdd(&forceBuffers[particles.x + PADDED_NUM_ATOMS],
               static_cast<unsigned long long>(
                  (long long)(forces[X].y * 0x100000000)));
            atomicAdd(&forceBuffers[particles.x + 2 * PADDED_NUM_ATOMS],
               static_cast<unsigned long long>(
                  (long long)(forces[X].z * 0x100000000)));
         }
         if ((axisType == 2 || axisType == 3) && particles.y > -1) {
            atomicAdd(&forceBuffers[particles.y],
               static_cast<unsigned long long>(
                  (long long)(forces[Y].x * 0x100000000)));
            atomicAdd(&forceBuffers[particles.y + PADDED_NUM_ATOMS],
               static_cast<unsigned long long>(
                  (long long)(forces[Y].y * 0x100000000)));
            atomicAdd(&forceBuffers[particles.y + 2 * PADDED_NUM_ATOMS],
               static_cast<unsigned long long>(
                  (long long)(forces[Y].z * 0x100000000)));
         }
         atomicAdd(&forceBuffers[atom],
            static_cast<unsigned long long>(
               (long long)(forces[I].x * 0x100000000)));
         atomicAdd(&forceBuffers[atom + PADDED_NUM_ATOMS],
            static_cast<unsigned long long>(
               (long long)(forces[I].y * 0x100000000)));
         atomicAdd(&forceBuffers[atom + 2 * PADDED_NUM_ATOMS],
            static_cast<unsigned long long>(
               (long long)(forces[I].z * 0x100000000)));
      }
   }
}


////////////////////////////// induced dipole stuff - zw

extern "C" __global__ void fphi_to_cphi_ind(const real* __restrict__ fphi,
   real* __restrict__ cphi,
   real3 recipBoxVecX, real3 recipBoxVecY,
   real3 recipBoxVecZ) {
// Build matrices for transforming the potential.

__shared__ real a[3][3];
if (threadIdx.x == 0) {
a[0][0] = GRID_SIZE_X * recipBoxVecX.x;
a[1][0] = GRID_SIZE_X * recipBoxVecY.x;
a[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
a[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
a[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
a[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
a[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
a[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
a[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
}
__syncthreads();
int index1[] = {0, 1, 2, 0, 0, 1};
int index2[] = {0, 1, 2, 1, 2, 2};
__shared__ real b[6][6];
if (threadIdx.x < 36) {
int i = threadIdx.x / 6;
int j = threadIdx.x - 6 * i;
b[i][j] = a[index1[i]][index1[j]] * a[index2[i]][index2[j]];
if (index1[j] != index2[j])
b[i][j] +=
(i < 3 ? b[i][j] : a[index1[i]][index2[j]] * a[index2[i]][index1[j]]);
}
__syncthreads();

// Transform the potential.

for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
i += blockDim.x * gridDim.x) {
cphi[10 * i] = fphi[i];
cphi[10 * i + 1] = a[0][0] * fphi[i + NUM_ATOMS * 1] +
a[0][1] * fphi[i + NUM_ATOMS * 2] + a[0][2] * fphi[i + NUM_ATOMS * 3];
cphi[10 * i + 2] = a[1][0] * fphi[i + NUM_ATOMS * 1] +
a[1][1] * fphi[i + NUM_ATOMS * 2] + a[1][2] * fphi[i + NUM_ATOMS * 3];
cphi[10 * i + 3] = a[2][0] * fphi[i + NUM_ATOMS * 1] +
a[2][1] * fphi[i + NUM_ATOMS * 2] + a[2][2] * fphi[i + NUM_ATOMS * 3];
for (int j = 0; j < 6; j++) {
cphi[10 * i + 4 + j] = 0;
for (int k = 0; k < 6; k++)
cphi[10 * i + 4 + j] += b[j][k] * fphi[i + NUM_ATOMS * (4 + k)];
}
}
}


extern "C" __global__ void
eprecip(real4* __restrict__ posq, unsigned long long* __restrict__ forceBuffers,
        long long* __restrict__ torqueBuffers, mixed* __restrict__ energyBuffer,
        const real* __restrict__ labFrameDipole,
        const real* __restrict__ labFrameQuadrupole,
        const real* __restrict__ fracDipole,
        const real* __restrict__ fracQuadrupole,
        const real* __restrict__ inducedDipole_global,
        const real* __restrict__ inducedDipolePolar_global,
        const real* __restrict__ phi, const real* __restrict__ phid,
        const real* __restrict__ phip, const real* __restrict__ phidp,
        const real* __restrict__ cphi_global, real3 recipBoxVecX,
        real3 recipBoxVecY, real3 recipBoxVecZ) {
  real multipole[10];
  real cinducedDipole[3], inducedDipole[3];
  real cinducedDipolePolar[3], inducedDipolePolar[3];
  const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
  const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
  const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
  // mixed energy = 0;
  __shared__ real fracToCart[3][3];
  if (threadIdx.x == 0) {
    fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
    fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
    fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
    fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
    fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
    fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
    fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
    fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
    fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
  }
  __syncthreads();

  real term = (4.0f/3.0f) * EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA / SQRT_PI;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
       i += blockDim.x * gridDim.x) {
    // Compute the torque.
    multipole[0] = posq[i].w;
    multipole[1] = labFrameDipole[i * 3];
    multipole[2] = labFrameDipole[i * 3 + 1];
    multipole[3] = labFrameDipole[i * 3 + 2];
    multipole[4] = labFrameQuadrupole[i * 5];
    multipole[5] = labFrameQuadrupole[i * 5 + 3];
    multipole[6] = -(multipole[4] + multipole[5]);
    multipole[7] = 2 * labFrameQuadrupole[i * 5 + 1];
    multipole[8] = 2 * labFrameQuadrupole[i * 5 + 2];
    multipole[9] = 2 * labFrameQuadrupole[i * 5 + 4];
    const real* cphi = &cphi_global[10 * i];

    // self-torque (don't need self-energy because i'm doing dot product)
    //term = (4.0f/3.0f) * EPSILON_FACTOR * EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA / SQRT_PI;
    real dix = multipole[1];
    real diy = multipole[2];
    real diz = multipole[3];

    real uix = inducedDipole_global[i * 3];
    real uiy = inducedDipole_global[i * 3 + 1];
    real uiz = inducedDipole_global[i * 3 + 2];

    real uip[3];
    uip[0] = inducedDipolePolar_global[i*3];
    uip[1] = inducedDipolePolar_global[i*3+1];
    uip[2] = inducedDipolePolar_global[i*3+2];
    printf(" OMM atom %3d, udir,udirp,%18.8lf%18.8lf%18.8lf%18.8lf%18.8lf%18.8lf\n", i+1,uix,uiy,uiz,uip[0],uip[1],uip[2]);


    int ii1 = NUM_ATOMS;
    int ii2 = NUM_ATOMS*2; int ii3 = NUM_ATOMS*3;
    printf(" omm atom %3d, phid %18.8lf%18.8lf%18.8lf%18.8lf\n", i+1, phid[i], phid[ii1+i], phid[ii2+i], phid[ii3+i]);
    printf(" omm atom %3d, phip %18.8lf%18.8lf%18.8lf%18.8lf\n", i+1, phip[i], phip[ii1+i], phip[ii2+i], phip[ii3+i]);
    printf(" omm atom %3d, phdp %18.8lf%18.8lf%18.8lf%18.8lf\n", i+1, phidp[i], phidp[ii1+i], phidp[ii2+i], phidp[ii3+i]);
    // self-torque
/*
    real tk1 = diy*uiz-diz*uiy;
    real tk2 = diz*uix-dix*uiz;
    real tk3 = dix*uiy-diy*uix;
    tk1 *= (term * EPSILON_FACTOR);
    tk2 *= (term * EPSILON_FACTOR);
    tk3 *= (term * EPSILON_FACTOR);
    printf(" OMM atom %3d, tk1,tk2,tk3,%18.8lf%18.8lf%18.8lf\n",i+1,tk1,tk2,tk3);
    torqueBuffers[i] +=
        (long long)(EPSILON_FACTOR*term*(diy*uiz-diz*uiy)*0x100000000);
    torqueBuffers[i + PADDED_NUM_ATOMS] +=
        (long long)(EPSILON_FACTOR*term*(diz*uix-dix*uiz)*0x100000000);
    torqueBuffers[i + PADDED_NUM_ATOMS * 2] +=
        (long long)(EPSILON_FACTOR*term*(dix*uiy-diy*uix)*0x100000000);
// */
//*
    torqueBuffers[i] +=
        (long long)( EPSILON_FACTOR *
                    (0.5f*(multipole[3] * cphi[2] - multipole[2] * cphi[3] +
                     2 * (multipole[6] - multipole[5]) * cphi[9] +
                     multipole[8] * cphi[7] + multipole[9] * cphi[5] -
                     multipole[7] * cphi[8] - multipole[9] * cphi[6]) +
                     term*(diy*uiz-diz*uiy)) *
                    0x100000000);

    torqueBuffers[i + PADDED_NUM_ATOMS] +=
        (long long)( EPSILON_FACTOR *
                    (0.5f*(multipole[1] * cphi[3] - multipole[3] * cphi[1] +
                     2 * (multipole[4] - multipole[6]) * cphi[8] +
                     multipole[7] * cphi[9] + multipole[8] * cphi[6] -
                     multipole[8] * cphi[4] - multipole[9] * cphi[7]) +
                     term * (diz*uix-dix*uiz)) *
                    0x100000000);

    torqueBuffers[i + PADDED_NUM_ATOMS * 2] +=
        (long long)( EPSILON_FACTOR *
                    (0.5f*(multipole[2] * cphi[1] - multipole[1] * cphi[2] +
                     2 * (multipole[5] - multipole[4]) * cphi[7] +
                     multipole[7] * cphi[4] + multipole[9] * cphi[8] -
                     multipole[7] * cphi[5] - multipole[8] * cphi[9]) +
                     term * (dix*uiy-diy*uix)) *
                    0x100000000);
// */
/*
    torqueBuffers[i] +=
        (long long)( EPSILON_FACTOR *
                    (multipole[3] * cphi[2] - multipole[2] * cphi[3] +
                     2 * (multipole[6] - multipole[5]) * cphi[9] +
                     multipole[8] * cphi[7] + multipole[9] * cphi[5] -
                     multipole[7] * cphi[8] - multipole[9] * cphi[6]) *
                    0x100000000);

    torqueBuffers[i + PADDED_NUM_ATOMS] +=
        (long long)( EPSILON_FACTOR *
                    (multipole[1] * cphi[3] - multipole[3] * cphi[1] +
                     2 * (multipole[4] - multipole[6]) * cphi[8] +
                     multipole[7] * cphi[9] + multipole[8] * cphi[6] -
                     multipole[8] * cphi[4] - multipole[9] * cphi[7]) *
                    0x100000000);

    torqueBuffers[i + PADDED_NUM_ATOMS * 2] +=
        (long long)( EPSILON_FACTOR *
                    (multipole[2] * cphi[1] - multipole[1] * cphi[2] +
                     2 * (multipole[5] - multipole[4]) * cphi[7] +
                     multipole[7] * cphi[4] + multipole[9] * cphi[8] -
                     multipole[7] * cphi[5] - multipole[8] * cphi[9]) *
                    0x100000000);
// */
    // Compute the force and energy.

    multipole[1] = fracDipole[i * 3];
    multipole[2] = fracDipole[i * 3 + 1];
    multipole[3] = fracDipole[i * 3 + 2];
    multipole[4] = fracQuadrupole[i * 6];
    multipole[5] = fracQuadrupole[i * 6 + 3];
    multipole[6] = fracQuadrupole[i * 6 + 5];
    multipole[7] = fracQuadrupole[i * 6 + 1];
    multipole[8] = fracQuadrupole[i * 6 + 2];
    multipole[9] = fracQuadrupole[i * 6 + 4];

    cinducedDipole[0] = inducedDipole_global[i * 3];
    cinducedDipole[1] = inducedDipole_global[i * 3 + 1];
    cinducedDipole[2] = inducedDipole_global[i * 3 + 2];
    cinducedDipolePolar[0] = inducedDipolePolar_global[i * 3];
    cinducedDipolePolar[1] = inducedDipolePolar_global[i * 3 + 1];
    cinducedDipolePolar[2] = inducedDipolePolar_global[i * 3 + 2];

    // Multiply the dipoles by cartToFrac, which is just the transpose of
    // fracToCart.

    inducedDipole[0] = cinducedDipole[0] * fracToCart[0][0] +
        cinducedDipole[1] * fracToCart[1][0] +
        cinducedDipole[2] * fracToCart[2][0];
    inducedDipole[1] = cinducedDipole[0] * fracToCart[0][1] +
        cinducedDipole[1] * fracToCart[1][1] +
        cinducedDipole[2] * fracToCart[2][1];
    inducedDipole[2] = cinducedDipole[0] * fracToCart[0][2] +
        cinducedDipole[1] * fracToCart[1][2] +
        cinducedDipole[2] * fracToCart[2][2];
    inducedDipolePolar[0] = cinducedDipolePolar[0] * fracToCart[0][0] +
        cinducedDipolePolar[1] * fracToCart[1][0] +
        cinducedDipolePolar[2] * fracToCart[2][0];
    inducedDipolePolar[1] = cinducedDipolePolar[0] * fracToCart[0][1] +
        cinducedDipolePolar[1] * fracToCart[1][1] +
        cinducedDipolePolar[2] * fracToCart[2][1];
    inducedDipolePolar[2] = cinducedDipolePolar[0] * fracToCart[0][2] +
        cinducedDipolePolar[1] * fracToCart[1][2] +
        cinducedDipolePolar[2] * fracToCart[2][2];
    real4 f = make_real4(0, 0, 0, 0);
/*
    energy += (inducedDipole[0] + inducedDipolePolar[0]) * phi[i + NUM_ATOMS];
    energy +=
        (inducedDipole[1] + inducedDipolePolar[1]) * phi[i + NUM_ATOMS * 2];
    energy +=
        (inducedDipole[2] + inducedDipolePolar[2]) * phi[i + NUM_ATOMS * 3];
*/
    for (int k = 0; k < 3; k++) {
      int j1 = deriv1[k + 1];
      int j2 = deriv2[k + 1];
      int j3 = deriv3[k + 1];
      f.x +=
          (inducedDipole[k] + inducedDipolePolar[k]) * phi[i + NUM_ATOMS * j1];
      f.y +=
          (inducedDipole[k] + inducedDipolePolar[k]) * phi[i + NUM_ATOMS * j2];
      f.z +=
          (inducedDipole[k] + inducedDipolePolar[k]) * phi[i + NUM_ATOMS * j3];
      f.x += (inducedDipole[k] * phip[i + NUM_ATOMS * j1] +
              inducedDipolePolar[k] * phid[i + NUM_ATOMS * j1]);
      f.y += (inducedDipole[k] * phip[i + NUM_ATOMS * j2] +
              inducedDipolePolar[k] * phid[i + NUM_ATOMS * j2]);
      f.z += (inducedDipole[k] * phip[i + NUM_ATOMS * j3] +
              inducedDipolePolar[k] * phid[i + NUM_ATOMS * j3]);
    }

    for (int k = 0; k < 10; k++) {
      f.x += multipole[k] * phidp[i + NUM_ATOMS * deriv1[k]];
      f.y += multipole[k] * phidp[i + NUM_ATOMS * deriv2[k]];
      f.z += multipole[k] * phidp[i + NUM_ATOMS * deriv3[k]];
    }

    f = make_real4(0.5f * EPSILON_FACTOR *
                       (f.x * fracToCart[0][0] + f.y * fracToCart[0][1] +
                        f.z * fracToCart[0][2]),
                   0.5f * EPSILON_FACTOR *
                       (f.x * fracToCart[1][0] + f.y * fracToCart[1][1] +
                        f.z * fracToCart[1][2]),
                   0.5f * EPSILON_FACTOR *
                       (f.x * fracToCart[2][0] + f.y * fracToCart[2][1] +
                        f.z * fracToCart[2][2]),
                   0);
//*
    forceBuffers[i] -=
        static_cast<unsigned long long>((long long)(f.x * 0x100000000));
    forceBuffers[i + PADDED_NUM_ATOMS] -=
        static_cast<unsigned long long>((long long)(f.y * 0x100000000));
    forceBuffers[i + PADDED_NUM_ATOMS * 2] -=
        static_cast<unsigned long long>((long long)(f.z * 0x100000000));
// */
  }
//  energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] +=
//      0.5f * EPSILON_FACTOR * energy;
// This coefficients was 0.25, changed to 0.5.
}


