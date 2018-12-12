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
      int  k      = i - 1;
      real denom  = RECIP(k);
      ARRAY(i, i) = denom * w * ARRAY(k, k);
      for (int j = 1; j <= i - 2; j++)
         ARRAY(i, i - j) = denom
            * ((w + j) * ARRAY(k, i - j - 1) + (i - j - w) * ARRAY(k, i - j));
      ARRAY(i, 1) = denom * (1 - w) * ARRAY(k, 1);
   }

   // get coefficients for the B-spline first derivative
   int k               = PME_ORDER - 1;
   ARRAY(k, PME_ORDER) = ARRAY(k, PME_ORDER - 1);
   for (int i = PME_ORDER - 1; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);

   // get coefficients for the B-spline second derivative
   k                       = PME_ORDER - 2;
   ARRAY(k, PME_ORDER - 1) = ARRAY(k, PME_ORDER - 2);
   for (int i = PME_ORDER - 2; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1)         = -ARRAY(k, 1);
   ARRAY(k, PME_ORDER) = ARRAY(k, PME_ORDER - 1);
   for (int i = PME_ORDER - 1; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1) = -ARRAY(k, 1);

   // get coefficients for the B-spline third derivative
   k                       = PME_ORDER - 3;
   ARRAY(k, PME_ORDER - 2) = ARRAY(k, PME_ORDER - 3);
   for (int i = PME_ORDER - 3; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1)             = -ARRAY(k, 1);
   ARRAY(k, PME_ORDER - 1) = ARRAY(k, PME_ORDER - 2);
   for (int i = PME_ORDER - 2; i >= 2; i--)
      ARRAY(k, i) = ARRAY(k, i - 1) - ARRAY(k, i);
   ARRAY(k, 1)         = -ARRAY(k, 1);
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
   int        index1[] = {0, 0, 0, 1, 1, 2};
   int        index2[] = {0, 1, 2, 1, 2, 2};
   __shared__ real b[6][6];
   if (threadIdx.x < 36) {
      int i   = threadIdx.x / 6;
      int j   = threadIdx.x - 6 * i;
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
   real*           array = &sharedArray[PME_ORDER * PME_ORDER * threadIdx.x];
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
      real atomCharge       = pos.w;
      real atomDipoleX      = fracDipole[m * 3];
      real atomDipoleY      = fracDipole[m * 3 + 1];
      real atomDipoleZ      = fracDipole[m * 3 + 2];
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
      real fr    = GRID_SIZE_X * (w - (int)(w + 0.5f) + 0.5f);
      int  ifr   = (int)floor(fr);
      w          = fr - ifr;
      int igrid1 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta1, w, array);
      w          = pos.y * recipBoxVecY.y + pos.z * recipBoxVecZ.y;
      fr         = GRID_SIZE_Y * (w - (int)(w + 0.5f) + 0.5f);
      ifr        = (int)floor(fr);
      w          = fr - ifr;
      int igrid2 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta2, w, array);
      w          = pos.z * recipBoxVecZ.z;
      fr         = GRID_SIZE_Z * (w - (int)(w + 0.5f) + 0.5f);
      ifr        = (int)floor(fr);
      w          = fr - ifr;
      int igrid3 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta3, w, array);
      igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
      igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
      igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

      // Spread the charge from this atom onto each grid point.

      for (int ix = 0; ix < PME_ORDER; ix++) {
         int xbase = igrid1 + ix;
         xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
         xbase   = xbase * GRID_SIZE_Y * GRID_SIZE_Z;
         real4 t = theta1[ix];

         for (int iy = 0; iy < PME_ORDER; iy++) {
            int ybase = igrid2 + iy;
            ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
            ybase   = xbase + ybase * GRID_SIZE_Z;
            real4 u = theta2[iy];

            for (int iz = 0; iz < PME_ORDER; iz++) {
               int zindex = igrid3 + iz;
               zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
               int   index = ybase + zindex;
               real4 v     = theta3[iz];

               real term0 = atomCharge * u.x * v.x + atomDipoleY * u.y * v.x
                  + atomDipoleZ * u.x * v.y + atomQuadrupoleYY * u.z * v.x
                  + atomQuadrupoleZZ * u.x * v.z + atomQuadrupoleYZ * u.y * v.y;
               real term1 = atomDipoleX * u.x * v.x
                  + atomQuadrupoleXY * u.y * v.x + atomQuadrupoleXZ * u.x * v.y;
               real term2 = atomQuadrupoleXX * u.x * v.x;
               real add   = term0 * t.x + term1 * t.y + term2 * t.z;
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

extern "C" __global__ void grid_convert_to_double(
   long long* __restrict__ pmeGrid) {
   real*              floatGrid = (real*)pmeGrid;
   const unsigned int gridSize  = 2 * GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
   real               scale     = 1 / (real)0x100000000;
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
   const unsigned int gridSize  = GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
   real               expFactor = M_PI * M_PI / (EWALD_ALPHA * EWALD_ALPHA);
   real               scaleFactor
      = RECIP(M_PI * periodicBoxSize.x * periodicBoxSize.y * periodicBoxSize.z);
   for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < gridSize;
        index += blockDim.x * gridDim.x) {
      int kx        = index / (GRID_SIZE_Y * GRID_SIZE_Z);
      int remainder = index - kx * GRID_SIZE_Y * GRID_SIZE_Z;
      int ky        = remainder / GRID_SIZE_Z;
      int kz        = remainder - ky * GRID_SIZE_Z;
      if (kx == 0 && ky == 0 && kz == 0) {
         pmeGrid[index] = make_real2(0, 0);
         continue;
      }
      int  mx  = (kx < (GRID_SIZE_X + 1) / 2) ? kx : (kx - GRID_SIZE_X);
      int  my  = (ky < (GRID_SIZE_Y + 1) / 2) ? ky : (ky - GRID_SIZE_Y);
      int  mz  = (kz < (GRID_SIZE_Z + 1) / 2) ? kz : (kz - GRID_SIZE_Z);
      real mhx = mx * recipBoxVecX.x;
      real mhy = mx * recipBoxVecY.x + my * recipBoxVecY.y;
      real mhz
         = mx * recipBoxVecZ.x + my * recipBoxVecZ.y + mz * recipBoxVecZ.z;
      real  bx       = pmeBsplineModuliX[kx];
      real  by       = pmeBsplineModuliY[ky];
      real  bz       = pmeBsplineModuliZ[kz];
      real2 grid     = pmeGrid[index];
      real  m2       = mhx * mhx + mhy * mhy + mhz * mhz;
      real  denom    = m2 * bx * by * bz;
      real  eterm    = scaleFactor * EXP(-expFactor * m2) / denom;
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
   real*           array = &sharedArray[PME_ORDER * PME_ORDER * threadIdx.x];
#endif
   real4      theta1[PME_ORDER];
   real4      theta2[PME_ORDER];
   real4      theta3[PME_ORDER];
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
      real fr    = GRID_SIZE_X * (w - (int)(w + 0.5f) + 0.5f);
      int  ifr   = (int)floor(fr);
      w          = fr - ifr;
      int igrid1 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta1, w, array);
      w          = pos.y * recipBoxVecY.y + pos.z * recipBoxVecZ.y;
      fr         = GRID_SIZE_Y * (w - (int)(w + 0.5f) + 0.5f);
      ifr        = (int)floor(fr);
      w          = fr - ifr;
      int igrid2 = ifr - PME_ORDER + 1;
      tinker_subroutine_bsplgen(theta2, w, array);
      w          = pos.z * recipBoxVecZ.z;
      fr         = GRID_SIZE_Z * (w - (int)(w + 0.5f) + 0.5f);
      ifr        = (int)floor(fr);
      w          = fr - ifr;
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
         int   i = igrid1 + ix - (igrid1 + ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
         real4 v = theta1[ix];
         real  tu00 = 0;
         real  tu10 = 0;
         real  tu01 = 0;
         real  tu20 = 0;
         real  tu11 = 0;
         real  tu02 = 0;
         real  tu30 = 0;
         real  tu21 = 0;
         real  tu12 = 0;
         real  tu03 = 0;
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
               real  tq   = pmeGrid[gridIndex].x;
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
      phi[m]                  = tuv000;
      phi[m + NUM_ATOMS]      = tuv100;
      phi[m + NUM_ATOMS * 2]  = tuv010;
      phi[m + NUM_ATOMS * 3]  = tuv001;
      phi[m + NUM_ATOMS * 4]  = tuv200;
      phi[m + NUM_ATOMS * 5]  = tuv020;
      phi[m + NUM_ATOMS * 6]  = tuv002;
      phi[m + NUM_ATOMS * 7]  = tuv110;
      phi[m + NUM_ATOMS * 8]  = tuv101;
      phi[m + NUM_ATOMS * 9]  = tuv011;
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
      long long fieldx     = (long long)((dipoleScale * labFrameDipole[m * 3]
                                        - tuv100 * fracToCart[0][0]
                                        - tuv010 * fracToCart[0][1]
                                        - tuv001 * fracToCart[0][2])
         * 0x100000000);
      fieldBuffers[m]      = fieldx;
      fieldPolarBuffers[m] = fieldx;
      long long fieldy = (long long)((dipoleScale * labFrameDipole[m * 3 + 1]
                                        - tuv100 * fracToCart[1][0]
                                        - tuv010 * fracToCart[1][1]
                                        - tuv001 * fracToCart[1][2])
         * 0x100000000);
      fieldBuffers[m + PADDED_NUM_ATOMS]      = fieldy;
      fieldPolarBuffers[m + PADDED_NUM_ATOMS] = fieldy;
      long long fieldz = (long long)((dipoleScale * labFrameDipole[m * 3 + 2]
                                        - tuv100 * fracToCart[2][0]
                                        - tuv010 * fracToCart[2][1]
                                        - tuv001 * fracToCart[2][2])
         * 0x100000000);
      fieldBuffers[m + 2 * PADDED_NUM_ATOMS]      = fieldz;
      fieldPolarBuffers[m + 2 * PADDED_NUM_ATOMS] = fieldz;
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
   int        index1[] = {0, 1, 2, 0, 0, 1};
   int        index2[] = {0, 1, 2, 1, 2, 2};
   __shared__ real b[6][6];
   if (threadIdx.x < 36) {
      int i   = threadIdx.x / 6;
      int j   = threadIdx.x - 6 * i;
      b[i][j] = a[index1[i]][index1[j]] * a[index2[i]][index2[j]];
      if (index1[j] != index2[j])
         b[i][j] += (i < 3 ? b[i][j]
                           : a[index1[i]][index2[j]] * a[index2[i]][index1[j]]);
   }
   __syncthreads();

   // transform the potential
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS;
        i += blockDim.x * gridDim.x) {
      cphi[10 * i]     = fphi[i];
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
