
typedef struct {
    real3 pos, force, torque, field;

    // charge penetration electrostatics
    real q;
    real3 dipole;
    real quadrupole[5]; // xx, xy, xz, yy, yz
    real pcore, pval, palpha;

    // induced dipole
    real3 inducedDipole;

} PolarAtomData;

inline __device__ void loadPolarAtomData(PolarAtomData& data, int atom, const real4* __restrict__ posq
    , const real* __restrict__ dpl, const real* __restrict__ quad
    , const real* __restrict__ pcore, const real* __restrict__ pval, const real* __restrict__ palpha
    , const real* __restrict__ inddpl)
{
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);

    // atomic partial charge

    data.q = atomPosq.w;

    // dipole and quadrupole moments in global frame
    data.dipole.x = dpl[atom*3];
    data.dipole.y = dpl[atom*3+1];
    data.dipole.z = dpl[atom*3+2];

    data.quadrupole[0] = quad[atom*5];     // xx
    data.quadrupole[1] = quad[atom*5+1];   // xy
    data.quadrupole[2] = quad[atom*5+2];   // xz
    data.quadrupole[3] = quad[atom*5+3];   // yy
    data.quadrupole[4] = quad[atom*5+4];   // yz

    // charge penetration electrostatics parameters

    // core and valence charges
    data.pcore = pcore[atom];
    data.pval = pval[atom];

    // charge penetration damping parameter
    data.palpha = palpha[atom]; 

    // induced dipole
    data.inducedDipole.x = inddpl[atom*3];
    data.inducedDipole.y = inddpl[atom*3+1];
    data.inducedDipole.z = inddpl[atom*3+2]; 
}


//__device__ real computeWScaleFactor(uint2 covalent, int index)
//{
//    int mask = 1 << index;
//    bool x = (covalent.x & mask);
//    bool y = (covalent.y & mask);
//    return (x ? (y ? (real)INDUCEDDIPOLE12SCALE : (real)INDUCEDDIPOLE13SCALE) : (y ? (real)INDUCEDDIPOLE14SCALE : (real)1.0));
//}


__device__ void computeOnePolarInteraction(PolarAtomData& atom1, PolarAtomData& atom2, real cscale, real wscale, real doubleCountingFactor, mixed& energyToBeAccumulated,
    real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ)
{
    // Compute the displacement.

    real3 delta;
    delta.x = atom2.pos.x - atom1.pos.x;
    delta.y = atom2.pos.y - atom1.pos.y;
    delta.z = atom2.pos.z - atom1.pos.z;
    APPLY_PERIODIC_TO_DELTA(delta)
    real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

 //   printf(" device %12.4f\n",100*CHGPEN_CUTOFF_SQUARED);

    if (r2 > CHGPEN_CUTOFF_SQUARED)
        return;

    real rInv = RSQRT(r2);
    real r = r2 * rInv;

    real xr = delta.x;
    real yr = delta.y;
    real zr = delta.z;

    real ci = atom1.q;
    real ck = atom2.q;

    real dix = atom1.dipole.x;
    real diy = atom1.dipole.y;
    real diz = atom1.dipole.z;
    
    real dkx = atom2.dipole.x;
    real dky = atom2.dipole.y;
    real dkz = atom2.dipole.z;

    real qixx = atom1.quadrupole[0];
    real qixy = atom1.quadrupole[1];
    real qixz = atom1.quadrupole[2];
    real qiyy = atom1.quadrupole[3];
    real qiyz = atom1.quadrupole[4];
    real qizz = -atom1.quadrupole[0] - atom1.quadrupole[3];

    real qkxx = atom2.quadrupole[0];
    real qkxy = atom2.quadrupole[1];
    real qkxz = atom2.quadrupole[2];
    real qkyy = atom2.quadrupole[3];
    real qkyz = atom2.quadrupole[4];
    real qkzz = -atom2.quadrupole[0] - atom2.quadrupole[3];

    real corei = atom1.pcore;
    real corek = atom2.pcore;

    real vali = atom1.pval;
    real valk = atom2.pval;

    real alphai = atom1.palpha;
    real alphak = atom2.palpha;

    real uix = atom1.inducedDipole.x;
    real uiy = atom1.inducedDipole.y;
    real uiz = atom1.inducedDipole.z;

    real ukx = atom2.inducedDipole.x;
    real uky = atom2.inducedDipole.y;
    real ukz = atom2.inducedDipole.z;

    //printf("CUDA dipoles:  %12.4f %12.4f %12.4f \n",uix,uiy,uiz);

    // intermediates involving moments and separation distance
    real dir = dix*xr + diy*yr + diz*zr;
    real qix = qixx*xr + qixy*yr + qixz*zr;
    real qiy = qixy*xr + qiyy*yr + qiyz*zr;
    real qiz = qixz*xr + qiyz*yr + qizz*zr;
    real qir = qix*xr + qiy*yr + qiz*zr;
    real dkr = dkx*xr + dky*yr + dkz*zr;
    real qkx = qkxx*xr + qkxy*yr + qkxz*zr;
    real qky = qkxy*xr + qkyy*yr + qkyz*zr;
    real qkz = qkxz*xr + qkyz*yr + qkzz*zr;
    real qkr = qkx*xr + qky*yr + qkz*zr;
    real uir = uix*xr + uiy*yr + uiz*zr;
    //real uirp = uixp*xr + uiyp*yr + uizp*zr;
    real ukr = ukx*xr + uky*yr + ukz*zr;
    //real ukrp = ukxp*xr + ukyp*yr + ukzp*zr;

    real rr1 = 1.0 / r;
    real rr3 = rr1 / r2;
    real rr5 = 3.0 * rr3 / r2;
    real rr7 = 5.0 * rr5 / r2;
    real rr9 = 7.0 * rr7 / r2;

    // calculate the real space Ewald error function terms

    real ralpha = EWALD_ALPHA*r;
    real exp2a = EXP(-ralpha*ralpha);

    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.
    const real t = RECIP(1.0f+0.3275911f*ralpha);
    const real erfAlphaR = 1-(0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
 
    real bn[5];
    bn[0] = (1 - erfAlphaR)/r;
    real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
    real alsq2n = 1.0f / (SQRT_PI*EWALD_ALPHA);
    real bfac = 0.0f;
    for (int i = 1; i < 5; ++i){
        bfac = (float) (i+i-1);
        alsq2n = alsq2*alsq2n;
        bn[i] = (bfac*bn[i-1]+alsq2n*exp2a) / r2;
    }

    // DAMPING FUNCTIONS

    real dampi = alphai*r;
    real dampk = alphak*r;
    real expi = EXP(-dampi);
    real expk = EXP(-dampk);

    real dampi2 = dampi*dampi;
    real dampi3 = dampi2*dampi;
    real dampi4 = dampi3*dampi;
    real dampi5 = dampi4*dampi;
    real dampi6 = dampi5*dampi;

    real dampk2 = dampk*dampk;
    real dampk3 = dampk2*dampk;
    real dampk4 = dampk3*dampk;
    real dampk5 = dampk4*dampk;
    real dampk6 = dampk5*dampk;

    real dmpi3 = 1 - (1 + dampi + 0.5f*dampi2)*expi;
    real dmpi5 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f)*expi;
    real dmpi7 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/30.0f)*expi;
    real dmpi9 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + 4*dampi4/105.0f + dampi5/210.0f)*expi;

    real dmpk3 = 1;
    real dmpk5 = 1;
    real dmpk7 = 1;
    real dmpk9 = 1;

    real dmpik5 = 1;
    real dmpik7 = 1;

    if (alphai == alphak){
        dmpk3 = dmpi3;
        dmpk5 = dmpi5;
        dmpk7 = dmpi7;
        dmpk9 = dmpi9;

        dmpik5 = 1  - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/24.0f + dampi5/144.0f)*expi;
        dmpik7 = 1  - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/24.0f + dampi5/120.0f + dampi6/720.0f)*expi;
    } else {
        dmpk3 = 1 - (1 + dampk + 0.5f*dampk2)*expk;
        dmpk5 = 1 - (1 + dampk + 0.5f*dampk2 + dampk3/6.0f)*expk;
        dmpk7 = 1 - (1 + dampk + 0.5f*dampk2 + dampk3/6.0f + dampk4/30.0f)*expk;
        dmpk9 = 1 - (1 + dampk + 0.5f*dampk2 + dampk3/6.0f + 4*dampk4/105.0f + dampk5/210.0f)*expk;

        real alphai2 = alphai*alphai;
        real alphak2 = alphak*alphak;
        real termi = alphak2 / (alphak2 - alphai2);
        real termk = alphai2 / (alphai2 - alphak2);
        real termi2 = termi * termi;
        real termk2 = termk * termk;

        dmpik5 = 1 - termi2*(1 + dampi + 0.5f*dampi2 + dampi3/6.0f)*expi
            - termk2*(1 + dampk + 0.5f*dampk2 + dampk3/6.0f)*expk
            - 2*termi2*termk*(1.0 + dampi + dampi2/3.0f)*expi
            - 2*termk2*termi*(1.0 + dampk + dampk2/3.0f)*expk;
        dmpik7 = 1 - termi2*(1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/30.0f)*expi
            - termk2*(1 + dampk + 0.5f*dampk2 + dampk3/6.0f + dampk4/30.0f)*expk
            - 2*termi2*termk*(1 + dampi + 2*dampi2/5.0f + dampi3/15.0f)*expi
            - 2*termk2*termi*(1 + dampk + 2*dampk2/5.0f + dampk3/15.0f)*expk;
    }

    //printf("damping: %12.4f%12.4f%12.4f%12.4f%12.4f",dmpi3,dmpi5,dmpi7,dmpi9);

    // apply charge penetration damping to scale factors
    real rr3core = bn[1] - (1-cscale)*rr3;
    real rr5core = bn[2] - (1-cscale)*rr5;
    real rr3i = bn[1] - (1-cscale*dmpi3)*rr3;
    real rr5i = bn[2] - (1-cscale*dmpi5)*rr5;
    real rr7i = bn[3] - (1-cscale*dmpi7)*rr7;
    real rr9i = bn[4] - (1-cscale*dmpi9)*rr9;
    real rr3k = bn[1] - (1-cscale*dmpk3)*rr3;
    real rr5k = bn[2] - (1-cscale*dmpk5)*rr5;
    real rr7k = bn[3] - (1-cscale*dmpk7)*rr7;
    real rr9k = bn[4] - (1-cscale*dmpk9)*rr9;

    printf("r: %12.4f, cscale: %12.4f wscale: %12.4f\n",r,cscale,wscale);
     // INSERT WSCALE IN HERE!!!!!!
    real rr5ik = bn[2] - (1-wscale*dmpik5)*rr5;
    real rr7ik = bn[3] - (1-wscale*dmpik7)*rr7;

    // get induced dipole field used for dipole torques
    real tix3 =  2*rr3i*ukx;
    real tiy3 =  2*rr3i*uky;
    real tiz3 =  2*rr3i*ukz;
    real tkx3 =  2*rr3k*uix;
    real tky3 =  2*rr3k*uiy;
    real tkz3 =  2*rr3k*uiz;
    real tuir = -2*rr5i*ukr;
    real tukr = -2*rr5k*uir;

    real ufldix = tix3 + xr*tuir;
    real ufldiy = tiy3 + yr*tuir;
    real ufldiz = tiz3 + zr*tuir;
    real ufldkx = tkx3 + xr*tukr;
    real ufldky = tky3 + yr*tukr;
    real ufldkz = tkz3 + zr*tukr;

    // get induced dipole field gradient used for quadrupole torques
    real tix5 = 4 * (rr5i*ukx);
    real tiy5 = 4 * (rr5i*uky);
    real tiz5 = 4 * (rr5i*ukz);
    real tkx5 = 4 * (rr5k*uix);
    real tky5 = 4 * (rr5k*uiy);
    real tkz5 = 4 * (rr5k*uiz);
    tuir = -2 * rr7i*ukr;
    tukr = -2 * rr7k*uir;

    real dufldixx = xr*tix5 + xr*xr*tuir;
    real dufldixy = xr*tiy5 + yr*tix5 + 2.0*xr*yr*tuir;
    real dufldiyy = yr*tiy5 + yr*yr*tuir;
    real dufldixz = xr*tiz5 + zr*tix5 + 2.0*xr*zr*tuir;
    real dufldiyz = yr*tiz5 + zr*tiy5 + 2.0*yr*zr*tuir;
    real dufldizz = zr*tiz5 + zr*zr*tuir;
    real dufldkxx = -xr*tkx5 - xr*xr*tukr;
    real dufldkxy = -xr*tky5 - yr*tkx5 - 2.0*xr*yr*tukr;
    real dufldkyy = -yr*tky5 - yr*yr*tukr;
    real dufldkxz = -xr*tkz5 - zr*tkx5 - 2.0*xr*zr*tukr;
    real dufldkyz = -yr*tkz5 - zr*tky5 - 2.0*yr*zr*tukr;
    real dufldkzz = -zr*tkz5 - zr*zr*tukr;

    // calculate torques on permanent dipoles and quadrupoles

    real tepix = diz*ufldiy - diy*ufldiz
                   + qixz*dufldixy - qixy*dufldixz
                   + 2*qiyz*(dufldiyy-dufldizz)
                   + (qizz-qiyy)*dufldiyz;
    real tepiy = dix*ufldiz - diz*ufldix
                   - qiyz*dufldixy + qixy*dufldiyz
                   + 2*qixz*(dufldizz-dufldixx)
                   + (qixx-qizz)*dufldixz;
    real tepiz = diy*ufldix - dix*ufldiy
                   + qiyz*dufldixz - qixz*dufldiyz
                   + 2*qixy*(dufldixx-dufldiyy)
                   + (qiyy-qixx)*dufldixy;

    real tepkx = dkz*ufldky - dky*ufldkz
                   + qkxz*dufldkxy - qkxy*dufldkxz
                   + 2*qkyz*(dufldkyy-dufldkzz)
                   + (qkzz-qkyy)*dufldkyz;
    real tepky = dkx*ufldkz - dkz*ufldkx
                   - qkyz*dufldkxy + qkxy*dufldkyz
                   + 2*qkxz*(dufldkzz-dufldkxx)
                   + (qkxx-qkzz)*dufldkxz;
    real tepkz = dky*ufldkx - dkx*ufldky
                   + qkyz*dufldkxz - qkxz*dufldkyz
                   + 2*qkxy*(dufldkxx-dufldkyy)
                   + (qkyy-qkxx)*dufldkxy;


    // get the field gradient for direct polarization force
    real term1i = rr3i - rr5i*xr*xr;
    real term1core = rr3core - rr5core*xr*xr;
    real term2i = 2.0*rr5i*xr;
    real term3i = rr7i*xr*xr - rr5i;
    real term4i = 2.0*rr5i;
    real term5i = 5.0*rr7i*xr;
    real term6i = rr9i*xr*xr;
    real term1k = rr3k - rr5k*xr*xr;
    real term2k = 2.0*rr5k*xr;
    real term3k = rr7k*xr*xr - rr5k;
    real term4k = 2.0*rr5k;
    real term5k = 5.0*rr7k*xr;
    real term6k = rr9k*xr*xr;
    real tixx = vali*term1i + corei*term1core  
                      + dix*term2i - dir*term3i
                      - qixx*term4i + qix*term5i - qir*term6i
                      + (qiy*yr+qiz*zr)*rr7i;
    real tkxx = valk*term1k + corek*term1core
                      - dkx*term2k + dkr*term3k
                      - qkxx*term4k + qkx*term5k - qkr*term6k
                      + (qky*yr+qkz*zr)*rr7k;
    term1i = rr3i - rr5i*yr*yr;
    term1core = rr3core - rr5core*yr*yr;
    term2i = 2.0*rr5i*yr;
    term3i = rr7i*yr*yr - rr5i;
    term4i = 2.0*rr5i;
    term5i = 5.0*rr7i*yr;
    term6i = rr9i*yr*yr;
    term1k = rr3k - rr5k*yr*yr;
    term2k = 2.0*rr5k*yr;
    term3k = rr7k*yr*yr - rr5k;
    term4k = 2.0*rr5k;
    term5k = 5.0*rr7k*yr;
    term6k = rr9k*yr*yr;
    real tiyy = vali*term1i + corei*term1core
                      + diy*term2i - dir*term3i
                      - qiyy*term4i + qiy*term5i - qir*term6i
                      + (qix*xr+qiz*zr)*rr7i;
    real tkyy = valk*term1k + corek*term1core
                      - dky*term2k + dkr*term3k
                      - qkyy*term4k + qky*term5k - qkr*term6k
                      + (qkx*xr+qkz*zr)*rr7k;
    term1i = rr3i - rr5i*zr*zr;
    term1core = rr3core - rr5core*zr*zr;
    term2i = 2.0*rr5i*zr;
    term3i = rr7i*zr*zr - rr5i;
    term4i = 2.0*rr5i;
    term5i = 5.0*rr7i*zr;
    term6i = rr9i*zr*zr;
    term1k = rr3k - rr5k*zr*zr;
    term2k = 2.0*rr5k*zr;
    term3k = rr7k*zr*zr - rr5k;
    term4k = 2.0*rr5k;
    term5k = 5.0*rr7k*zr;
    term6k = rr9k*zr*zr;
    real tizz = vali*term1i + corei*term1core
                      + diz*term2i - dir*term3i
                      - qizz*term4i + qiz*term5i - qir*term6i
                      + (qix*xr+qiy*yr)*rr7i;
    real tkzz = valk*term1k + corek*term1core
                      - dkz*term2k + dkr*term3k
                      - qkzz*term4k + qkz*term5k - qkr*term6k
                      + (qkx*xr+qky*yr)*rr7k;
    term2i = rr5i*xr ;
    term1i = yr * term2i;
    term1core = rr5core*xr*yr;
    term3i = rr5i*yr;
    term4i = yr * (rr7i*xr);
    term5i = 2.0*rr5i;
    term6i = 2.0*rr7i*xr;
    real term7i = 2.0*rr7i*yr;
    real term8i = yr*rr9i*xr;
    term2k = rr5k*xr;
    term1k = yr * term2k;
    term3k = rr5k*yr;
    term4k = yr * (rr7k*xr);
    term5k = 2.0*rr5k;
    term6k = 2.0*rr7k*xr;
    real term7k = 2.0*rr7k*yr;
    real term8k = yr*rr9k*xr;
    real tixy = -vali*term1i - corei*term1core 
                      + diy*term2i + dix*term3i
                      - dir*term4i - qixy*term5i + qiy*term6i
                      + qix*term7i - qir*term8i;
    real tkxy = -valk*term1k - corek*term1core 
                      - dky*term2k - dkx*term3k
                      + dkr*term4k - qkxy*term5k + qky*term6k
                      + qkx*term7k - qkr*term8k;
    term2i = rr5i*xr;
    term1i = zr * term2i;
    term1core = rr5core*xr*zr;
    term3i = rr5i*zr;
    term4i = zr * (rr7i*xr);
    term5i = 2.0*rr5i;
    term6i = 2.0*rr7i*xr;
    term7i = 2.0*rr7i*zr;
    term8i = zr*rr9i*xr;
    term2k = rr5k*xr;
    term1k = zr * term2k;
    term3k = rr5k*zr;
    term4k = zr * (rr7k*xr);
    term5k = 2.0*rr5k;
    term6k = 2.0*rr7k*xr;
    term7k = 2.0*rr7k*zr;
    term8k = zr*rr9k*xr;
    real tixz = -vali*term1i - corei*term1core
                      + diz*term2i + dix*term3i
                      - dir*term4i - qixz*term5i + qiz*term6i
                      + qix*term7i - qir*term8i;
    real tkxz = -valk*term1k - corek*term1core
                      - dkz*term2k - dkx*term3k
                      + dkr*term4k - qkxz*term5k + qkz*term6k
                      + qkx*term7k - qkr*term8k;
    term2i = rr5i*yr;
    term1i = zr * term2i;
    term1core = rr5core*yr*zr;
    term3i = rr5i*zr;
    term4i = zr * (rr7i*yr);
    term5i = 2.0*rr5i;
    term6i = 2.0*rr7i*yr;
    term7i = 2.0*rr7i*zr;
    term8i = zr*rr9i*yr;
    term2k = rr5k*yr;
    term1k = zr * term2k;
    term3k = rr5k*zr;
    term4k = zr * (rr7k*yr);
    term5k = 2.0*rr5k;
    term6k = 2.0*rr7k*yr;
    term7k = 2.0*rr7k*zr;
    term8k = zr*rr9k*yr;
    real tiyz = -vali*term1i - corei*term1core
                      + diz*term2i + diy*term3i
                      - dir*term4i - qiyz*term5i + qiz*term6i
                      + qiy*term7i - qir*term8i;
    real tkyz = -valk*term1k - corek*term1core
                      - dkz*term2k - dky*term3k
                      + dkr*term4k - qkyz*term5k + qkz*term6k
                      + qky*term7k - qkr*term8k;
    real depx = tixx*ukx + tixy*uky + tixz*ukz - tkxx*uix - tkxy*uiy - tkxz*uiz;
    real depy = tixy*ukx + tiyy*uky + tiyz*ukz - tkxy*uix - tkyy*uiy - tkyz*uiz;
    real depz = tixz*ukx + tiyz*uky + tizz*ukz - tkxz*uix - tkyz*uiy - tkzz*uiz;
    //real frcx = -2 * depx;
    //real frcy = -2 * depy;
    //real frcz = -2 * depz;
    real frcx = depx;
    real frcy = depy;
    real frcz = depz;


    // get the dtau/dr terms used for mutual polarization force

    real term1 = 2.0 * rr5ik;
    real term2 = term1*xr;
    real term3 = rr5ik - rr7ik*xr*xr;
    tixx = uix*term2 + uir*term3;
    tkxx = ukx*term2 + ukr*term3;
    term2 = term1*yr;
    term3 = rr5ik - rr7ik*yr*yr;
    tiyy = uiy*term2 + uir*term3;
    tkyy = uky*term2 + ukr*term3;
    term2 = term1*zr;
    term3 = rr5ik - rr7ik*zr*zr;
    tizz = uiz*term2 + uir*term3;
    tkzz = ukz*term2 + ukr*term3;
    term1 = rr5ik*yr;
    term2 = rr5ik*xr;
    term3 = yr * (rr7ik*xr);
    tixy = uix*term1 + uiy*term2 - uir*term3;
    tkxy = ukx*term1 + uky*term2 - ukr*term3;
    term1 = rr5ik * zr;
    term3 = zr * (rr7ik*xr);
    tixz = uix*term1 + uiz*term2 - uir*term3;
    tkxz = ukx*term1 + ukz*term2 - ukr*term3;
    term2 = rr5ik*yr;
    term3 = zr * (rr7ik*yr);
    tiyz = uiy*term1 + uiz*term2 - uir*term3;
    tkyz = uky*term1 + ukz*term2 - ukr*term3;
    //depx = tixx*ukxp + tixy*ukyp + tixz*ukzp
    //                  + tkxx*uixp + tkxy*uiyp + tkxz*uizp;
    //depy = tixy*ukxp + tiyy*ukyp + tiyz*ukzp
    //                  + tkxy*uixp + tkyy*uiyp + tkyz*uizp;
    //depz = tixz*ukxp + tiyz*ukyp + tizz*ukzp
    //                  + tkxz*uixp + tkyz*uiyp + tkzz*uizp;
    // replace p dipoles with d dipoles
    depx = tixx*ukx + tixy*uky + tixz*ukz
                      + tkxx*uix + tkxy*uiy + tkxz*uiz;
    depy = tixy*ukx + tiyy*uky + tiyz*ukz
                      + tkxy*uix + tkyy*uiy + tkyz*uiz;
    depz = tixz*ukx + tiyz*uky + tizz*ukz
                      + tkxz*uix + tkyz*uiy + tkzz*uiz;
    frcx += 0.5f * depx;
    frcy += 0.5f * depy;
    frcz += 0.5f * depz;

    // accumulate force

    //printf("do i have forces? %12.4f%12.4f%12.4f%12.4f\n",r,frcx,frcy,frcz);

    atom1.force -= make_real3(frcx, frcy, frcz) * EPSILON_FACTOR;
    if (doubleCountingFactor == 1) {
        atom2.force += make_real3(frcx, frcy, frcz) * EPSILON_FACTOR;
    }

    // accumulate torques

    atom1.torque += make_real3(tepix,tepiy,tepiz) * 0.5f * EPSILON_FACTOR;
    if (doubleCountingFactor == 1) {
        atom2.torque += make_real3(tepkx,tepky,tepkz) * 0.5f * EPSILON_FACTOR;
    }




    // energyToBeAccumulated += (mixed)doubleCountingFactor * EPSILON_FACTOR * energy;


}



extern "C" __global__ void computePolar(unsigned long long* __restrict__ forceBuffers, 
    unsigned long long* __restrict__ torqueBuffers, 
    mixed* __restrict__ energyBuffer,
    const real4* __restrict__ posq, const uint2* __restrict__ covalentFlags, const uint2* __restrict__ covalentFlags24, const ushort2* __restrict__ exclusionTiles, unsigned int startTileIndex,
    unsigned int numTileIndices,
#ifdef USE_CUTOFF
    const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX,
    real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
    const unsigned int* __restrict__ interactingAtoms,
#endif
    const real* __restrict__ dpl ,const real* __restrict__ quad,
    const real* __restrict__ pcore ,const real* __restrict__ pval, const real* __restrict__ palpha,
    const real* __restrict__ inddpl 
    )
{

    //printf("in the hood \n");

    const unsigned int totalWarps = (blockDim.x * gridDim.x) / TILE_SIZE;
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE - 1);
    const unsigned int tbx = threadIdx.x - tgx;
    mixed energy = 0;
    __shared__ PolarAtomData localData[THREAD_BLOCK_SIZE];
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE + warp * (LAST_EXCLUSION_TILE - FIRST_EXCLUSION_TILE) / totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE + (warp + 1) * (LAST_EXCLUSION_TILE - FIRST_EXCLUSION_TILE) / totalWarps;

    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        PolarAtomData data;
        unsigned int atom1 = x * TILE_SIZE + tgx;
        loadPolarAtomData(data, atom1, posq, dpl, quad, pcore, pval, palpha, inddpl);
        data.force = make_real3(0);
        data.torque = make_real3(0);
        uint2 covalent = covalentFlags[pos * TILE_SIZE + tgx];
        uint2 covalent24 = covalentFlags24[pos * TILE_SIZE + tgx];

//        printf("atom = %d, pcore = %12.4f, palpha = %12.4f, dipole = %12.4f%12.4f%12.4f\n", atom1+1, data.pcore, data.palpha, data.dipole.x, data.dipole.y, data.dipole.z);

        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].pos = data.pos;
            localData[threadIdx.x].q = data.q;
            localData[threadIdx.x].dipole = data.dipole;
            localData[threadIdx.x].quadrupole[0] = data.quadrupole[0];
            localData[threadIdx.x].quadrupole[1] = data.quadrupole[1];
            localData[threadIdx.x].quadrupole[2] = data.quadrupole[2];
            localData[threadIdx.x].quadrupole[3] = data.quadrupole[3];
            localData[threadIdx.x].quadrupole[4] = data.quadrupole[4];
            localData[threadIdx.x].pcore = data.pcore;
            localData[threadIdx.x].pval = data.pval;
            localData[threadIdx.x].palpha = data.palpha;
            localData[threadIdx.x].inducedDipole = data.inducedDipole;

            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real c = computeCScaleFactor(covalent, j);
                    real w = computeWScaleFactor(covalent24, j);
                    printf("w for atom %d and %d %12.4f\n",atom1+1,atom2+1,w);
                    computeOnePolarInteraction(
                        data, localData[tbx + j], c, w, (real)0.5, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    //printf (" x==y atom = %d atom = %d, energy = %12.4f, palphas: %12.4f%12.4f\n", atom1+1, atom2+1, energy, palpha[atom1], palpha[atom2]);

                }  
//                printf ("i = %d k = %d torque: %12.6f%12.6f%12.6f\n", atom1+1,atom2+1, data.torque); 
//                printf(" x == y atom = %d atom = %d, pcore %12.4f, palpha %12.4f, energy %12.4f\n", atom1+1, atom2+1, pcore[atom1], palpha[atom1], energy);
            }

            // In this block we are double counting, so we only accumulate force on atom1

            //printf ("atom1 = %d torque: %12.6f%12.6f%12.6f\n",atom1+1,data.torque.x,data.torque.y,data.torque.z);
            //data.force *= EPSILON_FACTOR;
            //data.torque *= EPSILON_FACTOR;

            //printf("bout to add to dem buffers, yo \n");

            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
            atomicAdd(&forceBuffers[atom1 + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
            atomicAdd(&forceBuffers[atom1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));
 
            atomicAdd(&torqueBuffers[atom1], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));

            printf ("POLAR nonPME forceBuffers for %d : %12.4f%12.4f%12.4f\n",atom1,data.force.x,data.force.y,data.force.z);

            //printf ("added to buffers \n");

            //printf ("field at atom: %d : %12.4f%12.4f%12.4f\n",atom1+1,data.field.x,data.field.y,data.field.z);

        } else {

            // This is an off-diagonal tile.
            unsigned int j = y * TILE_SIZE + tgx;
//            loadAtomData(localData[threadIdx.x], j, posq, chgct, dmpct);
            loadPolarAtomData(localData[threadIdx.x], j, posq, dpl, quad, pcore, pval, palpha, inddpl);
            localData[threadIdx.x].force = make_real3(0);
            localData[threadIdx.x].torque = make_real3(0);

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    float c = computeCScaleFactor(covalent, tj);
                    float w = computeWScaleFactor(covalent24, tj);
                    computeOnePolarInteraction(
                        data, localData[tbx + tj], c, w, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);    
                }
                //printf(" x != y atom = %d atom = %d, pcore %12.4f, palpha %12.4f\n", atom1+1, atom2+1, pcore[atom1], palpha[atom1]);
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            unsigned int offset = x * TILE_SIZE + tgx;

            // In this block we are not double counting, so we accumulate on
            // both atom1 and atom2

            //data.force *= EPSILON_FACTOR;
            //data.torque *= EPSILON_FACTOR;
            //localData[threadIdx.x].force *= EPSILON_FACTOR;
            //localData[threadIdx.x].torque *= EPSILON_FACTOR;
            
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));
            offset = y * TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.z * 0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.z*0x100000000)));
         }
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int)(numTiles > maxTiles ? startTileIndex + warp * (long long)numTileIndices / totalWarps : warp * (long long)numTiles / totalWarps);
    int end = (int)(numTiles > maxTiles ? startTileIndex + (warp + 1) * (long long)numTileIndices / totalWarps : (warp + 1) * (long long)numTiles / totalWarps);
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

            PolarAtomData data;
//            loadAtomData(data, atom1, posq, chgct, dmpct);
            loadPolarAtomData(data, atom1, posq, dpl, quad, pcore, pval, palpha, inddpl);
            data.force = make_real3(0);
            data.torque = make_real3(0);

#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos * TILE_SIZE + tgx];
#else
            unsigned int j = y * TILE_SIZE + tgx;
#endif

            atomIndices[threadIdx.x] = j;
//            loadAtomData(localData[threadIdx.x], j, posq, chgct, dmpct);
            loadPolarAtomData(localData[threadIdx.x], j, posq, dpl, quad, pcore, pval, palpha, inddpl);
            localData[threadIdx.x].force = make_real3(0);
            localData[threadIdx.x].torque = make_real3(0);

            // Compute forces.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = atomIndices[tbx + tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    computeOnePolarInteraction(
                        data, localData[tbx + tj], 1, 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                 }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }

            // Write results.

            //data.force *= EPSILON_FACTOR;
            //data.torque *= EPSILON_FACTOR;
            //localData[threadIdx.x].force *= EPSILON_FACTOR;
            //localData[threadIdx.x].torque *= EPSILON_FACTOR;

            unsigned int offset = x * TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));

#ifdef USE_CUTOFF
            offset = atomIndices[threadIdx.x];
#else
            offset = y * TILE_SIZE + tgx;
#endif

            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.z * 0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.z*0x100000000)));
 
        }
        pos++;
    }
    energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] += energy;
}