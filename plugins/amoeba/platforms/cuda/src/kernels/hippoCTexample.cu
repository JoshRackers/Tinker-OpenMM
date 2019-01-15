// test comment

typedef struct {
    real3 pos, force, torque, field;
    // charge transfer
    real chgct, dmpct;

    // repulsion
    real sizpr, elepr, dmppr;

    // charge penetration electrostatics
    real q;
    real3 dipole;
    real quadrupole[5]; // xx, xy, xz, yy, yz
    real pcore, pval, palpha;

    // dispersion
    real csix;
} AtomData;

inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ chgct, const real* __restrict__ dmpct
    , const real* __restrict__ dpl, const real* __restrict__ quad
    ,const real* __restrict__ sizpr,const real* __restrict__ elepr,const real* __restrict__ dmppr
    , const real* __restrict__ pcore, const real* __restrict__ pval, const real* __restrict__ palpha
    , const real* __restrict__ csix)
{
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    // charge transfer
    data.chgct = chgct[atom];
    data.dmpct = dmpct[atom];

    //repulsion
    data.sizpr = sizpr[atom];
    data.elepr = elepr[atom];
    data.dmppr = dmppr[atom];

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

    // dispersion c6 parameter
    data.csix = csix[atom];
}

__device__ real computeCScaleFactor(uint2 covalent, int index)
{
    int mask = 1 << index;
    bool x = (covalent.x & mask);
    bool y = (covalent.y & mask);
    return (x ? (y ? (real)CHARGETRANSFER13SCALE : (real)CHARGETRANSFER14SCALE) : (y ? (real)CHARGETRANSFER15SCALE : (real)1.0));
}

__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real cscale, real doubleCountingFactor, mixed& energyToBeAccumulated,
    real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ)
{
    // Compute the displacement.
    // The displacement could be computed inside the loop and passed to each function
    real3 delta;
    delta.x = atom1.pos.x - atom2.pos.x;
    delta.y = atom1.pos.y - atom2.pos.y;
    delta.z = atom1.pos.z - atom2.pos.z;
    APPLY_PERIODIC_TO_DELTA(delta);
    real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

    if (r2 > CHGTRN_CUTOFF_SQUARED)
        return;

    real rInv = RSQRT(r2);
    real r = r2 * rInv;

    real dmpcti = atom1.dmpct;
    real dmpctk = atom2.dmpct;
    real transferi = atom1.chgct;
    real transferk = atom2.chgct;
    real exptermi = EXP(-dmpcti * r);
    real exptermk = EXP(-dmpctk * r);
    real energy = -transferi * exptermk - transferk * exptermi;

    real de = transferi * dmpctk * exptermk + transferk * dmpcti * exptermi;

#ifdef USE_CUTOFF
    if (r > CHGTRN_TAPER) {
        real x = r - CHGTRN_TAPER;
        real taper = 1 + x * x * x * (CHGTRN_TAPER_C3 + x * (CHGTRN_TAPER_C4 + x * CHGTRN_TAPER_C5));
        real dtaper = x * x * (3 * CHGTRN_TAPER_C3 + x * (4 * CHGTRN_TAPER_C4 + x * 5 * CHGTRN_TAPER_C5));
        de = energy * dtaper + de * taper;
        energy *= taper;
    }
#endif

    energyToBeAccumulated += (mixed)doubleCountingFactor * cscale * energy;

    real frcx = de * delta.x * rInv * cscale;
    real frcy = de * delta.y * rInv * cscale;
    real frcz = de * delta.z * rInv * cscale;
    atom1.force -= make_real3(frcx, frcy, frcz);
    if (doubleCountingFactor == 1) {
        atom2.force += make_real3(frcx, frcy, frcz);
    }
}

__device__ void computeOneRepelInteraction(AtomData& atom1, AtomData& atom2, real cscale, real doubleCountingFactor, mixed& energyToBeAccumulated,
    real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) 
{
    // Compute the displacement.
    // The displacement could be computed inside the loop and passed to each function
    real3 delta;
    delta.x = atom2.pos.x - atom1.pos.x;
    delta.y = atom2.pos.y - atom1.pos.y;
    delta.z = atom2.pos.z - atom1.pos.z;
    APPLY_PERIODIC_TO_DELTA(delta);
    real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

    if (r2 > REPEL_CUTOFF_SQUARED)
        return;

    real rInv = RSQRT(r2);
    real r = r2 * rInv;

    real xi   = atom1.pos.x;
    real yi   = atom1.pos.y;
    real zi   = atom1.pos.z;
    real sizi = atom1.sizpr;
    real dmpi = atom1.dmppr;
    real vali = atom1.elepr;
    real ci   = atom1.q;
    real dix  = atom1.dipole.x;
    real diy  = atom1.dipole.y;
    real diz  = atom1.dipole.z;
    real qixx = atom1.quadrupole[0]; // xx, xy, xz, yy, yz
    real qixy = atom1.quadrupole[1];
    real qixz = atom1.quadrupole[2];
    real qiyy = atom1.quadrupole[3];
    real qiyz = atom1.quadrupole[4];
    real qizz = -(qixx+qiyy);

    printf("sizpr dmppr elepr %12.4f%12.4f%12.4f\n", sizi,dmpi,vali);
    // usei = use(i) ?
    // data for atom2 -- variables name matching erepel.f
    real sizk = atom2.sizpr;
    real dmpk = atom2.dmppr;
    real valk = atom2.elepr;
    real ck   = atom2.q;
    real dkx  = atom2.dipole.x;
    real dky  = atom2.dipole.y;
    real dkz  = atom2.dipole.z;
    real qkxx = atom2.quadrupole[0]; // xx, xy, xz, yy, yz
    real qkxy = atom2.quadrupole[1];
    real qkxz = atom2.quadrupole[2];
    real qkyy = atom2.quadrupole[3];
    real qkyz = atom2.quadrupole[4];
    real qkzz = -(qkxx+qkyy);
    //printf("%12.4e%12.4e%12.4e%12.4e%12.4e%12.4e\n", qixx,qixy,qixz,qiyy,qiyz,qizz);

    // get reciprocal distance terms for this interaction
    real rr3 = rInv / r2;
    real rr5 = 3.0f * rr3 / r2;
    real rr7 = 5.0f * rr5 / r2;
    real rr9 = 7.0f * rr7 / r2;
    real rr11 = 9.0f * rr9 / r2;

    //
    real xr = delta.x;
    real yr = delta.y;
    real zr = delta.z;
    //printf("%12.4f%12.4f%12.4f%12.4f\n", xr,yr,zr,r);
    //intermediates involving moments and distance separation
    real dikx = diy*dkz - diz*dky;
    real diky = diz*dkx - dix*dkz;
    real dikz = dix*dky - diy*dkx;
    real dirx = diy*zr - diz*yr;
    real diry = diz*xr - dix*zr;
    real dirz = dix*yr - diy*xr;
    real dkrx = dky*zr - dkz*yr;
    real dkry = dkz*xr - dkx*zr;
    real dkrz = dkx*yr - dky*xr;
    real dri = dix*xr + diy*yr + diz*zr;
    real drk = dkx*xr + dky*yr + dkz*zr;
    real dik = dix*dkx + diy*dky + diz*dkz;

    real qrix = qixx*xr + qixy*yr + qixz*zr;
    real qrkx = qkxx*xr + qkxy*yr + qkxz*zr;

    real qriy = qixy*xr + qiyy*yr + qiyz*zr;
    real qrky = qkxy*xr + qkyy*yr + qkyz*zr;

    real qriz = qixz*xr + qiyz*yr + qizz*zr; 
    real qrkz = qkxz*xr + qkyz*yr + qkzz*zr;

    real qrri = qrix*xr + qriy*yr + qriz*zr;
    real qrrk = qrkx*xr + qrky*yr + qrkz*zr;
    real qrrik = qrix*qrkx + qriy*qrky + qriz*qrkz;
    real qik = 2.0f*(qixy*qkxy+qixz*qkxz+qiyz*qkyz)+ qixx*qkxx + qiyy*qkyy + qizz*qkzz;
    real qrixr = qriz*yr - qriy*zr;
    real qriyr = qrix*zr - qriz*xr;
    real qrizr = qriy*xr - qrix*yr;
    real qrkxr = qrkz*yr - qrky*zr;
    real qrkyr = qrkx*zr - qrkz*xr;
    real qrkzr = qrky*xr - qrkx*yr;
    real qrrx = qrky*qriz - qrkz*qriy;
    real qrry = qrkz*qrix - qrkx*qriz;
    real qrrz = qrkx*qriy - qrky*qrix;
    real qikrx = qixx*qrkx + qixy*qrky + qixz*qrkz;
    real qikry = qixy*qrkx + qiyy*qrky + qiyz*qrkz;
    real qikrz = qixz*qrkx + qiyz*qrky + qizz*qrkz;
    real qkirx = qkxx*qrix + qkxy*qriy + qkxz*qriz;
    real qkiry = qkxy*qrix + qkyy*qriy + qkyz*qriz;
    real qkirz = qkxz*qrix + qkyz*qriy + qkzz*qriz;
    real qikrxr = qikrz*yr - qikry*zr;
    real qikryr = qikrx*zr - qikrz*xr;
    real qikrzr = qikry*xr - qikrx*yr;
    real qkirxr = qkirz*yr - qkiry*zr;
    real qkiryr = qkirx*zr - qkirz*xr;
    real qkirzr = qkiry*xr - qkirx*yr;
    real diqkx = dix*qkxx + diy*qkxy + diz*qkxz;
    real diqky = dix*qkxy + diy*qkyy + diz*qkyz;
    real diqkz = dix*qkxz + diy*qkyz + diz*qkzz;
    real dkqix = dkx*qixx + dky*qixy + dkz*qixz;
    real dkqiy = dkx*qixy + dky*qiyy + dkz*qiyz;
    real dkqiz = dkx*qixz + dky*qiyz + dkz*qizz;

    real diqrk = dix*qrkx + diy*qrky + diz*qrkz;
    real dkqri = dkx*qrix + dky*qriy + dkz*qriz;

    real diqkxr = diqkz*yr - diqky*zr;
    real diqkyr = diqkx*zr - diqkz*xr;
    real diqkzr = diqky*xr - diqkx*yr;
    real dkqixr = dkqiz*yr - dkqiy*zr;
    real dkqiyr = dkqix*zr - dkqiz*xr;
    real dkqizr = dkqiy*xr - dkqix*yr;
    real dqiqkx = diy*qrkz - diz*qrky + dky*qriz - dkz*qriy - 2.0f*(qixy*qkxz+qiyy*qkyz+qiyz*qkzz -qixz*qkxy-qiyz*qkyy)-qizz*qkyz; 
    real dqiqky = diz*qrkx - dix*qrkz + dkz*qrix - dkx*qriz - 2.0f*(qixz*qkxx+qiyz*qkxy-qixx*qkxz-qixy*qkyz-qixz*qkzz+qizz*qkxz);             
    real dqiqkz = dix*qrky - diy*qrkx + dkx*qriy - dky*qrix - 2.0f*(qixx*qkxy+qixy*qkyy+qixz*qkyz-qixy*qkxx-qiyy*qkxy-qiyz*qkxz);
                    
    // begin dumping function
    real r3, r4, r5, r6, r7, r8;
    real dmpi2, dampi, expi;
    real dmpi22, dmpi23, dmpi24, dmpi25, dmpi26,dmpi27;
    real pre, s, ds, d2s, d3s, d4s, d5s;
    real dmpik[6]; 
    if (dmpi == dmpk) {
        r3 = r2 * r;
        r4 = r3 * r;
        r5 = r4 * r;
        r6 = r5 * r;
        r7 = r6 * r;
        r8 = r7 * r;
        dmpi2 = 0.5f * dmpi;
        dampi = dmpi2 * r;
        expi = EXP(-dampi);
        dmpi22 = dmpi2 * dmpi2;
        dmpi23 = dmpi22 * dmpi2;
        dmpi24 = dmpi23 * dmpi2;
        dmpi25 = dmpi24 * dmpi2;
        dmpi26 = dmpi25 * dmpi2;
        dmpi27 = dmpi2 * dmpi26;
        pre = 128.0f;      
        s = (r + dmpi2*r2 + dmpi22*r3/3.0f) * expi;
        ds = (dmpi22*r3 + dmpi23*r4) * expi / 3.0f;
        d2s = dmpi24 * expi * r5 / 9.0f;
        d3s = dmpi25 * expi * r6 / 45.0f;
        d4s = (dmpi25*r6 + dmpi26*r7) * expi / 315.0f;
        d5s = (dmpi25*r6 + dmpi26*r7 + dmpi27*r8/3.0f)* expi / 945.0f;
    } else 
    {
        // treat the case where alpha damping exponents are unequal
        real dmpk2, dmpk22, dmpk23, dmpk24, dmpk25, dmpk26; 
        real term, tmp, expk, dampk;
        r3 = r2 * r;
        r4 = r3 * r;
        r5 = r4 * r;
        r6 = r5 * r;
        dmpi2 = 0.50f * dmpi;
        dmpk2 = 0.50f * dmpk;
        dampi = dmpi2 * r;
        dampk = dmpk2 * r;
        expi = EXP(-dampi);
        expk = EXP(-dampk);
        dmpi22 = dmpi2 * dmpi2;
        dmpi23 = dmpi22 * dmpi2;
        dmpi24 = dmpi23 * dmpi2;
        dmpi25 = dmpi24 * dmpi2;
        dmpi26 = dmpi25 * dmpi2;
        dmpk22 = dmpk2 * dmpk2;
        dmpk23 = dmpk22 * dmpk2;
        dmpk24 = dmpk23 * dmpk2;
        dmpk25 = dmpk24 * dmpk2;
        dmpk26 = dmpk25 * dmpk2;
        term = dmpi22 - dmpk22;
        pre = (8192.0f * dmpi23 * dmpk23)/(term*term*term*term);
        tmp = (4.0f * dmpi2 * dmpk2)/term;
        s = (dampi-tmp)*expk + (dampk+tmp)*expi;
        ds = (term*dmpk2*r2 - 4.0f*(dmpk22*r + dmpk2)) * dmpi2* expk/term
                + (term*dmpi2*r2 + 4.0f*(dmpi22*r + dmpi2)) * dmpk2 * expi/term; //changed

        d2s = (dmpk2*r2/3.0f + dmpk22*r3/3.0f - 4.0f/3.0f*dmpk23*r2/term
                - 4.0f*dmpk22*r/term - 4.0f*dmpk2/term) * dmpi2 * expk
                + ((dmpi2*r2 + dmpi22*r3)/3.0f + (4.0f/term)*(dmpi23*r2/3.0f
                + dmpi22*r + dmpi2)) * dmpk2 * expi; //changed
                
        d3s = ((dmpk23*r4/3.0f + dmpk22*r3 + dmpk2*r2)/5.0f
                + (4.0f/term)*(-dmpk24*r3/15.0f - (2.0f/5.0f)*dmpk23*r2
                - dmpk22*r) - (4.0f/term)*dmpk2) * dmpi2 * expk 
                + ((dmpi23*r4/3.0f + dmpi22*r3 + dmpi2*r2)/5.0f
                + (4.0f/term)*(dmpi24*r3/15.0f + 2.0f*dmpi23*r2/5.0f
                + dmpi22*r + dmpi2)) * dmpk2 * expi; //changed
                
        d4s = ((dmpk24*r5/15.0f + 2.0f/5.0f*dmpk23*r4 + dmpk22*r3 + dmpk2*r2)/7.0f
                + (4.0f/term)*(-dmpk25*r4/105.0f - 2.0f/21.0f*dmpk24*r3
                - 3.0f/7.0f*dmpk23*r2 - dmpk22*r - dmpk2)) * dmpi2 * expk
                + ((dmpi24*r5/15.0f + 2.0f/5.0f*dmpi23*r4 + dmpi22*r3 + dmpi2*r2)/7.0f            
                + (4.0f/term)*(dmpi25*r4/105.0f + 2.0f/21.0f*dmpi24*r3
                + 3.0f/7.0f*dmpi23*r2 + dmpi22*r + dmpi2)) * dmpk2 * expi;
            
        d5s = (dmpk25*r6/945.0f + 2.0f/189.0f*dmpk24*r5 + dmpk23*r4/21.0f
                + dmpk22*r3/9.0f + dmpk2*r2/9.0f
                + (4.0f/term)*(-dmpk26*r5/945.0f - dmpk25*r4/63.0f - dmpk24*r3/9.0f
                - 4.0f/9.0f*dmpk23*r2 - dmpk22*r - dmpk2)) * dmpi2 * expk
                + (dmpi25*r6/945.0f + 2.0f/189.0f*dmpi24*r5
                + dmpi23*r4/21.0f + dmpi22*r3/9.0f + dmpi2*r2/9.0f
                + (4.0f/term)*(dmpi26*r5/945.0f + dmpi25*r4/63.0f + dmpi24*r3/9.0f
                + 4.0f/9.0f*dmpi23*r2 + dmpi22*r + dmpi2)) * dmpk2 * expi;
    }
    // convert partial derivatives into full derivatives
    s = s * rInv;
    ds = ds * rr3;
    d2s = d2s * rr5;
    d3s = d3s * rr7;
    d4s = d4s * rr9;
    d5s = d5s * rr11;

    // dmpik is a vector -- why (1),(3),(5) .. and not (1),(2),... 

    //dmpik(1) == dmpik[0]
    //dmpik(3) == dmpik[1]
    //dmpik(5) == dmpik[2]
    //dmpik(7) == dmpik[3]
    //dmpik(9) == dmpik[4]
    //dmpik(11) == dmpik[5]
  
    dmpik[0] = 0.50f * pre * s * s;  
    dmpik[1] = pre * s * ds;
    dmpik[2] = pre * (s*d2s + ds*ds);
    dmpik[3] = pre * (s*d3s + 3.0f*ds*d2s);
    dmpik[4] = pre * (s*d4s + 4.0f*ds*d3s + 3.0f*d2s*d2s);
    dmpik[5] = pre * (s*d5s + 5.0f*ds*d4s + 10.0f*d2s*d3s);

    // End of damping function

    // compute the Pauli repulsion energy for this interaction
    real term1, term2, term3, term4, term5, eterm;
    real sizik, dterm, dterm1, dterm2, dterm3, dterm4, dterm5, dterm6;
    term1 = vali*valk;
    term2 = valk*dri - vali*drk + dik;
    term3 = vali*qrrk + valk*qrri - dri*drk + 2.0f*(dkqri-diqrk+qik);
    term4 = dri*qrrk - drk*qrri - 4.0f*qrrik;
    term5 = qrri*qrrk;
    eterm = term1*dmpik[0] + term2*dmpik[1] + term3*dmpik[2] + term4*dmpik[3] + term5*dmpik[4];
    sizik = sizi * sizk * cscale;
    real energy = sizik * eterm * rInv;
    //print *,"Rep terms",eterm,rr1,dmpik(1)

    //printf("Eterms ,%15.7e%15.7e%15.7e%15.7e%15.7e\n", term1*dmpik[0], term2*dmpik[1], 
    //    term3*dmpik[2], term4*dmpik[3], term5*dmpik[4]);
    //eterm is wrong -- review values of damping function and term1-5;

    // calculate intermediate terms for force and torque
    dterm = term1*dmpik[1] + term2*dmpik[2] + term3*dmpik[3] + term4*dmpik[4] + term5*dmpik[5];  
    dterm1 = -valk*dmpik[1] + drk*dmpik[2] - qrrk*dmpik[3];
    dterm2 = vali*dmpik[1] + dri*dmpik[2] + qrri*dmpik[3];   
    dterm3 = 2.0f * dmpik[2];
    dterm4 = 2.0f * (-valk*dmpik[2] + drk*dmpik[3] - qrrk*dmpik[4]);
    dterm5 = 2.0f * (-vali*dmpik[2] - dri*dmpik[3] - qrri*dmpik[4]);
    dterm6 = 4.0f * dmpik[3];

    //dmpik(1) == dmpik[0]
    //dmpik(3) == dmpik[1]
    //dmpik(5) == dmpik[2]
    //dmpik(7) == dmpik[3]
    //dmpik(9) == dmpik[4]
    //dmpik(11) == dmpik[5]
    // compute the force components for this interaction
    real frcx, frcy, frcz;
    frcx = dterm*xr + dterm1*dix + dterm2*dkx
                + dterm3*(diqkx-dkqix) + dterm4*qrix
                + dterm5*qrkx + dterm6*(qikrx+qkirx);
    frcy = dterm*yr + dterm1*diy + dterm2*dky
                + dterm3*(diqky-dkqiy) + dterm4*qriy
                + dterm5*qrky + dterm6*(qikry+qkiry);
    frcz = dterm*zr + dterm1*diz + dterm2*dkz
                + dterm3*(diqkz-dkqiz) + dterm4*qriz
                + dterm5*qrkz + dterm6*(qikrz+qkirz);
    frcx = frcx*rInv + eterm*rr3*xr;
    frcy = frcy*rInv + eterm*rr3*yr;
    frcz = frcz*rInv + eterm*rr3*zr;
    frcx = sizik * frcx;
    frcy = sizik * frcy;
    frcz = sizik * frcz;

    // compute the torque components for this interaction
    real ttrix, ttriy, ttriz, ttrkx, ttrky, ttrkz; 
    ttrix = -dmpik[1]*dikx + dterm1*dirx + dterm3*(dqiqkx+dkqixr) - dterm4*qrixr - dterm6*(qikrxr+qrrx);
    ttriy = -dmpik[1]*diky + dterm1*diry + dterm3*(dqiqky+dkqiyr) - dterm4*qriyr - dterm6*(qikryr+qrry);       
    ttriz = -dmpik[1]*dikz + dterm1*dirz + dterm3*(dqiqkz+dkqizr) - dterm4*qrizr - dterm6*(qikrzr+qrrz);
    ttrkx = dmpik[1]*dikx + dterm2*dkrx - dterm3*(dqiqkx+diqkxr) - dterm5*qrkxr - dterm6*(qkirxr-qrrx);
    ttrky = dmpik[1]*diky + dterm2*dkry - dterm3*(dqiqky+diqkyr) - dterm5*qrkyr - dterm6*(qkiryr-qrry);
    ttrkz = dmpik[1]*dikz + dterm2*dkrz - dterm3*(dqiqkz+diqkzr) - dterm5*qrkzr - dterm6*(qkirzr-qrrz);
    ttrix = sizik * ttrix * rInv;
    ttriy = sizik * ttriy * rInv;
    ttriz = sizik * ttriz * rInv;
    ttrkx = sizik * ttrkx * rInv;
    ttrky = sizik * ttrky * rInv;
    ttrkz = sizik * ttrkz * rInv;
#ifdef USE_CUTOFF
    if (r > REPEL_TAPER) 
    {
        real x = r - REPEL_TAPER;
        real taper = 1 + x * x * x * (REPEL_TAPER_C3 + x * (REPEL_TAPER_C4 + x * REPEL_TAPER_C5));
        real dtaper = x * x * (3 * REPEL_TAPER_C3 + x * (4 * REPEL_TAPER_C4 + x * 5 * REPEL_TAPER_C5));
        dtaper *= (energy * rInv);
        frcx = frcx*taper - dtaper*xr;
        frcy = frcy*taper - dtaper*yr;
        frcz = frcz*taper - dtaper*zr;
        ttrix *= taper;
        ttriy *= taper;
        ttriz *= taper;
        ttrkx *= taper;
        ttrky *= taper;
        ttrkz *= taper;
        energy = energy * taper;
    }
#endif
    printf("Eterm Fy Fz E %18.6e%18.6e%18.6e%18.6e\n", eterm, frcy/41.84, frcz/41.84,energy);

    energyToBeAccumulated += (mixed)doubleCountingFactor * energy;

    atom1.force -= make_real3(frcx, frcy, frcz);
    atom1.torque += make_real3(ttrix, ttriy, ttriz);
    // atom2.torque += make_real3(ttrkx, ttrky, ttrkz);
    if (doubleCountingFactor == 1) 
        atom2.force += make_real3(frcx, frcy, frcz);
        atom2.torque += make_real3(ttrkx, ttrky, ttrkz);
}

__device__ void computeOneChargePenetrationInteraction(AtomData& atom1, AtomData& atom2, real cscale, real doubleCountingFactor, mixed& energyToBeAccumulated,
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

//    printf ("multipoles i: %12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.8f%12.8f\n", ci, dix, diy, diz, qixx, qixy,qixz,qiyy,qiyz,qkyz);

//
//     get reciprocal distance terms for this interaction
//
//    real rr1 = EPSILON_FACTOR * cscale / r;
    real rr1 = 1.0f / r;
    real rr3 = rr1 / r2;
    real rr5 = 3 * rr3 / r2;
    real rr7 = 5 * rr5 / r2;
    real rr9 = 7 * rr7 / r2;
    real rr11 = 9 * rr9 / r2;
//
//     intermediates involving moments and distance separation
//
    real dikx = diy*dkz - diz*dky;
    real diky = diz*dkx - dix*dkz;
    real dikz = dix*dky - diy*dkx;
    real dirx = diy*zr - diz*yr;
    real diry = diz*xr - dix*zr;
    real dirz = dix*yr - diy*xr;
    real dkrx = dky*zr - dkz*yr;
    real dkry = dkz*xr - dkx*zr;
    real dkrz = dkx*yr - dky*xr;
    real dri = dix*xr + diy*yr + diz*zr;
    real drk = dkx*xr + dky*yr + dkz*zr;
    real dik = dix*dkx + diy*dky + diz*dkz;
    real qrix = qixx*xr + qixy*yr + qixz*zr;
    real qriy = qixy*xr + qiyy*yr + qiyz*zr;
    real qriz = qixz*xr + qiyz*yr + qizz*zr;
    real qrkx = qkxx*xr + qkxy*yr + qkxz*zr;
    real qrky = qkxy*xr + qkyy*yr + qkyz*zr;
    real qrkz = qkxz*xr + qkyz*yr + qkzz*zr;
    real qrri = qrix*xr + qriy*yr + qriz*zr;
    real qrrk = qrkx*xr + qrky*yr + qrkz*zr;
    real qrrik = qrix*qrkx + qriy*qrky + qriz*qrkz;
    real qik = 2*(qixy*qkxy+qixz*qkxz+qiyz*qkyz) + qixx*qkxx + qiyy*qkyy + qizz*qkzz;
    real qrixr = qriz*yr - qriy*zr;
    real qriyr = qrix*zr - qriz*xr;
    real qrizr = qriy*xr - qrix*yr;
    real qrkxr = qrkz*yr - qrky*zr;
    real qrkyr = qrkx*zr - qrkz*xr;
    real qrkzr = qrky*xr - qrkx*yr;
    real qrrx = qrky*qriz - qrkz*qriy;
    real qrry = qrkz*qrix - qrkx*qriz;
    real qrrz = qrkx*qriy - qrky*qrix;
    real qikrx = qixx*qrkx + qixy*qrky + qixz*qrkz;
    real qikry = qixy*qrkx + qiyy*qrky + qiyz*qrkz;
    real qikrz = qixz*qrkx + qiyz*qrky + qizz*qrkz;
    real qkirx = qkxx*qrix + qkxy*qriy + qkxz*qriz;
    real qkiry = qkxy*qrix + qkyy*qriy + qkyz*qriz;
    real qkirz = qkxz*qrix + qkyz*qriy + qkzz*qriz;
    real qikrxr = qikrz*yr - qikry*zr;
    real qikryr = qikrx*zr - qikrz*xr;
    real qikrzr = qikry*xr - qikrx*yr;
    real qkirxr = qkirz*yr - qkiry*zr;
    real qkiryr = qkirx*zr - qkirz*xr;
    real qkirzr = qkiry*xr - qkirx*yr;
    real diqkx = dix*qkxx + diy*qkxy + diz*qkxz;
    real diqky = dix*qkxy + diy*qkyy + diz*qkyz;
    real diqkz = dix*qkxz + diy*qkyz + diz*qkzz;
    real dkqix = dkx*qixx + dky*qixy + dkz*qixz;
    real dkqiy = dkx*qixy + dky*qiyy + dkz*qiyz;
    real dkqiz = dkx*qixz + dky*qiyz + dkz*qizz;
    real diqrk = dix*qrkx + diy*qrky + diz*qrkz;
    real dkqri = dkx*qrix + dky*qriy + dkz*qriz;
    real diqkxr = diqkz*yr - diqky*zr;
    real diqkyr = diqkx*zr - diqkz*xr;
    real diqkzr = diqky*xr - diqkx*yr;
    real dkqixr = dkqiz*yr - dkqiy*zr;
    real dkqiyr = dkqix*zr - dkqiz*xr;
    real dkqizr = dkqiy*xr - dkqix*yr;
    real dqiqkx = diy*qrkz - diz*qrky + dky*qriz - dkz*qriy - 2*(qixy*qkxz+qiyy*qkyz+qiyz*qkzz - qixz*qkxy-qiyz*qkyy-qizz*qkyz);
    real dqiqky = diz*qrkx - dix*qrkz + dkz*qrix - dkx*qriz - 2*(qixz*qkxx+qiyz*qkxy+qizz*qkxz - qixx*qkxz-qixy*qkyz-qixz*qkzz);
    real dqiqkz = dix*qrky - diy*qrkx + dkx*qriy - dky*qrix - 2*(qixx*qkxy+qixy*qkyy+qixz*qkyz - qixy*qkxx-qiyy*qkxy-qiyz*qkxz);

    real term1 = corei*corek;
    real term1i = corek*vali;
    real term2i = corek*dri;
    real term3i = corek*qrri;
    real term1k = corei*valk;
    real term2k = -corei*drk;
    real term3k = corei*qrrk;
    real term1ik = vali*valk;
    real term2ik = valk*dri - vali*drk + dik;
    real term3ik = vali*qrrk + valk*qrri - dri*drk + 2*(dkqri-diqrk+qik);
    real term4ik = dri*qrrk - drk*qrri - 4*qrrik;
    real term5ik = qrri*qrrk;

    // insert damping functions here
    // how do i handle this logic???
    // right now i'm punting and using a=b only (all alpha the same)

    real dampi = alphai*r;
    real dampk = alphak*r;
    real expi = EXP(-dampi);
    real expk = EXP(-dampk);

    real dampi2 = dampi*dampi;
    real dampi3 = dampi2*dampi;
    real dampi4 = dampi3*dampi;
    real dampi5 = dampi4*dampi;
    real dampi6 = dampi5*dampi;
    real dampi7 = dampi6*dampi;
    real dampi8 = dampi7*dampi;

    real dampk2 = dampk*dampk;
    real dampk3 = dampk2*dampk;
    real dampk4 = dampk3*dampk;
    real dampk5 = dampk4*dampk;
    real dampk6 = dampk5*dampk;
    real dampk7 = dampk6*dampk;
    real dampk8 = dampk7*dampk;

    real dmpi1 = 1 - (1 + 0.5f*dampi)*expi;
    real dmpi3 = 1 - (1 + dampi + 0.5f*dampi2)*expi;
    real dmpi5 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f)*expi;
    real dmpi7 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/30.0f)*expi;
    real dmpi9 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + 4*dampi4/105.0f + dampi5/210.0f)*expi;

    //printf ("Is this true? %d\n",alphai==alphak);

    real dmpk1 = 1;
    real dmpk3 = 1;
    real dmpk5 = 1;
    real dmpk7 = 1;
    real dmpk9 = 1;

    real dmpik1  = 1;
    real dmpik3  = 1;
    real dmpik5  = 1;
    real dmpik7  = 1;
    real dmpik9  = 1;
    real dmpik11 = 1;

    if (alphai == alphak){
        dmpk1 = dmpi1;
        dmpk3 = dmpi3;
        dmpk5 = dmpi5;
        dmpk7 = dmpi7;
        dmpk9 = dmpi9;

        dmpik1 = 1  - (1 + 11 *dampi/16  + 3*dampi2/16.0f + dampi3/48.0f)*expi;
        dmpik3 = 1  - (1 + dampi + 0.5f*dampi2 + 7*dampi3/48.0f + dampi4/48.0f)*expi;
        dmpik5 = 1  - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/24.0f + dampi5/144.0f)*expi;
        dmpik7 = 1  - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/24.0f + dampi5/120.0f + dampi6/720.0f)*expi;
        dmpik9 = 1  - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/24.0f + dampi5/120.0f + dampi6/720.0f + dampi7/5040.0f)*expi;
        dmpik11 = 1 - (1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/24.0f + dampi5/120.0f + dampi6/720.0f + dampi7/5040.0f + dampi8/45360.0f)*expi;        
    } else {
        dmpk1 = 1 - (1 + 0.5f*dampk)*expk;
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
        dmpik1 = 1 - termi2*(1 + 2*termk + 0.5f*dampi)*expi 
                   - termk2*(1 + 2*termi + 0.5f*dampk)*expk;
        dmpik3 = 1 - termi2*(1+dampi+0.5f*dampi2)*expi
                   - termk2*(1+dampk+0.5f*dampk2)*expk
                   - 2*termi2*termk*(1+dampi)*expi
                   - 2*termk2*termi*(1+dampk)*expk;
        dmpik5 = 1 - termi2*(1 + dampi + 0.5f*dampi2 + dampi3/6.0f)*expi
                   - termk2*(1 + dampk + 0.5f*dampk2 + dampk3/6.0f)*expk
                   - 2*termi2*termk*(1.0 + dampi + dampi2/3.0f)*expi
                   - 2*termk2*termi*(1.0 + dampk + dampk2/3.0f)*expk;
        dmpik7 = 1 - termi2*(1 + dampi + 0.5f*dampi2 + dampi3/6.0f + dampi4/30.0f)*expi
                   - termk2*(1 + dampk + 0.5f*dampk2 + dampk3/6.0f + dampk4/30.0f)*expk
                   - 2*termi2*termk*(1 + dampi + 2*dampi2/5.0f + dampi3/15.0f)*expi
                   - 2*termk2*termi*(1 + dampk + 2*dampk2/5.0f + dampk3/15.0f)*expk;
        dmpik9 = 1 - termi2*(1 + dampi + 0.5f*dampi2 + dampi3/6.0f + 4*dampi4/105.0f + dampi5/210.0f)*expi
                   - termk2*(1 + dampk + 0.5f*dampk2 + dampk3/6.0f + 4*dampk4/105.0f + dampk5/210.0f)*expk
                   - 2*termi2*termk*(1 + dampi + 3*dampi2/7.0f + 2*dampi3/21.0f + dampi4/105.0f)*expi 
                   - 2*termk2*termi*(1 + dampk + 3*dampk2/7.0f + 2*dampk3/21.0f + dampk4/105.0f)*expk;
        dmpik11 = 1 - termi2*(1 + dampi + 0.5f*dampi2 + dampi3/6.0f
                       + 5*dampi4/126.0f + 2*dampi5/315.0f + dampi6/1890.0f)*expi
                    - termk2*(1 + dampk + 0.5f*dampk2 + dampk3/6.0f
                       + 5*dampk4/126.0f + 2*dampk5/315.0f + dampk6/1890.0f)*expk
                    - 2*termi2*termk
                       *(1 + dampi + 4*dampi2/9.0f + dampi3/9.0f + dampi4/63.0f + dampi5/945.0f)*expi
                    - 2*termk2*termi
                       *(1 + dampk + 4*dampk2/9.0f + dampk3/9.0f + dampk4/63.0f + dampk5/945.0f)*expk; 
    }

    // calculate the real space Ewald error function terms

    real ralpha = EWALD_ALPHA*r;
    real exp2a = EXP(-ralpha*ralpha);

    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.
    const real t = RECIP(1.0f+0.3275911f*ralpha);
    const real erfAlphaR = 1-(0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
 
    real bn[6];
    bn[0] = (1 - erfAlphaR)/r;
    real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
    real alsq2n = 1.0f / (SQRT_PI*EWALD_ALPHA);
    real bfac = 0.0f;
    for (int i = 1; i < 6; ++i){
        bfac = (float) (i+i-1);
        alsq2n = alsq2*alsq2n;
        bn[i] = (bfac*bn[i-1]+alsq2n*exp2a) / r2;
    }

    real rr1i = bn[0] - (1-cscale*dmpi1)*rr1;
    real rr3i = bn[1] - (1-cscale*dmpi3)*rr3;
    real rr5i = bn[2] - (1-cscale*dmpi5)*rr5;
    real rr7i = bn[3] - (1-cscale*dmpi7)*rr7;
    real rr1k = bn[0] - (1-cscale*dmpk1)*rr1;
    real rr3k = bn[1] - (1-cscale*dmpk3)*rr3;
    real rr5k = bn[2] - (1-cscale*dmpk5)*rr5;
    real rr7k = bn[3] - (1-cscale*dmpk7)*rr7;
    real rr1ik = bn[0] - (1-cscale*dmpik1)*rr1;
    real rr3ik = bn[1] - (1-cscale*dmpik3)*rr3;
    real rr5ik = bn[2] - (1-cscale*dmpik5)*rr5;
    real rr7ik = bn[3] - (1-cscale*dmpik7)*rr7;
    real rr9ik = bn[4] - (1-cscale*dmpik9)*rr9;
    real rr11ik = bn[5] - (1-cscale*dmpik11)*rr11;
    rr1 = bn[0] - (1-cscale)*rr1;
    rr3 = bn[1] - (1-cscale)*rr3;

//    real rr1i = dmpi1*rr1;
//    real rr3i = dmpi3*rr3;
//    real rr5i = dmpi5*rr5;
//    real rr7i = dmpi7*rr7;
//    real rr1k = dmpk1*rr1;
//    real rr3k = dmpk3*rr3;
//    real rr5k = dmpk5*rr5;
//    real rr7k = dmpk7*rr7;
//    real rr1ik = dmpik1*rr1;
//    real rr3ik = dmpik3*rr3;
//    real rr5ik = dmpik5*rr5;
//    real rr7ik = dmpik7*rr7;
//    real rr9ik = dmpik9*rr9;
//    real rr11ik = dmpik11*rr11;
    real energy = term1*rr1 + term4ik*rr7ik + term5ik*rr9ik + 
                term1i*rr1i + term1k*rr1k + term1ik*rr1ik + 
                term2i*rr3i + term2k*rr3k + term2ik*rr3ik + 
                term3i*rr5i + term3k*rr5k + term3ik*rr5ik;

//  precision test
//    real energy = ci*ck*rr1;
//    real energy = vali*valk*rr1 + corei*corek*rr1 + vali*corek*rr1 + valk*corei*rr1;

//    printf ("some energy %12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f\n", 10*r, energy*0.239006f,vali,corei,alphai,valk,corek,alphak);
//    printf ("check: %12.4f%12.4f%12.4f%12.4f%12.4f%12.4f%12.8f\n", 0.3828f*0.3828f*332.063713f/2.8920f, ci*ck*332.063713f/(r*10), (term1/r + term1i/r + term1k/r + term1ik/r)*332.063713f/10.f, (term1*rr1 + term1i*rr1i + term1k*rr1k + term1ik*rr1ik)*0.239006f , rr1i, rr1k, vali+corei);

    energyToBeAccumulated += (mixed)doubleCountingFactor * EPSILON_FACTOR * energy;

    //  compute permanent electrostatic field

    real fieldix = -xr*(rr3*corek + rr3k*valk - rr5k*drk + rr7k*qrrk) - rr3k*dkx + 2*rr5k*qrkx;
    real fieldiy = -yr*(rr3*corek + rr3k*valk - rr5k*drk + rr7k*qrrk) - rr3k*dky + 2*rr5k*qrky;
    real fieldiz = -zr*(rr3*corek + rr3k*valk - rr5k*drk + rr7k*qrrk) - rr3k*dkz + 2*rr5k*qrkz;
    real fieldkx =  xr*(rr3*corei + rr3i*vali + rr5i*dri + rr7i*qrri) - rr3i*dix - 2*rr5i*qrix;
    real fieldky =  yr*(rr3*corei + rr3i*vali + rr5i*dri + rr7i*qrri) - rr3i*diy - 2*rr5i*qriy;
    real fieldkz =  zr*(rr3*corei + rr3i*vali + rr5i*dri + rr7i*qrri) - rr3i*diz - 2*rr5i*qriz;

    //
    //     find damped multipole intermediates for force and torque
    //

    real de = term1*rr3 + term4ik*rr9ik + term5ik*rr11ik 
        + term1i*rr3i + term1k*rr3k + term1ik*rr3ik
        + term2i*rr5i + term2k*rr5k + term2ik*rr5ik
        + term3i*rr7i + term3k*rr7k + term3ik*rr7ik;

    term1 = -corek*rr3i - valk*rr3ik + drk*rr5ik - qrrk*rr7ik;
    real term2 = corei*rr3k + vali*rr3ik + dri*rr5ik + qrri*rr7ik;
    real term3 = 2 * rr5ik;
    real term4 = -2 * (corek*rr5i+valk*rr5ik-drk*rr7ik+qrrk*rr9ik);
    real term5 = -2 * (corei*rr5k+vali*rr5ik+dri*rr7ik+qrri*rr9ik);
    real term6 = 4 * rr7ik;
    rr3 = rr3ik;

    // compute the force components

    real frcx = de*xr + term1*dix + term2*dkx + term3*(diqkx-dkqix) + term4*qrix
            + term5*qrkx + term6*(qikrx+qkirx);
    real frcy = de*yr + term1*diy + term2*dky + term3*(diqky-dkqiy) + term4*qriy
            + term5*qrky + term6*(qikry+qkiry);
    real frcz = de*zr + term1*diz + term2*dkz + term3*(diqkz-dkqiz) + term4*qriz
            + term5*qrkz + term6*(qikrz+qkirz);

    // compute the torque components

    real ttmi[3];
    real ttmk[3];

    ttmi[0] = -rr3*dikx + term1*dirx
                          + term3*(dqiqkx+dkqixr)
                          - term4*qrixr - term6*(qikrxr+qrrx);
    ttmi[1] = -rr3*diky + term1*diry
                          + term3*(dqiqky+dkqiyr)
                          - term4*qriyr - term6*(qikryr+qrry);
    ttmi[2] = -rr3*dikz + term1*dirz
                          + term3*(dqiqkz+dkqizr)
                          - term4*qrizr - term6*(qikrzr+qrrz);
    ttmk[0] = rr3*dikx + term2*dkrx
                          - term3*(dqiqkx+diqkxr)
                          - term5*qrkxr - term6*(qkirxr-qrrx);
    ttmk[1] = rr3*diky + term2*dkry
                          - term3*(dqiqky+diqkyr)
                          - term5*qrkyr - term6*(qkiryr-qrry);
    ttmk[2] = rr3*dikz + term2*dkrz
                          - term3*(dqiqkz+diqkzr)
                          - term5*qrkzr - term6*(qkirzr-qrrz);

    //printf ("ttmi %12.6f%12.6f%12.6f%12.6f\n",ci,ttmi[0],ttmi[1],ttmi[2]);
    //printf ("ttmk %12.6f%12.6f%12.6f%12.6f\n",ck,ttmk[0],ttmk[1],ttmk[2]);
    


    atom1.force -= make_real3(frcx, frcy, frcz) * EPSILON_FACTOR;
    if (doubleCountingFactor == 1) {
        atom2.force += make_real3(frcx, frcy, frcz) * EPSILON_FACTOR;
    }

    atom1.torque += make_real3(ttmi[0],ttmi[1],ttmi[2]) * EPSILON_FACTOR;
    if (doubleCountingFactor == 1) {
        atom2.torque += make_real3(ttmk[0],ttmk[1],ttmk[2]) * EPSILON_FACTOR;
    }

    atom1.field += make_real3(fieldix, fieldiy, fieldiz);
    if (doubleCountingFactor == 1) {
        atom2.field += make_real3(fieldkx, fieldky, fieldkz);
    }

}
__device__ void computeOneDispersionInteraction(AtomData& atom1, AtomData& atom2, real cscale, real doubleCountingFactor, mixed& energyToBeAccumulated,
    real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ)
{
    // Compute the displacement.

    real3 delta;
    delta.x = atom2.pos.x - atom1.pos.x;
    delta.y = atom2.pos.y - atom1.pos.y;
    delta.z = atom2.pos.z - atom1.pos.z;
    APPLY_PERIODIC_TO_DELTA(delta)
    real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

    if (r2 > DISP_CUTOFF_SQUARED)
        return;

    real rInv = RSQRT(r2);
    real r = r2 * rInv;   
    real r6 = r2*r2*r2; 

    printf ("cutoff: %12.4f, taper: %12.4f, r = %12.4f\n ",DISP_CUTOFF_SQUARED, DISP_TAPER, r);


    real ci = atom1.csix;
    real ck = atom2.csix;

    real e = -ci * ck / r6;
    real de = -6.0f * e / r;

    real ai = atom1.palpha;
    real ak = atom2.palpha;

    real di = ai * r;
    real di2 = di * di;
    real di3 = di * di2;
    real dk = ak * r;
    real expi = EXP(-di);
    real expk = EXP(-dk);

    real ddamp = 0;
    real damp3 = 1;
    real damp5 = 1;

    if (ai == ak) {
        real di4 = di2 * di2;
        real di5 = di2 * di3;
        damp3 = 1 - (1+di+0.5f*di2+7.0f*di3/48.0f+di4/48.0f)*expi;
        damp5 = 1 - (1+di+0.5f*di2+di3/6.0f+di4/24.0f+di5/144.0f)*expi;
        ddamp = ai * expi * (di5-3.0f*di3-3.0f*di2)/96.0f;
    }
    else{
        real ai2 = ai * ai;
        real ai3 = ai * ai2;
        real ak2 = ak * ak;
        real ak3 = ak * ak2;
        real dk2 = dk * dk;
        real dk3 = dk * dk2;
        real ti = ak2 / (ak2-ai2);
        real tk = ai2 / (ai2-ak2);
        real ti2 = ti * ti;
        real tk2 = tk * tk;
        damp3 = 1 - ti2*(1+di+0.5f*di2)*expi - tk2*(1+dk+0.5f*dk2)*expk
                       - 2.0f*ti2*tk*(1+di)*expi - 2.0f*tk2*ti*(1+dk)*expk;
        damp5 = 1 - ti2*(1+di+0.5f*di2+di3/6.0f)*expi - tk2*(1+dk+0.5f*dk2+dk3/6.0f)*expk
                       - 2.0f*ti2*tk*(1.0+di+di2/3.0f)*expi - 2.0f*tk2*ti*(1.0+dk+dk2/3.0f)*expk;
        ddamp = 0.25f * di2 * ti2 * ai * expi * (r*ai+4.0f*tk-1) + 0.25f * dk2 * tk2 * ak * expk * (r*ak+4.0f*ti-1);
    }
    real damp = 1.5f*damp5 - 0.5f*damp3;
    real damp2 = damp*damp;

    de = (de*damp2 + 2*e*damp*ddamp)*cscale;

    real energy = e*damp2*cscale;

    printf ("energy = %12.4f and cscale = %12.4f\n", energy, cscale);

    if (r > DISP_TAPER) {
        real x = r - DISP_TAPER;
        real taper = 1 + x * x * x * (DISP_TAPER_C3 + x * (DISP_TAPER_C4 + x * DISP_TAPER_C5));
        real dtaper = x * x * (3 * DISP_TAPER_C3 + x * (4 * DISP_TAPER_C4 + x * 5 * DISP_TAPER_C5));
        de = energy * dtaper + de * taper;
        energy *= taper;
    }

    // accumulate energy and force

    energyToBeAccumulated += (mixed)doubleCountingFactor * energy;

    real frcx = de * -delta.x/r;
    real frcy = de * -delta.y/r;
    real frcz = de * -delta.z/r;

    atom1.force -= make_real3(frcx, frcy, frcz);
    if (doubleCountingFactor == 1) {
        atom2.force += make_real3(frcx, frcy, frcz);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DEBUG_PRINT(AA2) { \
    printf("atom c,d %12d%12.6f%12.6f%12.6f%12.6f\n" \
           "q        %12.6f%12.6f%12.6f%12.6f%12.6f\n" \
           "chg dmp  %12.6f%12.6f%12.6f\n" \
           "fx fy xz %18.6e%18.6e%12.8e\n" \
        ,AA2+1,(float)posq[AA2].w,(float)dpl[3*AA2],(float)dpl[3*AA2+1],(float)dpl[3*AA2+2] \
        ,(float)quad[5*AA2],(float)quad[5*AA2+1],(float)quad[5*AA2+2],(float)quad[5*AA2+3],(float)quad[5*AA2+4] \
        ,(float)sizpr[AA2],(float)dmppr[AA2],(float)elepr[AA2] \
        ,(float)forceBuffers[3*AA2],(float)forceBuffers[3*AA2+1],(float)forceBuffers[3*AA2+2]); }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void computeChargeTransfer(unsigned long long* __restrict__ forceBuffers, unsigned long long* __restrict__ torqueBuffers, mixed* __restrict__ energyBuffer,
    unsigned long long* __restrict__ fieldBuffers, const real4* __restrict__ posq, const uint2* __restrict__ covalentFlags, const ushort2* __restrict__ exclusionTiles, unsigned int startTileIndex,
    unsigned int numTileIndices,
#ifdef USE_CUTOFF
    const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX,
    real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
    const unsigned int* __restrict__ interactingAtoms,
#endif
    const real* __restrict__ chgct, const real* __restrict__ dmpct
    ,const real* __restrict__ dpl ,const real* __restrict__ quad
    ,const real* __restrict__ sizpr ,const real* __restrict__ dmppr ,const real* __restrict__ elepr
    ,const real* __restrict__ pcore ,const real* __restrict__ pval, const real* __restrict__ palpha
    ,const real* __restrict__ csix 
    )
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
        loadAtomData(data, atom1, posq, chgct, dmpct, dpl, quad, sizpr, elepr, dmppr,pcore, pval, palpha, csix);
        data.force = make_real3(0);
        data.torque = make_real3(0);
        data.field = make_real3(0);
        uint2 covalent = covalentFlags[pos * TILE_SIZE + tgx];

        printf("atom = %d, csix = %12.4f\n", atom1+1, csix[atom1]);
//        printf("atom = %d, pcore = %12.4f, palpha = %12.4f, dipole = %12.4f%12.4f%12.4f\n", atom1+1, data.pcore, data.palpha, data.dipole.x, data.dipole.y, data.dipole.z);

        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].pos = data.pos;
            localData[threadIdx.x].dmpct = data.dmpct;
            localData[threadIdx.x].chgct = data.chgct;
            localData[threadIdx.x].q = data.q;
            localData[threadIdx.x].sizpr = data.sizpr;
            localData[threadIdx.x].elepr = data.elepr;
            localData[threadIdx.x].dmppr = data.dmppr;
            localData[threadIdx.x].dipole = data.dipole;
            localData[threadIdx.x].quadrupole[0] = data.quadrupole[0];
            localData[threadIdx.x].quadrupole[1] = data.quadrupole[1];
            localData[threadIdx.x].quadrupole[2] = data.quadrupole[2];
            localData[threadIdx.x].quadrupole[3] = data.quadrupole[3];
            localData[threadIdx.x].quadrupole[4] = data.quadrupole[4];
            localData[threadIdx.x].pcore = data.pcore;
            localData[threadIdx.x].pval = data.pval;
            localData[threadIdx.x].palpha = data.palpha;
            localData[threadIdx.x].csix = data.csix;

            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real c = computeCScaleFactor(covalent, j);
                    computeOneInteraction(
                        data, localData[tbx + j], c, (real)0.5, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneRepelInteraction(
                        data, localData[tbx + j], c, (real)0.5, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneChargePenetrationInteraction(
                        data, localData[tbx + j], c, (real)0.5, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneDispersionInteraction(
                        data, localData[tbx + j], c, (real)0.5, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    //printf (" x==y atom = %d atom = %d, energy = %12.4f, palphas: %12.4f%12.4f\n", atom1+1, atom2+1, energy, palpha[atom1], palpha[atom2]);

                }  
//                printf ("i = %d k = %d torque: %12.6f%12.6f%12.6f\n", atom1+1,atom2+1, data.torque); 
//                printf(" x == y atom = %d atom = %d, pcore %12.4f, palpha %12.4f, energy %12.4f\n", atom1+1, atom2+1, pcore[atom1], palpha[atom1], energy);
            }

            // In this block we are double counting, so we only accumulate force on atom1

            //printf ("atom1 = %d torque: %12.6f%12.6f%12.6f\n",atom1+1,data.torque.x,data.torque.y,data.torque.z);
            //data.force *= EPSILON_FACTOR;
            //data.torque *= EPSILON_FACTOR;
            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long)(data.force.x * 0x100000000)));
            atomicAdd(&forceBuffers[atom1 + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.y * 0x100000000)));
            atomicAdd(&forceBuffers[atom1 + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(data.force.z * 0x100000000)));
 
            atomicAdd(&torqueBuffers[atom1], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));

            atomicAdd(&fieldBuffers[atom1], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
            atomicAdd(&fieldBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
            atomicAdd(&fieldBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));

            //printf ("field at atom: %d : %12.4f%12.4f%12.4f\n",atom1+1,data.field.x,data.field.y,data.field.z);

        } else {

            // This is an off-diagonal tile.
            unsigned int j = y * TILE_SIZE + tgx;
//            loadAtomData(localData[threadIdx.x], j, posq, chgct, dmpct);
            loadAtomData(localData[threadIdx.x], j, posq, chgct, dmpct, dpl, quad, sizpr, elepr, dmppr,pcore, pval, palpha, csix);
            localData[threadIdx.x].force = make_real3(0);
            localData[threadIdx.x].torque = make_real3(0);
            localData[threadIdx.x].field = make_real3(0);

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    float c = computeCScaleFactor(covalent, tj);
                    computeOneInteraction(
                        data, localData[tbx + tj], c, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneRepelInteraction(
                        data, localData[tbx + tj], 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneChargePenetrationInteraction( 
                        data, localData[tbx + tj], c, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneDispersionInteraction(
                        data, localData[tbx + tj], c, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
    
                }
                printf(" x != y atom = %d atom = %d, pcore %12.4f, palpha %12.4f\n", atom1+1, atom2+1, pcore[atom1], palpha[atom1]);
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
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));
            offset = y * TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.x * 0x100000000)));
            atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.y * 0x100000000)));
            atomicAdd(&forceBuffers[offset + 2 * PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(localData[threadIdx.x].force.z * 0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.z*0x100000000)));
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
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

            AtomData data;
//            loadAtomData(data, atom1, posq, chgct, dmpct);
            loadAtomData(data, atom1, posq, chgct, dmpct, dpl, quad, sizpr, elepr, dmppr, pcore, pval, palpha, csix);
            data.force = make_real3(0);
            data.torque = make_real3(0);
            data.field = make_real3(0);

#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos * TILE_SIZE + tgx];
#else
            unsigned int j = y * TILE_SIZE + tgx;
#endif

            atomIndices[threadIdx.x] = j;
//            loadAtomData(localData[threadIdx.x], j, posq, chgct, dmpct);
            loadAtomData(localData[threadIdx.x], j, posq, chgct, dmpct, dpl, quad,sizpr, elepr, dmppr, pcore, pval, palpha, csix);
            localData[threadIdx.x].force = make_real3(0);
            localData[threadIdx.x].torque = make_real3(0);
            localData[threadIdx.x].field = make_real3(0);

            // Compute forces.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = atomIndices[tbx + tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    computeOneInteraction(
                        data, localData[tbx + tj], 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneRepelInteraction(
                        data, localData[tbx + tj], 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneChargePenetrationInteraction(
                        data, localData[tbx + tj], 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                    computeOneDispersionInteraction(
                        data, localData[tbx + tj], 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
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
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));

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
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));

        }
        pos++;
    }
    energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] += energy;
}

 inline __device__ real normVector(real3& v) {
    real n = SQRT(dot(v, v));
    v *= (n > 0 ? RECIP(n) : 0);
    return n;
}


extern "C" __global__ void mapTorqueToForce(unsigned long long* __restrict__ forceBuffers, const long long* __restrict__ torqueBuffers,
    const real4* __restrict__ posq, const int4* __restrict__ axisInfo) {
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

const real torqueScale = RECIP((double) 0x100000000);

real3 forces[4];
real norms[LastVectorIndex];
real3 vector[LastVectorIndex];
real angles[LastVectorIndex][2];

for (int atom = blockIdx.x*blockDim.x + threadIdx.x; atom < NUM_ATOMS; atom += gridDim.x*blockDim.x) {
    int4 particles = axisInfo[atom];
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
        }
        else {
            vector[V].x = 1;
            vector[V].y = 0;
            vector[V].z = 0;
            if (abs(vector[U].x/norms[U]) > 0.866) {
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
        angles[UV][1] = SQRT(1 - angles[UV][0]*angles[UV][0]);

        angles[UW][0] = dot(vector[U], vector[W]);
        angles[UW][1] = SQRT(1 - angles[UW][0]*angles[UW][0]);

        angles[VW][0] = dot(vector[V], vector[W]);
        angles[VW][1] = SQRT(1 - angles[VW][0]*angles[VW][0]);

        real dphi[3];
        real3 torque = make_real3(torqueScale*torqueBuffers[atom], torqueScale*torqueBuffers[atom+PADDED_NUM_ATOMS], torqueScale*torqueBuffers[atom+PADDED_NUM_ATOMS*2]);
        dphi[U] = -dot(vector[U], torque);
        dphi[V] = -dot(vector[V], torque);
        dphi[W] = -dot(vector[W], torque);

        // z-then-x and bisector

        if (axisType == 0 || axisType == 1) {
            real factor1 = dphi[V]/(norms[U]*angles[UV][1]);
            real factor2 = dphi[W]/(norms[U]);
            real factor3 = -dphi[U]/(norms[V]*angles[UV][1]);
            real factor4 = 0;
            if (axisType == 1) {
                factor2 *= 0.5f;
                factor4 = 0.5f*dphi[W]/(norms[V]);
            }
            forces[Z] = vector[UV]*factor1 + factor2*vector[UW];
            forces[X] = vector[UV]*factor3 + factor4*vector[VW];
            forces[I] = -(forces[X]+forces[Z]);
            forces[Y] = make_real3(0);
        }
        else if (axisType == 2) {
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
            angles[UR][1] = SQRT(1 - angles[UR][0]*angles[UR][0]);

            angles[US][0] = dot(vector[U], vector[S]);
            angles[US][1] = SQRT(1 - angles[US][0]*angles[US][0]);

            angles[VS][0] = dot(vector[V], vector[S]);
            angles[VS][1] = SQRT(1 - angles[VS][0]*angles[VS][0]);

            angles[WS][0] = dot(vector[W], vector[S]);
            angles[WS][1] = SQRT(1 - angles[WS][0]*angles[WS][0]);

            real3 t1 = vector[V] - vector[S]*angles[VS][0];
            real3 t2 = vector[W] - vector[S]*angles[WS][0];
            normVector(t1);
            normVector(t2);
            real ut1cos = dot(vector[U], t1);
            real ut1sin = SQRT(1 - ut1cos*ut1cos);
            real ut2cos = dot(vector[U], t2);
            real ut2sin = SQRT(1 - ut2cos*ut2cos);

            real dphiR = -dot(vector[R], torque);
            real dphiS = -dot(vector[S], torque);

            real factor1 = dphiR/(norms[U]*angles[UR][1]);
            real factor2 = dphiS/(norms[U]);
            real factor3 = dphi[U]/(norms[V]*(ut1sin+ut2sin));
            real factor4 = dphi[U]/(norms[W]*(ut1sin+ut2sin));
            forces[Z] = vector[UR]*factor1 + factor2*vector[US];
            forces[X] = (angles[VS][1]*vector[S] - angles[VS][0]*t1)*factor3;
            forces[Y] = (angles[WS][1]*vector[S] - angles[WS][0]*t2)*factor4;
            forces[I] = -(forces[X] + forces[Y] + forces[Z]);
        }
        else if (axisType == 3) {
            // 3-fold

            forces[Z] = (vector[UW]*dphi[W]/(norms[U]*angles[UW][1]) +
                        vector[UV]*dphi[V]/(norms[U]*angles[UV][1]) -
                        vector[UW]*dphi[U]/(norms[U]*angles[UW][1]) -
                        vector[UV]*dphi[U]/(norms[U]*angles[UV][1]))/3;

            forces[X] = (vector[VW]*dphi[W]/(norms[V]*angles[VW][1]) -
                        vector[UV]*dphi[U]/(norms[V]*angles[UV][1]) -
                        vector[VW]*dphi[V]/(norms[V]*angles[VW][1]) +
                        vector[UV]*dphi[V]/(norms[V]*angles[UV][1]))/3;

            forces[Y] = (-vector[UW]*dphi[U]/(norms[W]*angles[UW][1]) -
                        vector[VW]*dphi[V]/(norms[W]*angles[VW][1]) +
                        vector[UW]*dphi[W]/(norms[W]*angles[UW][1]) +
                        vector[VW]*dphi[W]/(norms[W]*angles[VW][1]))/3;
            forces[I] = -(forces[X] + forces[Y] + forces[Z]);
        }
        else if (axisType == 4) {
            // z-only

            forces[Z] = vector[UV]*dphi[V]/(norms[U]*angles[UV][1]) + vector[UW]*dphi[W]/norms[U];
            forces[X] = make_real3(0);
            forces[Y] = make_real3(0);
            forces[I] = -forces[Z];
        }
        else {
            forces[Z] = make_real3(0);
            forces[X] = make_real3(0);
            forces[Y] = make_real3(0);
            forces[I] = make_real3(0);
        }

        //printf ("atom: %d forcesZ: %12.6f%12.6f%12.6f \n",atom,forces[Z].x,forces[Z].y,forces[Z].z);
        //printf ("atom: %d forcesX: %12.6f%12.6f%12.6f \n",atom,forces[X].x,forces[X].y,forces[X].z);
        //printf ("atom: %d forcesY: %12.6f%12.6f%12.6f \n",atom,forces[Y].x,forces[Y].y,forces[Y].z);



        // Store results

        atomicAdd(&forceBuffers[particles.z], static_cast<unsigned long long>((long long) (forces[Z].x*0x100000000)));
        atomicAdd(&forceBuffers[particles.z+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[Z].y*0x100000000)));
        atomicAdd(&forceBuffers[particles.z+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[Z].z*0x100000000)));
        if (axisType != 4) {
            atomicAdd(&forceBuffers[particles.x], static_cast<unsigned long long>((long long) (forces[X].x*0x100000000)));
            atomicAdd(&forceBuffers[particles.x+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[X].y*0x100000000)));
            atomicAdd(&forceBuffers[particles.x+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[X].z*0x100000000)));
        }
        if ((axisType == 2 || axisType == 3) && particles.y > -1) {
            atomicAdd(&forceBuffers[particles.y], static_cast<unsigned long long>((long long) (forces[Y].x*0x100000000)));
            atomicAdd(&forceBuffers[particles.y+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[Y].y*0x100000000)));
            atomicAdd(&forceBuffers[particles.y+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[Y].z*0x100000000)));
        }
        atomicAdd(&forceBuffers[atom], static_cast<unsigned long long>((long long) (forces[I].x*0x100000000)));
        atomicAdd(&forceBuffers[atom+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[I].y*0x100000000)));
        atomicAdd(&forceBuffers[atom+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forces[I].z*0x100000000)));
    }
}
}
