package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 *
 */
@Platform(include = "NativeLapack.h", compiler = "cpp11", link = "nd4j", library = "jnind4j")
public class NativeLapack extends Pointer {


    native int dgbbrd(int matriint, char vect, int m, int n, int ncc, int kl, int ku, long ab, int ldab,
                      long d, long e, long q, int ldq,
                      long pt, int ldpt, long c, int ldc );
    native int cgbbrd(int matriint, char vect, int m, int n, int ncc, int kl, int ku, IComplexNumber ab, int ldab, long d, long e,
                      IComplexNumber q, int ldq,
                      IComplexNumber pt, int ldpt,
                      IComplexNumber c, int ldc );
    native int zgbbrd(int matriint, char vect, int m,int n, int ncc, int kl,int ku, IComplexNumber ab,int ldab, long d, long e,
                      IComplexNumber q, int ldq,
                      IComplexNumber pt, int ldpt,
                      IComplexNumber c, int ldc );

    native int sgbcon(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv, float anorm,
                      long rcond );
    native int dgbcon(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv,
                      double anorm, long rcond );
    native int cgbcon(int matriint, char norm, int n,int kl, int ku,
                      IComplexNumber ab, int ldab,int ipiv, float anorm, long rcond );
    native int zgbcon(int matriint, char norm, int n,int kl, int ku,
                      IComplexNumber ab, int ldab,int ipiv, double anorm,
                      long rcond );

    native int sgbequ(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c, long rowcnd,
                      long colcnd, long amax );
    native int dgbequ(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                      long rowcnd, long colcnd, long amax );
    native int cgbequ(int matriint, int m, int n,int kl, int ku,
                      IComplexNumber ab, int ldab,
                      long r, long c, long rowcnd, long colcnd,
                      long amax );
    native int zgbequ(int matriint, int m, int n,int kl, int ku,
                      IComplexNumber ab, int ldab,
                      long r, long c, long rowcnd, long colcnd,
                      long amax );

    native int sgbequb(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c, long rowcnd,
                       long colcnd, long amax );
    native int dgbequb(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                       long rowcnd, long colcnd, long amax );
    native int cgbequb(int matriint, int m, int n,int kl, int ku,
                       IComplexNumber ab, int ldab,
                       long r, long c, long rowcnd, long colcnd,
                       long amax );
    native int zgbequb(int matriint, int m, int n,int kl, int ku,
                       IComplexNumber ab, int ldab,
                       long r, long c, long rowcnd,
                       long colcnd, long amax );

    native int sgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      long ab, int ldab,  long afb,int ldafb,int ipiv,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int dgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      long ab, int ldab,  long afb,int ldafb,int ipiv,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int cgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber afb, int ldafb,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zgbrfs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber afb, int ldafb,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int sgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab, int ldab,
                       long afb, int ldafb,int ipiv,  long r,
                       long c,  long b, int ldb,
                       long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int dgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab, int ldab,
                       long afb, int ldafb,int ipiv,  long r,
                       long c,  long b, int ldb,
                       long x, int ldx, long rcond,
                       long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int cgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                       long r,  long c,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int zgbrfsx(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                       long r,  long c,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );

    native int sgbsv(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
    native int dgbsv(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
    native int cgbsv(int matriint, int n, int kl,int ku, int nrhs,
                     IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
    native int zgbsv(int matriint, int n, int kl,int ku, int nrhs,
                     IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );

    native int sgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                      long afb, int ldafb, int ipiv,
                      String equed, long r, long c, long b,int ldb, long x, int ldx,
                      long rcond, long ferr, long berr,
                      long rpivot );
    native int dgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                      long afb, int ldafb, int ipiv,
                      String equed, long r, long c, long b,int ldb, long x, int ldx,
                      long rcond, long ferr, long berr,
                      long rpivot );
    native int cgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                      long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                      long berr, long rpivot );
    native int zgbsvx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                      long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                      long berr, long rpivot );

    native int sgbsvxx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                       long afb, int ldafb, int ipiv,
                       String equed, long r, long c, long b,int ldb, long x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,long params );
    native int dgbsvxx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                       long afb, int ldafb, int ipiv,
                       String equed, long r, long c, long b,int ldb, long x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                                      long params );
    native int cgbsvxx(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                       long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                       long berr, int int_bnds,
                                       long erint_norm, long erint_comp,int nparams, long params );
    native int sgbtrf(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
    native int dgbtrf(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
    native int cgbtrf(int matriint, int m, int n,int kl, int ku,
                      IComplexNumber ab, int ldab,int ipiv );
    native int zgbtrf(int matriint, int m, int n,int kl, int ku,
                      IComplexNumber ab, int ldab,int ipiv );

    native int sgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      long ab, int ldab,int ipiv, long b, int ldb );
    native int dgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      long ab, int ldab,int ipiv, long b, int ldb );
    native int cgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
    native int zgbtrs(int matriint, char trans, int n,int kl, int ku, int nrhs,
                      IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );

    native int sgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, long v, int ldv );
    native int dgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, long v, int ldv );
    native int cgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, IComplexNumber v,int ldv );
    native int zgebak(int matriint, char job, char side, int n,int ilo, int ihi,  long scale,int m, IComplexNumber v,int ldv );

    native int sgebal(int matriint, char job, int n, long a,int lda, int ilo, int ihi,
                      long scale );
    native int dgebal(int matriint, char job, int n, long a,int lda, int ilo, int ihi,
                      long scale );
    native int cgebal(int matriint, char job, int n,
                      IComplexNumber a, int lda,int ilo, int ihi, long scale );
    native int zgebal(int matriint, char job, int n,
                      IComplexNumber a, int lda,int ilo, int ihi, long scale );

    native int sgebrd(int matriint, int m, int n,
                      long a, int lda, long d, long e,
                      long tauq, long taup );
    native int dgebrd(int matriint, int m, int n,
                      long a, int lda, long d, long e,
                      long tauq, long taup );
    native int cgebrd(int matriint, int m, int n,
                      IComplexNumber a, int lda, long d,
                      long e, IComplexNumber tauq,
                      IComplexNumber taup );
    native int zgebrd(int matriint, int m, int n,
                      IComplexNumber a, int lda, long d,
                      long e, IComplexNumber tauq,
                      IComplexNumber taup );

    native int sgecon(int matriint, char norm, int n,
                      long a, int lda, float anorm,
                      long rcond );
    native int dgecon(int matriint, char norm, int n,
                      long a, int lda, double anorm,
                      long rcond );
    native int cgecon(int matriint, char norm, int n,
                      IComplexNumber a, int lda,
                      float anorm, long rcond );
    native int zgecon(int matriint, char norm, int n,
                      IComplexNumber a, int lda,
                      double anorm, long rcond );

    native int sgeequ(int matriint, int m, int n,
                      long a, int lda, long r, long c,
                      long rowcnd, long colcnd, long amax );
    native int dgeequ(int matriint, int m, int n,
                      long a, int lda, long r,
                      long c, long rowcnd, long colcnd,
                      long amax );
    native int cgeequ(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      long r, long c, long rowcnd, long colcnd,
                      long amax );
    native int zgeequ(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      long r, long c, long rowcnd, long colcnd,
                      long amax );

    native int sgeequb(int matriint, int m, int n,
                       long a, int lda, long r, long c,
                       long rowcnd, long colcnd, long amax );
    native int dgeequb(int matriint, int m, int n,
                       long a, int lda, long r,
                       long c, long rowcnd, long colcnd,
                       long amax );
    native int cgeequb(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       long r, long c, long rowcnd, long colcnd,
                       long amax );
    native int zgeequb(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       long r, long c, long rowcnd,
                       long colcnd, long amax );

    native int sgees(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                     long wi, long vs, int ldvs );
    native int dgees(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                     long wi, long vs, int ldvs );
    native int cgees(int matriint, char jobvs, char sort,int select, int n,
                     IComplexNumber a, int lda,int sdim, IComplexNumber w,
                     IComplexNumber vs, int ldvs );
    native int zgees(int matriint, char jobvs, char sort,int select, int n,
                     IComplexNumber a, int lda,int sdim, IComplexNumber w,
                     IComplexNumber vs, int ldvs );

    native int sgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                      long a, int lda, int sdim,
                      long wr, long wi, long vs, int ldvs,
                      long rconde, long rcondv );
    native int dgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                      long a, int lda, int sdim,
                      long wr, long wi, long vs, int ldvs,
                      long rconde, long rcondv );
    native int cgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                      IComplexNumber a, int lda,int sdim, IComplexNumber w,
                      IComplexNumber vs, int ldvs,
                      long rconde, long rcondv );
    native int zgeesx(int matriint, char jobvs, char sort,int select, char sense, int n,
                      IComplexNumber a, int lda,int sdim, IComplexNumber w,
                      IComplexNumber vs, int ldvs,
                      long rconde, long rcondv );

    native int sgeev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long wr,
                     long wi, long vl, int ldvl, long vr,int ldvr );
    native int dgeev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long wr,
                     long wi, long vl, int ldvl, long vr,int ldvr );
    native int cgeev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a, int lda,
                     IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr );
    native int zgeev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber w,
                     IComplexNumber vl, int ldvl,
                     IComplexNumber vr, int ldvr );

    native int sgeevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n, long a,int lda, long wr, long wi, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long scale,
                      long abnrm, long rconde, long rcondv );
    native int dgeevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n, long a,int lda, long wr, long wi, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long scale,
                      long abnrm, long rconde, long rcondv );
    native int cgeevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int ilo, int ihi,
                      long scale, long abnrm, long rconde,
                      long rcondv );
    native int zgeevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int ilo, int ihi,
                      long scale, long abnrm, long rconde,
                      long rcondv );

    native int sgehrd(int matriint, int n, int ilo,int ihi, long a, int lda,
                      long tau );
    native int dgehrd(int matriint, int n, int ilo,int ihi, long a, int lda,
                      long tau );
    native int cgehrd(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau );
    native int zgehrd(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau );

    native int sgejsv(int matriint, char joba, char jobu, char jobv,
                      char jobr, char jobt, char jobp, int m,int n, long a, int lda, long sva,
                      long u, int ldu, long v, int ldv,
                      long stat, int istat );
    native int dgejsv(int matriint, char joba, char jobu, char jobv,
                      char jobr, char jobt, char jobp, int m,int n, long a, int lda, long sva,
                      long u, int ldu, long v, int ldv,
                      long stat, int istat );
    native int cgejsv(int matriint, char joba, char jobu, char jobv,
                      char jobr, char jobt, char jobp, int m,int n, IComplexNumber a, int lda, long sva,
                      IComplexNumber u, int ldu, IComplexNumber v, int ldv,
                      long stat, int istat );
    native int zgejsv(int matriint, char joba, char jobu, char jobv,
                      char jobr, char jobt, char jobp, int m,int n, IComplexNumber a, int lda, long sva,
                      IComplexNumber u, int ldu, IComplexNumber v, int ldv,
                      long stat, int istat );

    native int sgelq2(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dgelq2(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int cgelq2(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zgelq2(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int sgelqf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dgelqf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int cgelqf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zgelqf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int sgels(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb );
    native int dgels(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb );
    native int cgels(int matriint, char trans, int m,int n, int nrhs,
                     IComplexNumber a, int lda,
                     IComplexNumber b, int ldb );
    native int zgels(int matriint, char trans, int m,int n, int nrhs,
                     IComplexNumber a, int lda,
                     IComplexNumber b, int ldb );

    native int sgelsd(int matriint, int m, int n,int nrhs, long a, int lda, long b,int ldb, long s, float rcond,int rank );
    native int dgelsd(int matriint, int m, int n,int nrhs, long a, int lda,
                      long b, int ldb, long s, double rcond,int rank );
    native int cgelsd(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank );
    native int zgelsd(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank );

    native int sgelss(int matriint, int m, int n,int nrhs, long a, int lda, long b,int ldb, long s, float rcond,int rank );
    native int dgelss(int matriint, int m, int n,int nrhs, long a, int lda,
                      long b, int ldb, long s, double rcond,int rank );
    native int cgelss(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank );
    native int zgelss(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank );

    native int sgelsy(int matriint, int m, int n,int nrhs, long a, int lda, long b,int ldb, int jpvt, float rcond,int rank );
    native int dgelsy(int matriint, int m, int n,int nrhs, long a, int lda,
                      long b, int ldb, int jpvt,
                      double rcond, int rank );
    native int cgelsy(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, float rcond,int rank );
    native int zgelsy(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, double rcond,int rank );

    native int sgeqlf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dgeqlf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int cgeqlf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zgeqlf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int sgeqp3(int matriint, int m, int n,
                      long a, int lda, int jpvt,
                      long tau );
    native int dgeqp3(int matriint, int m, int n,
                      long a, int lda, int jpvt,
                      long tau );
    native int cgeqp3(int matriint, int m, int n,
                      IComplexNumber a, int lda,int jpvt, IComplexNumber tau );
    native int zgeqp3(int matriint, int m, int n,
                      IComplexNumber a, int lda,int jpvt, IComplexNumber tau );

    native int sgeqpf(int matriint, int m, int n,
                      long a, int lda, int jpvt,
                      long tau );
    native int dgeqpf(int matriint, int m, int n,
                      long a, int lda, int jpvt,
                      long tau );
    native int cgeqpf(int matriint, int m, int n,
                      IComplexNumber a, int lda,int jpvt, IComplexNumber tau );
    native int zgeqpf(int matriint, int m, int n,
                      IComplexNumber a, int lda,int jpvt, IComplexNumber tau );

    native int sgeqr2(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dgeqr2(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int cgeqr2(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zgeqr2(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int sgeqrf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dgeqrf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int cgeqrf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zgeqrf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int sgeqrfp(int matriint, int m, int n,
                       long a, int lda, long tau );
    native int dgeqrfp(int matriint, int m, int n,
                       long a, int lda, long tau );
    native int cgeqrfp(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    native int zgeqrfp(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    native int sgerfs(int matriint, char trans, int n,int nrhs,  long a, int lda,
                      long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int dgerfs(int matriint, char trans, int n,int nrhs,  long a, int lda,
                      long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int cgerfs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zgerfs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int sgerfsx(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long r,
                       long c,  long b, int ldb,
                       long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int dgerfsx(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long r,
                       long c,  long b, int ldb,
                       long x, int ldx, long rcond,
                       long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int cgerfsx(int matriint, char trans, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv,  long r,
                       long c,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int zgerfsx(int matriint, char trans, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv,  long r,
                       long c,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );

    native int sgerqf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dgerqf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int cgerqf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zgerqf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int sgesdd(int matriint, char jobz, int m,int n, long a, int lda, long s,
                      long u, int ldu, long vt,int ldvt );
    native int dgesdd(int matriint, char jobz, int m,int n, long a, int lda, long s,
                      long u, int ldu, long vt,int ldvt );
    native int cgesdd(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt );
    native int zgesdd(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt );

    native int sgesv(int matriint, int n, int nrhs,
                     long a, int lda, int ipiv, long b,int ldb );
    native int dgesv(int matriint, int n, int nrhs,
                     long a, int lda, int ipiv,
                     long b, int ldb );
    native int cgesv(int matriint, int n, int nrhs,
                     IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
    native int zgesv(int matriint, int n, int nrhs,
                     IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
    native int dsgesv(int matriint, int n, int nrhs,
                      long a, int lda, int ipiv,
                      long b, int ldb, long x, int ldx,int iter );
    native int zcgesv(int matriint, int n, int nrhs,
                      IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, IComplexNumber x,int ldx, int iter );

    native int sgesvd(int matriint, char jobu, char jobvt,int m, int n, long a, int lda,
                      long s, long u, int ldu, long vt,int ldvt, long superb );
    native int dgesvd(int matriint, char jobu, char jobvt,int m, int n, long a,int lda, long s, long u, int ldu,
                      long vt, int ldvt, long superb );
    native int cgesvd(int matriint, char jobu, char jobvt,int m, int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, long superb );
    native int zgesvd(int matriint, char jobu, char jobvt,int m, int n, IComplexNumber a,int lda, long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, long superb );

    native int sgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                       long s, long u, int ldu,
                       long vt, int ldvt,int superb );
    native int dgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                       long s, long u, int ldu,
                       long vt, int ldvt,int superb );
    native int cgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                       long s, IComplexNumber u, int ldu,
                       IComplexNumber vt, int ldvt,int superb );
    native int zgesvdx(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                       long s, IComplexNumber u, int ldu,
                       IComplexNumber vt, int ldvt,int superb );

    native int sgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, long a, int lda,
                      long sva, int mv, long v, int ldv,
                      long stat );
    native int dgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, long a,int lda, long sva, int mv,
                      long v, int ldv, long stat );
    native int cgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, IComplexNumber a,int lda, long sva, int mv,
                      IComplexNumber v, int ldv, long stat );
    native int zgesvj(int matriint, char joba, char jobu, char jobv,int m, int n, IComplexNumber a,int lda, long sva, int mv,
                      IComplexNumber v, int ldv, long stat );

    native int sgesvx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                      long b, int ldb, long x, int ldx,
                      long rcond, long ferr, long berr,
                      long rpivot );
    native int dgesvx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                      long b, int ldb, long x, int ldx,
                      long rcond, long ferr, long berr,
                      long rpivot );
    native int cgesvx(int matriint, char fact, char trans,int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr,
                      long rpivot );
    native int zgesvx(int matriint, char fact, char trans,int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr,
                      long rpivot );

    native int sgesvxx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                       long b, int ldb, long x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int dgesvxx(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                       long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                       long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int cgesvxx(int matriint, char fact, char trans,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int zgesvxx(int matriint, char fact, char trans,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );

    native int sgetf2(int matriint, int m, int n,
                      long a, int lda, int ipiv );
    native int dgetf2(int matriint, int m, int n,
                      long a, int lda, int ipiv );
    native int cgetf2(int matriint, int m, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zgetf2(int matriint, int m, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int sgetrf(int matriint, int m, int n,
                      long a, int lda, int ipiv );
    native int dgetrf(int matriint, int m, int n,
                      long a, int lda, int ipiv );
    native int cgetrf(int matriint, int m, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zgetrf(int matriint, int m, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int sgetrf2(int matriint, int m, int n,
                       long a, int lda, int ipiv );
    native int dgetrf2(int matriint, int m, int n,
                       long a, int lda, int ipiv );
    native int cgetrf2(int matriint, int m, int n,
                       IComplexNumber a, int lda,int ipiv );
    native int zgetrf2(int matriint, int m, int n,
                       IComplexNumber a, int lda,int ipiv );

    native int sgetri(int matriint, int n, long a,int lda,int ipiv );
    native int dgetri(int matriint, int n, long a,int lda,int ipiv );
    native int cgetri(int matriint, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zgetri(int matriint, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int sgetrs(int matriint, char trans, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int dgetrs(int matriint, char trans, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int cgetrs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                      IComplexNumber b, int ldb );
    native int zgetrs(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                      IComplexNumber b, int ldb );

    native int sggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                      long rscale, int m, long v,int ldv );
    native int dggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                      long rscale, int m, long v,int ldv );
    native int cggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                      long rscale, int m,
                      IComplexNumber v, int ldv );
    native int zggbak(int matriint, char job, char side, int n,int ilo, int ihi,  long lscale,
                      long rscale, int m,
                      IComplexNumber v, int ldv );

    native int sggbal(int matriint, char job, int n, long a,int lda, long b, int ldb,int ilo, int ihi, long lscale,
                      long rscale );
    native int dggbal(int matriint, char job, int n, long a,int lda, long b, int ldb,int ilo, int ihi, long lscale,
                      long rscale );
    native int cggbal(int matriint, char job, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,int ilo, int ihi, long lscale,
                      long rscale );
    native int zggbal(int matriint, char job, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,int ilo, int ihi, long lscale,
                      long rscale );

    native int sgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n, long a,int lda, long b, int ldb,int sdim, long alphar, long alphai,
                     long beta, long vsl, int ldvsl, long vsr,int ldvsr );
    native int dgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n, long a,int lda, long b, int ldb,int sdim, long alphar, long alphai,
                     long beta, long vsl, int ldvsl,
                     long vsr, int ldvsr );
    native int cgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n,
                     IComplexNumber a, int lda,
                     IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                     IComplexNumber beta, IComplexNumber vsl,int ldvsl, IComplexNumber vsr,int ldvsr );
    native int zgges(int matriint, char jobvsl, char jobvsr, char sort,int selctg, int n,
                     IComplexNumber a, int lda,
                     IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                     IComplexNumber beta,
                     IComplexNumber vsl, int ldvsl,
                     IComplexNumber vsr, int ldvsr );

    native int sgges3(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, int n,
                      long a, int lda, long b, int ldb,int sdim, long alphar, long alphai,
                      long beta, long vsl, int ldvsl,
                      long vsr, int ldvsr );
    native int dgges3(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, int n,
                      long a, int lda, long b, int ldb,int sdim, long alphar, long alphai,
                      long beta, long vsl, int ldvsl,
                      long vsr, int ldvsr );
    native int cgges3(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vsl, int ldvsl,
                      IComplexNumber vsr, int ldvsr );
    native int zgges3(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vsl, int ldvsl,
                      IComplexNumber vsr, int ldvsr );

    native int sggesx(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, char sense,int n, long a, int lda, long b,int ldb, int sdim, long alphar,
                      long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                      long rconde, long rcondv );
    native int dggesx(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, char sense,int n, long a, int lda, long b,int ldb, int sdim, long alphar,
                      long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                      long rconde, long rcondv );
    native int cggesx(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vsl, int ldvsl,
                      IComplexNumber vsr, int ldvsr,
                      long rconde, long rcondv );
    native int zggesx(int matriint, char jobvsl, char jobvsr,
                      char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vsl, int ldvsl,
                      IComplexNumber vsr, int ldvsr,
                      long rconde, long rcondv );

    native int sggev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long b,int ldb, long alphar, long alphai,
                     long beta, long vl, int ldvl, long vr,int ldvr );
    native int dggev(int matriint, char jobvl, char jobvr,int n, long a, int lda, long b,int ldb, long alphar, long alphai,
                     long beta, long vl, int ldvl, long vr,int ldvr );
    native int cggev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a, int lda,
                     IComplexNumber b, int ldb,
                     IComplexNumber alpha,
                     IComplexNumber beta, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr );
    native int zggev(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber alpha,
                     IComplexNumber beta,
                     IComplexNumber vl, int ldvl,
                     IComplexNumber vr, int ldvr );

    native int sggev3(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                      long b, int ldb,
                      long alphar, long alphai, long beta,
                      long vl, int ldvl,
                      long vr, int ldvr );
    native int dggev3(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                      long b, int ldb,
                      long alphar, long alphai, long beta,
                      long vl, int ldvl,
                      long vr, int ldvr );
    native int cggev3(int matriint, char jobvl, char jobvr,int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr );
    native int zggev3(int matriint, char jobvl, char jobvr,int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr );

    native int sggevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                      long alphar, long alphai, long beta, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long lscale,
                      long rscale, long abnrm, long bbnrm,
                      long rconde, long rcondv );
    native int dggevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                      long alphar, long alphai, long beta,
                      long vl, int ldvl, long vr,int ldvr, int ilo, int ihi,
                      long lscale, long rscale, long abnrm,
                      long bbnrm, long rconde, long rcondv );
    native int cggevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int ilo, int ihi,
                      long lscale, long rscale, long abnrm,
                      long bbnrm, long rconde, long rcondv );
    native int zggevx(int matriint, char balanc, char jobvl,
                      char jobvr, char sense, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,int  ilo, int ihi, long lscale,
                      long rscale, long abnrm, long bbnrm,
                      long rconde, long rcondv );

    native int sggglm(int matriint, int n, int m,int p, long a, int lda, long b,int ldb, long d, long x, long y );
    native int dggglm(int matriint, int n, int m,int p, long a, int lda, long b,int ldb, long d, long x, long y );
    native int cggglm(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                      IComplexNumber x, IComplexNumber y );
    native int zggglm(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                      IComplexNumber x, IComplexNumber y );

    native int sgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      long a, int lda, long b, int ldb,
                      long q, int ldq, long z, int ldz );
    native int dgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      long a, int lda, long b, int ldb,
                      long q, int ldq, long z,int ldz );
    native int cgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz );
    native int zgghrd(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz );

    native int sgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      long a, int lda, long b, int ldb,
                      long q, int ldq, long z, int ldz );
    native int dgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      long a, int lda, long b, int ldb,
                      long q, int ldq, long z,int ldz );
    native int cgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz );
    native int zgghd3(int matriint, char compq, char compz,int n, int ilo, int ihi,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz );

    native int sgglse(int matriint, int m, int n,int p, long a, int lda, long b,int ldb, long c, long d, long x );
    native int dgglse(int matriint, int m, int n,int p, long a, int lda, long b,int ldb, long c, long d, long x );
    native int cgglse(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                      IComplexNumber d, IComplexNumber x );
    native int zgglse(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                      IComplexNumber d, IComplexNumber x );

    native int sggqrf(int matriint, int n, int m,int p, long a, int lda, long taua,
                      long b, int ldb, long taub );
    native int dggqrf(int matriint, int n, int m,int p, long a, int lda,
                      long taua, long b, int ldb,
                      long taub );
    native int cggqrf(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                      IComplexNumber b, int ldb,
                      IComplexNumber taub );
    native int zggqrf(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                      IComplexNumber b, int ldb,
                      IComplexNumber taub );

    native int sggrqf(int matriint, int m, int p,int n, long a, int lda, long taua,
                      long b, int ldb, long taub );
    native int dggrqf(int matriint, int m, int p,int n, long a, int lda,
                      long taua, long b, int ldb,
                      long taub );
    native int cggrqf(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                      IComplexNumber b, int ldb,
                      IComplexNumber taub );
    native int zggrqf(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                      IComplexNumber b, int ldb,
                      IComplexNumber taub );

    native int sggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                      long alpha, long beta, long u, int ldu,
                      long v, int ldv, long q, int ldq,int iwork );
    native int dggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                      long alpha, long beta, long u,int ldu, long v, int ldv, long q,int ldq, int iwork );
    native int cggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      long alpha, long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, int iwork );
    native int zggsvd(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      long alpha, long beta,
                      IComplexNumber u, int ldu,
                      IComplexNumber v, int ldv,
                      IComplexNumber q, int ldq,int iwork );

    native int sggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                       long alpha, long beta, long u, int ldu,
                       long v, int ldv, long q, int ldq,int iwork );
    native int dggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l, long a,int lda, long b, int ldb,
                       long alpha, long beta, long u,int ldu, long v, int ldv, long q,int ldq, int iwork );
    native int cggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       long alpha, long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, int iwork );
    native int zggsvd3(int matriint, char jobu, char jobv, char jobq,int m, int n, int p,int k, int l,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       long alpha, long beta,
                       IComplexNumber u, int ldu,
                       IComplexNumber v, int ldv,
                       IComplexNumber q, int ldq,int iwork );

    native int sggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb, float tola,
                      float tolb, int k, int l, long u,int ldu, long v, int ldv, long q,int ldq );
    native int dggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb,
                      double tola, double tolb, int k,int l, long u, int ldu, long v,int ldv, long q, int ldq );
    native int cggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb, float tola,
                      float tolb, int k, int l,
                      IComplexNumber u, int ldu,
                      IComplexNumber v, int ldv,
                      IComplexNumber q, int ldq );
    native int zggsvp(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      double tola, double tolb, int k,int l, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq );

    native int sggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb, float tola,
                       float tolb, int k, int l, long u,int ldu, long v, int ldv, long q,int ldq );
    native int dggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n, long a,int lda, long b, int ldb,
                       double tola, double tolb, int k,int l, long u, int ldu, long v,int ldv, long q, int ldq );
    native int cggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb, float tola,
                       float tolb, int k, int l,
                       IComplexNumber u, int ldu,
                       IComplexNumber v, int ldv,
                       IComplexNumber q, int ldq );
    native int zggsvp3(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       double tola, double tolb, int k,int l, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq );

    native int sgtcon( char norm, int n,  long dl,
                       long d,  long du,  long du2,int ipiv, float anorm, long rcond );
    native int dgtcon( char norm, int n,  long dl,
                       long d,  long du,  long du2,int ipiv, double anorm,
                       long rcond );
    native int cgtcon( char norm, int n,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber du2,int ipiv, float anorm, long rcond );
    native int zgtcon( char norm, int n,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber du2,int ipiv, double anorm,
                       long rcond );

    native int sgtrfs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                      long du,  long dlf,  long df,
                      long duf,  long du2,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int dgtrfs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                      long du,  long dlf,
                      long df,  long duf,
                      long du2,int ipiv,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int cgtrfs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                      IComplexNumber d,
                      IComplexNumber du,
                      IComplexNumber dlf,
                      IComplexNumber df,
                      IComplexNumber duf,
                      IComplexNumber du2,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zgtrfs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                      IComplexNumber d,
                      IComplexNumber du,
                      IComplexNumber dlf,
                      IComplexNumber df,
                      IComplexNumber duf,
                      IComplexNumber du2,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int sgtsv(int matriint, int n, int nrhs,
                     long dl, long d, long du, long b,int ldb );
    native int dgtsv(int matriint, int n, int nrhs,
                     long dl, long d, long du, long b,int ldb );
    native int cgtsv(int matriint, int n, int nrhs,
                     IComplexNumber dl, IComplexNumber d,
                     IComplexNumber du, IComplexNumber b,int ldb );
    native int zgtsv(int matriint, int n, int nrhs,
                     IComplexNumber dl, IComplexNumber d,
                     IComplexNumber du, IComplexNumber b,int ldb );

    native int sgtsvx(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                      long d,  long du, long dlf,
                      long df, long duf, long du2, int ipiv,
                      long b, int ldb, long x,int ldx, long rcond, long ferr,
                      long berr );
    native int dgtsvx(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                      long d,  long du, long dlf,
                      long df, long duf, long du2,int ipiv,  long b, int ldb,
                      long x, int ldx, long rcond,
                      long ferr, long berr );
    native int cgtsvx(int matriint, char fact, char trans,int n, int nrhs,
                      IComplexNumber dl,
                      IComplexNumber d,
                      IComplexNumber du,
                      IComplexNumber dlf, IComplexNumber df,
                      IComplexNumber duf, IComplexNumber du2,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                      long berr );
    native int zgtsvx(int matriint, char fact, char trans,int n, int nrhs,
                      IComplexNumber dl,
                      IComplexNumber d,
                      IComplexNumber du,
                      IComplexNumber dlf,
                      IComplexNumber df,
                      IComplexNumber duf,
                      IComplexNumber du2, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int sgttrf(int n, long dl, long d, long du,
                      long du2, int ipiv );
    native int dgttrf(int n, long dl, long d, long du,
                      long du2, int ipiv );
    native int cgttrf(int n, IComplexNumber dl,
                      IComplexNumber d, IComplexNumber du,
                      IComplexNumber du2, int ipiv );
    native int zgttrf(int n, IComplexNumber dl,
                      IComplexNumber d, IComplexNumber du,
                      IComplexNumber du2, int ipiv );

    native int sgttrs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                      long du,  long du2,int ipiv, long b, int ldb );
    native int dgttrs(int matriint, char trans, int n,int nrhs,  long dl,  long d,
                      long du,  long du2,int ipiv, long b, int ldb );
    native int cgttrs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                      IComplexNumber d,
                      IComplexNumber du,
                      IComplexNumber du2,int ipiv, IComplexNumber b,int ldb );
    native int zgttrs(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                      IComplexNumber d,
                      IComplexNumber du,
                      IComplexNumber du2,int ipiv, IComplexNumber b,int ldb );

    native int chbev(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );
    native int zhbev(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );

    native int chbevd(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );
    native int zhbevd(int matriint, char jobz, char uplo, int n,int kd, IComplexNumber ab,int ldab, long w, IComplexNumber z,int ldz );

    native int chbevx(int matriint, char jobz, char range, char uplo,int n, int kd,
                      IComplexNumber ab, int ldab,
                      IComplexNumber q, int ldq, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
    native int zhbevx(int matriint, char jobz, char range, char uplo,int n, int kd,
                      IComplexNumber ab, int ldab,
                      IComplexNumber q, int ldq, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );

    native int chbgst(int matriint, char vect, char uplo, int n,int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb,
                      IComplexNumber x, int ldx );
    native int zhbgst(int matriint, char vect, char uplo, int n,int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb,
                      IComplexNumber x, int ldx );

    native int chbgv(int matriint, char jobz, char uplo, int n,int ka, int kb,
                     IComplexNumber ab, int ldab,
                     IComplexNumber bb, int ldbb, long w,
                     IComplexNumber z, int ldz );
    native int zhbgv(int matriint, char jobz, char uplo, int n,int ka, int kb,
                     IComplexNumber ab, int ldab,
                     IComplexNumber bb, int ldbb, long w,
                     IComplexNumber z, int ldz );

    native int chbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb, long w,
                      IComplexNumber z, int ldz );
    native int zhbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb,
                      long w, IComplexNumber z,int ldz );

    native int chbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb,
                      IComplexNumber q, int ldq, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
    native int zhbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb,
                      IComplexNumber q, int ldq, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );

    native int chbtrd(int matriint, char vect, char uplo, int n,int kd, IComplexNumber ab,int ldab, long d, long e,
                      IComplexNumber q, int ldq );
    native int zhbtrd(int matriint, char vect, char uplo, int n,int kd, IComplexNumber ab,int ldab, long d, long e,
                      IComplexNumber q, int ldq );

    native int checon(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv, float anorm, long rcond );
    native int zhecon(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv, double anorm,
                      long rcond );

    native int cheequb(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       long s, long scond, long amax );
    native int zheequb(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       long s, long scond, long amax );

    native int cheev(int matriint, char jobz, char uplo, int n,
                     IComplexNumber a, int lda, long w );
    native int zheev(int matriint, char jobz, char uplo, int n,
                     IComplexNumber a, int lda, long w );

    native int cheevd(int matriint, char jobz, char uplo, int n,
                      IComplexNumber a, int lda, long w );
    native int zheevd(int matriint, char jobz, char uplo, int n,
                      IComplexNumber a, int lda,
                      long w );

    native int cheevr(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, float vl, float vu, int il,int iu, float abstol, int m, long w,
                      IComplexNumber z, int ldz,int isuppz );
    native int zheevr(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, double vl, double vu, int il,int iu, double abstol, int m,
                      long w, IComplexNumber z, int ldz,int isuppz );

    native int cheevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, float vl, float vu, int il,int iu, float abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );
    native int zheevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber a,int lda, double vl, double vu, int il,int iu, double abstol, int m,
                      long w, IComplexNumber z, int ldz,int ifail );

    native int chegst(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );
    native int zhegst(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );

    native int chegv(int matriint, int itype, char jobz,
                     char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );
    native int zhegv(int matriint, int itype, char jobz,
                     char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );

    native int chegvd(int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );
    native int zhegvd(int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w );

    native int chegvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
    native int zhegvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );

    native int cherfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zherfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int cherfsx(int matriint, char uplo, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv,  long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int zherfsx(int matriint, char uplo, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv,  long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );

    native int chesv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                     IComplexNumber b, int ldb );
    native int zhesv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                     IComplexNumber b, int ldb );

    native int chesvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zhesvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int chesvxx(int matriint, char fact, char uplo,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int zhesvxx(int matriint, char fact, char uplo,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );

    native int chetrd(int matriint, char uplo, int n,
                      IComplexNumber a, int lda, long d,
                      long e, IComplexNumber tau );
    native int zhetrd(int matriint, char uplo, int n,
                      IComplexNumber a, int lda, long d,
                      long e, IComplexNumber tau );

    native int chetrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zhetrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int chetri(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zhetri(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int chetrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                      IComplexNumber b, int ldb );
    native int zhetrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                      IComplexNumber b, int ldb );

    native int chfrk(int matriint, char transr, char uplo, char trans,int n, int k, float alpha,
                     IComplexNumber a, int lda,
                     float beta, IComplexNumber c );
    native int zhfrk(int matriint, char transr, char uplo, char trans,int n, int k, double alpha,
                     IComplexNumber a, int lda,
                     double beta, IComplexNumber c );

    native int shgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                      long h, int ldh, long t, int ldt,
                      long alphar, long alphai, long beta, long q,int ldq, long z, int ldz );
    native int dhgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                      long h, int ldh, long t, int ldt,
                      long alphar, long alphai, long beta,
                      long q, int ldq, long z,int ldz );
    native int chgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                      IComplexNumber h, int ldh,
                      IComplexNumber t, int ldt,
                      IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber q,int ldq, IComplexNumber z,int ldz );
    native int zhgeqz(int matriint, char job, char compq, char compz,int n, int ilo, int ihi,
                      IComplexNumber h, int ldh,
                      IComplexNumber t, int ldt,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz );

    native int chpcon(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv, float anorm, long rcond );
    native int zhpcon(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv, double anorm,
                      long rcond );

    native int chpev(int matriint, char jobz, char uplo, int n,
                     IComplexNumber ap, long w,
                     IComplexNumber z, int ldz );
    native int zhpev(int matriint, char jobz, char uplo, int n,
                     IComplexNumber ap, long w,
                     IComplexNumber z, int ldz );

    native int chpevd(int matriint, char jobz, char uplo, int n,
                      IComplexNumber ap, long w,
                      IComplexNumber z, int ldz );
    native int zhpevd(int matriint, char jobz, char uplo, int n,
                      IComplexNumber ap, long w,
                      IComplexNumber z, int ldz );

    native int chpevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber ap, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int ifail );
    native int zhpevx(int matriint, char jobz, char range, char uplo,int n, IComplexNumber ap, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );

    native int chpgst(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                      IComplexNumber bp );
    native int zhpgst(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                      IComplexNumber bp );

    native int chpgv(int matriint, int itype, char jobz,
                     char uplo, int n, IComplexNumber ap,
                     IComplexNumber bp, long w,
                     IComplexNumber z, int ldz );
    native int zhpgv(int matriint, int itype, char jobz,
                     char uplo, int n, IComplexNumber ap,
                     IComplexNumber bp, long w,
                     IComplexNumber z, int ldz );

    native int chpgvd(int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber ap,
                      IComplexNumber bp, long w,
                      IComplexNumber z, int ldz );
    native int zhpgvd(int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber ap,
                      IComplexNumber bp, long w,
                      IComplexNumber z, int ldz );

    native int chpgvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n,
                      IComplexNumber ap, IComplexNumber bp,
                      float vl, float vu, int il, int iu,
                      float abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );
    native int zhpgvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n,
                      IComplexNumber ap, IComplexNumber bp,
                      double vl, double vu, int il, int iu,
                      double abstol, int m, long w,
                      IComplexNumber z, int ldz,int ifail );

    native int chprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zhprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int chpsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zhpsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

    native int chpsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zhpsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int chptrd(int matriint, char uplo, int n,
                      IComplexNumber ap, long d, long e,
                      IComplexNumber tau );
    native int zhptrd(int matriint, char uplo, int n,
                      IComplexNumber ap, long d, long e,
                      IComplexNumber tau );

    native int chptrf(int matriint, char uplo, int n,
                      IComplexNumber ap, int ipiv );
    native int zhptrf(int matriint, char uplo, int n,
                      IComplexNumber ap, int ipiv );

    native int chptri(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv );
    native int zhptri(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv );

    native int chptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zhptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

    native int shsein(int matriint, char job, char eigsrc, char initv,int select, int n,  long h,int ldh, long wr,  long wi,
                      long vl, int ldvl, long vr,int ldvr, int mm, int m,int ifaill, int ifailr );
    native int dhsein(int matriint, char job, char eigsrc, char initv,int select, int n,
                      long h, int ldh, long wr,
                      long wi, long vl, int ldvl,
                      long vr, int ldvr, int mm,int m, int ifaill,int ifailr );
    native int chsein(int matriint, char job, char eigsrc, char initv,int select, int n,
                      IComplexNumber h, int ldh,
                      IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int mm, int m,int ifaill, int ifailr );
    native int zhsein(int matriint, char job, char eigsrc, char initv,int select, int n,
                      IComplexNumber h, int ldh,
                      IComplexNumber w, IComplexNumber vl,int ldvl, IComplexNumber vr,int ldvr, int mm, int m,int ifaill, int ifailr );

    native int shseqr(int matriint, char job, char compz, int n,int ilo, int ihi, long h,int ldh, long wr, long wi, long z,int ldz );
    native int dhseqr(int matriint, char job, char compz, int n,int ilo, int ihi, long h,int ldh, long wr, long wi, long z,int ldz );
    native int chseqr(int matriint, char job, char compz, int n,int ilo, int ihi,
                      IComplexNumber h, int ldh,
                      IComplexNumber w, IComplexNumber z,int ldz );
    native int zhseqr(int matriint, char job, char compz, int n,int ilo, int ihi,
                      IComplexNumber h, int ldh,
                      IComplexNumber w, IComplexNumber z,int ldz );


    native int slacpy(int matriint, char uplo, int m,int n,  long a, int lda, long b,int ldb );
    native int dlacpy(int matriint, char uplo, int m,int n,  long a, int lda, long b,int ldb );
    native int clacpy(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int zlacpy(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

    native int clacp2(int matriint, char uplo, int m,int n,  long a, int lda,
                      IComplexNumber b, int ldb );
    native int zlacp2(int matriint, char uplo, int m,int n,  long a, int lda,
                      IComplexNumber b, int ldb );

    native int zlag2c(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber sa, int ldsa );

    native int slag2d(int matriint, int m, int n,
                      long sa, int ldsa, long a,int lda );

    native int dlag2s(int matriint, int m, int n,
                      long a, int lda, long sa,int ldsa );

    native int clag2z(int matriint, int m, int n,
                      IComplexNumber sa, int ldsa,
                      IComplexNumber a, int lda );

    native int slagge(int matriint, int m, int n,int kl, int ku,  long d,
                      long a, int lda, int iseed );
    native int dlagge(int matriint, int m, int n,int kl, int ku,  long d,
                      long a, int lda, int iseed );
    native int clagge(int matriint, int m, int n,int kl, int ku,  long d,
                      IComplexNumber a, int lda,int iseed );
    native int zlagge(int matriint, int m, int n,int kl, int ku,  long d,
                      IComplexNumber a, int lda,int iseed );

 
    native int slarfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k,  long v, int ldv,
                      long t, int ldt, long c,int ldc );
    native int dlarfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k,  long v, int ldv,
                      long t, int ldt, long c,int ldc );
    native int clarfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );
    native int zlarfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );



    native int slarft(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
    native int dlarft(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
    native int clarft(int matriint, char direct, char storev,int n, int k,
                      IComplexNumber v, int ldv,
                      IComplexNumber tau,
                      IComplexNumber t, int ldt );
    native int zlarft(int matriint, char direct, char storev,int n, int k,
                      IComplexNumber v, int ldv,
                      IComplexNumber tau,
                      IComplexNumber t, int ldt );

    native int slarfx(int matriint, char side, int m,int n,  long v, float tau, long c,int ldc, long work );
    native int dlarfx(int matriint, char side, int m,int n,  long v, double tau, long c,int ldc, long work );
    native int clarfx(int matriint, char side, int m,int n,  IComplexNumber v,
                      IComplexNumber tau, IComplexNumber c,int ldc, IComplexNumber work );
    native int zlarfx(int matriint, char side, int m,int n,  IComplexNumber v,
                      IComplexNumber tau, IComplexNumber c,int ldc, IComplexNumber work );



    native int slascl(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, long a,int lda );
    native int dlascl(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, long a,int lda );
    native int clascl(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, IComplexNumber a,int lda );
    native int zlascl(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, IComplexNumber a,int lda );

    native int slaset(int matriint, char uplo, int m,int n, float alpha, float beta, long a,int lda );
    native int dlaset(int matriint, char uplo, int m,int n, double alpha, double beta, long a,int lda );
    native int claset(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber a,int lda );
    native int zlaset(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber a,int lda );

    native int slasrt( char id, int n, long d );
    native int dlasrt( char id, int n, long d );

    native int slaswp(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
    native int dlaswp(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
    native int claswp(int matriint, int n,
                      IComplexNumber a, int lda,int k1, int k2,int ipiv,int incx );
    native int zlaswp(int matriint, int n,
                      IComplexNumber a, int lda,int k1, int k2,int ipiv,int incx );

    native int slatms(int matriint, int m, int n,
                      char dist, int iseed, char sym, long d,int mode, float cond, float dmax,int kl, int ku, char pack, long a,int lda );
    native int dlatms(int matriint, int m, int n,
                      char dist, int iseed, char sym, long d,int mode, double cond, double dmax,int kl, int ku, char pack, long a,int lda );
    native int clatms(int matriint, int m, int n,
                      char dist, int iseed, char sym, long d,int mode, float cond, float dmax,int kl, int ku, char pack,
                      IComplexNumber a, int lda );
    native int zlatms(int matriint, int m, int n,
                      char dist, int iseed, char sym, long d,int mode, double cond, double dmax,int kl, int ku, char pack,
                      IComplexNumber a, int lda );

    native int slauum(int matriint, char uplo, int n, long a,int lda );
    native int dlauum(int matriint, char uplo, int n, long a,int lda );
    native int clauum(int matriint, char uplo, int n,
                      IComplexNumber a, int lda );
    native int zlauum(int matriint, char uplo, int n,
                      IComplexNumber a, int lda );

    native int sopgtr(int matriint, char uplo, int n,
                      long ap,  long tau, long q,int ldq );
    native int dopgtr(int matriint, char uplo, int n,
                      long ap,  long tau, long q,int ldq );

    native int sopmtr(int matriint, char side, char uplo, char trans,int m, int n,  long ap,
                      long tau, long c, int ldc );
    native int dopmtr(int matriint, char side, char uplo, char trans,int m, int n,  long ap,
                      long tau, long c, int ldc );

    native int sorgbr(int matriint, char vect, int m,int n, int k, long a, int lda,
                      long tau );
    native int dorgbr(int matriint, char vect, int m,int n, int k, long a,int lda,  long tau );

    native int sorghr(int matriint, int n, int ilo,int ihi, long a, int lda,
                      long tau );
    native int dorghr(int matriint, int n, int ilo,int ihi, long a, int lda,
                      long tau );

    native int sorglq(int matriint, int m, int n,int k, long a, int lda,
                      long tau );
    native int dorglq(int matriint, int m, int n,int k, long a, int lda,
                      long tau );

    native int sorgql(int matriint, int m, int n,int k, long a, int lda,
                      long tau );
    native int dorgql(int matriint, int m, int n,int k, long a, int lda,
                      long tau );

    native int sorgqr(int matriint, int m, int n,int k, long a, int lda,
                      long tau );
    native int dorgqr(int matriint, int m, int n,int k, long a, int lda,
                      long tau );

    native int sorgrq(int matriint, int m, int n,int k, long a, int lda,
                      long tau );
    native int dorgrq(int matriint, int m, int n,int k, long a, int lda,
                      long tau );

    native int sorgtr(int matriint, char uplo, int n, long a,int lda,  long tau );
    native int dorgtr(int matriint, char uplo, int n, long a,int lda,  long tau );

    native int sormbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );
    native int dormbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );

    native int sormhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                      long tau, long c, int ldc );
    native int dormhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                      long tau, long c, int ldc );

    native int sormlq(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );
    native int dormlq(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );

    native int sormql(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );
    native int dormql(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );

    native int sormqr(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );
    native int dormqr(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );

    native int sormrq(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );
    native int dormrq(int matriint, char side, char trans,int m, int n, int k,
                      long a, int lda,  long tau,
                      long c, int ldc );

    native int sormrz(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                      long tau, long c, int ldc );
    native int dormrz(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                      long tau, long c, int ldc );

    native int sormtr(int matriint, char side, char uplo, char trans,int m, int n,  long a,int lda,  long tau, long c,int ldc );
    native int dormtr(int matriint, char side, char uplo, char trans,int m, int n,  long a,int lda,  long tau, long c,int ldc );

    native int spbcon(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                      float anorm, long rcond );
    native int dpbcon(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                      double anorm, long rcond );
    native int cpbcon(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, float anorm, long rcond );
    native int zpbcon(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, double anorm, long rcond );

    native int spbequ(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                      long s, long scond, long amax );
    native int dpbequ(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                      long s, long scond, long amax );
    native int cpbequ(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                      long amax );
    native int zpbequ(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                      long amax );

    native int spbrfs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab,  long afb, int ldafb,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int dpbrfs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab,  long afb, int ldafb,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int cpbrfs(int matriint, char uplo, int n,int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber afb, int ldafb,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zpbrfs(int matriint, char uplo, int n,int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber afb, int ldafb,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int spbstf(int matriint, char uplo, int n,int kb, long bb, int ldbb );
    native int dpbstf(int matriint, char uplo, int n,int kb, long bb, int ldbb );
    native int cpbstf(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );
    native int zpbstf(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );

    native int spbsv(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
    native int dpbsv(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
    native int cpbsv(int matriint, char uplo, int n,int kd, int nrhs,
                     IComplexNumber ab, int ldab,
                     IComplexNumber b, int ldb );
    native int zpbsv(int matriint, char uplo, int n,int kd, int nrhs,
                     IComplexNumber ab, int ldab,
                     IComplexNumber b, int ldb );

    native int spbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs, long ab,int ldab, long afb, int ldafb,
                      String equed, long s, long b, int ldb,
                      long x, int ldx, long rcond, long ferr,
                      long berr );
    native int dpbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs, long ab,int ldab, long afb, int ldafb,
                      String equed, long s, long b, int ldb,
                      long x, int ldx, long rcond,
                      long ferr, long berr );
    native int cpbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber afb, int ldafb,
                      String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                      long berr );
    native int zpbsvx(int matriint, char fact, char uplo, int n,int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber afb, int ldafb,
                      String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                      long berr );

    native int spbtrf(int matriint, char uplo, int n,int kd, long ab, int ldab );
    native int dpbtrf(int matriint, char uplo, int n,int kd, long ab, int ldab );
    native int cpbtrf(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );
    native int zpbtrf(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );

    native int spbtrs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb );
    native int dpbtrs(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb );
    native int cpbtrs(int matriint, char uplo, int n,int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb );
    native int zpbtrs(int matriint, char uplo, int n,int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb );

    native int spftrf(int matriint, char transr, char uplo,int n, long a );
    native int dpftrf(int matriint, char transr, char uplo,int n, long a );
    native int cpftrf(int matriint, char transr, char uplo,int n, IComplexNumber a );
    native int zpftrf(int matriint, char transr, char uplo,int n, IComplexNumber a );

    native int spftri(int matriint, char transr, char uplo,int n, long a );
    native int dpftri(int matriint, char transr, char uplo,int n, long a );
    native int cpftri(int matriint, char transr, char uplo,int n, IComplexNumber a );
    native int zpftri(int matriint, char transr, char uplo,int n, IComplexNumber a );

    native int spftrs(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                      long b, int ldb );
    native int dpftrs(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                      long b, int ldb );
    native int cpftrs(int matriint, char transr, char uplo,int n, int nrhs,
                      IComplexNumber a,
                      IComplexNumber b, int ldb );
    native int zpftrs(int matriint, char transr, char uplo,int n, int nrhs,
                      IComplexNumber a,
                      IComplexNumber b, int ldb );

    native int spocon(int matriint, char uplo, int n,
                      long a, int lda, float anorm,
                      long rcond );
    native int dpocon(int matriint, char uplo, int n,
                      long a, int lda, double anorm,
                      long rcond );
    native int cpocon(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,
                      float anorm, long rcond );
    native int zpocon(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,
                      double anorm, long rcond );

    native int spoequ(int matriint, int n,  long a,int lda, long s, long scond,
                      long amax );
    native int dpoequ(int matriint, int n,  long a,int lda, long s, long scond,
                      long amax );
    native int cpoequ(int matriint, int n,
                      IComplexNumber a, int lda,
                      long s, long scond, long amax );
    native int zpoequ(int matriint, int n,
                      IComplexNumber a, int lda,
                      long s, long scond, long amax );

    native int spoequb(int matriint, int n,  long a,int lda, long s, long scond,
                       long amax );
    native int dpoequb(int matriint, int n,  long a,int lda, long s, long scond,
                       long amax );
    native int cpoequb(int matriint, int n,
                       IComplexNumber a, int lda,
                       long s, long scond, long amax );
    native int zpoequb(int matriint, int n,
                       IComplexNumber a, int lda,
                       long s, long scond, long amax );

    native int sporfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                      long af, int ldaf,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int dporfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                      long af, int ldaf,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int cporfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr );
    native int zporfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr );

    native int sporfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,
                       long s,  long b, int ldb,
                       long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int dporfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,
                       long s,  long b, int ldb,
                       long x, int ldx, long rcond,
                       long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int cporfsx(int matriint, char uplo, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,
                       long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int zporfsx(int matriint, char uplo, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,
                       long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );

    native int sposv(int matriint, char uplo, int n,int nrhs, long a, int lda, long b,int ldb );
    native int dposv(int matriint, char uplo, int n,int nrhs, long a, int lda, long b,int ldb );
    native int cposv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int zposv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int dsposv(int matriint, char uplo, int n,int nrhs, long a, int lda,
                      long b, int ldb, long x, int ldx,int iter );
    native int zcposv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber x,int ldx, int iter );

    native int sposvx(int matriint, char fact, char uplo, int n,int nrhs, long a, int lda, long af,int ldaf, String equed, long s, long b,int ldb, long x, int ldx,
                      long rcond, long ferr, long berr );
    native int dposvx(int matriint, char fact, char uplo, int n,int nrhs, long a, int lda,
                      long af, int ldaf, String equed, long s,
                      long b, int ldb, long x, int ldx,
                      long rcond, long ferr, long berr );
    native int cposvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber af,int ldaf, String equed, long s,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zposvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber af,int ldaf, String equed, long s,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int sposvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                       String equed, long s, long b, int ldb,
                       long x, int ldx, long rcond,
                       long rpvgrw, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int dposvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                       String equed, long s, long b, int ldb,
                       long x, int ldx, long rcond,
                       long rpvgrw, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int cposvxx(int matriint, char fact, char uplo,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,
                       String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                       long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int zposvxx(int matriint, char fact, char uplo,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,
                       String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                       long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );

    native int spotrf2(int matriint, char uplo, int n, long a,int lda );
    native int dpotrf2(int matriint, char uplo, int n, long a,int lda );
    native int cpotrf2(int matriint, char uplo, int n,
                       IComplexNumber a, int lda );
    native int zpotrf2(int matriint, char uplo, int n,
                       IComplexNumber a, int lda );

    native int spotrf(int matriint, char uplo, int n, long a,int lda );
    native int dpotrf(int matriint, char uplo, int n, long a,int lda );
    native int cpotrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda );
    native int zpotrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda );

    native int spotri(int matriint, char uplo, int n, long a,int lda );
    native int dpotri(int matriint, char uplo, int n, long a,int lda );
    native int cpotri(int matriint, char uplo, int n,
                      IComplexNumber a, int lda );
    native int zpotri(int matriint, char uplo, int n,
                      IComplexNumber a, int lda );

    native int spotrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                      long b, int ldb );
    native int dpotrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                      long b, int ldb );
    native int cpotrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int zpotrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

    native int sppcon(int matriint, char uplo, int n,
                      long ap, float anorm, long rcond );
    native int dppcon(int matriint, char uplo, int n,
                      long ap, double anorm, long rcond );
    native int cppcon(int matriint, char uplo, int n,
                      IComplexNumber ap, float anorm,
                      long rcond );
    native int zppcon(int matriint, char uplo, int n,
                      IComplexNumber ap, double anorm,
                      long rcond );

    native int sppequ(int matriint, char uplo, int n,
                      long ap, long s, long scond,
                      long amax );
    native int dppequ(int matriint, char uplo, int n,
                      long ap, long s, long scond,
                      long amax );
    native int cppequ(int matriint, char uplo, int n,
                      IComplexNumber ap, long s,
                      long scond, long amax );
    native int zppequ(int matriint, char uplo, int n,
                      IComplexNumber ap, long s,
                      long scond, long amax );

    native int spprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int dpprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,
                      long b, int ldb, long x,int ldx, long ferr, long berr );
    native int cpprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zpprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int sppsv(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
    native int dppsv(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
    native int cppsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                     IComplexNumber b, int ldb );
    native int zppsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                     IComplexNumber b, int ldb );

    native int sppsvx(int matriint, char fact, char uplo, int n,int nrhs, long ap, long afp, String equed,
                      long s, long b, int ldb, long x,int ldx, long rcond, long ferr,
                      long berr );
    native int dppsvx(int matriint, char fact, char uplo, int n,int nrhs, long ap, long afp,
                      String equed, long s, long b, int ldb,
                      long x, int ldx, long rcond,
                      long ferr, long berr );
    native int cppsvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber ap,
                      IComplexNumber afp, String equed, long s,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zppsvx(int matriint, char fact, char uplo, int n,int nrhs, IComplexNumber ap,
                      IComplexNumber afp, String equed, long s,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int spptrf(int matriint, char uplo, int n,
                      long ap );
    native int dpptrf(int matriint, char uplo, int n,
                      long ap );
    native int cpptrf(int matriint, char uplo, int n,
                      IComplexNumber ap );
    native int zpptrf(int matriint, char uplo, int n,
                      IComplexNumber ap );

    native int spptri(int matriint, char uplo, int n,
                      long ap );
    native int dpptri(int matriint, char uplo, int n,
                      long ap );
    native int cpptri(int matriint, char uplo, int n,
                      IComplexNumber ap );
    native int zpptri(int matriint, char uplo, int n,
                      IComplexNumber ap );

    native int spptrs(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
    native int dpptrs(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
    native int cpptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber b, int ldb );
    native int zpptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber b, int ldb );

    native int spstrf(int matriint, char uplo, int n, long a,int lda, int piv, int rank,
                      float tol );
    native int dpstrf(int matriint, char uplo, int n, long a,int lda, int piv, int rank,
                      double tol );
    native int cpstrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int piv, int rank, float tol );
    native int zpstrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int piv, int rank, double tol );

    native int sptcon(int n,  long d,  long e,
                      float anorm, long rcond );
    native int dptcon(int n,  long d,  long e,
                      double anorm, long rcond );
    native int cptcon(int n,  long d,
                      IComplexNumber e, float anorm,
                      long rcond );
    native int zptcon(int n,  long d,
                      IComplexNumber e, double anorm,
                      long rcond );

    native int spteqr(int matriint, char compz, int n, long d,
                      long e, long z, int ldz );
    native int dpteqr(int matriint, char compz, int n,
                      long d, long e, long z, int ldz );
    native int cpteqr(int matriint, char compz, int n, long d,
                      long e, IComplexNumber z, int ldz );
    native int zpteqr(int matriint, char compz, int n,
                      long d, long e, IComplexNumber z,int ldz );

    native int sptrfs(int matriint, int n, int nrhs,
                      long d,  long e,  long df,
                      long ef,  long b, int ldb,
                      long x, int ldx, long ferr, long berr );
    native int dptrfs(int matriint, int n, int nrhs,
                      long d,  long e,  long df,
                      long ef,  long b, int ldb,
                      long x, int ldx, long ferr,
                      long berr );
    native int cptrfs(int matriint, char uplo, int n,int nrhs,  long d,
                      IComplexNumber e,  long df,
                      IComplexNumber ef,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zptrfs(int matriint, char uplo, int n,int nrhs,  long d,
                      IComplexNumber e,  long df,
                      IComplexNumber ef,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int sptsv(int matriint, int n, int nrhs,
                     long d, long e, long b, int ldb );
    native int dptsv(int matriint, int n, int nrhs,
                     long d, long e, long b, int ldb );
    native int cptsv(int matriint, int n, int nrhs,
                     long d, IComplexNumber e,
                     IComplexNumber b, int ldb );
    native int zptsv(int matriint, int n, int nrhs,
                     long d, IComplexNumber e,
                     IComplexNumber b, int ldb );

    native int sptsvx(int matriint, char fact, int n,int nrhs,  long d,  long e,
                      long df, long ef,  long b, int ldb,
                      long x, int ldx, long rcond, long ferr,
                      long berr );
    native int dptsvx(int matriint, char fact, int n,int nrhs,  long d,  long e,
                      long df, long ef,  long b,int ldb, long x, int ldx,
                      long rcond, long ferr, long berr );
    native int cptsvx(int matriint, char fact, int n,int nrhs,  long d,
                      IComplexNumber e, long df,
                      IComplexNumber ef,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zptsvx(int matriint, char fact, int n,int nrhs,  long d,
                      IComplexNumber e, long df,
                      IComplexNumber ef,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int spttrf(int n, long d, long e );
    native int dpttrf(int n, long d, long e );
    native int cpttrf(int n, long d, IComplexNumber e );
    native int zpttrf(int n, long d, IComplexNumber e );

    native int spttrs(int matriint, int n, int nrhs,
                      long d,  long e, long b,int ldb );
    native int dpttrs(int matriint, int n, int nrhs,
                      long d,  long e, long b,int ldb );
    native int cpttrs(int matriint, char uplo, int n,int nrhs,  long d,
                      IComplexNumber e,
                      IComplexNumber b, int ldb );
    native int zpttrs(int matriint, char uplo, int n,int nrhs,  long d,
                      IComplexNumber e,
                      IComplexNumber b, int ldb );

    native int ssbev(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab, long w,
                     long z, int ldz );
    native int dsbev(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab, long w,
                     long z, int ldz );

    native int ssbevd(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab, long w,
                      long z, int ldz );
    native int dsbevd(int matriint, char jobz, char uplo, int n,int kd, long ab, int ldab,
                      long w, long z, int ldz );

    native int ssbevx(int matriint, char jobz, char range, char uplo,int n, int kd, long ab,int ldab, long q, int ldq, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
    native int dsbevx(int matriint, char jobz, char range, char uplo,int n, int kd, long ab,int ldab, long q, int ldq,
                      double vl, double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int ifail );

    native int ssbgst(int matriint, char vect, char uplo, int n,int ka, int kb, long ab,int ldab,  long bb, int ldbb,
                      long x, int ldx );
    native int dsbgst(int matriint, char vect, char uplo, int n,int ka, int kb, long ab,int ldab,  long bb, int ldbb,
                      long x, int ldx );

    native int ssbgv(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb, long w,
                     long z, int ldz );
    native int dsbgv(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb,
                     long w, long z, int ldz );

    native int ssbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb,
                      long w, long z, int ldz );
    native int dsbgvd(int matriint, char jobz, char uplo, int n,int ka, int kb, long ab,int ldab, long bb, int ldbb,
                      long w, long z, int ldz );

    native int ssbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                      long ab, int ldab, long bb,int ldbb, long q, int ldq, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
    native int dsbgvx(int matriint, char jobz, char range, char uplo,int n, int ka, int kb,
                      long ab, int ldab, long bb,int ldbb, long q, int ldq,
                      double vl, double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int ifail );

    native int ssbtrd(int matriint, char vect, char uplo, int n,int kd, long ab, int ldab, long d,
                      long e, long q, int ldq );
    native int dsbtrd(int matriint, char vect, char uplo, int n,int kd, long ab, int ldab,
                      long d, long e, long q, int ldq );

    native int ssfrk(int matriint, char transr, char uplo, char trans,int n, int k, float alpha,
                     long a, int lda, float beta,
                     long c );
    native int dsfrk(int matriint, char transr, char uplo, char trans,int n, int k, double alpha,
                     long a, int lda, double beta,
                     long c );

    native int sspcon(int matriint, char uplo, int n,
                      long ap,int ipiv, float anorm,
                      long rcond );
    native int dspcon(int matriint, char uplo, int n,
                      long ap,int ipiv,
                      double anorm, long rcond );
    native int cspcon(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv, float anorm, long rcond );
    native int zspcon(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv, double anorm,
                      long rcond );

    native int sspev(int matriint, char jobz, char uplo, int n,
                     long ap, long w, long z, int ldz );
    native int dspev(int matriint, char jobz, char uplo, int n,
                     long ap, long w, long z, int ldz );

    native int sspevd(int matriint, char jobz, char uplo, int n,
                      long ap, long w, long z, int ldz );
    native int dspevd(int matriint, char jobz, char uplo, int n,
                      long ap, long w, long z, int ldz );

    native int sspevx(int matriint, char jobz, char range, char uplo,int n, long ap, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
    native int dspevx(int matriint, char jobz, char range, char uplo,int n, long ap, double vl, double vu,int il, int iu, double abstol,int m, long w, long z, int ldz,int ifail );

    native int sspgst(int matriint, int itype, char uplo,int n, long ap,  long bp );
    native int dspgst(int matriint, int itype, char uplo,int n, long ap,  long bp );

    native int sspgv(int matriint, int itype, char jobz,
                     char uplo, int n, long ap, long bp,
                     long w, long z, int ldz );
    native int dspgv(int matriint, int itype, char jobz,
                     char uplo, int n, long ap, long bp,
                     long w, long z, int ldz );

    native int sspgvd(int matriint, int itype, char jobz,
                      char uplo, int n, long ap, long bp,
                      long w, long z, int ldz );
    native int dspgvd(int matriint, int itype, char jobz,
                      char uplo, int n, long ap, long bp,
                      long w, long z, int ldz );

    native int sspgvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n, long ap,
                      long bp, float vl, float vu, int il,int iu, float abstol, int m, long w,
                      long z, int ldz, int ifail );
    native int dspgvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n, long ap,
                      long bp, double vl, double vu, int il,int iu, double abstol, int m,
                      long w, long z, int ldz,int ifail );

    native int ssprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int dsprfs(int matriint, char uplo, int n,int nrhs,  long ap,  long afp,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int csprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zsprfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int sspsv(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                     long b, int ldb );
    native int dspsv(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                     long b, int ldb );
    native int cspsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zspsv(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

    native int sspsvx(int matriint, char fact, char uplo, int n,int nrhs,  long ap, long afp,int ipiv,  long b, int ldb,
                      long x, int ldx, long rcond, long ferr,
                      long berr );
    native int dspsvx(int matriint, char fact, char uplo, int n,int nrhs,  long ap, long afp,int ipiv,  long b, int ldb,
                      long x, int ldx, long rcond,
                      long ferr, long berr );
    native int cspsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zspsvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber ap,
                      IComplexNumber afp, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int ssptrd(int matriint, char uplo, int n, long ap,
                      long d, long e, long tau );
    native int dsptrd(int matriint, char uplo, int n,
                      long ap, long d, long e, long tau );

    native int ssptrf(int matriint, char uplo, int n, long ap,int ipiv );
    native int dsptrf(int matriint, char uplo, int n,
                      long ap, int ipiv );
    native int csptrf(int matriint, char uplo, int n,
                      IComplexNumber ap, int ipiv );
    native int zsptrf(int matriint, char uplo, int n,
                      IComplexNumber ap, int ipiv );

    native int ssptri(int matriint, char uplo, int n, long ap,int ipiv );
    native int dsptri(int matriint, char uplo, int n,
                      long ap,int ipiv );
    native int csptri(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv );
    native int zsptri(int matriint, char uplo, int n,
                      IComplexNumber ap,int ipiv );

    native int ssptrs(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b, int ldb );
    native int dsptrs(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b, int ldb );
    native int csptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zsptrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

    native int sstebz( char range, char order, int n, float vl,
                       float vu, int il, int iu, float abstol,
                       long d,  long e, int m,int nsplit, long w, int iblock,int isplit );
    native int dstebz( char range, char order, int n, double vl,
                       double vu, int il, int iu,
                       double abstol,  long d,  long e,int m, int nsplit, long w,int iblock, int isplit );

    native int sstedc(int matriint, char compz, int n, long d,
                      long e, long z, int ldz );
    native int dstedc(int matriint, char compz, int n,
                      long d, long e, long z, int ldz );
    native int cstedc(int matriint, char compz, int n, long d,
                      long e, IComplexNumber z, int ldz );
    native int zstedc(int matriint, char compz, int n,
                      long d, long e, IComplexNumber z,int ldz );

    native int sstegr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int isuppz );
    native int dstegr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int isuppz );
    native int cstegr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, IComplexNumber z,int ldz, int isuppz );
    native int zstegr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w,
                      IComplexNumber z, int ldz,int isuppz );

    native int sstein(int matriint, int n,  long d,
                      long e, int m,  long w,int iblock,int isplit,
                      long z, int ldz, int ifailv );
    native int dstein(int matriint, int n,  long d,
                      long e, int m,  long w,int iblock,int isplit,
                      long z, int ldz, int ifailv );
    native int cstein(int matriint, int n,  long d,
                      long e, int m,  long w,int iblock,int isplit,
                      IComplexNumber z, int ldz,int ifailv );
    native int zstein(int matriint, int n,  long d,
                      long e, int m,  long w,int iblock,int isplit,
                      IComplexNumber z, int ldz,int ifailv );

    native int sstemr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, int m,
                      long w, long z, int ldz, int nzc,int isuppz, int tryrac );
    native int dstemr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                      double vu, int il, int iu,int m, long w, long z, int ldz,int nzc, int isuppz,int tryrac );
    native int cstemr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, int m,
                      long w, IComplexNumber z, int ldz,int nzc, int isuppz,int tryrac );
    native int zstemr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                      double vu, int il, int iu,int m, long w, IComplexNumber z,int ldz, int nzc, int isuppz,int tryrac );

    native int ssteqr(int matriint, char compz, int n, long d,
                      long e, long z, int ldz );
    native int dsteqr(int matriint, char compz, int n,
                      long d, long e, long z, int ldz );
    native int csteqr(int matriint, char compz, int n, long d,
                      long e, IComplexNumber z, int ldz );
    native int zsteqr(int matriint, char compz, int n,
                      long d, long e, IComplexNumber z,int ldz );

    native int ssterf(int n, long d, long e );
    native int dsterf(int n, long d, long e );

    native int sstev(int matriint, char jobz, int n, long d,
                     long e, long z, int ldz );
    native int dstev(int matriint, char jobz, int n, long d,
                     long e, long z, int ldz );

    native int sstevd(int matriint, char jobz, int n, long d,
                      long e, long z, int ldz );
    native int dstevd(int matriint, char jobz, int n, long d,
                      long e, long z, int ldz );

    native int sstevr(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int isuppz );
    native int dstevr(int matriint, char jobz, char range,int n, long d, long e, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int isuppz );

    native int sstevx(int matriint, char jobz, char range,int n, long d, long e, float vl, float vu,int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
    native int dstevx(int matriint, char jobz, char range,int n, long d, long e, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int ifail );

    native int ssycon(int matriint, char uplo, int n,
                      long a, int lda,int ipiv, float anorm, long rcond );
    native int dsycon(int matriint, char uplo, int n,
                      long a, int lda,int ipiv, double anorm,
                      long rcond );
    native int csycon(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv, float anorm, long rcond );
    native int zsycon(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv, double anorm,
                      long rcond );

    native int ssyequb(int matriint, char uplo, int n,
                       long a, int lda, long s,
                       long scond, long amax );
    native int dsyequb(int matriint, char uplo, int n,
                       long a, int lda, long s,
                       long scond, long amax );
    native int csyequb(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       long s, long scond, long amax );
    native int zsyequb(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       long s, long scond, long amax );

    native int ssyev(int matriint, char jobz, char uplo, int n,
                     long a, int lda, long w );
    native int dsyev(int matriint, char jobz, char uplo, int n,
                     long a, int lda, long w );

    native int ssyevd(int matriint, char jobz, char uplo, int n,
                      long a, int lda, long w );
    native int dsyevd(int matriint, char jobz, char uplo, int n,
                      long a, int lda, long w );

    native int ssyevr(int matriint, char jobz, char range, char uplo,int n, long a, int lda, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int isuppz );
    native int dsyevr(int matriint, char jobz, char range, char uplo,int n, long a, int lda, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int isuppz );

    native int ssyevx(int matriint, char jobz, char range, char uplo,int n, long a, int lda, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
    native int dsyevx(int matriint, char jobz, char range, char uplo,int n, long a, int lda, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int ifail );

    native int ssygst(int matriint, int itype, char uplo,int n, long a, int lda,
                      long b, int ldb );
    native int dsygst(int matriint, int itype, char uplo,int n, long a, int lda,
                      long b, int ldb );

    native int ssygv(int matriint, int itype, char jobz,
                     char uplo, int n, long a, int lda,
                     long b, int ldb, long w );
    native int dsygv(int matriint, int itype, char jobz,
                     char uplo, int n, long a, int lda,
                     long b, int ldb, long w );

    native int ssygvd(int matriint, int itype, char jobz,
                      char uplo, int n, long a, int lda,
                      long b, int ldb, long w );
    native int dsygvd(int matriint, int itype, char jobz,
                      char uplo, int n, long a, int lda,
                      long b, int ldb, long w );

    native int ssygvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n, long a,int lda, long b, int ldb, float vl,
                      float vu, int il, int iu, float abstol,int m, long w, long z, int ldz,int ifail );
    native int dsygvx(int matriint, int itype, char jobz,
                      char range, char uplo, int n, long a,int lda, long b, int ldb, double vl,
                      double vu, int il, int iu,
                      double abstol, int m, long w, long z,int ldz, int ifail );

    native int ssyrfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                      long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int dsyrfs(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                      long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                      long ferr, long berr );
    native int csyrfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx, long ferr,
                      long berr );
    native int zsyrfs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int ssyrfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long s,
                       long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int dsyrfsx(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long s,
                       long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int csyrfsx(int matriint, char uplo, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv,  long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );
    native int zsyrfsx(int matriint, char uplo, char equed,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv,  long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long berr, int int_bnds,
                       long erint_norm, long erint_comp,int nparams, long params );

    native int ssysv(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
    native int dsysv(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
    native int csysv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                     IComplexNumber b, int ldb );
    native int zsysv(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                     IComplexNumber b, int ldb );

    native int ssysvx(int matriint, char fact, char uplo, int n,int nrhs,  long a, int lda,
                      long af, int ldaf, int ipiv,
                      long b, int ldb, long x,int ldx, long rcond, long ferr,
                      long berr );
    native int dsysvx(int matriint, char fact, char uplo, int n,int nrhs,  long a, int lda,
                      long af, int ldaf, int ipiv,
                      long b, int ldb, long x,int ldx, long rcond, long ferr,
                      long berr );
    native int csysvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );
    native int zsysvx(int matriint, char fact, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber af,int ldaf, int ipiv,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long rcond, long ferr, long berr );

    native int ssysvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int dsysvxx(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int csysvxx(int matriint, char fact, char uplo,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );
    native int zsysvxx(int matriint, char fact, char uplo,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                       long erint_comp, int nparams,
                       long params );

    native int ssytrd(int matriint, char uplo, int n, long a,int lda, long d, long e, long tau );
    native int dsytrd(int matriint, char uplo, int n, long a,int lda, long d, long e, long tau );

    native int ssytrf(int matriint, char uplo, int n, long a,int lda, int ipiv );
    native int dsytrf(int matriint, char uplo, int n, long a,int lda, int ipiv );
    native int csytrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zsytrf(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int ssytri(int matriint, char uplo, int n, long a,int lda,int ipiv );
    native int dsytri(int matriint, char uplo, int n, long a,int lda,int ipiv );
    native int csytri(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );
    native int zsytri(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,int ipiv );

    native int ssytrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int dsytrs(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int csytrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                      IComplexNumber b, int ldb );
    native int zsytrs(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                      IComplexNumber b, int ldb );

    native int stbcon(int matriint, char norm, char uplo, char diag,int n, int kd,  long ab,int ldab, long rcond );
    native int dtbcon(int matriint, char norm, char uplo, char diag,int n, int kd,  long ab,int ldab, long rcond );
    native int ctbcon(int matriint, char norm, char uplo, char diag,int n, int kd,
                      IComplexNumber ab, int ldab,
                      long rcond );
    native int ztbcon(int matriint, char norm, char uplo, char diag,int n, int kd,
                      IComplexNumber ab, int ldab,
                      long rcond );

    native int stbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      long ab, int ldab,  long b,int ldb,  long x, int ldx,
                      long ferr, long berr );
    native int dtbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      long ab, int ldab,  long b,int ldb,  long x, int ldx,
                      long ferr, long berr );
    native int ctbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );
    native int ztbrfs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int stbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      long ab, int ldab, long b,int ldb );
    native int dtbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      long ab, int ldab, long b,int ldb );
    native int ctbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb );
    native int ztbtrs(int matriint, char uplo, char trans, char diag,int n, int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb );

    native int stfsm(int matriint, char transr, char side, char uplo,
                     char trans, char diag, int m, int n,
                     float alpha,  long a, long b,int ldb );
    native int dtfsm(int matriint, char transr, char side, char uplo,
                     char trans, char diag, int m, int n,
                     double alpha,  long a, long b,int ldb );
    native int ctfsm(int matriint, char transr, char side, char uplo,
                     char trans, char diag, int m, int n,
                     IComplexNumber alpha,
                     IComplexNumber a,
                     IComplexNumber b, int ldb );
    native int ztfsm(int matriint, char transr, char side, char uplo,
                     char trans, char diag, int m, int n,
                     IComplexNumber alpha,
                     IComplexNumber a,
                     IComplexNumber b, int ldb );

    native int stftri(int matriint, char transr, char uplo, char diag,int n, long a );
    native int dtftri(int matriint, char transr, char uplo, char diag,int n, long a );
    native int ctftri(int matriint, char transr, char uplo, char diag,int n, IComplexNumber a );
    native int ztftri(int matriint, char transr, char uplo, char diag,int n, IComplexNumber a );

    native int stfttp(int matriint, char transr, char uplo,int n,  long arf, long ap );
    native int dtfttp(int matriint, char transr, char uplo,int n,  long arf, long ap );
    native int ctfttp(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                      IComplexNumber ap );
    native int ztfttp(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                      IComplexNumber ap );

    native int stfttr(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
    native int dtfttr(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
    native int ctfttr(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                      IComplexNumber a, int lda );
    native int ztfttr(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                      IComplexNumber a, int lda );

    native int stgevc(int matriint, char side, char howmny,int select, int n,
                      long s, int lds,  long p,int ldp, long vl, int ldvl,
                      long vr, int ldvr, int mm,int m );
    native int dtgevc(int matriint, char side, char howmny,int select, int n,
                      long s, int lds,  long p,int ldp, long vl, int ldvl,
                      long vr, int ldvr, int mm,int m );
    native int ctgevc(int matriint, char side, char howmny,int select, int n,
                      IComplexNumber s, int lds,
                      IComplexNumber p, int ldp,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,int mm, int m );
    native int ztgevc(int matriint, char side, char howmny,int select, int n,
                      IComplexNumber s, int lds,
                      IComplexNumber p, int ldp,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,int mm, int m );

    native int stgexc(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb, long q,int ldq, long z, int ldz,int ifst, int ilst );
    native int dtgexc(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb, long q,int ldq, long z, int ldz,int ifst, int ilst );
    native int ctgexc(int matriint, int wantq,int wantz, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz,int ifst, int ilst );
    native int ztgexc(int matriint, int wantq,int wantz, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz,int ifst, int ilst );

    native int stgsen(int matriint, int ijob,int wantq, int wantz,int select, int n, long a,int lda, long b, int ldb,
                      long alphar, long alphai, long beta, long q,int ldq, long z, int ldz,int m, long pl, long pr, long dif );
    native int dtgsen(int matriint, int ijob,int wantq, int wantz,int select, int n,
                      long a, int lda, long b, int ldb,
                      long alphar, long alphai, long beta,
                      long q, int ldq, long z, int ldz,int m, long pl, long pr, long dif );
    native int ctgsen(int matriint, int ijob,int wantq, int wantz,int select, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber q,int ldq, IComplexNumber z,int ldz, int m, long pl, long pr,
                      long dif );
    native int ztgsen(int matriint, int ijob,int wantq, int wantz,int select, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber q, int ldq,
                      IComplexNumber z, int ldz,int m, long pl, long pr, long dif );

    native int stgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, long a, int lda,
                      long b, int ldb, float tola, float tolb,
                      long alpha, long beta, long u, int ldu,
                      long v, int ldv, long q, int ldq,int ncycle );
    native int dtgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, long a,int lda, long b, int ldb,
                      double tola, double tolb, long alpha,
                      long beta, long u, int ldu, long v,int ldv, long q, int ldq,int ncycle );
    native int ctgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, IComplexNumber a,int lda, IComplexNumber b,int ldb, float tola, float tolb, long alpha,
                      long beta, IComplexNumber u, int ldu,
                      IComplexNumber v, int ldv,
                      IComplexNumber q, int ldq,int ncycle );
    native int ztgsja(int matriint, char jobu, char jobv, char jobq,int m, int p, int n,int k, int l, IComplexNumber a,int lda, IComplexNumber b,int ldb, double tola, double tolb,
                      long alpha, long beta,
                      IComplexNumber u, int ldu,
                      IComplexNumber v, int ldv,
                      IComplexNumber q, int ldq,int ncycle );

    native int stgsna(int matriint, char job, char howmny,int select, int n,
                      long a, int lda,  long b,int ldb,  long vl, int ldvl,
                      long vr, int ldvr, long s,
                      long dif, int mm, int m );
    native int dtgsna(int matriint, char job, char howmny,int select, int n,
                      long a, int lda,  long b,int ldb,  long vl, int ldvl,
                      long vr, int ldvr, long s,
                      long dif, int mm, int m );
    native int ctgsna(int matriint, char job, char howmny,int select, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,
                      long s, long dif, int mm, int m );
    native int ztgsna(int matriint, char job, char howmny,int select, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,
                      long s, long dif, int mm,int m );

    native int stgsyl(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                      long c, int ldc,  long d,int ldd,  long e, int lde,
                      long f, int ldf, long scale, long dif );
    native int dtgsyl(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                      long c, int ldc,  long d,int ldd,  long e, int lde,
                      long f, int ldf, long scale,
                      long dif );
    native int ctgsyl(int matriint, char trans, int ijob,int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber c, int ldc,
                      IComplexNumber d, int ldd,
                      IComplexNumber e, int lde,
                      IComplexNumber f, int ldf,
                      long scale, long dif );
    native int ztgsyl(int matriint, char trans, int ijob,int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber c, int ldc,
                      IComplexNumber d, int ldd,
                      IComplexNumber e, int lde,
                      IComplexNumber f, int ldf,
                      long scale, long dif );

    native int stpcon(int matriint, char norm, char uplo, char diag,int n,  long ap, long rcond );
    native int dtpcon(int matriint, char norm, char uplo, char diag,int n,  long ap, long rcond );
    native int ctpcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber ap,
                      long rcond );
    native int ztpcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber ap,
                      long rcond );

    native int stprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                      long b, int ldb,  long x,int ldx, long ferr, long berr );
    native int dtprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                      long b, int ldb,  long x,int ldx, long ferr, long berr );
    native int ctprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber ap,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );
    native int ztprfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber ap,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int stptri(int matriint, char uplo, char diag, int n,
                      long ap );
    native int dtptri(int matriint, char uplo, char diag, int n,
                      long ap );
    native int ctptri(int matriint, char uplo, char diag, int n,
                      IComplexNumber ap );
    native int ztptri(int matriint, char uplo, char diag, int n,
                      IComplexNumber ap );

    native int stptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                      long b, int ldb );
    native int dtptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long ap,
                      long b, int ldb );
    native int ctptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber ap,
                      IComplexNumber b, int ldb );
    native int ztptrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber ap,
                      IComplexNumber b, int ldb );

    native int stpttf(int matriint, char transr, char uplo,int n,  long ap, long arf );
    native int dtpttf(int matriint, char transr, char uplo,int n,  long ap, long arf );
    native int ctpttf(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                      IComplexNumber arf );
    native int ztpttf(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                      IComplexNumber arf );

    native int stpttr(int matriint, char uplo, int n,
                      long ap, long a, int lda );
    native int dtpttr(int matriint, char uplo, int n,
                      long ap, long a, int lda );
    native int ctpttr(int matriint, char uplo, int n,
                      IComplexNumber ap,
                      IComplexNumber a, int lda );
    native int ztpttr(int matriint, char uplo, int n,
                      IComplexNumber ap,
                      IComplexNumber a, int lda );

    native int strcon(int matriint, char norm, char uplo, char diag,int n,  long a, int lda,
                      long rcond );
    native int dtrcon(int matriint, char norm, char uplo, char diag,int n,  long a, int lda,
                      long rcond );
    native int ctrcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber a,int lda, long rcond );
    native int ztrcon(int matriint, char norm, char uplo, char diag,int n,  IComplexNumber a,int lda, long rcond );

    native int strevc(int matriint, char side, char howmny,int select, int n,  long t,int ldt, long vl, int ldvl,
                      long vr, int ldvr, int mm,int m );
    native int dtrevc(int matriint, char side, char howmny,int select, int n,
                      long t, int ldt, long vl,int ldvl, long vr, int ldvr,int mm, int m );
    native int ctrevc(int matriint, char side, char howmny,int select, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,int mm, int m );
    native int ztrevc(int matriint, char side, char howmny,int select, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,int mm, int m );

    native int strexc(int matriint, char compq, int n, long t,int ldt, long q, int ldq,int ifst, int ilst );
    native int dtrexc(int matriint, char compq, int n,
                      long t, int ldt, long q, int ldq,int ifst, int ilst );
    native int ctrexc(int matriint, char compq, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber q, int ldq,int ifst, int ilst );
    native int ztrexc(int matriint, char compq, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber q, int ldq,int ifst, int ilst );

    native int strrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda,  long b, int ldb,
                      long x, int ldx, long ferr,
                      long berr );
    native int dtrrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda,  long b, int ldb,
                      long x, int ldx, long ferr,
                      long berr );
    native int ctrrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );
    native int ztrrfs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber x, int ldx,
                      long ferr, long berr );

    native int strsen(int matriint, char job, char compq,int select, int n, long t,int ldt, long q, int ldq, long wr,
                      long wi, int m, long s, long sep );
    native int dtrsen(int matriint, char job, char compq,int select, int n,
                      long t, int ldt, long q, int ldq,
                      long wr, long wi, int m, long s,
                      long sep );
    native int ctrsen(int matriint, char job, char compq,int select, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber q, int ldq,
                      IComplexNumber w, int m, long s,
                      long sep );
    native int ztrsen(int matriint, char job, char compq,int select, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber q, int ldq,
                      IComplexNumber w, int m, long s,
                      long sep );

    native int strsna(int matriint, char job, char howmny,int select, int n,
                      long t, int ldt,  long vl,int ldvl,  long vr, int ldvr,
                      long s, long sep, int mm, int m );
    native int dtrsna(int matriint, char job, char howmny,int select, int n,
                      long t, int ldt,  long vl,int ldvl,  long vr, int ldvr,
                      long s, long sep, int mm,int m );
    native int ctrsna(int matriint, char job, char howmny,int select, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,
                      long s, long sep, int mm, int m );
    native int ztrsna(int matriint, char job, char howmny,int select, int n,
                      IComplexNumber t, int ldt,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr,
                      long s, long sep, int mm,int m );

    native int strsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                      long a, int lda,  long b,int ldb, long c, int ldc,
                      long scale );
    native int dtrsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                      long a, int lda,  long b,int ldb, long c, int ldc,
                      long scale );
    native int ctrsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber c, int ldc,
                      long scale );
    native int ztrsyl(int matriint, char trana, char tranb,int isgn, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber c, int ldc,
                      long scale );

    native int strtri(int matriint, char uplo, char diag, int n,
                      long a, int lda );
    native int dtrtri(int matriint, char uplo, char diag, int n,
                      long a, int lda );
    native int ctrtri(int matriint, char uplo, char diag, int n,
                      IComplexNumber a, int lda );
    native int ztrtri(int matriint, char uplo, char diag, int n,
                      IComplexNumber a, int lda );

    native int strtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda, long b, int ldb );
    native int dtrtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,  long a,int lda, long b, int ldb );
    native int ctrtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb );
    native int ztrtrs(int matriint, char uplo, char trans, char diag,int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb );

    native int strttf(int matriint, char transr, char uplo,int n,  long a, int lda,
                      long arf );
    native int dtrttf(int matriint, char transr, char uplo,int n,  long a, int lda,
                      long arf );
    native int ctrttf(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );
    native int ztrttf(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );

    native int strttp(int matriint, char uplo, int n,
                      long a, int lda, long ap );
    native int dtrttp(int matriint, char uplo, int n,
                      long a, int lda, long ap );
    native int ctrttp(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber ap );
    native int ztrttp(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber ap );

    native int stzrzf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int dtzrzf(int matriint, int m, int n,
                      long a, int lda, long tau );
    native int ctzrzf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int ztzrzf(int matriint, int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int cungbr(int matriint, char vect, int m,int n, int k, IComplexNumber a,int lda,  IComplexNumber tau );
    native int zungbr(int matriint, char vect, int m,int n, int k, IComplexNumber a,int lda,  IComplexNumber tau );

    native int cunghr(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,  IComplexNumber tau );
    native int zunghr(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,  IComplexNumber tau );

    native int cunglq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
    native int zunglq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

    native int cungql(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
    native int zungql(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

    native int cungqr(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
    native int zungqr(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

    native int cungrq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );
    native int zungrq(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau );

    native int cungtr(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );
    native int zungtr(int matriint, char uplo, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau );

    native int cunmbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmbr(int matriint, char vect, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,  IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmhr(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,  IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmlq(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmlq(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmql(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmql(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmqr(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmqr(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmrq(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmrq(int matriint, char side, char trans,int m, int n, int k,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmrz(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmrz(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cunmtr(int matriint, char side, char uplo, char trans,int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zunmtr(int matriint, char side, char uplo, char trans,int m, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int cupgtr(int matriint, char uplo, int n,
                      IComplexNumber ap,
                      IComplexNumber tau,
                      IComplexNumber q, int ldq );
    native int zupgtr(int matriint, char uplo, int n,
                      IComplexNumber ap,
                      IComplexNumber tau,
                      IComplexNumber q, int ldq );

    native int cupmtr(int matriint, char side, char uplo, char trans,int m, int n,
                      IComplexNumber ap,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );
    native int zupmtr(int matriint, char side, char uplo, char trans,int m, int n,
                      IComplexNumber ap,
                      IComplexNumber tau,
                      IComplexNumber c, int ldc );

    native int sbdsdint(int matriint, char uplo, char compq,int n, long d, long e, long u,int ldu, long vt, int ldvt,
                        long q, int iq, long work,int iwork );
    native int dbdsdint(int matriint, char uplo, char compq,int n, long d, long e, long u,int ldu, long vt, int ldvt,
                        long q, int iq, long work,int iwork );

    native int sbdsvdint(int matriint, char uplo, char jobz, char range,int n, long d, long e,int vl, int vu,int il, int iu, int ns,
                         long s, long z, int ldz,
                         long work, int iwork );
    native int dbdsvdint(int matriint, char uplo, char jobz, char range,int n, long d, long e,int vl, int vu,int il, int iu, int ns,
                         long s, long z, int ldz,
                         long work, int iwork );

    native int sbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                        long d, long e, long vt, int ldvt,
                        long u, int ldu, long c,int ldc, long work );
    native int dbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                        long d, long e, long vt,int ldvt, long u, int ldu,
                        long c, int ldc, long work );
    native int cbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                        long d, long e, IComplexNumber vt,int ldvt, IComplexNumber u,int ldu, IComplexNumber c,int ldc, long work );
    native int zbdsqint(int matriint, char uplo, int n,int ncvt, int nru, int ncc,
                        long d, long e, IComplexNumber vt,int ldvt, IComplexNumber u,int ldu, IComplexNumber c,int ldc, long work );

    native int sdisnint( char job, int m, int n,
                         long d, long sep );
    native int ddisnint( char job, int m, int n,
                         long d, long sep );

    native int sgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, long ab, int ldab,
                        long d, long e, long q, int ldq,
                        long pt, int ldpt, long c,int ldc, long work );
    native int dgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, long ab, int ldab,
                        long d, long e, long q, int ldq,
                        long pt, int ldpt, long c,int ldc, long work );
    native int cgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, IComplexNumber ab,int ldab, long d, long e,
                        IComplexNumber q, int ldq,
                        IComplexNumber pt, int ldpt,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, long rwork );
    native int zgbbrint(int matriint, char vect, int m,int n, int ncc, int kl,int ku, IComplexNumber ab,int ldab, long d, long e,
                        IComplexNumber q, int ldq,
                        IComplexNumber pt, int ldpt,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, long rwork );

    native int sgbcoint(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv,
                        float anorm, long rcond, long work,int iwork );
    native int dgbcoint(int matriint, char norm, int n,int kl, int ku,  long ab,int ldab,int ipiv,
                        double anorm, long rcond, long work,int iwork );
    native int cgbcoint(int matriint, char norm, int n,int kl, int ku,
                        IComplexNumber ab, int ldab,int ipiv, float anorm,
                        long rcond, IComplexNumber work,
                        long rwork );
    native int zgbcoint(int matriint, char norm, int n,int kl, int ku,
                        IComplexNumber ab,int ldab,int ipiv,
                        double anorm, long rcond,
                        IComplexNumber work, long rwork );

    native int sgbeqint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                        long rowcnd, long colcnd, long amax );
    native int dgbeqint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                        long rowcnd, long colcnd, long amax );
    native int cgbeqint(int matriint, int m, int n,int kl, int ku,
                        IComplexNumber ab, int ldab,
                        long r, long c, long rowcnd,
                        long colcnd, long amax );
    native int zgbeqint(int matriint, int m, int n,int kl, int ku,
                        IComplexNumber ab,int ldab, long r, long c,
                        long rowcnd, long colcnd, long amax );

    native int sgbequint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                         long rowcnd, long colcnd, long amax );
    native int dgbequint(int matriint, int m, int n,int kl, int ku,  long ab,int ldab, long r, long c,
                         long rowcnd, long colcnd, long amax );
    native int cgbequint(int matriint, int m, int n,int kl, int ku,
                         IComplexNumber ab,int ldab, long r, long c,
                         long rowcnd, long colcnd, long amax );
    native int zgbequint(int matriint, int m, int n,int kl, int ku,
                         IComplexNumber ab,int ldab, long r, long c,
                         long rowcnd, long colcnd, long amax );

    native int sgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        long ab, int ldab,
                        long afb, int ldafb,int ipiv,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        long ab, int ldab,
                        long afb, int ldafb,int ipiv,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int cgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber afb,int ldafb,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zgbrfint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        IComplexNumber ab,int ldab,
                        IComplexNumber afb,int ldafb,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab,int ldab,  long afb,int ldafb,int ipiv,
                         long r,  long c,  long b,int ldb, long x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int dgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,  long ab,int ldab,  long afb,int ldafb,int ipiv,
                         long r,  long c,
                         long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int cgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,
                         IComplexNumber ab,int ldab,
                         IComplexNumber afb,int ldafb,int ipiv,
                         long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zgbrfsint(int matriint, char trans, char equed,int n, int kl, int ku,int nrhs,
                         IComplexNumber ab,int ldab,
                         IComplexNumber afb,int ldafb,int ipiv,
                         long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int sgbsint(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
    native int dgbsint(int matriint, int n, int kl,int ku, int nrhs, long ab,int ldab, int ipiv, long b,int ldb );
    native int cgbsint(int matriint, int n, int kl,int ku, int nrhs,
                       IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
    native int zgbsint(int matriint, int n, int kl,int ku, int nrhs,
                       IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );

    native int sgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                        long afb, int ldafb, int ipiv,
                        String equed, long r, long c, long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int dgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                        long afb, int ldafb, int ipiv,
                        String equed, long r, long c, long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int cgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                        long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );
    native int zgbsvint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed,
                        long r, long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );

    native int sgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                         long afb, int ldafb, int ipiv,
                         String equed, long r, long c, long b,int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int dgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, long ab, int ldab,
                         long afb, int ldafb,int ipiv, String equed, long r,
                         long c, long b, int ldb,
                         long x, int ldx, long rcond,
                         long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int cgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv,
                         String equed, long r, long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zgbsvxint(int matriint, char fact, char trans,int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv,
                         String equed, long r, long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int sgbtrint(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
    native int dgbtrint(int matriint, int m, int n,int kl, int ku, long ab,int ldab, int ipiv );
    native int cgbtrint(int matriint, int m, int n,int kl, int ku,
                        IComplexNumber ab, int ldab,int ipiv );
    native int zgbtrint(int matriint, int m, int n,int kl, int ku,
                        IComplexNumber ab, int ldab,int ipiv );

    native int sgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        long ab, int ldab,int ipiv, long b,int ldb );
    native int dgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        long ab, int ldab,int ipiv, long b,int ldb );
    native int cgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        IComplexNumber ab, int ldab,int ipiv, IComplexNumber b,int ldb );
    native int zgbtrint(int matriint, char trans, int n,int kl, int ku, int nrhs,
                        IComplexNumber ab,int ldab,int ipiv,
                        IComplexNumber b, int ldb );

    native int sgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long scale, int m, long v,int ldv );
    native int dgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long scale, int m, long v,int ldv );
    native int cgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long scale, int m,
                        IComplexNumber v, int ldv );
    native int zgebaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long scale, int m,
                        IComplexNumber v, int ldv );

    native int sgebaint(int matriint, char job, int n,
                        long a, int lda, int ilo,int ihi, long scale );
    native int dgebaint(int matriint, char job, int n,
                        long a, int lda, int ilo,int ihi, long scale );
    native int cgebaint(int matriint, char job, int n,
                        IComplexNumber a, int lda,int ilo, int ihi,
                        long scale );
    native int zgebaint(int matriint, char job, int n,
                        IComplexNumber a, int lda,int ilo, int ihi,
                        long scale );

    native int sgebrint(int matriint, int m, int n,
                        long a, int lda, long d, long e,
                        long tauq, long taup, long work,int lwork );
    native int dgebrint(int matriint, int m, int n,
                        long a, int lda, long d, long e,
                        long tauq, long taup, long work,int lwork );
    native int cgebrint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        long d, long e, IComplexNumber tauq,
                        IComplexNumber taup,
                        IComplexNumber work, int lwork );
    native int zgebrint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        long d, long e,
                        IComplexNumber tauq,
                        IComplexNumber taup,
                        IComplexNumber work, int lwork );

    native int sgecoint(int matriint, char norm, int n,
                        long a, int lda, float anorm,
                        long rcond, long work, int iwork );
    native int dgecoint(int matriint, char norm, int n,
                        long a, int lda, double anorm,
                        long rcond, long work,int iwork );
    native int cgecoint(int matriint, char norm, int n,
                        IComplexNumber a, int lda,
                        float anorm, long rcond,
                        IComplexNumber work, long rwork );
    native int zgecoint(int matriint, char norm, int n,
                        IComplexNumber a, int lda,
                        double anorm, long rcond,
                        IComplexNumber work, long rwork );

    native int sgeeqint(int matriint, int m, int n,
                        long a, int lda, long r,
                        long c, long rowcnd, long colcnd,
                        long amax );
    native int dgeeqint(int matriint, int m, int n,
                        long a, int lda, long r,
                        long c, long rowcnd, long colcnd,
                        long amax );
    native int cgeeqint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        long r, long c, long rowcnd,
                        long colcnd, long amax );
    native int zgeeqint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        long r, long c, long rowcnd,
                        long colcnd, long amax );

    native int sgeequint(int matriint, int m, int n,
                         long a, int lda, long r,
                         long c, long rowcnd, long colcnd,
                         long amax );
    native int dgeequint(int matriint, int m, int n,
                         long a, int lda, long r,
                         long c, long rowcnd, long colcnd,
                         long amax );
    native int cgeequint(int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         long r, long c, long rowcnd,
                         long colcnd, long amax );
    native int zgeequint(int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         long r, long c, long rowcnd,
                         long colcnd, long amax );

    native int sgeeint(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                       long wi, long vs, int ldvs,
                       long work, int lwork,int bwork );
    native int dgeeint(int matriint, char jobvs, char sort,int select, int n, long a,int lda, int sdim, long wr,
                       long wi, long vs, int ldvs,
                       long work, int lwork,int bwork );
    native int cgeeint(int matriint, char jobvs, char sort,int select, int n,
                       IComplexNumber a, int lda,int sdim, IComplexNumber w,
                       IComplexNumber vs, int ldvs,
                       IComplexNumber work, int lwork,
                       long rwork, int bwork );
    native int zgeeint(int matriint, char jobvs, char sort,int select, int n,
                       IComplexNumber a, int lda,int sdim, IComplexNumber w,
                       IComplexNumber vs, int ldvs,
                       IComplexNumber work, int lwork,
                       long rwork, int bwork );

    native int sgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, long a, int lda,int sdim, long wr, long wi,
                        long vs, int ldvs, long rconde,
                        long rcondv, long work, int lwork,int iwork, int liwork,int bwork );
    native int dgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, long a, int lda,int sdim, long wr, long wi,
                        long vs, int ldvs, long rconde,
                        long rcondv, long work, int lwork,int iwork, int liwork,int bwork );
    native int cgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, IComplexNumber a,int lda, int sdim,
                        IComplexNumber w,
                        IComplexNumber vs, int ldvs,
                        long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork, int bwork );
    native int zgeesint(int matriint, char jobvs, char sort,int select, char sense,int n, IComplexNumber a,int lda, int sdim,
                        IComplexNumber w,
                        IComplexNumber vs, int ldvs,
                        long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork, int bwork );

    native int sgeeint(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                       long wr, long wi, long vl, int ldvl,
                       long vr, int ldvr, long work,int lwork );
    native int dgeeint(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                       long wr, long wi, long vl,int ldvl, long vr, int ldvr,
                       long work, int lwork );
    native int cgeeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber w,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       IComplexNumber work, int lwork,
                       long rwork );
    native int zgeeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber w,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       IComplexNumber work, int lwork,
                       long rwork );

    native int sgeevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n, long a,int lda, long wr, long wi, long vl,int ldvl, long vr, int ldvr,int ilo, int ihi, long scale,
                        long abnrm, long rconde, long rcondv,
                        long work, int lwork,int iwork );
    native int dgeevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n, long a,int lda, long wr, long wi,
                        long vl, int ldvl, long vr,int ldvr, int ilo,int ihi, long scale, long abnrm,
                        long rconde, long rcondv, long work,int lwork, int iwork );
    native int cgeevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber w,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int ilo, int ihi, long scale,
                        long abnrm, long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork );
    native int zgeevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber w,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int ilo, int ihi, long scale,
                        long abnrm, long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork );

    native int sgehrint(int matriint, int n, int ilo,int ihi, long a, int lda,
                        long tau, long work, int lwork );
    native int dgehrint(int matriint, int n, int ilo,int ihi, long a, int lda,
                        long tau, long work, int lwork );
    native int cgehrint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zgehrint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda, IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int sgejsint(int matriint, char joba, char jobu,
                        char jobv, char jobr, char jobt, char jobp,int m, int n, long a,int lda, long sva, long u,int ldu, long v, int ldv,
                        long work, int lwork,int iwork );
    native int dgejsint(int matriint, char joba, char jobu,
                        char jobv, char jobr, char jobt, char jobp,int m, int n, long a,int lda, long sva, long u,int ldu, long v, int ldv,
                        long work, int lwork,int iwork );
    native int cgejsint(int matriint, char joba, char jobu,
                        char jobv, char jobr, char jobt, char jobp,int m, int n, IComplexNumber a,int lda, long sva, IComplexNumber u,int ldu, IComplexNumber v, int ldv,
                        IComplexNumber cwork, int lwork,
                        long work, int lrwork,int iwork );
    native int zgejsint(int matriint, char joba, char jobu,
                        char jobv, char jobr, char jobt, char jobp,int m, int n, IComplexNumber a,int lda, long sva, IComplexNumber u,int ldu, IComplexNumber v, int ldv,
                        IComplexNumber cwork, int lwork,
                        long work, int lrwork,int iwork );

    native int sgelq2_work(int matriint, int m, int n,
                           long a, int lda, long tau,
                           long work );
    native int dgelq2_work(int matriint, int m, int n,
                           long a, int lda, long tau,
                           long work );
    native int cgelq2_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work );
    native int zgelq2_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work );

    native int sgelqint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int dgelqint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int cgelqint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zgelqint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int sgelint(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb,
                       long work, int lwork );
    native int dgelint(int matriint, char trans, int m,int n, int nrhs, long a,int lda, long b, int ldb,
                       long work, int lwork );
    native int cgelint(int matriint, char trans, int m,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork );
    native int zgelint(int matriint, char trans, int m,int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork );

    native int sgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                        long b, int ldb, long s, float rcond,int rank, long work, int lwork,int iwork );
    native int dgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                        long b, int ldb, long s,
                        double rcond, int rank, long work,int lwork, int iwork );
    native int cgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank, IComplexNumber work,int lwork, long rwork,int iwork );
    native int zgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank, IComplexNumber work,int lwork, long rwork,int iwork );

    native int sgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                        long b, int ldb, long s, float rcond,int rank, long work,int lwork );
    native int dgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                        long b, int ldb, long s,
                        double rcond, int rank, long work,int lwork );
    native int cgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, float rcond,int rank, IComplexNumber work,int lwork, long rwork );
    native int zgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, long s, double rcond,int rank, IComplexNumber work,int lwork, long rwork );

    native int sgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                        long b, int ldb, int jpvt,
                        float rcond, int rank, long work,int lwork );
    native int dgelsint(int matriint, int m, int n,int nrhs, long a, int lda,
                        long b, int ldb, int jpvt,
                        double rcond, int rank, long work,int lwork );
    native int cgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, float rcond,int rank, IComplexNumber work,int lwork, long rwork );
    native int zgelsint(int matriint, int m, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, int jpvt, double rcond,int rank, IComplexNumber work,int lwork, long rwork );

    native int sgeqlint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int dgeqlint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int cgeqlint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zgeqlint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int sgeqp3_work(int matriint, int m, int n,
                           long a, int lda, int jpvt,
                           long tau, long work, int lwork );
    native int dgeqp3_work(int matriint, int m, int n,
                           long a, int lda, int jpvt,
                           long tau, long work, int lwork );
    native int cgeqp3_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                           IComplexNumber work, int lwork,
                           long rwork );
    native int zgeqp3_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                           IComplexNumber work, int lwork,
                           long rwork );

    native int sgeqpint(int matriint, int m, int n,
                        long a, int lda, int jpvt,
                        long tau, long work );
    native int dgeqpint(int matriint, int m, int n,
                        long a, int lda, int jpvt,
                        long tau, long work );
    native int cgeqpint(int matriint, int m, int n,
                        IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                        IComplexNumber work, long rwork );
    native int zgeqpint(int matriint, int m, int n,
                        IComplexNumber a, int lda,int jpvt, IComplexNumber tau,
                        IComplexNumber work, long rwork );

    native int sgeqr2_work(int matriint, int m, int n,
                           long a, int lda, long tau,
                           long work );
    native int dgeqr2_work(int matriint, int m, int n,
                           long a, int lda, long tau,
                           long work );
    native int cgeqr2_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work );
    native int zgeqr2_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,
                           IComplexNumber tau,
                           IComplexNumber work );

    native int sgeqrint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int dgeqrint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int cgeqrint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zgeqrint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int sgeqrfint(int matriint, int m, int n,
                         long a, int lda, long tau,
                         long work, int lwork );
    native int dgeqrfint(int matriint, int m, int n,
                         long a, int lda, long tau,
                         long work, int lwork );
    native int cgeqrfint(int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    native int zgeqrfint(int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work,int lwork );

    native int sgerfint(int matriint, char trans, int n,int nrhs,  long a, int lda,
                        long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dgerfint(int matriint, char trans, int n,int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work, int iwork );
    native int cgerfint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zgerfint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sgerfsint(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                         long r,  long c,  long b,int ldb, long x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int dgerfsint(int matriint, char trans, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                         long r,  long c,
                         long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int cgerfsint(int matriint, char trans, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,int ipiv,
                         long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zgerfsint(int matriint, char trans, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,int ipiv,
                         long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int sgerqint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int dgerqint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int cgerqint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zgerqint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int sgesdint(int matriint, char jobz, int m,int n, long a, int lda,
                        long s, long u, int ldu, long vt,int ldvt, long work, int lwork,int iwork );
    native int dgesdint(int matriint, char jobz, int m,int n, long a, int lda,
                        long s, long u, int ldu,
                        long vt, int ldvt, long work,int lwork, int iwork );
    native int cgesdint(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s,
                        IComplexNumber u, int ldu,
                        IComplexNumber vt, int ldvt,
                        IComplexNumber work, int lwork,
                        long rwork, int iwork );
    native int zgesdint(int matriint, char jobz, int m,int n, IComplexNumber a,int lda, long s,
                        IComplexNumber u, int ldu,
                        IComplexNumber vt, int ldvt,
                        IComplexNumber work, int lwork,
                        long rwork, int iwork );

    native int sgesint(int matriint, int n, int nrhs,
                       long a, int lda, int ipiv,
                       long b, int ldb );
    native int dgesint(int matriint, int n, int nrhs,
                       long a, int lda, int ipiv,
                       long b, int ldb );
    native int cgesint(int matriint, int n, int nrhs,
                       IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
    native int zgesint(int matriint, int n, int nrhs,
                       IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb );
    native int dsgesint(int matriint, int n, int nrhs,
                        long a, int lda, int ipiv,
                        long b, int ldb, long x,int ldx, long work, long swork,int iter );
    native int zcgesint(int matriint, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, IComplexNumber x,int ldx, IComplexNumber work,
                        IComplexNumber swork, long rwork,int iter );

    native int sgesvint(int matriint, char jobu, char jobvt,int m, int n, long a,int lda, long s, long u,int ldu, long vt, int ldvt,
                        long work, int lwork );
    native int dgesvint(int matriint, char jobu, char jobvt,int m, int n, long a,int lda, long s, long u,int ldu, long vt, int ldvt,
                        long work, int lwork );
    native int cgesvint(int matriint, char jobu, char jobvt,int m, int n,
                        IComplexNumber a, int lda,
                        long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, IComplexNumber work,int lwork, long rwork );
    native int zgesvint(int matriint, char jobu, char jobvt,int m, int n,
                        IComplexNumber a, int lda,
                        long s, IComplexNumber u,int ldu, IComplexNumber vt,int ldvt, IComplexNumber work,int lwork, long rwork );

    native int sgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                         long s, long u, int ldu,
                         long vt, int ldvt,
                         long work, int lwork, int iwork );
    native int dgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, long a,int lda, int vl, int vu,int il, int iu, int ns,
                         long s, long u, int ldu,
                         long vt, int ldvt,
                         long work, int lwork, int iwork );
    native int cgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                         long s, IComplexNumber u, int ldu,
                         IComplexNumber vt, int ldvt,
                         IComplexNumber work, int lwork,
                         long rwork, int iwork );
    native int zgesvdint(int matriint, char jobu, char jobvt, char range,int m, int n, IComplexNumber a,int lda, int vl, int vu,int il, int iu, int ns,
                         long s, IComplexNumber u, int ldu,
                         IComplexNumber vt, int ldvt,
                         IComplexNumber work, int lwork,
                         long rwork, int iwork );

    native int sgesvint(int matriint, char joba, char jobu,
                        char jobv, int m, int n, long a,int lda, long sva, int mv,
                        long v, int ldv, long work,int lwork );
    native int dgesvint(int matriint, char joba, char jobu,
                        char jobv, int m, int n,
                        long a, int lda, long sva,int mv, long v, int ldv,
                        long work, int lwork );
    native int cgesvint(int matriint, char joba, char jobu,
                        char jobv, int m, int n, IComplexNumber a,int lda, long sva, int mv,
                        IComplexNumber v, int ldv,
                        IComplexNumber cwork, int lwork,
                        long rwork,int lrwork );
    native int zgesvint(int matriint, char joba, char jobu,
                        char jobv, int m, int n,
                        IComplexNumber a, int lda, long sva,int mv, IComplexNumber v, int ldv,
                        IComplexNumber cwork, int lwork,
                        long rwork, int lrwork );

    native int sgesvint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                        long c, long b, int ldb, long x,int ldx, long rcond, long ferr,
                        long berr, long work, int iwork );
    native int dgesvint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                        long c, long b, int ldb, long x,int ldx, long rcond, long ferr,
                        long berr, long work, int iwork );
    native int cgesvint(int matriint, char fact, char trans,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                        long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );
    native int zgesvint(int matriint, char fact, char trans,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                        long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );

    native int sgesvxint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                         long c, long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork );
    native int dgesvxint(int matriint, char fact, char trans,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long r,
                         long c, long b, int ldb,
                         long x, int ldx, long rcond,
                         long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int cgesvxint(int matriint, char fact, char trans,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                         long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork );
    native int zgesvxint(int matriint, char fact, char trans,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long r,
                         long c, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork );

    native int sgetf2_work(int matriint, int m, int n,
                           long a, int lda, int ipiv );
    native int dgetf2_work(int matriint, int m, int n,
                           long a, int lda, int ipiv );
    native int cgetf2_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,int ipiv );
    native int zgetf2_work(int matriint, int m, int n,
                           IComplexNumber a, int lda,int ipiv );

    native int sgetrint(int matriint, int m, int n,
                        long a, int lda, int ipiv );
    native int dgetrint(int matriint, int m, int n,
                        long a, int lda, int ipiv );
    native int cgetrint(int matriint, int m, int n,
                        IComplexNumber a, int lda,int ipiv );
    native int zgetrint(int matriint, int m, int n,
                        IComplexNumber a, int lda,int ipiv );

    native int sgetrf2_work(int matriint, int m, int n,
                            long a, int lda, int ipiv );
    native int dgetrf2_work(int matriint, int m, int n,
                            long a, int lda, int ipiv );
    native int cgetrf2_work(int matriint, int m, int n,
                            IComplexNumber a, int lda,int ipiv );
    native int zgetrf2_work(int matriint, int m, int n,
                            IComplexNumber a, int lda,int ipiv );

    native int sgetrint(int matriint, int n, long a,int lda,int ipiv,
                        long work, int lwork );
    native int dgetrint(int matriint, int n, long a,int lda,int ipiv,
                        long work, int lwork );
    native int cgetrint(int matriint, int n,
                        IComplexNumber a, int lda,int ipiv,
                        IComplexNumber work, int lwork );
    native int zgetrint(int matriint, int n,
                        IComplexNumber a, int lda,int ipiv,
                        IComplexNumber work, int lwork );

    native int sgetrint(int matriint, char trans, int n,int nrhs,  long a, int lda,int ipiv, long b,int ldb );
    native int dgetrint(int matriint, char trans, int n,int nrhs,  long a,int lda,int ipiv,
                        long b, int ldb );
    native int cgetrint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                        IComplexNumber b, int ldb );
    native int zgetrint(int matriint, char trans, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                        IComplexNumber b, int ldb );

    native int sggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long lscale,  long rscale,int m, long v, int ldv );
    native int dggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long lscale,  long rscale,int m, long v, int ldv );
    native int cggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long lscale,  long rscale,int m, IComplexNumber v,int ldv );
    native int zggbaint(int matriint, char job, char side,int n, int ilo, int ihi,
                        long lscale,  long rscale,int m, IComplexNumber v,int ldv );

    native int sggbaint(int matriint, char job, int n,
                        long a, int lda, long b,int ldb, int ilo,int ihi, long lscale, long rscale,
                        long work );
    native int dggbaint(int matriint, char job, int n,
                        long a, int lda, long b,int ldb, int ilo,int ihi, long lscale, long rscale,
                        long work );
    native int cggbaint(int matriint, char job, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,int ilo, int ihi, long lscale,
                        long rscale, long work );
    native int zggbaint(int matriint, char job, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,int ilo, int ihi,
                        long lscale, long rscale, long work );

    native int sggeint(int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       long a, int lda, long b,int ldb, int sdim, long alphar,
                       long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                       long work, int lwork,int bwork );
    native int dggeint(int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       long a, int lda, long b,int ldb, int sdim, long alphar,
                       long alphai, long beta, long vsl,int ldvsl, long vsr, int ldvsr,
                       long work, int lwork,int bwork );
    native int cggeint(int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vsl, int ldvsl,
                       IComplexNumber vsr, int ldvsr,
                       IComplexNumber work, int lwork,
                       long rwork, int bwork );
    native int zggeint(int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vsl, int ldvsl,
                       IComplexNumber vsr, int ldvsr,
                       IComplexNumber work, int lwork,
                       long rwork, int bwork );

    native int sgges3_work(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg,int n,
                           long a, int lda,
                           long b, int ldb, int sdim,
                           long alphar, long alphai, long beta,
                           long vsl, int ldvsl,
                           long vsr, int ldvsr,
                           long work, int lwork,int bwork );
    native int dgges3_work(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg,int n,
                           long a, int lda,
                           long b, int ldb, int sdim,
                           long alphar, long alphai, long beta,
                           long vsl, int ldvsl,
                           long vsr, int ldvsr,
                           long work, int lwork,int bwork );
    native int cgges3_work(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg,int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vsl, int ldvsl,
                           IComplexNumber vsr, int ldvsr,
                           IComplexNumber work, int lwork,
                           long rwork, int bwork );
    native int zgges3_work(int matriint, char jobvsl, char jobvsr,
                           char sort, int selctg,int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,int sdim, IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vsl, int ldvsl,
                           IComplexNumber vsr, int ldvsr,
                           IComplexNumber work, int lwork,
                           long rwork, int bwork );

    native int sggesint(int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, char sense,int n, long a, int lda,
                        long b, int ldb, int sdim,
                        long alphar, long alphai, long beta,
                        long vsl, int ldvsl, long vsr,int ldvsr, long rconde, long rcondv,
                        long work, int lwork,int iwork, int liwork,int bwork );
    native int dggesint(int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, char sense,int n, long a, int lda,
                        long b, int ldb, int sdim,
                        long alphar, long alphai, long beta,
                        long vsl, int ldvsl, long vsr,int ldvsr, long rconde,
                        long rcondv, long work, int lwork,int iwork, int liwork,int bwork );
    native int cggesint(int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                        IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vsl, int ldvsl,
                        IComplexNumber vsr, int ldvsr,
                        long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork, int iwork,int liwork, int bwork );
    native int zggesint(int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, char sense,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, int sdim,
                        IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vsl, int ldvsl,
                        IComplexNumber vsr, int ldvsr,
                        long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork, int iwork,int liwork, int bwork );

    native int sggeint(int matriint, char jobvl, char jobvr,int n, long a, int lda, long b,int ldb, long alphar, long alphai,
                       long beta, long vl, int ldvl,
                       long vr, int ldvr, long work,int lwork );
    native int dggeint(int matriint, char jobvl, char jobvr,int n, long a, int lda,
                       long b, int ldb, long alphar,
                       long alphai, long beta, long vl,int ldvl, long vr, int ldvr,
                       long work, int lwork );
    native int cggeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       IComplexNumber work, int lwork,
                       long rwork );
    native int zggeint(int matriint, char jobvl, char jobvr,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       IComplexNumber work, int lwork,
                       long rwork );

    native int sggev3_work(int matriint, char jobvl, char jobvr,int n,
                           long a, int lda,
                           long b, int ldb,
                           long alphar, long alphai, long beta,
                           long vl, int ldvl,
                           long vr, int ldvr,
                           long work, int lwork );
    native int dggev3_work(int matriint, char jobvl, char jobvr,int n,
                           long a, int lda,
                           long b, int ldb,
                           long alphar, long alphai, long beta,
                           long vl, int ldvl,
                           long vr, int ldvr,
                           long work, int lwork );
    native int cggev3_work(int matriint, char jobvl, char jobvr,int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vl, int ldvl,
                           IComplexNumber vr, int ldvr,
                           IComplexNumber work, int lwork,
                           long rwork );
    native int zggev3_work(int matriint, char jobvl, char jobvr,int n,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber alpha,
                           IComplexNumber beta,
                           IComplexNumber vl, int ldvl,
                           IComplexNumber vr, int ldvr,
                           IComplexNumber work, int lwork,
                           long rwork );

    native int sggevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                        long alphar, long alphai, long beta,
                        long vl, int ldvl, long vr,int ldvr, int ilo,int ihi, long lscale, long rscale,
                        long abnrm, long bbnrm, long rconde,
                        long rcondv, long work, int lwork,int iwork, int bwork );
    native int dggevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n, long a,int lda, long b, int ldb,
                        long alphar, long alphai, long beta,
                        long vl, int ldvl, long vr,int ldvr, int ilo,int ihi, long lscale, long rscale,
                        long abnrm, long bbnrm, long rconde,
                        long rcondv, long work, int lwork,int iwork, int bwork );
    native int cggevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int ilo, int ihi, long lscale,
                        long rscale, long abnrm, long bbnrm,
                        long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork, int iwork,int bwork );
    native int zggevint(int matriint, char balanc, char jobvl,
                        char jobvr, char sense, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int ilo, int ihi,
                        long lscale, long rscale, long abnrm,
                        long bbnrm, long rconde, long rcondv,
                        IComplexNumber work, int lwork,
                        long rwork, int iwork,int bwork );

    native int sggglint(int matriint, int n, int m,int p, long a, int lda,
                        long b, int ldb, long d, long x,
                        long y, long work, int lwork );
    native int dggglint(int matriint, int n, int m,int p, long a, int lda,
                        long b, int ldb, long d, long x,
                        long y, long work, int lwork );
    native int cggglint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                        IComplexNumber x,
                        IComplexNumber y,
                        IComplexNumber work, int lwork );
    native int zggglint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber d,
                        IComplexNumber x,
                        IComplexNumber y,
                        IComplexNumber work, int lwork );

    native int sgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                        long a, int lda, long b,int ldb, long q, int ldq,
                        long z, int ldz );
    native int dgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                        long a, int lda, long b,int ldb, long q, int ldq,
                        long z, int ldz );
    native int cgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz );
    native int zgghrint(int matriint, char compq, char compz,int n, int ilo, int ihi,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz );

    native int sgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                           long a, int lda,
                           long b, int ldb,
                           long q, int ldq,
                           long z, int ldz,
                           long work, int lwork );
    native int dgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                           long a, int lda,
                           long b, int ldb,
                           long q, int ldq,
                           long z, int ldz,
                           long work, int lwork );
    native int cgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber q, int ldq,
                           IComplexNumber z, int ldz,
                           IComplexNumber work, int lwork );
    native int zgghd3_work(int matriint, char compq, char compz,int n, int ilo, int ihi,
                           IComplexNumber a, int lda,
                           IComplexNumber b, int ldb,
                           IComplexNumber q, int ldq,
                           IComplexNumber z, int ldz,
                           IComplexNumber work,int lwork );

    native int sgglsint(int matriint, int m, int n,int p, long a, int lda,
                        long b, int ldb, long c, long d,
                        long x, long work, int lwork );
    native int dgglsint(int matriint, int m, int n,int p, long a, int lda,
                        long b, int ldb, long c, long d,
                        long x, long work, int lwork );
    native int cgglsint(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                        IComplexNumber d,
                        IComplexNumber x,
                        IComplexNumber work, int lwork );
    native int zgglsint(int matriint, int m, int n,int p, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber c,
                        IComplexNumber d,
                        IComplexNumber x,
                        IComplexNumber work, int lwork );

    native int sggqrint(int matriint, int n, int m,int p, long a, int lda,
                        long taua, long b, int ldb,
                        long taub, long work, int lwork );
    native int dggqrint(int matriint, int n, int m,int p, long a, int lda,
                        long taua, long b, int ldb,
                        long taub, long work, int lwork );
    native int cggqrint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                        IComplexNumber b, int ldb,
                        IComplexNumber taub,
                        IComplexNumber work, int lwork );
    native int zggqrint(int matriint, int n, int m,int p, IComplexNumber a,int lda, IComplexNumber taua,
                        IComplexNumber b, int ldb,
                        IComplexNumber taub,
                        IComplexNumber work, int lwork );

    native int sggrqint(int matriint, int m, int p,int n, long a, int lda,
                        long taua, long b, int ldb,
                        long taub, long work, int lwork );
    native int dggrqint(int matriint, int m, int p,int n, long a, int lda,
                        long taua, long b, int ldb,
                        long taub, long work, int lwork );
    native int cggrqint(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                        IComplexNumber b, int ldb,
                        IComplexNumber taub,
                        IComplexNumber work, int lwork );
    native int zggrqint(int matriint, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber taua,
                        IComplexNumber b, int ldb,
                        IComplexNumber taub,
                        IComplexNumber work, int lwork );

    native int sggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int n,int p, int k, int l,
                        long a, int lda, long b,int ldb, long alpha, long beta,
                        long u, int ldu, long v,int ldv, long q, int ldq,
                        long work, int iwork );
    native int dggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int n,int p, int k, int l,
                        long a, int lda, long b,int ldb, long alpha, long beta,
                        long u, int ldu, long v,int ldv, long q, int ldq,
                        long work, int iwork );
    native int cggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int n,int p, int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        long alpha, long beta,
                        IComplexNumber u, int ldu,
                        IComplexNumber v, int ldv,
                        IComplexNumber q, int ldq,
                        IComplexNumber work, long rwork,int iwork );
    native int zggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int n,int p, int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        long alpha, long beta,
                        IComplexNumber u, int ldu,
                        IComplexNumber v, int ldv,
                        IComplexNumber q, int ldq,
                        IComplexNumber work, long rwork,int iwork );

    native int sggsvd3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            long a, int lda, long b,int ldb, long alpha, long beta,
                            long u, int ldu, long v,int ldv, long q, int ldq,
                            long work, int lwork,int iwork );
    native int dggsvd3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            long a, int lda, long b,int ldb, long alpha, long beta,
                            long u, int ldu, long v,int ldv, long q, int ldq,
                            long work, int lwork,int iwork );
    native int cggsvd3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            long alpha, long beta,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork );
    native int zggsvd3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int n,int p, int k, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            long alpha, long beta,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,
                            IComplexNumber work, int lwork,
                            long rwork, int iwork );

    native int sggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, long a, int lda,
                        long b, int ldb, float tola,
                        float tolb, int k, int l,
                        long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau, long work );
    native int dggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, long a, int lda,
                        long b, int ldb, double tola,
                        double tolb, int k, int l,
                        long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau, long work );
    native int cggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, float tola, float tolb,int k, int l,
                        IComplexNumber u, int ldu,
                        IComplexNumber v, int ldv,
                        IComplexNumber q, int ldq,int iwork, long rwork,
                        IComplexNumber tau,
                        IComplexNumber work );
    native int zggsvint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, double tola, double tolb,int k, int l,
                        IComplexNumber u, int ldu,
                        IComplexNumber v, int ldv,
                        IComplexNumber q, int ldq,int iwork, long rwork,
                        IComplexNumber tau,
                        IComplexNumber work );

    native int sggsvp3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, long a, int lda,
                            long b, int ldb, float tola,
                            float tolb, int k, int l,
                            long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau,
                            long work, int lwork );
    native int dggsvp3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, long a, int lda,
                            long b, int ldb, double tola,
                            double tolb, int k, int l,
                            long u, int ldu, long v,int ldv, long q, int ldq,int iwork, long tau, long work,int lwork );
    native int cggsvp3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, float tola, float tolb,int k, int l,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,int iwork, long rwork,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );
    native int zggsvp3_work(int matriint, char jobu, char jobv,
                            char jobq, int m, int p,int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, double tola, double tolb,int k, int l,
                            IComplexNumber u, int ldu,
                            IComplexNumber v, int ldv,
                            IComplexNumber q, int ldq,int iwork, long rwork,
                            IComplexNumber tau,
                            IComplexNumber work, int lwork );

    native int sgtcoint( char norm, int n,  long dl,
                         long d,  long du,
                         long du2,int ipiv,
                         float anorm, long rcond, long work,int iwork );
    native int dgtcoint( char norm, int n,  long dl,
                         long d,  long du,
                         long du2,int ipiv,
                         double anorm, long rcond, long work,int iwork );
    native int cgtcoint( char norm, int n,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2,int ipiv, float anorm,
                         long rcond, IComplexNumber work );
    native int zgtcoint( char norm, int n,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2,int ipiv, double anorm,
                         long rcond, IComplexNumber work );

    native int sgtrfint(int matriint, char trans, int n,int nrhs,  long dl,
                        long d,  long du,
                        long dlf,  long df,
                        long duf,  long du2,int ipiv,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dgtrfint(int matriint, char trans, int n,int nrhs,  long dl,
                        long d,  long du,
                        long dlf,  long df,
                        long duf,  long du2,int ipiv,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int cgtrfint(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber dlf,
                        IComplexNumber df,
                        IComplexNumber duf,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zgtrfint(int matriint, char trans, int n,int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber dlf,
                        IComplexNumber df,
                        IComplexNumber duf,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sgtsint(int matriint, int n, int nrhs,
                       long dl, long d, long du, long b,int ldb );
    native int dgtsint(int matriint, int n, int nrhs,
                       long dl, long d, long du, long b,int ldb );
    native int cgtsint(int matriint, int n, int nrhs,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber b, int ldb );
    native int zgtsint(int matriint, int n, int nrhs,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber b, int ldb );

    native int sgtsvint(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                        long d,  long du, long dlf,
                        long df, long duf, long du2,int ipiv,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int dgtsvint(int matriint, char fact, char trans,int n, int nrhs,  long dl,
                        long d,  long du, long dlf,
                        long df, long duf, long du2,int ipiv,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int cgtsvint(int matriint, char fact, char trans,int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber dlf,
                        IComplexNumber df,
                        IComplexNumber duf,
                        IComplexNumber du2, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zgtsvint(int matriint, char fact, char trans,int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber dlf,
                        IComplexNumber df,
                        IComplexNumber duf,
                        IComplexNumber du2, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sgttrint(int n, long dl, long d, long du,
                        long du2, int ipiv );
    native int dgttrint(int n, long dl, long d, long du,
                        long du2, int ipiv );
    native int cgttrint(int n, IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2, int ipiv );
    native int zgttrint(int n, IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2, int ipiv );

    native int sgttrint(int matriint, char trans, int n,int nrhs,  long dl,
                        long d,  long du,
                        long du2,int ipiv,
                        long b, int ldb );
    native int dgttrint(int matriint, char trans, int n,int nrhs,  long dl,
                        long d,  long du,
                        long du2,int ipiv,
                        long b, int ldb );
    native int cgttrint(int matriint, char trans, int n,int nrhs,  IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2,int ipiv, IComplexNumber b,int ldb );
    native int zgttrint(int matriint, char trans, int n,int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb );

    native int chbeint(int matriint, char jobz, char uplo,int n, int kd,
                       IComplexNumber ab, int ldab,
                       long w, IComplexNumber z,int ldz, IComplexNumber work,
                       long rwork );
    native int zhbeint(int matriint, char jobz, char uplo,int n, int kd,
                       IComplexNumber ab, int ldab,
                       long w, IComplexNumber z,int ldz, IComplexNumber work,
                       long rwork );

    native int chbevint(int matriint, char jobz, char uplo,int n, int kd,
                        IComplexNumber ab, int ldab,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
    native int zhbevint(int matriint, char jobz, char uplo,int n, int kd,
                        IComplexNumber ab, int ldab,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

    native int chbevint(int matriint, char jobz, char range,
                        char uplo, int n, int kd,
                        IComplexNumber ab, int ldab,
                        IComplexNumber q, int ldq,
                        float vl, float vu, int il,int iu, float abstol, int m,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,
                        long rwork, int iwork,int ifail );
    native int zhbevint(int matriint, char jobz, char range,
                        char uplo, int n, int kd,
                        IComplexNumber ab, int ldab,
                        IComplexNumber q, int ldq,
                        double vl, double vu, int il,int iu, double abstol, int m,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,
                        long rwork, int iwork,int ifail );

    native int chbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                        IComplexNumber ab, int ldab,
                        IComplexNumber bb, int ldbb,
                        IComplexNumber x, int ldx,
                        IComplexNumber work, long rwork );
    native int zhbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                        IComplexNumber ab, int ldab,
                        IComplexNumber bb,int ldbb, IComplexNumber x,int ldx, IComplexNumber work,
                        long rwork );

    native int chbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       long w, IComplexNumber z,int ldz, IComplexNumber work,
                       long rwork );
    native int zhbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       long w, IComplexNumber z,int ldz, IComplexNumber work,
                       long rwork );

    native int chbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                        IComplexNumber ab, int ldab,
                        IComplexNumber bb, int ldbb,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
    native int zhbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                        IComplexNumber ab, int ldab,
                        IComplexNumber bb, int ldbb,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

    native int chbgvint(int matriint, char jobz, char range,
                        char uplo, int n, int ka,int kb, IComplexNumber ab,int ldab, IComplexNumber bb,int ldbb, IComplexNumber q,int ldq, float vl, float vu,int il, int iu, float abstol,int m, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, long rwork,int iwork, int ifail );
    native int zhbgvint(int matriint, char jobz, char range,
                        char uplo, int n, int ka,int kb, IComplexNumber ab,int ldab, IComplexNumber bb,int ldbb, IComplexNumber q,int ldq, double vl, double vu,int il, int iu, double abstol,int m, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, long rwork,int iwork, int ifail );

    native int chbtrint(int matriint, char vect, char uplo,int n, int kd,
                        IComplexNumber ab, int ldab,
                        long d, long e, IComplexNumber q,int ldq, IComplexNumber work );
    native int zhbtrint(int matriint, char vect, char uplo,int n, int kd,
                        IComplexNumber ab, int ldab,
                        long d, long e, IComplexNumber q,int ldq, IComplexNumber work );

    native int checoint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, float anorm,
                        long rcond, IComplexNumber work );
    native int zhecoint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, double anorm,
                        long rcond, IComplexNumber work );

    native int cheequint(int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         long s, long scond, long amax,
                         IComplexNumber work );
    native int zheequint(int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         long s, long scond, long amax,
                         IComplexNumber work );

    native int cheeint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                       IComplexNumber work, int lwork,
                       long rwork );
    native int zheeint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                       IComplexNumber work, int lwork,
                       long rwork );

    native int cheevint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork, int liwork );
    native int zheevint(int matriint, char jobz, char uplo,int n, IComplexNumber a,int lda, long w,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork, int liwork );

    native int cheevint(int matriint, char jobz, char range,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        float vl, float vu, int il,int iu, float abstol, int m,
                        long w, IComplexNumber z,int ldz, int isuppz,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork, int liwork );
    native int zheevint(int matriint, char jobz, char range,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        double vl, double vu, int il,int iu, double abstol, int m,
                        long w, IComplexNumber z,int ldz, int isuppz,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork, int liwork );

    native int cheevint(int matriint, char jobz, char range,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        float vl, float vu, int il,int iu, float abstol, int m,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );
    native int zheevint(int matriint, char jobz, char range,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        double vl, double vu, int il,int iu, double abstol, int m,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );

    native int chegsint(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );
    native int zhegsint(int matriint, int itype, char uplo,int n, IComplexNumber a,int lda,  IComplexNumber b,int ldb );

    native int chegint(int matriint, int itype, char jobz,
                       char uplo, int n, IComplexNumber a,int lda, IComplexNumber b,int ldb, long w,
                       IComplexNumber work, int lwork,
                       long rwork );
    native int zhegint(int matriint, int itype, char jobz,
                       char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       long w, IComplexNumber work,int lwork, long rwork );

    native int chegvint(int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        long w, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
    native int zhegvint(int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        long w, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

    native int chegvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        float vl, float vu, int il,int iu, float abstol, int m,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );
    native int zhegvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        double vl, double vu, int il,int iu, double abstol, int m,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int iwork, int ifail );

    native int cherfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zherfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int cherfsint(int matriint, char uplo, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,int ipiv,
                         long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zherfsint(int matriint, char uplo, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,int ipiv,
                         long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int chesint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork );
    native int zhesint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork );

    native int chesvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,int lwork, long rwork );
    native int zhesvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, int lwork,
                        long rwork );

    native int chesvxint(int matriint, char fact, char uplo,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zhesvxint(int matriint, char fact, char uplo,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int chetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        long d, long e, IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zhetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        long d, long e,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int chetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
    native int zhetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

    native int chetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv,
                        IComplexNumber work );
    native int zhetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv,
                        IComplexNumber work );


    native int chfrint(int matriint, char transr, char uplo,
                       char trans, int n, int k,
                       float alpha,  IComplexNumber a,int lda, float beta,
                       IComplexNumber c );
    native int zhfrint(int matriint, char transr, char uplo,
                       char trans, int n, int k,
                       double alpha,  IComplexNumber a,int lda, double beta,
                       IComplexNumber c );

    native int shgeqint(int matriint, char job, char compq,
                        char compz, int n, int ilo,int ihi, long h, int ldh,
                        long t, int ldt, long alphar,
                        long alphai, long beta, long q,int ldq, long z, int ldz,
                        long work, int lwork );
    native int dhgeqint(int matriint, char job, char compq,
                        char compz, int n, int ilo,int ihi, long h, int ldh,
                        long t, int ldt, long alphar,
                        long alphai, long beta, long q,int ldq, long z, int ldz,
                        long work, int lwork );
    native int chgeqint(int matriint, char job, char compq,
                        char compz, int n, int ilo,int ihi, IComplexNumber h,int ldh, IComplexNumber t,int ldt, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, int lwork,
                        long rwork );
    native int zhgeqint(int matriint, char job, char compq,
                        char compz, int n, int ilo,int ihi, IComplexNumber h,int ldh, IComplexNumber t,int ldt, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, int lwork,
                        long rwork );

    native int chpcoint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv, float anorm,
                        long rcond, IComplexNumber work );
    native int zhpcoint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv, double anorm,
                        long rcond, IComplexNumber work );

    native int chpeint(int matriint, char jobz, char uplo,int n, IComplexNumber ap, long w,
                       IComplexNumber z, int ldz,
                       IComplexNumber work, long rwork );
    native int zhpeint(int matriint, char jobz, char uplo,int n, IComplexNumber ap,
                       long w, IComplexNumber z,int ldz, IComplexNumber work,
                       long rwork );

    native int chpevint(int matriint, char jobz, char uplo,int n, IComplexNumber ap,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
    native int zhpevint(int matriint, char jobz, char uplo,int n, IComplexNumber ap,
                        long w, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

    native int chpevint(int matriint, char jobz, char range,
                        char uplo, int n,
                        IComplexNumber ap, float vl, float vu,int il, int iu, float abstol,int m, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, long rwork,int iwork, int ifail );
    native int zhpevint(int matriint, char jobz, char range,
                        char uplo, int n,
                        IComplexNumber ap, double vl, double vu,int il, int iu, double abstol,int m, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, long rwork,int iwork, int ifail );

    native int chpgsint(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                        IComplexNumber bp );
    native int zhpgsint(int matriint, int itype, char uplo,int n, IComplexNumber ap,
                        IComplexNumber bp );

    native int chpgint(int matriint, int itype, char jobz,
                       char uplo, int n,
                       IComplexNumber ap,
                       IComplexNumber bp, long w,
                       IComplexNumber z, int ldz,
                       IComplexNumber work, long rwork );
    native int zhpgint(int matriint, int itype, char jobz,
                       char uplo, int n,
                       IComplexNumber ap,
                       IComplexNumber bp, long w,
                       IComplexNumber z, int ldz,
                       IComplexNumber work, long rwork );

    native int chpgvint(int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber bp, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork, int liwork );
    native int zhpgvint(int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber bp, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork, int liwork );

    native int chpgvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber bp, float vl, float vu,int il, int iu, float abstol,int m, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, long rwork,int iwork, int ifail );
    native int zhpgvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber bp, double vl, double vu,int il, int iu, double abstol,int m, long w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, long rwork,int iwork, int ifail );

    native int chprfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zhprfint(int matriint, char uplo, int n,int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int chpsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zhpsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

    native int chpsvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zhpsvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int chptrint(int matriint, char uplo, int n,
                        IComplexNumber ap, long d, long e,
                        IComplexNumber tau );
    native int zhptrint(int matriint, char uplo, int n,
                        IComplexNumber ap, long d, long e,
                        IComplexNumber tau );

    native int chptrint(int matriint, char uplo, int n,
                        IComplexNumber ap, int ipiv );
    native int zhptrint(int matriint, char uplo, int n,
                        IComplexNumber ap, int ipiv );

    native int chptrint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber work );
    native int zhptrint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber work );

    native int chptrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zhptrint(int matriint, char uplo, int n,int nrhs,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber b, int ldb );

    native int shseiint(int matriint, char job, char eigsrc,
                        char initv, int select,int n,  long h, int ldh,
                        long wr,  long wi, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work,int ifaill, int ifailr );
    native int dhseiint(int matriint, char job, char eigsrc,
                        char initv, int select,int n,  long h, int ldh,
                        long wr,  long wi, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work,int ifaill, int ifailr );
    native int chseiint(int matriint, char job, char eigsrc,
                        char initv,int select,int n,  IComplexNumber h,int ldh, IComplexNumber w,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int mm, int m,
                        IComplexNumber work, long rwork,int ifaill, int ifailr );
    native int zhseiint(int matriint, char job, char eigsrc,
                        char initv,int select,int n,  IComplexNumber h,int ldh, IComplexNumber w,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int mm, int m,
                        IComplexNumber work, long rwork,int ifaill, int ifailr );

    native int shseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                        long h, int ldh, long wr, long wi,
                        long z, int ldz, long work,int lwork );
    native int dhseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                        long h, int ldh, long wr,
                        long wi, long z, int ldz,
                        long work, int lwork );
    native int chseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                        IComplexNumber h, int ldh,
                        IComplexNumber w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, int lwork );
    native int zhseqint(int matriint, char job, char compz,int n, int ilo, int ihi,
                        IComplexNumber h, int ldh,
                        IComplexNumber w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, int lwork );

    native int clacgint(int n, IComplexNumber x,int incx );
    native int zlacgint(int n, IComplexNumber x,int incx );

    native int slacn2_work(int n, long v, long x,int isgn, long est, int kase,int isave );
    native int dlacn2_work(int n, long v, long x,int isgn, long est, int kase,int isave );
    native int clacn2_work(int n, IComplexNumber v,
                           IComplexNumber x,
                           long est, int kase,int isave );
    native int zlacn2_work(int n, IComplexNumber v,
                           IComplexNumber x,
                           long est, int kase,int isave );

    native int slacpint(int matriint, char uplo, int m,int n,  long a, int lda,
                        long b, int ldb );
    native int dlacpint(int matriint, char uplo, int m,int n,  long a, int lda,
                        long b, int ldb );
    native int clacpint(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int zlacpint(int matriint, char uplo, int m,int n,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

    native int clacp2_work(int matriint, char uplo, int m,int n,  long a, int lda,
                           IComplexNumber b, int ldb );
    native int zlacp2_work(int matriint, char uplo, int m,int n,  long a, int lda,
                           IComplexNumber b, int ldb );

    native int zlag2int(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber sa, int ldsa );

    native int slag2int(int matriint, int m, int n,
                        long sa, int ldsa, long a,int lda );

    native int dlag2int(int matriint, int m, int n,
                        long a, int lda, long sa,int ldsa );

    native int clag2int(int matriint, int m, int n,
                        IComplexNumber sa, int ldsa,
                        IComplexNumber a, int lda );

    native int slaggint(int matriint, int m, int n,int kl, int ku,  long d,
                        long a, int lda, int iseed,
                        long work );
    native int dlaggint(int matriint, int m, int n,int kl, int ku,  long d,
                        long a, int lda, int iseed,
                        long work );
    native int claggint(int matriint, int m, int n,int kl, int ku,  long d,
                        IComplexNumber a, int lda,int iseed, IComplexNumber work );
    native int zlaggint(int matriint, int m, int n,int kl, int ku,  long d,
                        IComplexNumber a, int lda,int iseed,
                        IComplexNumber work );

    native int claghint(int matriint, int n, int k,
                        long d, IComplexNumber a,int lda, int iseed,
                        IComplexNumber work );
    native int zlaghint(int matriint, int n, int k,
                        long d, IComplexNumber a,int lda, int iseed,
                        IComplexNumber work );

    native int slagsint(int matriint, int n, int k,
                        long d, long a, int lda,int iseed, long work );
    native int dlagsint(int matriint, int n, int k,
                        long d, long a, int lda,int iseed, long work );
    native int clagsint(int matriint, int n, int k,
                        long d, IComplexNumber a,int lda, int iseed,
                        IComplexNumber work );
    native int zlagsint(int matriint, int n, int k,
                        long d, IComplexNumber a,int lda, int iseed,
                        IComplexNumber work );

    native int slapmint(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
    native int dlapmint(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
    native int clapmint(int matriint, int forwrd,int m, int n,
                        IComplexNumber x, int ldx,int k );
    native int zlapmint(int matriint, int forwrd,int m, int n,
                        IComplexNumber x, int ldx,int k );


    native int slartgint( float f, float g, long cs, long sn,
                          long r );
    native int dlartgint( double f, double g, long cs, long sn,
                          long r );

    native int slartgint( float x, float y, float sigma, long cs,
                          long sn );
    native int dlartgint( double x, double y, double sigma, long cs,
                          long sn );

    native float slapy2_work( float x, float y );
    native double dlapy2_work( double x, double y );

    native float slapy3_work( float x, float y, float z );
    native  double dlapy3_work( double x, double y, double z );

    native   float slamcint( char cmach );
    native  double dlamcint( char cmach );

    native float slangint(int matriint, char norm, int m,int n,  long a, int lda,
                   long work );
    native double dlangint(int matriint, char norm, int m,int n,  long a, int lda,
                    long work );
    native float clangint(int matriint, char norm, int m,int n,  IComplexNumber a,int lda, long work );
    native double zlangint(int matriint, char norm, int m,int n,  IComplexNumber a,int lda, long work );

    native float clanhint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );
    native double zlanhint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );

    native float slansint(int matriint, char norm, char uplo,int n,  long a, int lda,
                   long work );
    native  double dlansint(int matriint, char norm, char uplo,int n,  long a, int lda,
                    long work );
    native  float clansint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );
    native  double zlansint(int matriint, char norm, char uplo,int n,  IComplexNumber a,int lda, long work );

    native float slantint(int matriint, char norm, char uplo,
                   char diag, int m, int n,  long a,int lda, long work );
    native  double dlantint(int matriint, char norm, char uplo,
                    char diag, int m, int n,
                    long a, int lda, long work );
    native float clantint(int matriint, char norm, char uplo,
                   char diag, int m, int n,
                   IComplexNumber a, int lda,
                   long work );
    native double zlantint(int matriint, char norm, char uplo,
                    char diag, int m, int n,
                    IComplexNumber a, int lda,
                    long work );

    native int slarfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k,  long v,int ldv,  long t, int ldt,
                        long c, int ldc, long work,int ldwork );
    native int dlarfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k,  long v,int ldv,  long t, int ldt,
                        long c, int ldc, long work,int ldwork );
    native int clarfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k,
                        IComplexNumber v, int ldv,
                        IComplexNumber t, int ldt,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int ldwork );
    native int zlarfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k,
                        IComplexNumber v, int ldv,
                        IComplexNumber t, int ldt,
                        IComplexNumber c, int ldc,
                        IComplexNumber work,int ldwork );

    native int slarfint(int n, long alpha, long x,int incx, long tau );
    native int dlarfint(int n, long alpha, long x,int incx, long tau );
    native int clarfint(int n, IComplexNumber alpha,
                        IComplexNumber x, int incx,
                        IComplexNumber tau );
    native int zlarfint(int n, IComplexNumber alpha,
                        IComplexNumber x, int incx,
                        IComplexNumber tau );

    native int slarfint(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
    native int dlarfint(int matriint, char direct, char storev,int n, int k,  long v,int ldv,  long tau, long t,int ldt );
    native int clarfint(int matriint, char direct, char storev,int n, int k,
                        IComplexNumber v, int ldv,
                        IComplexNumber tau,
                        IComplexNumber t, int ldt );
    native int zlarfint(int matriint, char direct, char storev,int n, int k,
                        IComplexNumber v, int ldv,
                        IComplexNumber tau,
                        IComplexNumber t, int ldt );

    native int slarfint(int matriint, char side, int m,int n,  long v, float tau,
                        long c, int ldc, long work );
    native int dlarfint(int matriint, char side, int m,int n,  long v, double tau,
                        long c, int ldc, long work );
    native int clarfint(int matriint, char side, int m,int n,  IComplexNumber v,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work );
    native int zlarfint(int matriint, char side, int m,int n,  IComplexNumber v,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work );

    native int slarnint(int idist, int iseed,int n, long x );
    native int dlarnint(int idist, int iseed,int n, long x );
    native int clarnint(int idist, int iseed,int n, IComplexNumber x );
    native int zlarnint(int idist, int iseed,int n, IComplexNumber x );


    native int slascint(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, long a,int lda );
    native int dlascint(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, long a,int lda );
    native int clascint(int matriint, char type, int kl,int ku, float cfrom, float cto,int m, int n, IComplexNumber a,int lda );
    native int zlascint(int matriint, char type, int kl,int ku, double cfrom, double cto,int m, int n, IComplexNumber a,int lda );

    native int slaseint(int matriint, char uplo, int m,int n, float alpha, float beta, long a,int lda );
    native int dlaseint(int matriint, char uplo, int m,int n, double alpha, double beta,
                        long a, int lda );
    native int claseint(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber a, int lda );
    native int zlaseint(int matriint, char uplo, int m,int n, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber a, int lda );

    native int slasrint( char id, int n, long d );
    native int dlasrint( char id, int n, long d );

    native int slaswint(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
    native int dlaswint(int matriint, int n, long a,int lda, int k1, int k2,int ipiv, int incx );
    native int claswint(int matriint, int n,
                        IComplexNumber a, int lda,int k1, int k2,int ipiv, int incx );
    native int zlaswint(int matriint, int n,
                        IComplexNumber a, int lda,int k1, int k2,int ipiv, int incx );

    native int slatmint(int matriint, int m, int n,
                        char dist, int iseed, char sym,
                        long d, int mode, float cond,
                        float dmax, int kl, int ku,
                        char pack, long a, int lda,
                        long work );
    native int dlatmint(int matriint, int m, int n,
                        char dist, int iseed, char sym,
                        long d, int mode, double cond,
                        double dmax, int kl, int ku,
                        char pack, long a, int lda,
                        long work );
    native int clatmint(int matriint, int m, int n,
                        char dist, int iseed, char sym,
                        long d, int mode, float cond,
                        float dmax, int kl, int ku,
                        char pack, IComplexNumber a,int lda, IComplexNumber work );
    native int zlatmint(int matriint, int m, int n,
                        char dist, int iseed, char sym,
                        long d, int mode, double cond,
                        double dmax, int kl, int ku,
                        char pack, IComplexNumber a,int lda, IComplexNumber work );

    native int slauuint(int matriint, char uplo, int n,
                        long a, int lda );
    native int dlauuint(int matriint, char uplo, int n,
                        long a, int lda );
    native int clauuint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda );
    native int zlauuint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda );

    native int sopgtint(int matriint, char uplo, int n,
                        long ap,  long tau, long q,int ldq, long work );
    native int dopgtint(int matriint, char uplo, int n,
                        long ap,  long tau, long q,int ldq, long work );

    native int sopmtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        long ap,  long tau, long c,int ldc, long work );
    native int dopmtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        long ap,  long tau, long c,int ldc, long work );

    native int sorgbint(int matriint, char vect, int m,int n, int k, long a,int lda,  long tau, long work,int lwork );
    native int dorgbint(int matriint, char vect, int m,int n, int k, long a,int lda,  long tau, long work,int lwork );

    native int sorghint(int matriint, int n, int ilo,int ihi, long a, int lda,
                        long tau, long work,int lwork );
    native int dorghint(int matriint, int n, int ilo,int ihi, long a, int lda,
                        long tau, long work,int lwork );

    native int sorglint(int matriint, int m, int n,int k, long a, int lda,
                        long tau, long work,int lwork );
    native int dorglint(int matriint, int m, int n,int k, long a, int lda,
                        long tau, long work,int lwork );

    native int sorgqint(int matriint, int m, int n,int k, long a, int lda,
                        long tau, long work,int lwork );
    native int dorgqint(int matriint, int m, int n,int k, long a, int lda,
                        long tau, long work,int lwork );



    native int sorgrint(int matriint, int m, int n,int k, long a, int lda,
                        long tau, long work,int lwork );
    native int dorgrint(int matriint, int m, int n,int k, long a, int lda,
                        long tau, long work,int lwork );

    native int sorgtint(int matriint, char uplo, int n,
                        long a, int lda,  long tau,
                        long work, int lwork );
    native int dorgtint(int matriint, char uplo, int n,
                        long a, int lda,  long tau,
                        long work, int lwork );

    native int sormbint(int matriint, char vect, char side,
                        char trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormbint(int matriint, char vect, char side,
                        char trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );

    native int sormhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );

    native int sormlint(int matriint, char side, char trans,int m, int n, int k,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormlint(int matriint, char side, char trans,int m, int n, int k,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );

    native int sormqint(int matriint, char side, char trans,int m, int n, int k,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormqint(int matriint, char side, char trans,int m, int n, int k,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );



    native int sormrint(int matriint, char side, char trans,int m, int n, int k,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormrint(int matriint, char side, char trans,int m, int n, int k,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );

    native int sormrint(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormrint(int matriint, char side, char trans,int m, int n, int k,int l,  long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );

    native int sormtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );
    native int dormtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        long a, int lda,
                        long tau, long c, int ldc,
                        long work, int lwork );

    native int spbcoint(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                        float anorm, long rcond, long work,int iwork );
    native int dpbcoint(int matriint, char uplo, int n,int kd,  long ab,int ldab, double anorm, long rcond,
                        long work, int iwork );
    native int cpbcoint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, float anorm, long rcond,
                        IComplexNumber work, long rwork );
    native int zpbcoint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, double anorm, long rcond,
                        IComplexNumber work, long rwork );

    native int spbeqint(int matriint, char uplo, int n,int kd,  long ab, int ldab,
                        long s, long scond, long amax );
    native int dpbeqint(int matriint, char uplo, int n,int kd,  long ab,int ldab, long s, long scond,
                        long amax );
    native int cpbeqint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                        long amax );
    native int zpbeqint(int matriint, char uplo, int n,int kd,  IComplexNumber ab,int ldab, long s, long scond,
                        long amax );

    native int spbrfint(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab,  long afb,int ldafb,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dpbrfint(int matriint, char uplo, int n,int kd, int nrhs,
                        long ab, int ldab,
                        long afb, int ldafb,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work, int iwork );
    native int cpbrfint(int matriint, char uplo, int n,int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber afb,int ldafb,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zpbrfint(int matriint, char uplo, int n,int kd, int nrhs,
                        IComplexNumber ab,int ldab,
                        IComplexNumber afb,int ldafb,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int spbstint(int matriint, char uplo, int n,int kb, long bb, int ldbb );
    native int dpbstint(int matriint, char uplo, int n,int kb, long bb, int ldbb );
    native int cpbstint(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );
    native int zpbstint(int matriint, char uplo, int n,int kb, IComplexNumber bb,int ldbb );

    native int spbsint(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
    native int dpbsint(int matriint, char uplo, int n,int kd, int nrhs, long ab,int ldab, long b, int ldb );
    native int cpbsint(int matriint, char uplo, int n,int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb );
    native int zpbsint(int matriint, char uplo, int n,int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb );

    native int spbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                        long ab, int ldab, long afb,int ldafb, String equed, long s,
                        long b, int ldb, long x,int ldx, long rcond, long ferr,
                        long berr, long work, int iwork );
    native int dpbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                        long ab, int ldab, long afb,int ldafb, String equed, long s,
                        long b, int ldb, long x,int ldx, long rcond, long ferr,
                        long berr, long work, int iwork );
    native int cpbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber afb, int ldafb,
                        String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );
    native int zpbsvint(int matriint, char fact, char uplo,int n, int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber afb, int ldafb,
                        String equed, long s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int spbtrint(int matriint, char uplo, int n,int kd, long ab, int ldab );
    native int dpbtrint(int matriint, char uplo, int n,int kd, long ab, int ldab );
    native int cpbtrint(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );
    native int zpbtrint(int matriint, char uplo, int n,int kd, IComplexNumber ab,int ldab );

    native int spbtrint(int matriint, char uplo, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb );
    native int dpbtrint(int matriint, char uplo, int n,int kd, int nrhs,
                        long ab, int ldab, long b,int ldb );
    native int cpbtrint(int matriint, char uplo, int n,int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb );
    native int zpbtrint(int matriint, char uplo, int n,int kd, int nrhs,
                        IComplexNumber ab,int ldab, IComplexNumber b,int ldb );

    native int spftrint(int matriint, char transr, char uplo,int n, long a );
    native int dpftrint(int matriint, char transr, char uplo,int n, long a );
    native int cpftrint(int matriint, char transr, char uplo,int n, IComplexNumber a );
    native int zpftrint(int matriint, char transr, char uplo,int n, IComplexNumber a );



    native int spftrint(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                        long b, int ldb );
    native int dpftrint(int matriint, char transr, char uplo,int n, int nrhs,  long a,
                        long b, int ldb );
    native int cpftrint(int matriint, char transr, char uplo,int n, int nrhs,
                        IComplexNumber a,
                        IComplexNumber b, int ldb );
    native int zpftrint(int matriint, char transr, char uplo,int n, int nrhs,
                        IComplexNumber a,
                        IComplexNumber b, int ldb );

    native int spocoint(int matriint, char uplo, int n,
                        long a, int lda, float anorm,
                        long rcond, long work, int iwork );
    native int dpocoint(int matriint, char uplo, int n,
                        long a, int lda, double anorm,
                        long rcond, long work,int iwork );
    native int cpocoint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        float anorm, long rcond,
                        IComplexNumber work, long rwork );
    native int zpocoint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        double anorm, long rcond,
                        IComplexNumber work, long rwork );

    native int spoeqint(int matriint, int n,  long a,int lda, long s, long scond,
                        long amax );
    native int dpoeqint(int matriint, int n,  long a,int lda, long s, long scond,
                        long amax );
    native int cpoeqint(int matriint, int n,
                        IComplexNumber a, int lda,
                        long s, long scond, long amax );
    native int zpoeqint(int matriint, int n,
                        IComplexNumber a, int lda,
                        long s, long scond, long amax );

    native int spoequint(int matriint, int n,  long a,int lda, long s, long scond,
                         long amax );
    native int dpoequint(int matriint, int n,
                         long a, int lda, long s,
                         long scond, long amax );
    native int cpoequint(int matriint, int n,
                         IComplexNumber a, int lda,
                         long s, long scond, long amax );
    native int zpoequint(int matriint, int n,
                         IComplexNumber a, int lda,
                         long s, long scond, long amax );

    native int sporfint(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                        long af, int ldaf,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work, int iwork );
    native int dporfint(int matriint, char uplo, int n,int nrhs,  long a,int lda,  long af,int ldaf,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int cporfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zporfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sporfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,  long s,
                         long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int dporfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,  long s,
                         long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int cporfsint(int matriint, char uplo, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,  long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zporfsint(int matriint, char uplo, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,  long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int sposint(int matriint, char uplo, int n,int nrhs, long a, int lda,
                       long b, int ldb );
    native int dposint(int matriint, char uplo, int n,int nrhs, long a, int lda,
                       long b, int ldb );
    native int cposint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int zposint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int dsposint(int matriint, char uplo, int n,int nrhs, long a, int lda,
                        long b, int ldb, long x,int ldx, long work, long swork,int iter );
    native int zcposint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, IComplexNumber b,int ldb, IComplexNumber x,int ldx, IComplexNumber work,
                        IComplexNumber swork, long rwork,int iter );

    native int sposvint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                        String equed, long s, long b, int ldb,
                        long x, int ldx, long rcond,
                        long ferr, long berr, long work,int iwork );
    native int dposvint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                        String equed, long s, long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int cposvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );
    native int zposvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        String equed, long s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sposvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                         String equed, long s, long b,int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int dposvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,
                         String equed, long s, long b,int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int cposvxint(int matriint, char fact, char uplo,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         String equed, long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork );
    native int zposvxint(int matriint, char fact, char uplo,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int spotrf2_work(int matriint, char uplo, int n,
                            long a, int lda );
    native int dpotrf2_work(int matriint, char uplo, int n,
                            long a, int lda );
    native int cpotrf2_work(int matriint, char uplo, int n,
                            IComplexNumber a, int lda );
    native int zpotrf2_work(int matriint, char uplo, int n,
                            IComplexNumber a, int lda );

    native int spotrint(int matriint, char uplo, int n,
                        long a, int lda );
    native int dpotrint(int matriint, char uplo, int n,
                        long a, int lda );
    native int cpotrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda );
    native int zpotrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda );



    native int spotrint(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                        long b, int ldb );
    native int dpotrint(int matriint, char uplo, int n,int nrhs,  long a,int lda, long b, int ldb );
    native int cpotrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );
    native int zpotrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b,int ldb );

    native int sppcoint(int matriint, char uplo, int n,
                        long ap, float anorm, long rcond,
                        long work, int iwork );
    native int dppcoint(int matriint, char uplo, int n,
                        long ap, double anorm, long rcond,
                        long work, int iwork );
    native int cppcoint(int matriint, char uplo, int n,
                        IComplexNumber ap, float anorm,
                        long rcond, IComplexNumber work,
                        long rwork );
    native int zppcoint(int matriint, char uplo, int n,
                        IComplexNumber ap, double anorm,
                        long rcond, IComplexNumber work,
                        long rwork );

    native int sppeqint(int matriint, char uplo, int n,
                        long ap, long s, long scond,
                        long amax );
    native int dppeqint(int matriint, char uplo, int n,
                        long ap, long s, long scond,
                        long amax );
    native int cppeqint(int matriint, char uplo, int n,
                        IComplexNumber ap, long s,
                        long scond, long amax );
    native int zppeqint(int matriint, char uplo, int n,
                        IComplexNumber ap, long s,
                        long scond, long amax );

    native int spprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                        long afp,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dpprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                        long afp,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int cpprfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber afp,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zpprfint(int matriint, char uplo, int n,int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sppsint(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
    native int dppsint(int matriint, char uplo, int n,int nrhs, long ap, long b,int ldb );
    native int cppsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                       IComplexNumber b, int ldb );
    native int zppsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,
                       IComplexNumber b, int ldb );

    native int sppsvint(int matriint, char fact, char uplo,int n, int nrhs, long ap,
                        long afp, String equed, long s, long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int dppsvint(int matriint, char fact, char uplo,int n, int nrhs, long ap,
                        long afp, String equed, long s, long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int cppsvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp, String equed,
                        long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );
    native int zppsvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp, String equed,
                        long s, IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,
                        long rwork );

    native int spptrint(int matriint, char uplo, int n,
                        long ap );
    native int dpptrint(int matriint, char uplo, int n,
                        long ap );
    native int cpptrint(int matriint, char uplo, int n,
                        IComplexNumber ap );
    native int zpptrint(int matriint, char uplo, int n,
                        IComplexNumber ap );



    native int spptrint(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
    native int dpptrint(int matriint, char uplo, int n,int nrhs,  long ap, long b,int ldb );
    native int cpptrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber b, int ldb );
    native int zpptrint(int matriint, char uplo, int n,int nrhs,
                        IComplexNumber ap,
                        IComplexNumber b, int ldb );

    native int spstrint(int matriint, char uplo, int n,
                        long a, int lda, int piv,int rank, float tol, long work );
    native int dpstrint(int matriint, char uplo, int n,
                        long a, int lda, int piv,int rank, double tol, long work );
    native int cpstrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int piv, int rank, float tol,
                        long work );
    native int zpstrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int piv, int rank, double tol,
                        long work );

    native int sptcoint(int n,  long d,  long e,
                        float anorm, long rcond, long work );
    native int dptcoint(int n,  long d,  long e,
                        double anorm, long rcond, long work );
    native int cptcoint(int n,  long d,
                        IComplexNumber e, float anorm,
                        long rcond, long work );
    native int zptcoint(int n,  long d,
                        IComplexNumber e, double anorm,
                        long rcond, long work );

    native int spteqint(int matriint, char compz, int n,
                        long d, long e, long z, int ldz,
                        long work );
    native int dpteqint(int matriint, char compz, int n,
                        long d, long e, long z, int ldz,
                        long work );
    native int cpteqint(int matriint, char compz, int n,
                        long d, long e, IComplexNumber z,int ldz, long work );
    native int zpteqint(int matriint, char compz, int n,
                        long d, long e, IComplexNumber z,int ldz, long work );

    native int sptrfint(int matriint, int n, int nrhs,
                        long d,  long e,  long df,
                        long ef,  long b, int ldb,
                        long x, int ldx, long ferr,
                        long berr, long work );
    native int dptrfint(int matriint, int n, int nrhs,
                        long d,  long e,
                        long df,  long ef,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work );
    native int cptrfint(int matriint, char uplo, int n,int nrhs,  long d,
                        IComplexNumber e,  long df,
                        IComplexNumber ef,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zptrfint(int matriint, char uplo, int n,int nrhs,  long d,
                        IComplexNumber e,
                        long df,
                        IComplexNumber ef,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sptsint(int matriint, int n, int nrhs,
                       long d, long e, long b, int ldb );
    native int dptsint(int matriint, int n, int nrhs,
                       long d, long e, long b,int ldb );
    native int cptsint(int matriint, int n, int nrhs,
                       long d, IComplexNumber e,
                       IComplexNumber b, int ldb );
    native int zptsint(int matriint, int n, int nrhs,
                       long d, IComplexNumber e,
                       IComplexNumber b, int ldb );

    native int sptsvint(int matriint, char fact, int n,int nrhs,  long d,  long e,
                        long df, long ef,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work );
    native int dptsvint(int matriint, char fact, int n,int nrhs,  long d,
                        long e, long df, long ef,
                        long b, int ldb, long x,int ldx, long rcond, long ferr,
                        long berr, long work );
    native int cptsvint(int matriint, char fact, int n,int nrhs,  long d,
                        IComplexNumber e, long df,
                        IComplexNumber ef,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zptsvint(int matriint, char fact, int n,int nrhs,  long d,
                        IComplexNumber e, long df,
                        IComplexNumber ef,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int spttrint(int n, long d, long e );
    native int dpttrint(int n, long d, long e );
    native int cpttrint(int n, long d,
                        IComplexNumber e );
    native int zpttrint(int n, long d,
                        IComplexNumber e );

    native int spttrint(int matriint, int n, int nrhs,
                        long d,  long e, long b,int ldb );
    native int dpttrint(int matriint, int n, int nrhs,
                        long d,  long e, long b,int ldb );
    native int cpttrint(int matriint, char uplo, int n,int nrhs,  long d,
                        IComplexNumber e,
                        IComplexNumber b, int ldb );
    native int zpttrint(int matriint, char uplo, int n,int nrhs,  long d,
                        IComplexNumber e,
                        IComplexNumber b, int ldb );

    native int ssbeint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work );
    native int dsbeint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work );

    native int ssbevint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );
    native int dsbevint(int matriint, char jobz, char uplo,int n, int kd, long ab,int ldab, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );

    native int ssbevint(int matriint, char jobz, char range,
                        char uplo, int n, int kd,
                        long ab, int ldab, long q,int ldq, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );
    native int dsbevint(int matriint, char jobz, char range,
                        char uplo, int n, int kd,
                        long ab, int ldab, long q,int ldq, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );

    native int ssbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                        long ab, int ldab,  long bb,int ldbb, long x, int ldx,
                        long work );
    native int dsbgsint(int matriint, char vect, char uplo,int n, int ka, int kb,
                        long ab, int ldab,  long bb,int ldbb, long x, int ldx,
                        long work );

    native int ssbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                       long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work );
    native int dsbgint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                       long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work );

    native int ssbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                        long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );
    native int dsbgvint(int matriint, char jobz, char uplo,int n, int ka, int kb,
                        long ab, int ldab, long bb,int ldbb, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );

    native int ssbgvint(int matriint, char jobz, char range,
                        char uplo, int n, int ka,int kb, long ab, int ldab,
                        long bb, int ldbb, long q,int ldq, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );
    native int dsbgvint(int matriint, char jobz, char range,
                        char uplo, int n, int ka,int kb, long ab, int ldab,
                        long bb, int ldbb, long q,int ldq, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, long work, int iwork,int ifail );

    native int ssbtrint(int matriint, char vect, char uplo,int n, int kd, long ab,int ldab, long d, long e, long q,int ldq, long work );
    native int dsbtrint(int matriint, char vect, char uplo,int n, int kd, long ab,int ldab, long d, long e,
                        long q, int ldq, long work );

    native int ssfrint(int matriint, char transr, char uplo,
                       char trans, int n, int k,
                       float alpha,  long a, int lda,
                       float beta, long c );
    native int dsfrint(int matriint, char transr, char uplo,
                       char trans, int n, int k,
                       double alpha,  long a, int lda,
                       double beta, long c );

    native int sspcoint(int matriint, char uplo, int n,
                        long ap,int ipiv,
                        float anorm, long rcond, long work,int iwork );
    native int dspcoint(int matriint, char uplo, int n,
                        long ap,int ipiv,
                        double anorm, long rcond, long work,int iwork );
    native int cspcoint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv, float anorm,
                        long rcond, IComplexNumber work );
    native int zspcoint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv, double anorm,
                        long rcond, IComplexNumber work );

    native int sspeint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work );
    native int dspeint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work );

    native int sspevint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );
    native int dspevint(int matriint, char jobz, char uplo,int n, long ap, long w, long z,int ldz, long work, int lwork,int iwork, int liwork );

    native int sspevint(int matriint, char jobz, char range,
                        char uplo, int n, long ap, float vl,
                        float vu, int il, int iu,
                        float abstol, int m, long w, long z,int ldz, long work, int iwork,int ifail );
    native int dspevint(int matriint, char jobz, char range,
                        char uplo, int n, long ap, double vl,
                        double vu, int il, int iu,
                        double abstol, int m, long w,
                        long z, int ldz, long work,int iwork, int ifail );

    native int sspgsint(int matriint, int itype, char uplo,int n, long ap,  long bp );
    native int dspgsint(int matriint, int itype, char uplo,int n, long ap,  long bp );

    native int sspgint(int matriint, int itype, char jobz,
                       char uplo, int n, long ap, long bp,
                       long w, long z, int ldz,
                       long work );
    native int dspgint(int matriint, int itype, char jobz,
                       char uplo, int n, long ap, long bp,
                       long w, long z, int ldz,
                       long work );

    native int sspgvint(int matriint, int itype, char jobz,
                        char uplo, int n, long ap, long bp,
                        long w, long z, int ldz, long work,int lwork, int iwork,int liwork );
    native int dspgvint(int matriint, int itype, char jobz,
                        char uplo, int n, long ap, long bp,
                        long w, long z, int ldz,
                        long work, int lwork,int iwork, int liwork );

    native int sspgvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n, long ap,
                        long bp, float vl, float vu, int il,int iu, float abstol, int m,
                        long w, long z, int ldz, long work,int iwork, int ifail );
    native int dspgvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n, long ap,
                        long bp, double vl, double vu, int il,int iu, double abstol, int m,
                        long w, long z, int ldz,
                        long work, int iwork,int ifail );

    native int ssprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                        long afp,int ipiv,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work, int iwork );
    native int dsprfint(int matriint, char uplo, int n,int nrhs,  long ap,
                        long afp,int ipiv,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work, int iwork );
    native int csprfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zsprfint(int matriint, char uplo, int n,int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int sspsint(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                       long b, int ldb );
    native int dspsint(int matriint, char uplo, int n,int nrhs, long ap, int ipiv,
                       long b, int ldb );
    native int cspsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zspsint(int matriint, char uplo, int n,int nrhs, IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );

    native int sspsvint(int matriint, char fact, char uplo,int n, int nrhs,  long ap,
                        long afp, int ipiv,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int dspsvint(int matriint, char fact, char uplo,int n, int nrhs,  long ap,
                        long afp, int ipiv,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int iwork );
    native int cspsvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zspsvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int ssptrint(int matriint, char uplo, int n,
                        long ap, long d, long e, long tau );
    native int dsptrint(int matriint, char uplo, int n,
                        long ap, long d, long e, long tau );

    native int ssptrint(int matriint, char uplo, int n,
                        long ap, int ipiv );
    native int dsptrint(int matriint, char uplo, int n,
                        long ap, int ipiv );
    native int csptrint(int matriint, char uplo, int n,
                        IComplexNumber ap, int ipiv );
    native int zsptrint(int matriint, char uplo, int n,
                        IComplexNumber ap, int ipiv );

    native int ssptrint(int matriint, char uplo, int n,
                        long ap,int ipiv,
                        long work );
    native int dsptrint(int matriint, char uplo, int n,
                        long ap,int ipiv,
                        long work );
    native int csptrint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber work );
    native int zsptrint(int matriint, char uplo, int n,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber work );

    native int ssptrint(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b,int ldb );
    native int dsptrint(int matriint, char uplo, int n,int nrhs,  long ap,int ipiv, long b,int ldb );
    native int csptrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber ap,int ipiv, IComplexNumber b,int ldb );
    native int zsptrint(int matriint, char uplo, int n,int nrhs,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber b, int ldb );

    native int sstebint( char range, char order, int n, float vl,
                         float vu, int il, int iu,
                         float abstol,  long d,  long e,int m, int nsplit, long w,int iblock, int isplit,
                         long work, int iwork );
    native int dstebint( char range, char order, int n, double vl,
                         double vu, int il, int iu,
                         double abstol,  long d,  long e,int m, int nsplit, long w,int iblock, int isplit,
                         long work, int iwork );

    native int sstedint(int matriint, char compz, int n,
                        long d, long e, long z, int ldz,
                        long work, int lwork,int iwork, int liwork );
    native int dstedint(int matriint, char compz, int n,
                        long d, long e, long z, int ldz,
                        long work, int lwork,int iwork, int liwork );
    native int cstedint(int matriint, char compz, int n,
                        long d, long e, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );
    native int zstedint(int matriint, char compz, int n,
                        long d, long e, IComplexNumber z,int ldz, IComplexNumber work,int lwork, long rwork,int lrwork, int iwork,int liwork );

    native int sstegint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                        float vu, int il, int iu,
                        float abstol, int m, long w, long z,int ldz, int isuppz, long work,int lwork, int iwork,int liwork );
    native int dstegint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                        double vu, int il, int iu,
                        double abstol, int m, long w,
                        long z, int ldz, int isuppz,
                        long work, int lwork,int iwork, int liwork );
    native int cstegint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                        float vu, int il, int iu,
                        float abstol, int m, long w,
                        IComplexNumber z, int ldz,int isuppz, long work,int lwork, int iwork,int liwork );
    native int zstegint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                        double vu, int il, int iu,
                        double abstol, int m, long w,
                        IComplexNumber z, int ldz,int isuppz, long work,int lwork, int iwork,int liwork );

    native int ssteiint(int matriint, int n,  long d,
                        long e, int m,  long w,int iblock,int isplit, long z,int ldz, long work, int iwork,int ifailv );
    native int dsteiint(int matriint, int n,  long d,
                        long e, int m,  long w,int iblock,int isplit, long z,int ldz, long work, int iwork,int ifailv );
    native int csteiint(int matriint, int n,  long d,
                        long e, int m,  long w,int iblock,int isplit,
                        IComplexNumber z, int ldz,
                        long work, int iwork,int ifailv );
    native int zsteiint(int matriint, int n,  long d,
                        long e, int m,  long w,int iblock,int isplit,
                        IComplexNumber z, int ldz,
                        long work, int iwork,int ifailv );

    native int sstemint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                        float vu, int il, int iu,int m, long w, long z,int ldz, int nzc,int isuppz, int tryrac,
                        long work, int lwork,int iwork, int liwork );
    native int dstemint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                        double vu, int il, int iu,int m, long w, long z,int ldz, int nzc,int isuppz, int tryrac,
                        long work, int lwork,int iwork, int liwork );
    native int cstemint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                        float vu, int il, int iu,int m, long w,
                        IComplexNumber z, int ldz,int nzc, int isuppz,int tryrac, long work,int lwork, int iwork,int liwork );
    native int zstemint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                        double vu, int il, int iu,int m, long w,
                        IComplexNumber z, int ldz,int nzc, int isuppz,int tryrac, long work,int lwork, int iwork,int liwork );

    native int ssteqint(int matriint, char compz, int n,
                        long d, long e, long z, int ldz,
                        long work );
    native int dsteqint(int matriint, char compz, int n,
                        long d, long e, long z, int ldz,
                        long work );
    native int csteqint(int matriint, char compz, int n,
                        long d, long e, IComplexNumber z,int ldz, long work );
    native int zsteqint(int matriint, char compz, int n,
                        long d, long e, IComplexNumber z,int ldz, long work );

    native int ssterint(int n, long d, long e );
    native int dsterint(int n, long d, long e );

    native int ssteint(int matriint, char jobz, int n,
                       long d, long e, long z, int ldz,
                       long work );
    native int dsteint(int matriint, char jobz, int n,
                       long d, long e, long z, int ldz,
                       long work );

    native int sstevint(int matriint, char jobz, int n,
                        long d, long e, long z, int ldz,
                        long work, int lwork,int iwork, int liwork );
    native int dstevint(int matriint, char jobz, int n,
                        long d, long e, long z, int ldz,
                        long work, int lwork,int iwork, int liwork );

    native int sstevint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                        float vu, int il, int iu,
                        float abstol, int m, long w, long z,int ldz, int isuppz, long work,int lwork, int iwork,int liwork );
    native int dstevint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                        double vu, int il, int iu,
                        double abstol, int m, long w,
                        long z, int ldz, int isuppz,
                        long work, int lwork,int iwork, int liwork );

    native int sstevint(int matriint, char jobz, char range,int n, long d, long e, float vl,
                        float vu, int il, int iu,
                        float abstol, int m, long w, long z,int ldz, long work, int iwork,int ifail );
    native int dstevint(int matriint, char jobz, char range,int n, long d, long e, double vl,
                        double vu, int il, int iu,
                        double abstol, int m, long w,
                        long z, int ldz, long work,int iwork, int ifail );

    native int ssycoint(int matriint, char uplo, int n,
                        long a, int lda,int ipiv, float anorm,
                        long rcond, long work, int iwork );
    native int dsycoint(int matriint, char uplo, int n,
                        long a, int lda,int ipiv, double anorm,
                        long rcond, long work,int iwork );
    native int csycoint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, float anorm,
                        long rcond, IComplexNumber work );
    native int zsycoint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, double anorm,
                        long rcond, IComplexNumber work );

    native int ssyequint(int matriint, char uplo, int n,
                         long a, int lda, long s,
                         long scond, long amax, long work );
    native int dsyequint(int matriint, char uplo, int n,
                         long a, int lda, long s,
                         long scond, long amax, long work );
    native int csyequint(int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         long s, long scond, long amax,
                         IComplexNumber work );
    native int zsyequint(int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         long s, long scond, long amax,
                         IComplexNumber work );

    native int ssyeint(int matriint, char jobz, char uplo,int n, long a, int lda, long w,
                       long work, int lwork );
    native int dsyeint(int matriint, char jobz, char uplo,int n, long a, int lda,
                       long w, long work, int lwork );

    native int ssyevint(int matriint, char jobz, char uplo,int n, long a, int lda,
                        long w, long work, int lwork,int iwork, int liwork );
    native int dsyevint(int matriint, char jobz, char uplo,int n, long a, int lda,
                        long w, long work, int lwork,int iwork, int liwork );

    native int ssyevint(int matriint, char jobz, char range,
                        char uplo, int n, long a,int lda, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, int isuppz, long work,int lwork, int iwork,int liwork );
    native int dsyevint(int matriint, char jobz, char range,
                        char uplo, int n, long a,int lda, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, int isuppz,
                        long work, int lwork,int iwork, int liwork );

    native int ssyevint(int matriint, char jobz, char range,
                        char uplo, int n, long a,int lda, float vl, float vu,int il, int iu, float abstol,int m, long w, long z,int ldz, long work, int lwork,int iwork, int ifail );
    native int dsyevint(int matriint, char jobz, char range,
                        char uplo, int n, long a,int lda, double vl, double vu,int il, int iu, double abstol,int m, long w, long z,int ldz, long work, int lwork,int iwork, int ifail );

    native int ssygsint(int matriint, int itype, char uplo,int n, long a, int lda,
                        long b, int ldb );
    native int dsygsint(int matriint, int itype, char uplo,int n, long a, int lda,
                        long b, int ldb );

    native int ssygint(int matriint, int itype, char jobz,
                       char uplo, int n, long a,int lda, long b, int ldb,
                       long w, long work, int lwork );
    native int dsygint(int matriint, int itype, char jobz,
                       char uplo, int n, long a,int lda, long b, int ldb,
                       long w, long work, int lwork );

    native int ssygvint(int matriint, int itype, char jobz,
                        char uplo, int n, long a,int lda, long b, int ldb,
                        long w, long work, int lwork,int iwork, int liwork );
    native int dsygvint(int matriint, int itype, char jobz,
                        char uplo, int n, long a,int lda, long b, int ldb,
                        long w, long work, int lwork,int iwork, int liwork );

    native int ssygvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n, long a,int lda, long b, int ldb,
                        float vl, float vu, int il,int iu, float abstol, int m,
                        long w, long z, int ldz, long work,int lwork, int iwork,int ifail );
    native int dsygvint(int matriint, int itype, char jobz,
                        char range, char uplo, int n, long a,int lda, long b, int ldb,
                        double vl, double vu, int il,int iu, double abstol, int m,
                        long w, long z, int ldz,
                        long work, int lwork,int iwork, int ifail );

    native int ssyrfint(int matriint, char uplo, int n,int nrhs,  long a, int lda,
                        long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dsyrfint(int matriint, char uplo, int n,int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                        long b, int ldb, long x,int ldx, long ferr, long berr,
                        long work, int iwork );
    native int csyrfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int zsyrfint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber af,int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int ssyrfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                         long s,  long b, int ldb,
                         long x, int ldx, long rcond,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork );
    native int dsyrfsint(int matriint, char uplo, char equed,int n, int nrhs,  long a,int lda,  long af,int ldaf,int ipiv,
                         long s,  long b,int ldb, long x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, long work,int iwork );
    native int csyrfsint(int matriint, char uplo, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,int ipiv,
                         long s,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zsyrfsint(int matriint, char uplo, char equed,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af,int ldaf,int ipiv,
                         long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int ssysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                       long work, int lwork );
    native int dsysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                       long work, int lwork );
    native int csysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork );
    native int zsysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork );

    native int ssysvint(int matriint, char fact, char uplo,int n, int nrhs,  long a,int lda, long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int lwork,int iwork );
    native int dsysvint(int matriint, char fact, char uplo,int n, int nrhs,  long a,int lda, long af, int ldaf,int ipiv,  long b,int ldb, long x, int ldx,
                        long rcond, long ferr, long berr,
                        long work, int lwork,int iwork );
    native int csysvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x,int ldx, long rcond, long ferr,
                        long berr, IComplexNumber work,int lwork, long rwork );
    native int zsysvint(int matriint, char fact, char uplo,int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, int lwork,
                        long rwork );

    native int ssysvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s,
                         long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork );
    native int dsysvxint(int matriint, char fact, char uplo,int n, int nrhs, long a,int lda, long af, int ldaf,int ipiv, String equed, long s,
                         long b, int ldb, long x,int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         long work, int iwork );
    native int csysvxint(int matriint, char fact, char uplo,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );
    native int zsysvxint(int matriint, char fact, char uplo,int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams,
                         long params, IComplexNumber work,
                         long rwork );

    native int ssytrint(int matriint, char uplo, int n,
                        long a, int lda, long d, long e,
                        long tau, long work, int lwork );
    native int dsytrint(int matriint, char uplo, int n,
                        long a, int lda, long d, long e,
                        long tau, long work, int lwork );

    native int ssytrint(int matriint, char uplo, int n,
                        long a, int lda, int ipiv,
                        long work, int lwork );
    native int dsytrint(int matriint, char uplo, int n,
                        long a, int lda, int ipiv,
                        long work, int lwork );
    native int csytrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
    native int zsytrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

    native int ssytrint(int matriint, char uplo, int n,
                        long a, int lda,int ipiv, long work );
    native int dsytrint(int matriint, char uplo, int n,
                        long a, int lda,int ipiv, long work );
    native int csytrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv,
                        IComplexNumber work );
    native int zsytrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv,
                        IComplexNumber work );



    native int stbcoint(int matriint, char norm, char uplo,
                        char diag, int n, int kd,
                        long ab, int ldab, long rcond,
                        long work, int iwork );
    native int dtbcoint(int matriint, char norm, char uplo,
                        char diag, int n, int kd,
                        long ab, int ldab,
                        long rcond, long work,int iwork );
    native int ctbcoint(int matriint, char norm, char uplo,
                        char diag, int n, int kd,
                        IComplexNumber ab, int ldab,
                        long rcond, IComplexNumber work,
                        long rwork );
    native int ztbcoint(int matriint, char norm, char uplo,
                        char diag, int n, int kd,
                        IComplexNumber ab,int ldab, long rcond,
                        IComplexNumber work, long rwork );

    native int stbrfint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,  long ab,int ldab,  long b, int ldb,
                        long x, int ldx, long ferr,
                        long berr, long work, int iwork );
    native int dtbrfint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,  long ab,int ldab,  long b,int ldb,  long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int ctbrfint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int ztbrfint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,
                        IComplexNumber ab,int ldab,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int stbtrint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,  long ab,int ldab, long b, int ldb );
    native int dtbtrint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,  long ab,int ldab, long b, int ldb );
    native int ctbtrint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,  IComplexNumber ab,int ldab, IComplexNumber b,int ldb );
    native int ztbtrint(int matriint, char uplo, char trans,
                        char diag, int n, int kd,int nrhs,
                        IComplexNumber ab,int ldab, IComplexNumber b,int ldb );

    native int stfsint(int matriint, char transr, char side,
                       char uplo, char trans, char diag, int m,int n, float alpha,  long a,
                       long b, int ldb );
    native int dtfsint(int matriint, char transr, char side,
                       char uplo, char trans, char diag, int m,int n, double alpha,  long a,
                       long b, int ldb );
    native int ctfsint(int matriint, char transr, char side,
                       char uplo, char trans, char diag, int m,int n, IComplexNumber alpha,
                       IComplexNumber a,
                       IComplexNumber b, int ldb );
    native int ztfsint(int matriint, char transr, char side,
                       char uplo, char trans, char diag, int m,int n, IComplexNumber alpha,
                       IComplexNumber a,
                       IComplexNumber b, int ldb );

    native int stftrint(int matriint, char transr, char uplo,
                        char diag, int n, long a );
    native int dtftrint(int matriint, char transr, char uplo,
                        char diag, int n, long a );
    native int ctftrint(int matriint, char transr, char uplo,
                        char diag, int n,
                        IComplexNumber a );
    native int ztftrint(int matriint, char transr, char uplo,
                        char diag, int n,
                        IComplexNumber a );

    native int stfttint(int matriint, char transr, char uplo,int n,  long arf, long ap );
    native int dtfttint(int matriint, char transr, char uplo,int n,  long arf, long ap );
    native int ctfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                        IComplexNumber ap );
    native int ztfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                        IComplexNumber ap );

    native int stfttint(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
    native int dtfttint(int matriint, char transr, char uplo,int n,  long arf, long a,int lda );
    native int ctfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                        IComplexNumber a, int lda );
    native int ztfttint(int matriint, char transr, char uplo,int n,  IComplexNumber arf,
                        IComplexNumber a, int lda );

    native int stgevint(int matriint, char side, char howmny,int select, int n,
                        long s, int lds,  long p,int ldp, long vl, int ldvl,
                        long vr, int ldvr, int mm,int m, long work );
    native int dtgevint(int matriint, char side, char howmny,int select, int n,
                        long s, int lds,
                        long p, int ldp, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work );
    native int ctgevint(int matriint, char side, char howmny,int select, int n,
                        IComplexNumber s, int lds,
                        IComplexNumber p, int ldp,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int mm, int m,
                        IComplexNumber work, long rwork );
    native int ztgevint(int matriint, char side, char howmny,int select, int n,
                        IComplexNumber s, int lds,
                        IComplexNumber p, int ldp,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int mm, int m,
                        IComplexNumber work, long rwork );

    native int stgexint(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb,
                        long q, int ldq, long z,int ldz, int ifst,int ilst, long work,int lwork );
    native int dtgexint(int matriint, int wantq,int wantz, int n, long a,int lda, long b, int ldb,
                        long q, int ldq, long z,int ldz, int ifst,int ilst, long work,int lwork );
    native int ctgexint(int matriint, int wantq,int wantz, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz,int ifst, int ilst );
    native int ztgexint(int matriint, int wantq,int wantz, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz,int ifst, int ilst );

    native int stgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                        long a, int lda, long b,int ldb, long alphar, long alphai,
                        long beta, long q, int ldq, long z,int ldz, int m, long pl,
                        long pr, long dif, long work,int lwork, int iwork,int liwork );
    native int dtgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                        long a, int lda, long b,int ldb, long alphar, long alphai,
                        long beta, long q, int ldq,
                        long z, int ldz, int m,
                        long pl, long pr, long dif,
                        long work, int lwork,int iwork, int liwork );
    native int ctgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz,int m, long pl, long pr, long dif,
                        IComplexNumber work, int lwork,int iwork, int liwork );
    native int ztgseint(int matriint, int ijob,int wantq, int wantz,int select, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber q, int ldq,
                        IComplexNumber z, int ldz,int m, long pl, long pr,
                        long dif, IComplexNumber work,int lwork, int iwork,int liwork );

    native int stgsjint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, int k, int l,
                        long a, int lda, long b,int ldb, float tola, float tolb,
                        long alpha, long beta, long u,int ldu, long v, int ldv,
                        long q, int ldq, long work,int ncycle );
    native int dtgsjint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, int k, int l,
                        long a, int lda, long b,int ldb, double tola, double tolb,
                        long alpha, long beta, long u,int ldu, long v, int ldv,
                        long q, int ldq, long work,int ncycle );
    native int ctgsjint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        float tola, float tolb, long alpha,
                        long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, IComplexNumber work,int ncycle );
    native int ztgsjint(int matriint, char jobu, char jobv,
                        char jobq, int m, int p,int n, int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        double tola, double tolb, long alpha,
                        long beta, IComplexNumber u,int ldu, IComplexNumber v,int ldv, IComplexNumber q,int ldq, IComplexNumber work,int ncycle );

    native int stgsnint(int matriint, char job, char howmny,int select, int n,
                        long a, int lda,  long b,int ldb,  long vl,int ldvl,  long vr,int ldvr, long s, long dif,int mm, int m, long work,int lwork, int iwork );
    native int dtgsnint(int matriint, char job, char howmny,int select, int n,
                        long a, int lda,
                        long b, int ldb,
                        long vl, int ldvl,
                        long vr, int ldvr, long s,
                        long dif, int mm, int m,
                        long work, int lwork,int iwork );
    native int ctgsnint(int matriint, char job, char howmny,int select, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,
                        long s, long dif, int mm,int m, IComplexNumber work,int lwork, int iwork );
    native int ztgsnint(int matriint, char job, char howmny,int select, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber vl,int ldvl,
                        IComplexNumber vr,int ldvr, long s, long dif,int mm, int m,
                        IComplexNumber work, int lwork,int iwork );

    native int stgsyint(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                        long c, int ldc,  long d,int ldd,  long e, int lde,
                        long f, int ldf, long scale,
                        long dif, long work, int lwork,int iwork );
    native int dtgsyint(int matriint, char trans, int ijob,int m, int n,  long a,int lda,  long b, int ldb,
                        long c, int ldc,  long d,int ldd,  long e, int lde,
                        long f, int ldf, long scale,
                        long dif, long work, int lwork,int iwork );
    native int ctgsyint(int matriint, char trans, int ijob,int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber c, int ldc,
                        IComplexNumber d, int ldd,
                        IComplexNumber e, int lde,
                        IComplexNumber f, int ldf,
                        long scale, long dif,
                        IComplexNumber work, int lwork,int iwork );
    native int ztgsyint(int matriint, char trans, int ijob,int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber c, int ldc,
                        IComplexNumber d, int ldd,
                        IComplexNumber e, int lde,
                        IComplexNumber f, int ldf,
                        long scale, long dif,
                        IComplexNumber work, int lwork,int iwork );

    native int stpcoint(int matriint, char norm, char uplo,
                        char diag, int n,  long ap,
                        long rcond, long work, int iwork );
    native int dtpcoint(int matriint, char norm, char uplo,
                        char diag, int n,  long ap,
                        long rcond, long work,int iwork );
    native int ctpcoint(int matriint, char norm, char uplo,
                        char diag, int n,
                        IComplexNumber ap, long rcond,
                        IComplexNumber work, long rwork );
    native int ztpcoint(int matriint, char norm, char uplo,
                        char diag, int n,
                        IComplexNumber ap, long rcond,
                        IComplexNumber work, long rwork );

    native int stprfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long ap,  long b, int ldb,
                        long x, int ldx, long ferr,
                        long berr, long work, int iwork );
    native int dtprfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long ap,  long b,int ldb,  long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int ctprfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int ztprfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int stptrint(int matriint, char uplo, char diag,int n, long ap );
    native int dtptrint(int matriint, char uplo, char diag,int n, long ap );
    native int ctptrint(int matriint, char uplo, char diag,int n, IComplexNumber ap );
    native int ztptrint(int matriint, char uplo, char diag,int n, IComplexNumber ap );

    native int stptrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long ap, long b, int ldb );
    native int dtptrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long ap, long b, int ldb );
    native int ctptrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber b, int ldb );
    native int ztptrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber b, int ldb );

    native int stpttint(int matriint, char transr, char uplo,int n,  long ap, long arf );
    native int dtpttint(int matriint, char transr, char uplo,int n,  long ap, long arf );
    native int ctpttint(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                        IComplexNumber arf );
    native int ztpttint(int matriint, char transr, char uplo,int n,  IComplexNumber ap,
                        IComplexNumber arf );

    native int stpttint(int matriint, char uplo, int n,
                        long ap, long a, int lda );
    native int dtpttint(int matriint, char uplo, int n,
                        long ap, long a, int lda );
    native int ctpttint(int matriint, char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber a, int lda );
    native int ztpttint(int matriint, char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber a, int lda );

    native int strcoint(int matriint, char norm, char uplo,
                        char diag, int n,  long a,int lda, long rcond, long work,int iwork );
    native int dtrcoint(int matriint, char norm, char uplo,
                        char diag, int n,  long a,int lda, long rcond, long work,int iwork );
    native int ctrcoint(int matriint, char norm, char uplo,
                        char diag, int n,
                        IComplexNumber a, int lda,
                        long rcond, IComplexNumber work,
                        long rwork );
    native int ztrcoint(int matriint, char norm, char uplo,
                        char diag, int n,
                        IComplexNumber a, int lda,
                        long rcond, IComplexNumber work,
                        long rwork );

    native int strevint(int matriint, char side, char howmny,int select, int n,
                        long t, int ldt, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work );
    native int dtrevint(int matriint, char side, char howmny,int select, int n,
                        long t, int ldt, long vl,int ldvl, long vr, int ldvr,int mm, int m, long work );
    native int ctrevint(int matriint, char side, char howmny,int select, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int mm, int m,
                        IComplexNumber work, long rwork );
    native int ztrevint(int matriint, char side, char howmny,int select, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,int mm, int m,
                        IComplexNumber work, long rwork );

    native int strexint(int matriint, char compq, int n,
                        long t, int ldt, long q,int ldq, int ifst,int ilst, long work );
    native int dtrexint(int matriint, char compq, int n,
                        long t, int ldt, long q,int ldq, int ifst,int ilst, long work );
    native int ctrexint(int matriint, char compq, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber q, int ldq,int ifst, int ilst );
    native int ztrexint(int matriint, char compq, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber q, int ldq,int ifst, int ilst );

    native int strrfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long a, int lda,  long b,int ldb,  long x, int ldx,
                        long ferr, long berr, long work,int iwork );
    native int dtrrfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long a, int lda,
                        long b, int ldb,
                        long x, int ldx, long ferr,
                        long berr, long work, int iwork );
    native int ctrrfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );
    native int ztrrfint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr,
                        IComplexNumber work, long rwork );

    native int strseint(int matriint, char job, char compq,int select, int n,
                        long t, int ldt, long q,int ldq, long wr, long wi,int m, long s, long sep,
                        long work, int lwork,int iwork, int liwork );
    native int dtrseint(int matriint, char job, char compq,int select, int n,
                        long t, int ldt, long q,int ldq, long wr, long wi,int m, long s, long sep,
                        long work, int lwork,int iwork, int liwork );
    native int ctrseint(int matriint, char job, char compq,int select, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber q, int ldq,
                        IComplexNumber w, int m,
                        long s, long sep,
                        IComplexNumber work, int lwork );
    native int ztrseint(int matriint, char job, char compq,int select, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber q, int ldq,
                        IComplexNumber w, int m,
                        long s, long sep,
                        IComplexNumber work, int lwork );

    native int strsnint(int matriint, char job, char howmny,int select, int n,
                        long t, int ldt,  long vl,int ldvl,  long vr,int ldvr, long s, long sep,int mm, int m, long work,int ldwork, int iwork );
    native int dtrsnint(int matriint, char job, char howmny,int select, int n,
                        long t, int ldt,
                        long vl, int ldvl,
                        long vr, int ldvr, long s,
                        long sep, int mm, int m,
                        long work, int ldwork,int iwork );
    native int ctrsnint(int matriint, char job, char howmny,int select, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,
                        long s, long sep, int mm,int m, IComplexNumber work,int ldwork, long rwork );
    native int ztrsnint(int matriint, char job, char howmny,int select, int n,
                        IComplexNumber t, int ldt,
                        IComplexNumber vl,int ldvl,
                        IComplexNumber vr,int ldvr, long s, long sep,int mm, int m,
                        IComplexNumber work, int ldwork,
                        long rwork );

    native int strsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                        long a, int lda,  long b,int ldb, long c, int ldc,
                        long scale );
    native int dtrsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                        long a, int lda,
                        long b, int ldb, long c,int ldc, long scale );
    native int ctrsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber c, int ldc,
                        long scale );
    native int ztrsyint(int matriint, char trana, char tranb,int isgn, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber c, int ldc,
                        long scale );

    native int strtrint(int matriint, char uplo, char diag,int n, long a, int lda );
    native int dtrtrint(int matriint, char uplo, char diag,int n, long a, int lda );
    native int ctrtrint(int matriint, char uplo, char diag,int n, IComplexNumber a,int lda );
    native int ztrtrint(int matriint, char uplo, char diag,int n, IComplexNumber a,int lda );

    native int strtrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long a, int lda, long b,int ldb );
    native int dtrtrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        long a, int lda, long b,int ldb );
    native int ctrtrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb );
    native int ztrtrint(int matriint, char uplo, char trans,
                        char diag, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb );

    native int strttint(int matriint, char transr, char uplo,int n,  long a, int lda,
                        long arf );
    native int dtrttint(int matriint, char transr, char uplo,int n,  long a, int lda,
                        long arf );
    native int ctrttint(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );
    native int ztrttint(int matriint, char transr, char uplo,int n,  IComplexNumber a,int lda, IComplexNumber arf );

    native int strttint(int matriint, char uplo, int n,
                        long a, int lda, long ap );
    native int dtrttint(int matriint, char uplo, int n,
                        long a, int lda, long ap );
    native int ctrttint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber ap );
    native int ztrttint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber ap );

    native int stzrzint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int dtzrzint(int matriint, int m, int n,
                        long a, int lda, long tau,
                        long work, int lwork );
    native int ctzrzint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int ztzrzint(int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int cungbint(int matriint, char vect, int m,int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zungbint(int matriint, char vect, int m,int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int cunghint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zunghint(int matriint, int n, int ilo,int ihi, IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int cunglint(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zunglint(int matriint, int m, int n,int k, IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int cungqint(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zungqint(int matriint, int m, int n,int k, IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );


    native int cungrint(int matriint, int m, int n,int k, IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zungrint(int matriint, int m, int n,int k, IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int cungtint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );
    native int zungtint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber work, int lwork );

    native int cunmbint(int matriint, char vect, char side,
                        char trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmbint(int matriint, char vect, char side,
                        char trans, int m, int n,int k,  IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );

    native int cunmhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmhint(int matriint, char side, char trans,int m, int n, int ilo,int ihi,  IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );

    native int cunmlint(int matriint, char side, char trans,int m, int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmlint(int matriint, char side, char trans,int m, int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );

    native int cunmqint(int matriint, char side, char trans,int m, int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmqint(int matriint, char side, char trans,int m, int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );



    native int cunmrint(int matriint, char side, char trans,int m, int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmrint(int matriint, char side, char trans,int m, int n, int k,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );

    native int cunmrint(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmrint(int matriint, char side, char trans,int m, int n, int k,int l,  IComplexNumber a,int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );

    native int cunmtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );
    native int zunmtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork );

    native int cupgtint(int matriint, char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber tau,
                        IComplexNumber q, int ldq,
                        IComplexNumber work );
    native int zupgtint(int matriint, char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber tau,
                        IComplexNumber q, int ldq,
                        IComplexNumber work );

    native int cupmtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        IComplexNumber ap,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work );
    native int zupmtint(int matriint, char side, char uplo,
                        char trans, int m, int n,
                        IComplexNumber ap,
                        IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work );

    native int claghe(int matriint, int n, int k,
                      long d, IComplexNumber a,int lda, int iseed );
    native int zlaghe(int matriint, int n, int k,
                      long d, IComplexNumber a,int lda, int iseed );

    native int slagsy(int matriint, int n, int k,
                      long d, long a, int lda,int iseed );
    native int dlagsy(int matriint, int n, int k,
                      long d, long a, int lda,int iseed );
    native int clagsy(int matriint, int n, int k,
                      long d, IComplexNumber a,int lda, int iseed );
    native int zlagsy(int matriint, int n, int k,
                      long d, IComplexNumber a,int lda, int iseed );

    native int slapmr(int matriint, int forwrd,int m, int n, long x, int ldx,int k );
    native int dlapmr(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
    native int clapmr(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );
    native int zlapmr(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );

    native int slapmt(int matriint, int forwrd,int m, int n, long x, int ldx,int k );
    native int dlapmt(int matriint, int forwrd,int m, int n, long x,int ldx, int k );
    native int clapmt(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );
    native int zlapmt(int matriint, int forwrd,int m, int n, IComplexNumber x,int ldx, int k );

    native float slapy2( float x, float y );
    native double dlapy2( double x, double y );

    native float slapy3( float x, float y, float z );
    native double dlapy3( double x, double y, double z );

    native int slartgp( float f, float g, long cs, long sn, long r );
    native int dlartgp( double f, double g, long cs, long sn,
                        long r );

    native int slartgs( float x, float y, float sigma, long cs,
                        long sn );
    native int dlartgs( double x, double y, double sigma, long cs,
                        long sn );


    //LAPACK 3.3.0
    native int cbbcsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta, long phi,
                      IComplexNumber u1, int ldu1,
                      IComplexNumber u2, int ldu2,
                      IComplexNumber v1t, int ldv1t,
                      IComplexNumber v2t, int ldv2t,
                      long b11d, long b11e, long b12d, long b12e,
                      long b21d, long b21e, long b22d, long b22e );
    native int cbbcsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,int m, int p, int q,
                        long theta, long phi,
                        IComplexNumber u1, int ldu1,
                        IComplexNumber u2, int ldu2,
                        IComplexNumber v1t, int ldv1t,
                        IComplexNumber v2t, int ldv2t,
                        long b11d, long b11e, long b12d,
                        long b12e, long b21d, long b21e,
                        long b22d, long b22e, long rwork,int lrwork );
    native int cheswapr(int matriint, char uplo, int n,
                        IComplexNumber a, int i1,int i2 );
    native int cheswapint(int matriint, char uplo, int n,
                          IComplexNumber a, int i1,int i2 );
    native int chetri2(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,int ipiv );
    native int chetri2_work(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
    native int chetri2x(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, int nb );
    native int chetri2int(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv,
                          IComplexNumber work, int nb );
    native int chetrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                       IComplexNumber b, int ldb );
    native int chetrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber work );
    native int csyconv(int matriint, char uplo, char way, int n,
                       IComplexNumber a, int lda,int ipiv, IComplexNumber work  );
    native int csyconint(int matriint, char uplo, char way,int n, IComplexNumber a,int lda,int ipiv,
                         IComplexNumber work );
    native int csyswapr(int matriint, char uplo, int n,
                        IComplexNumber a, int i1,int i2 );
    native int csyswapint(int matriint, char uplo, int n,
                          IComplexNumber a, int i1,int i2 );
    native int csytri2(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,int ipiv );
    native int csytri2_work(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
    native int csytri2x(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, int nb );
    native int csytri2int(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv,
                          IComplexNumber work, int nb );
    native int csytrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                       IComplexNumber b, int ldb );
    native int csytrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber work );
    native int cunbdb(int matriint, char trans, char signs,int m, int p, int q,
                      IComplexNumber x11, int ldx11,
                      IComplexNumber x12, int ldx12,
                      IComplexNumber x21, int ldx21,
                      IComplexNumber x22, int ldx22,
                      long theta, long phi,
                      IComplexNumber taup1,
                      IComplexNumber taup2,
                      IComplexNumber tauq1,
                      IComplexNumber tauq2 );
    native int cunbdint(int matriint, char trans, char signs,int m, int p, int q,
                        IComplexNumber x11, int ldx11,
                        IComplexNumber x12, int ldx12,
                        IComplexNumber x21, int ldx21,
                        IComplexNumber x22, int ldx22,
                        long theta, long phi,
                        IComplexNumber taup1,
                        IComplexNumber taup2,
                        IComplexNumber tauq1,
                        IComplexNumber tauq2,
                        IComplexNumber work, int lwork );
    native int cuncsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q,
                      IComplexNumber x11, int ldx11,
                      IComplexNumber x12, int ldx12,
                      IComplexNumber x21, int ldx21,
                      IComplexNumber x22, int ldx22,
                      long theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber v2t,int ldv2t );
    native int cuncsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,
                        char signs, int m, int p,int q, IComplexNumber x11,int ldx11, IComplexNumber x12,int ldx12, IComplexNumber x21,int ldx21, IComplexNumber x22,int ldx22, long theta,
                        IComplexNumber u1, int ldu1,
                        IComplexNumber u2, int ldu2,
                        IComplexNumber v1t, int ldv1t,
                        IComplexNumber v2t, int ldv2t,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork );
    native int cuncsd2by1(int matriint, char jobu1, char jobu2,
                          char jobv1t, int m, int p, int q,
                          IComplexNumber x11, int ldx11,
                          IComplexNumber x21, int ldx21,
                          IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t, int ldv1t );
    native int cuncsd2by1_work(int matriint, char jobu1, char jobu2,
                               char jobv1t, int m, int p,int q, IComplexNumber x11, int ldx11,
                               IComplexNumber x21, int ldx21,
                               IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber work,int lwork, long rwork, int lrwork,int iwork );
    native int dbbcsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta,
                      long phi, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                      long v2t, int ldv2t, long b11d,
                      long b11e, long b12d, long b12e,
                      long b21d, long b21e, long b22d,
                      long b22e );
    native int dbbcsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,int m, int p, int q,
                        long theta, long phi, long u1,int ldu1, long u2, int ldu2,
                        long v1t, int ldv1t, long v2t,int ldv2t, long b11d, long b11e,
                        long b12d, long b12e, long b21d,
                        long b21e, long b22d, long b22e,
                        long work, int lwork );
    native int dorbdb(int matriint, char trans, char signs,int m, int p, int q,
                      long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                      long x22, int ldx22, long theta,
                      long phi, long taup1, long taup2,
                      long tauq1, long tauq2 );
    native int dorbdint(int matriint, char trans, char signs,int m, int p, int q,
                        long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                        long x22, int ldx22, long theta,
                        long phi, long taup1, long taup2,
                        long tauq1, long tauq2, long work,int lwork );
    native int dorcsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q,
                      long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                      long x22, int ldx22, long theta,
                      long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                      long v2t, int ldv2t );
    native int dorcsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,
                        char signs, int m, int p,int q, long x11, int ldx11,
                        long x12, int ldx12, long x21,int ldx21, long x22, int ldx22,
                        long theta, long u1, int ldu1,
                        long u2, int ldu2, long v1t,int ldv1t, long v2t, int ldv2t,
                        long work, int lwork,int iwork );
    native int dorcsd2by1(int matriint, char jobu1, char jobu2,
                          char jobv1t, int m, int p, int q,
                          long x11, int ldx11, long x21, int ldx21,
                          long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t);
    native int dorcsd2by1_work(int matriint, char jobu1, char jobu2,
                               char jobv1t, int m, int p, int q,
                               long x11, int ldx11, long x21, int ldx21,
                               long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                               long work, int lwork, int iwork );
    native int dsyconv(int matriint, char uplo, char way, int n,
                       long a, int lda,int ipiv, long work);
    native int dsyconint(int matriint, char uplo, char way,int n, long a, int lda,int ipiv, long work );
    native int dsyswapr(int matriint, char uplo, int n,
                        long a, int i1, int i2 );
    native int dsyswapint(int matriint, char uplo, int n,
                          long a, int i1, int i2 );
    native int dsytri2(int matriint, char uplo, int n,
                       long a, int lda,int ipiv );
    native int dsytri2_work(int matriint, char uplo, int n,
                            long a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
    native int dsytri2x(int matriint, char uplo, int n,
                        long a, int lda,int ipiv,int nb );
    native int dsytri2int(int matriint, char uplo, int n,
                          long a, int lda,int ipiv, long work,int nb );
    native int dsytrs2(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int dsytrs2_work(int matriint, char uplo, int n,int nrhs,  long a,int lda,int ipiv,
                            long b, int ldb, long work );
    native int sbbcsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta, long phi,
                      long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                      long v2t, int ldv2t, long b11d,
                      long b11e, long b12d, long b12e, long b21d,
                      long b21e, long b22d, long b22e );
    native int sbbcsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,int m, int p, int q,
                        long theta, long phi, long u1,int ldu1, long u2, int ldu2,
                        long v1t, int ldv1t, long v2t,int ldv2t, long b11d, long b11e,
                        long b12d, long b12e, long b21d,
                        long b21e, long b22d, long b22e,
                        long work, int lwork );
    native int sorbdb(int matriint, char trans, char signs,int m, int p, int q, long x11,int ldx11, long x12, int ldx12,
                      long x21, int ldx21, long x22,int ldx22, long theta, long phi,
                      long taup1, long taup2, long tauq1,
                      long tauq2 );
    native int sorbdint(int matriint, char trans, char signs,int m, int p, int q,
                        long x11, int ldx11, long x12,int ldx12, long x21, int ldx21,
                        long x22, int ldx22, long theta,
                        long phi, long taup1, long taup2,
                        long tauq1, long tauq2, long work,int lwork );
    native int sorcsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q, long x11,int ldx11, long x12, int ldx12,
                      long x21, int ldx21, long x22,int ldx22, long theta, long u1,int ldu1, long u2, int ldu2,
                      long v1t, int ldv1t, long v2t,int ldv2t );
    native int sorcsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,
                        char signs, int m, int p,int q, long x11, int ldx11,
                        long x12, int ldx12, long x21,int ldx21, long x22, int ldx22,
                        long theta, long u1, int ldu1,
                        long u2, int ldu2, long v1t,int ldv1t, long v2t, int ldv2t,
                        long work, int lwork,int iwork );
    native int sorcsd2by1(int matriint, char jobu1, char jobu2,
                          char jobv1t, int m, int p, int q,
                          long x11, int ldx11, long x21, int ldx21,
                          long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t);
    native int sorcsd2by1_work(int matriint, char jobu1, char jobu2,
                               char jobv1t, int m, int p, int q,
                               long x11, int ldx11, long x21, int ldx21,
                               long theta, long u1, int ldu1, long u2,int ldu2, long v1t, int ldv1t,
                               long work, int lwork, int iwork );
    native int ssyconv(int matriint, char uplo, char way, int n,
                       long a, int lda,int ipiv, long work );
    native int ssyconint(int matriint, char uplo, char way,int n, long a, int lda,int ipiv, long work );
    native int ssyswapr(int matriint, char uplo, int n,
                        long a, int i1, int i2 );
    native int ssyswapint(int matriint, char uplo, int n,
                          long a, int i1, int i2 );
    native int ssytri2(int matriint, char uplo, int n, long a,int lda,int ipiv );
    native int ssytri2_work(int matriint, char uplo, int n,
                            long a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
    native int ssytri2x(int matriint, char uplo, int n,
                        long a, int lda,int ipiv,int nb );
    native int ssytri2int(int matriint, char uplo, int n,
                          long a, int lda,int ipiv, long work,int nb );
    native int ssytrs2(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int ssytrs2_work(int matriint, char uplo, int n,int nrhs,  long a,int lda,int ipiv,
                            long b, int ldb, long work );
    native int zbbcsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, int m,int p, int q, long theta,
                      long phi, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber v2t,int ldv2t, long b11d, long b11e,
                      long b12d, long b12e, long b21d,
                      long b21e, long b22d, long b22e );
    native int zbbcsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,int m, int p, int q,
                        long theta, long phi,
                        IComplexNumber u1, int ldu1,
                        IComplexNumber u2, int ldu2,
                        IComplexNumber v1t, int ldv1t,
                        IComplexNumber v2t, int ldv2t,
                        long b11d, long b11e, long b12d,
                        long b12e, long b21d, long b21e,
                        long b22d, long b22e, long rwork,int lrwork );
    native int zheswapr(int matriint, char uplo, int n,
                        IComplexNumber a, int i1,int i2 );
    native int zheswapint(int matriint, char uplo, int n,
                          IComplexNumber a, int i1,int i2 );
    native int zhetri2(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,int ipiv );
    native int zhetri2_work(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
    native int zhetri2x(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, int nb );
    native int zhetri2int(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv,
                          IComplexNumber work, int nb );
    native int zhetrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                       IComplexNumber b, int ldb );
    native int zhetrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber work );
    native int zsyconv(int matriint, char uplo, char way, int n,
                       IComplexNumber a, int lda,int ipiv, IComplexNumber work );
    native int zsyconint(int matriint, char uplo, char way,int n, IComplexNumber a,int lda,int ipiv,
                         IComplexNumber work );
    native int zsyswapr(int matriint, char uplo, int n,
                        IComplexNumber a, int i1,int i2 );
    native int zsyswapint(int matriint, char uplo, int n,
                          IComplexNumber a, int i1,int i2 );
    native int zsytri2(int matriint, char uplo, int n,
                       IComplexNumber a, int lda,int ipiv );
    native int zsytri2_work(int matriint, char uplo, int n,
                            IComplexNumber a, int lda,int ipiv,
                            IComplexNumber work, int lwork );
    native int zsytri2x(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv, int nb );
    native int zsytri2int(int matriint, char uplo, int n,
                          IComplexNumber a, int lda,int ipiv,
                          IComplexNumber work, int nb );
    native int zsytrs2(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                       IComplexNumber b, int ldb );
    native int zsytrs2_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber work );
    native int zunbdb(int matriint, char trans, char signs,int m, int p, int q,
                      IComplexNumber x11, int ldx11,
                      IComplexNumber x12, int ldx12,
                      IComplexNumber x21, int ldx21,
                      IComplexNumber x22, int ldx22,
                      long theta, long phi,
                      IComplexNumber taup1,
                      IComplexNumber taup2,
                      IComplexNumber tauq1,
                      IComplexNumber tauq2 );
    native int zunbdint(int matriint, char trans, char signs,int m, int p, int q,
                        IComplexNumber x11, int ldx11,
                        IComplexNumber x12, int ldx12,
                        IComplexNumber x21, int ldx21,
                        IComplexNumber x22, int ldx22,
                        long theta, long phi,
                        IComplexNumber taup1,
                        IComplexNumber taup2,
                        IComplexNumber tauq1,
                        IComplexNumber tauq2,
                        IComplexNumber work, int lwork );
    native int zuncsd(int matriint, char jobu1, char jobu2,
                      char jobv1t, char jobv2t, char trans, char signs,int m, int p, int q,
                      IComplexNumber x11, int ldx11,
                      IComplexNumber x12, int ldx12,
                      IComplexNumber x21, int ldx21,
                      IComplexNumber x22, int ldx22,
                      long theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber v2t,int ldv2t );
    native int zuncsint(int matriint, char jobu1, char jobu2,
                        char jobv1t, char jobv2t, char trans,
                        char signs, int m, int p,int q, IComplexNumber x11,int ldx11, IComplexNumber x12,int ldx12, IComplexNumber x21,int ldx21, IComplexNumber x22,int ldx22, long theta,
                        IComplexNumber u1, int ldu1,
                        IComplexNumber u2, int ldu2,
                        IComplexNumber v1t, int ldv1t,
                        IComplexNumber v2t, int ldv2t,
                        IComplexNumber work, int lwork,
                        long rwork, int lrwork,int iwork );
    native int zuncsd2by1(int matriint, char jobu1, char jobu2,
                          char jobv1t, int m, int p, int q,
                          IComplexNumber x11, int ldx11,
                          IComplexNumber x21, int ldx21,
                          IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t, int ldv1t );
    native int zuncsd2by1_work(int matriint, char jobu1, char jobu2,
                               char jobv1t, int m, int p,int q, IComplexNumber x11, int ldx11,
                               IComplexNumber x21, int ldx21,
                               IComplexNumber theta, IComplexNumber u1,int ldu1, IComplexNumber u2,int ldu2, IComplexNumber v1t,int ldv1t, IComplexNumber work,int lwork, long rwork, int lrwork,int iwork );

    //LAPACK 3.4.0
    native int sgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                       long t, int ldt, long c,int ldc );
    native int dgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                       long t, int ldt, long c,int ldc );
    native int cgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );
    native int zgemqrt(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc );

    native int sgeqrt(int matriint, int m, int n,int nb, long a, int lda, long t,int ldt );
    native int dgeqrt(int matriint, int m, int n,int nb, long a, int lda, long t,int ldt );
    native int cgeqrt(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt );
    native int zgeqrt(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt );

    native int sgeqrt2(int matriint, int m, int n,
                       long a, int lda, long t,int ldt );
    native int dgeqrt2(int matriint, int m, int n,
                       long a, int lda, long t,int ldt );
    native int cgeqrt2(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber t, int ldt );
    native int zgeqrt2(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber t, int ldt );

    native int sgeqrt3(int matriint, int m, int n,
                       long a, int lda, long t,int ldt );
    native int dgeqrt3(int matriint, int m, int n,
                       long a, int lda, long t,int ldt );
    native int cgeqrt3(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber t, int ldt );
    native int zgeqrt3(int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber t, int ldt );

    native int stpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t, int ldt,
                       long a, int lda, long b,int ldb );
    native int dtpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t, int ldt,
                       long a, int lda, long b,int ldb );
    native int ctpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                       IComplexNumber v, int ldv,
                       IComplexNumber t, int ldt,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb );
    native int ztpmqrt(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                       IComplexNumber v, int ldv,
                       IComplexNumber t, int ldt,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb );

    native int stpqrt(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb, long t,int ldt );

    native int dtpqrt(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb, long t,int ldt );
    native int ctpqrt(int matriint, int m, int n,int l, int nb,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber t, int ldt );
    native int ztpqrt(int matriint, int m, int n,int l, int nb,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber t, int ldt );

    native int stpqrt2(int matriint,int m, int n, int l,
                       long a, int lda,
                       long b, int ldb,
                       long t, int ldt );
    native int dtpqrt2(int matriint,int m, int n, int l,
                       long a, int lda,
                       long b, int ldb,
                       long t, int ldt );
    native int ctpqrt2(int matriint,int m, int n, int l,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber t, int ldt );
    native int ztpqrt2(int matriint,int m, int n, int l,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber t, int ldt );

    native int stprfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k, int l,  long v,int ldv,  long t, int ldt,
                      long a, int lda, long b, int ldb );
    native int dtprfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k, int l,  long v,int ldv,  long t, int ldt,
                      long a, int lda, long b, int ldb );
    native int ctprfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k, int l,
                      IComplexNumber v, int ldv,
                      IComplexNumber t, int ldt,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb );
    native int ztprfb(int matriint, char side, char trans, char direct,
                      char storev, int m, int n,int k, int l,
                      IComplexNumber v, int ldv,
                      IComplexNumber t, int ldt,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb );

    native int sgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                         long t, int ldt, long c,int ldc, long work );
    native int dgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  long v, int ldv,
                         long t, int ldt, long c,int ldc, long work );
    native int cgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc, IComplexNumber work );
    native int zgemqrint(int matriint, char side, char trans,int m, int n, int k,int nb,  IComplexNumber v,int ldv,  IComplexNumber t,int ldt, IComplexNumber c,int ldc, IComplexNumber work );

    native int sgeqrint(int matriint, int m, int n,int nb, long a, int lda,
                        long t, int ldt, long work );
    native int dgeqrint(int matriint, int m, int n,int nb, long a, int lda,
                        long t, int ldt, long work );
    native int cgeqrint(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt, IComplexNumber work );
    native int zgeqrint(int matriint, int m, int n,int nb, IComplexNumber a,int lda, IComplexNumber t,int ldt, IComplexNumber work );

    native int sgeqrt2_work(int matriint, int m, int n,
                            long a, int lda, long t,int ldt );
    native int dgeqrt2_work(int matriint, int m, int n,
                            long a, int lda, long t,int ldt );
    native int cgeqrt2_work(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber t, int ldt );
    native int zgeqrt2_work(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber t, int ldt );

    native int sgeqrt3_work(int matriint, int m, int n,
                            long a, int lda, long t,int ldt );
    native int dgeqrt3_work(int matriint, int m, int n,
                            long a, int lda, long t,int ldt );
    native int cgeqrt3_work(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber t, int ldt );
    native int zgeqrt3_work(int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber t, int ldt );

    native int stpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t, int ldt,
                         long a, int lda, long b,int ldb, long work );
    native int dtpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,  long v,int ldv,  long t,int ldt, long a, int lda,
                         long b, int ldb, long work );
    native int ctpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                         IComplexNumber v, int ldv,
                         IComplexNumber t, int ldt,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber work );
    native int ztpmqrint(int matriint, char side, char trans,int m, int n, int k,int l, int nb,
                         IComplexNumber v, int ldv,
                         IComplexNumber t, int ldt,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber work );

    native int stpqrint(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb,
                        long t, int ldt, long work );
    native int dtpqrint(int matriint, int m, int n,int l, int nb, long a,int lda, long b, int ldb,
                        long t, int ldt, long work );
    native int ctpqrint(int matriint, int m, int n,int l, int nb,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber t, int ldt,
                        IComplexNumber work );
    native int ztpqrint(int matriint, int m, int n,int l, int nb,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber t, int ldt,
                        IComplexNumber work );

    native int stpqrt2_work(int matriint,int m, int n, int l,
                            long a, int lda,
                            long b, int ldb,
                            long t, int ldt );
    native int dtpqrt2_work(int matriint,int m, int n, int l,
                            long a, int lda,
                            long b, int ldb,
                            long t, int ldt );
    native int ctpqrt2_work(int matriint,int m, int n, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber t, int ldt );
    native int ztpqrt2_work(int matriint,int m, int n, int l,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber t, int ldt );

    native int stprfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k, int l,
                        long v, int ldv,  long t,int ldt, long a, int lda,
                        long b, int ldb,  long work,int ldwork );
    native int dtprfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k, int l,
                        long v, int ldv,
                        long t, int ldt, long a,int lda, long b, int ldb,
                        long work, int ldwork );
    native int ctprfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k, int l,
                        IComplexNumber v, int ldv,
                        IComplexNumber t, int ldt,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int ldwork );
    native int ztprfint(int matriint, char side, char trans,
                        char direct, char storev, int m,int n, int k, int l,
                        IComplexNumber v, int ldv,
                        IComplexNumber t, int ldt,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int ldwork );
    //LAPACK 3.X.X
    native int ssysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
    native int dsysint(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb );
    native int csysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                       IComplexNumber b, int ldb );
    native int zsysint(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                       IComplexNumber b, int ldb );

    native int ssytrint(int matriint, char uplo, int n, long a,int lda, int ipiv );
    native int dsytrint(int matriint, char uplo, int n, long a,int lda, int ipiv );
    native int csytrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv );
    native int zsytrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv );

    native int ssytrint(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int dsytrint(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b, int ldb );
    native int csytrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                        IComplexNumber b, int ldb );
    native int zsytrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                        IComplexNumber b, int ldb );

    native int chetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv );
    native int zhetrint(int matriint, char uplo, int n,
                        IComplexNumber a, int lda,int ipiv );

    native int chetrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                        IComplexNumber b, int ldb );
    native int zhetrint(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                        IComplexNumber b, int ldb );

    native int csyr(int matriint, char uplo, int n,
                    IComplexNumber alpha,
                    IComplexNumber x, int incx,
                    IComplexNumber a, int lda );
    native int zsyr(int matriint, char uplo, int n,
                    IComplexNumber alpha,
                    IComplexNumber x, int incx,
                    IComplexNumber a, int lda );

    native int ssysint_work(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                            long work, int lwork );
    native int dsysint_work(int matriint, char uplo, int n,int nrhs, long a, int lda,int ipiv, long b, int ldb,
                            long work, int lwork );
    native int csysint_work(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber work,int lwork );
    native int zsysint_work(int matriint, char uplo, int n,int nrhs, IComplexNumber a,int lda, int ipiv,
                            IComplexNumber b, int ldb,
                            IComplexNumber work,int lwork );

    native int ssytrint_work(int matriint, char uplo, int n,
                             long a, int lda, int ipiv,
                             long work, int lwork );
    native int dsytrint_work(int matriint, char uplo, int n,
                             long a, int lda, int ipiv,
                             long work, int lwork );
    native int csytrint_work(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
    native int zsytrint_work(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

    native int ssytrint_work(int matriint, char uplo, int n,int nrhs,  long a, int lda,int ipiv, long b,int ldb );
    native int dsytrint_work(int matriint, char uplo, int n,int nrhs,  long a,int lda,int ipiv,
                             long b, int ldb );
    native int csytrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                             IComplexNumber b, int ldb );
    native int zsytrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                             IComplexNumber b, int ldb );

    native int chetrint_work(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );
    native int zhetrint_work(int matriint, char uplo, int n,
                             IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork );

    native int chetrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                             IComplexNumber b, int ldb );
    native int zhetrint_work(int matriint, char uplo, int n,int nrhs,  IComplexNumber a,int lda,int ipiv,
                             IComplexNumber b, int ldb );


    native int csyint(int matriint, char uplo, int n,
                      IComplexNumber alpha,
                      IComplexNumber x,int incx, IComplexNumber a,int lda );
    native int zsyint(int matriint, char uplo, int n,
                      IComplexNumber alpha,
                      IComplexNumber x,int incx, IComplexNumber a,int lda );

    native void dgetrf(int m, int n, long a, int lda,int ipiv, int info );
    native void cgetrf(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
    native void zgetrf(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
    native void sgetrf2(int m, int n, long a, int lda,int ipiv, int info );
    native void dgetrf2(int m, int n, long a, int lda,int ipiv, int info );
    native void cgetrf2(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
    native void zgetrf2(int m, int n, IComplexNumber a,int lda, int ipiv, int info );
    native void sgbtrf(int m, int n, int kl,int ku, long ab, int ldab,int ipiv, int info );
    native void dgbtrf(int m, int n, int kl,int ku, long ab, int ldab,int ipiv, int info );
    native void cgbtrf(int m, int n, int kl,int ku, IComplexNumber ab, int ldab,int ipiv, int info );
    native void zgbtrf(int m, int n, int kl,int ku, IComplexNumber ab, int ldab,int ipiv, int info );
    native void sgttrf(int n, long dl, long d, long du, long du2,int ipiv, int info );
    native void dgttrf(int n, long dl, long d, long du,
                       long du2, int ipiv, int info );
    native void cgttrf(int n, IComplexNumber dl,
                       IComplexNumber d, IComplexNumber du,
                       IComplexNumber du2, int ipiv,int info );
    native void zgttrf(int n, IComplexNumber dl,
                       IComplexNumber d, IComplexNumber du,
                       IComplexNumber du2, int ipiv,int info );
    native void spotrf2( String uplo, int n, long a, int lda,int info );
    native void dpotrf2( String uplo, int n, long a, int lda,int info );
    native void cpotrf2( String uplo, int n, IComplexNumber a,int lda, int info );
    native void zpotrf2( String uplo, int n, IComplexNumber a,int lda, int info );
    native void spotrf( String uplo, int n, long a, int lda,int info );
    native void dpotrf( String uplo, int n, long a, int lda,int info );
    native void cpotrf( String uplo, int n, IComplexNumber a,int lda, int info );
    native void zpotrf( String uplo, int n, IComplexNumber a,int lda, int info );
    native void dpstrf( String uplo, int n, long a, int lda,int piv, int rank, long tol,
                        long work, int info );
    native void spstrf( String uplo, int n, long a, int lda,int piv, int rank, long tol, long work,int info );
    native void zpstrf( String uplo, int n, IComplexNumber a,int lda, int piv, int rank,
                        long tol, long work, int info );
    native void cpstrf( String uplo, int n, IComplexNumber a,int lda, int piv, int rank,
                        long tol, long work, int info );
    native void dpftrf( String transr, String uplo, int n, long a,int info );
    native void spftrf( String transr, String uplo, int n, long a,int info );
    native void zpftrf( String transr, String uplo, int n,
                        IComplexNumber a, int info );
    native void cpftrf( String transr, String uplo, int n,
                        IComplexNumber a, int info );
    native void spptrf( String uplo, int n, long ap, int info );
    native void dpptrf( String uplo, int n, long ap, int info );
    native void cpptrf( String uplo, int n, IComplexNumber ap,int info );
    native void zpptrf( String uplo, int n, IComplexNumber ap,int info );
    native void spbtrf( String uplo, int n, int kd, long ab,int ldab, int info );
    native void dpbtrf( String uplo, int n, int kd, long ab,int ldab, int info );
    native void cpbtrf( String uplo, int n, int kd,
                        IComplexNumber ab, int ldab,int info );
    native void zpbtrf( String uplo, int n, int kd,
                        IComplexNumber ab, int ldab,int info );
    native void spttrf(int n, long d, long e, int info );
    native void dpttrf(int n, long d, long e, int info );
    native void cpttrf(int n, long d, IComplexNumber e,int info );
    native void zpttrf(int n, long d, IComplexNumber e,int info );
    native void ssytrf( String uplo, int n, long a, int lda,int ipiv, long work, int lwork,int info );
    native void dsytrf( String uplo, int n, long a, int lda,int ipiv, long work, int lwork,int info );
    native void csytrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                        IComplexNumber work, int lwork,int info );
    native void zsytrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                        IComplexNumber work, int lwork,int info );
    native void chetrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                        IComplexNumber work, int lwork,int info );
    native void zhetrf( String uplo, int n, IComplexNumber a,int lda, int ipiv,
                        IComplexNumber work, int lwork,int info );
    native void ssptrf( String uplo, int n, long ap, int ipiv,int info );
    native void dsptrf( String uplo, int n, long ap, int ipiv,int info );
    native void csptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
    native void zsptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
    native void chptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
    native void zhptrf( String uplo, int n, IComplexNumber ap,int ipiv, int info );
    native void sgetrs( String trans, int n, int nrhs,
                        long a, int lda,int ipiv,
                        long b, int ldb, int info );
    native void dgetrs( String trans, int n, int nrhs,
                        long a, int lda,int ipiv,
                        long b, int ldb, int info );
    native void cgetrs( String trans, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
    native void zgetrs( String trans, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
    native void sgbtrs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,int ipiv, long b, int ldb,int info );
    native void dgbtrs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,int ipiv, long b, int ldb,int info );
    native void cgbtrs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void zgbtrs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void sgttrs( String trans, int n, int nrhs,
                        long dl,  long d,  long du,
                        long du2,int ipiv, long b,int ldb, int info );
    native void dgttrs( String trans, int n, int nrhs,
                        long dl,  long d,  long du,
                        long du2,int ipiv, long b,int ldb, int info );
    native void cgttrs( String trans, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void zgttrs( String trans, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void spotrs( String uplo, int n, int nrhs,  long a,int lda, long b, int ldb,int info );
    native void dpotrs( String uplo, int n, int nrhs,
                        long a, int lda, long b,int ldb, int info );
    native void cpotrs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,int info );
    native void zpotrs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,int info );
    native void dpftrs( String transr, String uplo, int n, int nrhs,
                        long a, long b, int ldb,int info );
    native void spftrs( String transr, String uplo, int n, int nrhs,
                        long a, long b, int ldb,int info );
    native void zpftrs( String transr, String uplo, int n, int nrhs,
                        IComplexNumber a, IComplexNumber b,int ldb, int info );
    native void cpftrs( String transr, String uplo, int n, int nrhs,
                        IComplexNumber a, IComplexNumber b,int ldb, int info );
    native void spptrs( String uplo, int n, int nrhs,
                        long ap, long b, int ldb,int info );
    native void dpptrs( String uplo, int n, int nrhs,
                        long ap, long b, int ldb,int info );
    native void cpptrs( String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber b,int ldb, int info );
    native void zpptrs( String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber b,int ldb, int info );
    native void spbtrs( String uplo, int n, int kd, int nrhs,
                        long ab, int ldab, long b,int ldb, int info );
    native void dpbtrs( String uplo, int n, int kd, int nrhs,
                        long ab, int ldab, long b,int ldb, int info );
    native void cpbtrs( String uplo, int n, int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb,int info );
    native void zpbtrs( String uplo, int n, int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb,int info );
    native void spttrs(int n, int nrhs,  long d,
                       long e, long b, int ldb,int info );
    native void dpttrs(int n, int nrhs,  long d,
                       long e, long b, int ldb,int info );
    native void cpttrs( String uplo, int n, int nrhs,  long d,
                        IComplexNumber e, IComplexNumber b,int ldb, int info );
    native void zpttrs( String uplo, int n, int nrhs,
                        long d,  IComplexNumber e,
                        IComplexNumber b, int ldb,int info );
    native void ssytrs( String uplo, int n, int nrhs,  long a,int lda,int ipiv, long b,int ldb, int info );
    native void dsytrs( String uplo, int n, int nrhs,
                        long a, int lda,int ipiv,
                        long b, int ldb, int info );
    native void csytrs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
    native void zsytrs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
    native void chetrs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
    native void zhetrs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,int ipiv, IComplexNumber b,int ldb, int info );
    native void ssptrs( String uplo, int n, int nrhs,
                        long ap,int ipiv, long b,int ldb, int info );
    native void dsptrs( String uplo, int n, int nrhs,
                        long ap,int ipiv, long b,int ldb, int info );
    native void csptrs( String uplo, int n, int nrhs,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void zsptrs( String uplo, int n, int nrhs,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void chptrs( String uplo, int n, int nrhs,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void zhptrs( String uplo, int n, int nrhs,
                        IComplexNumber ap,int ipiv,
                        IComplexNumber b, int ldb,int info );
    native void strtrs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda, long b,int ldb, int info );
    native void dtrtrs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda,
                        long b, int ldb, int info );
    native void ctrtrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b, int ldb,int info );
    native void ztrtrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda, IComplexNumber b, int ldb,int info );
    native void stptrs( String uplo, String trans, String diag, int n,int nrhs,  long ap, long b,int ldb, int info );
    native void dtptrs( String uplo, String trans, String diag, int n,int nrhs,  long ap, long b,int ldb, int info );
    native void ctptrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber b, int ldb,int info );
    native void ztptrs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber b, int ldb,int info );
    native void stbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb,int info );
    native void dtbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab, long b, int ldb,int info );
    native void ctbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb,int info );
    native void ztbtrs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb,int info );
    native void sgecon( String norm, int n,  long a, int lda,
                        long anorm, long rcond, long work, int iwork,int info );
    native void dgecon( String norm, int n,  long a, int lda,
                        long anorm, long rcond, long work,int iwork, int info );
    native void cgecon( String norm, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void zgecon( String norm, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void sgbcon( String norm, int n, int kl, int ku,
                        long ab, int ldab,int ipiv,
                        long anorm, long rcond, long work, int iwork,int info );
    native void dgbcon( String norm, int n, int kl, int ku,
                        long ab, int ldab,int ipiv,
                        long anorm, long rcond, long work,int iwork, int info );
    native void cgbcon( String norm, int n, int kl, int ku,
                        IComplexNumber ab, int ldab,int ipiv, long anorm, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void zgbcon( String norm, int n, int kl, int ku,
                        IComplexNumber ab, int ldab,int ipiv, long anorm, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void sgtcon( String norm, int n,  long dl,  long d,
                        long du,  long du2,int ipiv,
                        long anorm, long rcond, long work, int iwork,int info );
    native void dgtcon( String norm, int n,  long dl,
                        long d,  long du,  long du2,int ipiv, long anorm, long rcond,
                        long work, int iwork, int info );
    native void cgtcon( String norm, int n,  IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2,int ipiv,
                        long anorm, long rcond, IComplexNumber work,int info );
    native void zgtcon( String norm, int n,  IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber du2,int ipiv,
                        long anorm, long rcond, IComplexNumber work,int info );
    native void spocon( String uplo, int n,  long a, int lda,
                        long anorm, long rcond, long work, int iwork,int info );
    native void dpocon( String uplo, int n,  long a, int lda,
                        long anorm, long rcond, long work,int iwork, int info );
    native void cpocon( String uplo, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void zpocon( String uplo, int n,  IComplexNumber a,int lda, long anorm, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void sppcon( String uplo, int n,  long ap, long anorm,
                        long rcond, long work, int iwork,int info );
    native void dppcon( String uplo, int n,  long ap, long anorm,
                        long rcond, long work, int iwork,int info );
    native void cppcon( String uplo, int n,  IComplexNumber ap,
                        long anorm, long rcond, IComplexNumber work,
                        long rwork, int info );
    native void zppcon( String uplo, int n,  IComplexNumber ap,
                        long anorm, long rcond, IComplexNumber work,
                        long rwork, int info );
    native void spbcon( String uplo, int n, int kd,  long ab,int ldab, long anorm, long rcond, long work,int iwork, int info );
    native void dpbcon( String uplo, int n, int kd,  long ab,int ldab, long anorm, long rcond,
                        long work, int iwork, int info );
    native void cpbcon( String uplo, int n, int kd,
                        IComplexNumber ab, int ldab,
                        long anorm, long rcond, IComplexNumber work,
                        long rwork, int info );
    native void zpbcon( String uplo, int n, int kd,
                        IComplexNumber ab, int ldab,
                        long anorm, long rcond, IComplexNumber work,
                        long rwork, int info );
    native void sptcon(int n,  long d,  long e, long anorm,
                       long rcond, long work, int info );
    native void dptcon(int n,  long d,  long e,
                       long anorm, long rcond, long work,int info );
    native void cptcon(int n,  long d,
                       IComplexNumber e, long anorm, long rcond,
                       long work, int info );
    native void zptcon(int n,  long d,
                       IComplexNumber e, long anorm,
                       long rcond, long work, int info );
    native void ssycon( String uplo, int n,  long a, int lda,int ipiv, long anorm, long rcond,
                        long work, int iwork, int info );
    native void dsycon( String uplo, int n,  long a, int lda,int ipiv, long anorm, long rcond,
                        long work, int iwork, int info );
    native void csycon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                        long rcond, IComplexNumber work,int info );
    native void zsycon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                        long rcond, IComplexNumber work,int info );
    native void checon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                        long rcond, IComplexNumber work,int info );
    native void zhecon( String uplo, int n,  IComplexNumber a,int lda,int ipiv, long anorm,
                        long rcond, IComplexNumber work,int info );
    native void sspcon( String uplo, int n,  long ap,int ipiv, long anorm, long rcond,
                        long work, int iwork, int info );
    native void dspcon( String uplo, int n,  long ap,int ipiv, long anorm, long rcond,
                        long work, int iwork, int info );
    native void cspcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                        IComplexNumber work, int info );
    native void zspcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                        IComplexNumber work, int info );
    native void chpcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                        IComplexNumber work, int info );
    native void zhpcon( String uplo, int n,  IComplexNumber ap,int ipiv, long anorm, long rcond,
                        IComplexNumber work, int info );
    native void strcon( String norm, String uplo, String diag, int n,
                        long a, int lda, long rcond, long work,int iwork, int info );
    native void dtrcon( String norm, String uplo, String diag, int n,
                        long a, int lda, long rcond,
                        long work, int iwork, int info );
    native void ctrcon( String norm, String uplo, String diag, int n,
                        IComplexNumber a, int lda,
                        long rcond, IComplexNumber work, long rwork,int info );
    native void ztrcon( String norm, String uplo, String diag, int n,
                        IComplexNumber a, int lda,
                        long rcond, IComplexNumber work, long rwork,int info );
    native void stpcon( String norm, String uplo, String diag, int n,
                        long ap, long rcond, long work,int iwork, int info );
    native void dtpcon( String norm, String uplo, String diag, int n,
                        long ap, long rcond, long work,int iwork, int info );
    native void ctpcon( String norm, String uplo, String diag, int n,
                        IComplexNumber ap, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void ztpcon( String norm, String uplo, String diag, int n,
                        IComplexNumber ap, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void stbcon( String norm, String uplo, String diag, int n,int kd,  long ab, int ldab,
                        long rcond, long work, int iwork,int info );
    native void dtbcon( String norm, String uplo, String diag, int n,int kd,  long ab, int ldab,
                        long rcond, long work, int iwork,int info );
    native void ctbcon( String norm, String uplo, String diag, int n,int kd,  IComplexNumber ab,int ldab, long rcond, IComplexNumber work,
                        long rwork, int info );
    native void ztbcon( String norm, String uplo, String diag, int n,int kd,  IComplexNumber ab,int ldab, long rcond,
                        IComplexNumber work, long rwork,int info );
    native void sgerfs( String trans, int n, int nrhs,
                        long a, int lda,  long af,int ldaf,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void dgerfs( String trans, int n, int nrhs,
                        long a, int lda,  long af,int ldaf,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void cgerfs( String trans, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zgerfs( String trans, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void dgerfsx( String trans, String equed, int n, int nrhs,
                         long a, int lda,  long af,int ldaf,int ipiv,  long r,
                         long c,  long b, int ldb,
                         long x, int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void sgerfsx( String trans, String equed, int n, int nrhs,
                         long a, int lda,  long af,int ldaf,int ipiv,  long r,
                         long c,  long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void zgerfsx( String trans, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv,  long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cgerfsx( String trans, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv,  long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void sgbrfs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,
                        long afb, int ldafb,int ipiv,
                        long b, int ldb, long x, int ldx,
                        long ferr, long berr, long work, int iwork,int info );
    native void dgbrfs( String trans, int n, int kl, int ku,int nrhs,  long ab, int ldab,
                        long afb, int ldafb,int ipiv,  long b, int ldb,
                        long x, int ldx, long ferr, long berr,
                        long work, int iwork, int info );
    native void cgbrfs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zgbrfs( String trans, int n, int kl, int ku,int nrhs,  IComplexNumber ab,int ldab,  IComplexNumber afb,int ldafb,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void dgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,  long ab,int ldab,  long afb, int ldafb,int ipiv,  long r,  long c,
                         long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void sgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,  long ab,int ldab,  long afb, int ldafb,int ipiv,  long r,  long c,
                         long b, int ldb, long x, int ldx,
                         long rcond, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
    native void zgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb, int ldafb,int ipiv,  long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cgbrfsx( String trans, String equed, int n, int kl,int ku, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb, int ldafb,int ipiv,  long r,  long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void sgtrfs( String trans, int n, int nrhs,
                        long dl,  long d,  long du,
                        long dlf,  long df,  long duf,
                        long du2,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void dgtrfs( String trans, int n, int nrhs,
                        long dl,  long d,  long du,
                        long dlf,  long df,  long duf,
                        long du2,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void cgtrfs( String trans, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber dlf,
                        IComplexNumber df,
                        IComplexNumber duf,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zgtrfs( String trans, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber dlf,
                        IComplexNumber df,
                        IComplexNumber duf,
                        IComplexNumber du2,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void sporfs( String uplo, int n, int nrhs,  long a,int lda,  long af, int ldaf,
                        long b, int ldb, long x, int ldx,
                        long ferr, long berr, long work, int iwork,int info );
    native void dporfs( String uplo, int n, int nrhs,
                        long a, int lda,  long af,int ldaf,  long b, int ldb,
                        long x, int ldx, long ferr, long berr,
                        long work, int iwork, int info );
    native void cporfs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zporfs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void dporfsx( String uplo, String equed, int n, int nrhs,
                         long a, int lda,  long af,int ldaf,  long s,  long b,int ldb, long x, int ldx, long rcond,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
    native void sporfsx( String uplo, String equed, int n, int nrhs,
                         long a, int lda,  long af,int ldaf,  long s,  long b,int ldb, long x, int ldx, long rcond,
                         long berr, int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void zporfsx( String uplo, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         long s,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                         long rcond, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cporfsx( String uplo, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         long s,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                         long rcond, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void spprfs( String uplo, int n, int nrhs,
                        long ap,  long afp,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void dpprfs( String uplo, int n, int nrhs,
                        long ap,  long afp,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void cpprfs( String uplo, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zpprfs( String uplo, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void spbrfs( String uplo, int n, int kd, int nrhs,
                        long ab, int ldab,  long afb,int ldafb,  long b, int ldb,
                        long x, int ldx, long ferr, long berr,
                        long work, int iwork, int info );
    native void dpbrfs( String uplo, int n, int kd, int nrhs,
                        long ab, int ldab,  long afb,int ldafb,  long b, int ldb,
                        long x, int ldx, long ferr, long berr,
                        long work, int iwork, int info );
    native void cpbrfs( String uplo, int n, int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber afb, int ldafb,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zpbrfs( String uplo, int n, int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber afb, int ldafb,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void sptrfs(int n, int nrhs,  long d,
                       long e,  long df,  long ef,
                       long b, int ldb, long x, int ldx,
                       long ferr, long berr, long work, int info );
    native void dptrfs(int n, int nrhs,  long d,
                       long e,  long df,  long ef,
                       long b, int ldb, long x,int ldx, long ferr, long berr, long work,int info );
    native void cptrfs( String uplo, int n, int nrhs,  long d,
                        IComplexNumber e,  long df,
                        IComplexNumber ef,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zptrfs( String uplo, int n, int nrhs,
                        long d,  IComplexNumber e,
                        long df,  IComplexNumber ef,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void ssyrfs( String uplo, int n, int nrhs,  long a,int lda,  long af, int ldaf,int ipiv,  long b, int ldb,
                        long x, int ldx, long ferr, long berr,
                        long work, int iwork, int info );
    native void dsyrfs( String uplo, int n, int nrhs,
                        long a, int lda,  long af,int ldaf,int ipiv,  long b,int ldb, long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void csyrfs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zsyrfs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void dsyrfsx( String uplo, String equed, int n, int nrhs,
                         long a, int lda,  long af,int ldaf,int ipiv,  long s,
                         long b, int ldb, long x,int ldx, long rcond, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void ssyrfsx( String uplo, String equed, int n, int nrhs,
                         long a, int lda,  long af,int ldaf,int ipiv,  long s,
                         long b, int ldb, long x, int ldx,
                         long rcond, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
    native void zsyrfsx( String uplo, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv,  long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void csyrfsx( String uplo, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv,  long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cherfs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zherfs( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zherfsx( String uplo, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv,  long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cherfsx( String uplo, String equed, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv,  long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long berr, int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void ssprfs( String uplo, int n, int nrhs,
                        long ap,  long afp,int ipiv,
                        long b, int ldb, long x, int ldx,
                        long ferr, long berr, long work, int iwork,int info );
    native void dsprfs( String uplo, int n, int nrhs,
                        long ap,  long afp,int ipiv,
                        long b, int ldb, long x,int ldx, long ferr, long berr, long work,int iwork, int info );
    native void csprfs( String uplo, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zsprfs( String uplo, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void chprfs( String uplo, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void zhprfs( String uplo, int n, int nrhs,
                        IComplexNumber ap,
                        IComplexNumber afp,int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void strrfs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda,
                        long b, int ldb,  long x,int ldx, long ferr, long berr, long work,int iwork, int info );
    native void dtrrfs( String uplo, String trans, String diag, int n,int nrhs,  long a, int lda,
                        long b, int ldb,  long x,int ldx, long ferr, long berr, long work,int iwork, int info );
    native void ctrrfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void ztrrfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber a,int lda,  IComplexNumber b,int ldb,  IComplexNumber x,int ldx, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void stprfs( String uplo, String trans, String diag, int n,int nrhs,  long ap,  long b,int ldb,  long x, int ldx,
                        long ferr, long berr, long work, int iwork,int info );
    native void dtprfs( String uplo, String trans, String diag, int n,int nrhs,  long ap,  long b,int ldb,  long x, int ldx,
                        long ferr, long berr, long work, int iwork,int info );
    native void ctprfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void ztprfs( String uplo, String trans, String diag, int n,int nrhs,  IComplexNumber ap,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void stbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab,  long b, int ldb,
                        long x, int ldx, long ferr, long berr,
                        long work, int iwork, int info );
    native void dtbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,  long ab,int ldab,  long b, int ldb,
                        long x, int ldx, long ferr,
                        long berr, long work, int iwork,int info );
    native void ctbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long ferr,
                        long berr, IComplexNumber work, long rwork,int info );
    native void ztbrfs( String uplo, String trans, String diag, int n,int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void sgetri(int n, long a, int lda,int ipiv, long work, int lwork,int info );
    native void dgetri(int n, long a, int lda,int ipiv, long work, int lwork,int info );
    native void cgetri(int n, IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork, int info );
    native void zgetri(int n, IComplexNumber a, int lda,int ipiv, IComplexNumber work,int lwork, int info );
    native void spotri( String uplo, int n, long a, int lda,int info );
    native void dpotri( String uplo, int n, long a, int lda,int info );
    native void cpotri( String uplo, int n, IComplexNumber a,int lda, int info );
    native void zpotri( String uplo, int n, IComplexNumber a,int lda, int info );
    native void dpftri( String transr, String uplo, int n, long a,int info );
    native void spftri( String transr, String uplo, int n, long a,int info );
    native void zpftri( String transr, String uplo, int n,
                        IComplexNumber a, int info );
    native void cpftri( String transr, String uplo, int n,
                        IComplexNumber a, int info );
    native void spptri( String uplo, int n, long ap, int info );
    native void dpptri( String uplo, int n, long ap, int info );
    native void cpptri( String uplo, int n, IComplexNumber ap,int info );
    native void zpptri( String uplo, int n, IComplexNumber ap,int info );
    native void ssytri( String uplo, int n, long a, int lda,int ipiv, long work, int info );
    native void dsytri( String uplo, int n, long a, int lda,int ipiv, long work, int info );
    native void csytri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                        IComplexNumber work, int info );
    native void zsytri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                        IComplexNumber work, int info );
    native void chetri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                        IComplexNumber work, int info );
    native void zhetri( String uplo, int n, IComplexNumber a,int lda,int ipiv,
                        IComplexNumber work, int info );
    native void ssptri( String uplo, int n, long ap,int ipiv, long work, int info );
    native void dsptri( String uplo, int n, long ap,int ipiv, long work, int info );
    native void csptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
    native void zsptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
    native void chptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
    native void zhptri( String uplo, int n, IComplexNumber ap,int ipiv, IComplexNumber work,int info );
    native void strtri( String uplo, String diag, int n, long a,int lda, int info );
    native void dtrtri( String uplo, String diag, int n, long a,int lda, int info );
    native void ctrtri( String uplo, String diag, int n,
                        IComplexNumber a, int lda,int info );
    native void ztrtri( String uplo, String diag, int n,
                        IComplexNumber a, int lda,int info );
    native void dtftri( String transr, String uplo, String diag, int n,
                        long a, int info );
    native void stftri( String transr, String uplo, String diag, int n,
                        long a, int info );
    native void ztftri( String transr, String uplo, String diag, int n,
                        IComplexNumber a, int info );
    native void ctftri( String transr, String uplo, String diag, int n,
                        IComplexNumber a, int info );
    native void stptri( String uplo, String diag, int n, long ap,int info );
    native void dtptri( String uplo, String diag, int n, long ap,int info );
    native void ctptri( String uplo, String diag, int n,
                        IComplexNumber ap, int info );
    native void ztptri( String uplo, String diag, int n,
                        IComplexNumber ap, int info );
    native void sgeequ(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                       long colcnd, long amax, int info );
    native void dgeequ(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                       long colcnd, long amax, int info );
    native void cgeequ(int m, int n,  IComplexNumber a,int lda, long r, long c, long rowcnd,
                       long colcnd, long amax, int info );
    native void zgeequ(int m, int n,
                       IComplexNumber a, int lda, long r,
                       long c, long rowcnd, long colcnd, long amax,int info );
    native void dgeequb(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                        long colcnd, long amax, int info );
    native void sgeequb(int m, int n,  long a,int lda, long r, long c, long rowcnd,
                        long colcnd, long amax, int info );
    native void zgeequb(int m, int n,
                        IComplexNumber a, int lda, long r,
                        long c, long rowcnd, long colcnd, long amax,int info );
    native void cgeequb(int m, int n,
                        IComplexNumber a, int lda, long r,
                        long c, long rowcnd, long colcnd, long amax,int info );
    native void sgbequ(int m, int n, int kl,int ku,  long ab, int ldab, long r,
                       long c, long rowcnd, long colcnd, long amax,int info );
    native void dgbequ(int m, int n, int kl,int ku,  long ab, int ldab,
                       long r, long c, long rowcnd, long colcnd,
                       long amax, int info );
    native void cgbequ(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                       long colcnd, long amax, int info );
    native void zgbequ(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                       long colcnd, long amax, int info );
    native void dgbequb(int m, int n, int kl,int ku,  long ab, int ldab,
                        long r, long c, long rowcnd, long colcnd,
                        long amax, int info );
    native void sgbequb(int m, int n, int kl,int ku,  long ab, int ldab,
                        long r, long c, long rowcnd, long colcnd,
                        long amax, int info );
    native void zgbequb(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                        long colcnd, long amax, int info );
    native void cgbequb(int m, int n, int kl,int ku,  IComplexNumber ab,int ldab, long r, long c, long rowcnd,
                        long colcnd, long amax, int info );
    native void spoequ(int n,  long a, int lda, long s,
                       long scond, long amax, int info );
    native void dpoequ(int n,  long a, int lda, long s,
                       long scond, long amax, int info );
    native void cpoequ(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
    native void zpoequ(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
    native void dpoequb(int n,  long a, int lda, long s,
                        long scond, long amax, int info );
    native void spoequb(int n,  long a, int lda, long s,
                        long scond, long amax, int info );
    native void zpoequb(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
    native void cpoequb(int n,  IComplexNumber a,int lda, long s, long scond, long amax,int info );
    native void sppequ( String uplo, int n,  long ap, long s,
                        long scond, long amax, int info );
    native void dppequ( String uplo, int n,  long ap, long s,
                        long scond, long amax, int info );
    native void cppequ( String uplo, int n,  IComplexNumber ap,
                        long s, long scond, long amax, int info );
    native void zppequ( String uplo, int n,  IComplexNumber ap,
                        long s, long scond, long amax, int info );
    native void spbequ( String uplo, int n, int kd,  long ab,int ldab, long s, long scond, long amax,int info );
    native void dpbequ( String uplo, int n, int kd,  long ab,int ldab, long s, long scond, long amax,int info );
    native void cpbequ( String uplo, int n, int kd,
                        IComplexNumber ab, int ldab, long s,
                        long scond, long amax, int info );
    native void zpbequ( String uplo, int n, int kd,
                        IComplexNumber ab, int ldab,
                        long s, long scond, long amax, int info );
    native void dsyequb( String uplo, int n,  long a,int lda, long s, long scond, long amax,
                         long work, int info );
    native void ssyequb( String uplo, int n,  long a, int lda,
                         long s, long scond, long amax, long work,int info );
    native void zsyequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                         IComplexNumber work, int info );
    native void csyequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                         IComplexNumber work, int info );
    native void zheequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                         IComplexNumber work, int info );
    native void cheequb( String uplo, int n,  IComplexNumber a,int lda, long s, long scond, long amax,
                         IComplexNumber work, int info );
    native void sgesv(int n, int nrhs, long a, int lda,int ipiv, long b, int ldb,int info );
    native void dgesv(int n, int nrhs, long a, int lda,int ipiv, long b, int ldb,int info );
    native void cgesv(int n, int nrhs, IComplexNumber a,int lda, int ipiv, IComplexNumber b,int ldb, int info );
    native void zgesv(int n, int nrhs, IComplexNumber a,int lda, int ipiv, IComplexNumber b,int ldb, int info );
    native void dsgesv(int n, int nrhs, long a, int lda,int ipiv, long b, int ldb, long x,int ldx, long work, long swork,int iter, int info );
    native void zcgesv(int n, int nrhs, IComplexNumber a,int lda, int ipiv, IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                       IComplexNumber work, IComplexNumber swork,
                       long rwork, int iter, int info );
    native void sgesvx( String fact, String trans, int n, int nrhs,
                        long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c, long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int iwork,int info );
    native void dgesvx( String fact, String trans, int n, int nrhs,
                        long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                        long b, int ldb, long x, int ldx,
                        long rcond, long ferr, long berr, long work,int iwork, int info );
    native void cgesvx( String fact, String trans, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zgesvx( String fact, String trans, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void dgesvxx( String fact, String trans, int n, int nrhs,
                         long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                         long b, int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void sgesvxx( String fact, String trans, int n, int nrhs,
                         long a, int lda, long af, int ldaf,int ipiv, String equed, long r, long c,
                         long b, int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void zgesvxx( String fact, String trans, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cgesvxx( String fact, String trans, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long r, long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void sgbsv(int n, int kl, int ku,int nrhs, long ab, int ldab,int ipiv, long b, int ldb,int info );
    native void dgbsv(int n, int kl, int ku,int nrhs, long ab, int ldab,int ipiv, long b, int ldb,int info );
    native void cgbsv(int n, int kl, int ku,int nrhs, IComplexNumber ab, int ldab,int ipiv, IComplexNumber b, int ldb,int info );
    native void zgbsv(int n, int kl, int ku,int nrhs, IComplexNumber ab,int ldab, int ipiv, IComplexNumber b,int ldb, int info );
    native void sgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c, long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int iwork,int info );
    native void dgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c,
                        long b, int ldb, long x, int ldx,
                        long rcond, long ferr, long berr, long work,int iwork, int info );
    native void cgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed, long r,
                        long c, IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zgbsvx( String fact, String trans, int n, int kl,int ku, int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed, long r,
                        long c, IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void dgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c,
                         long b, int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void sgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs, long ab,int ldab, long afb, int ldafb,int ipiv, String equed, long r, long c,
                         long b, int ldb, long x, int ldx,
                         long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void zgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb, int ldafb,int ipiv, String equed, long r, long c,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cgbsvxx( String fact, String trans, int n, int kl,int ku, int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, int ipiv, String equed, long r,
                         long c, IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void sgtsv(int n, int nrhs, long dl, long d,
                      long du, long b, int ldb, int info );
    native void dgtsv(int n, int nrhs, long dl, long d,
                      long du, long b, int ldb, int info );
    native void cgtsv(int n, int nrhs, IComplexNumber dl,
                      IComplexNumber d, IComplexNumber du,
                      IComplexNumber b, int ldb, int info );
    native void zgtsv(int n, int nrhs, IComplexNumber dl,
                      IComplexNumber d, IComplexNumber du,
                      IComplexNumber b, int ldb,int info );
    native void sgtsvx( String fact, String trans, int n, int nrhs,
                        long dl,  long d,  long du,
                        long dlf, long df, long duf, long du2,int ipiv,  long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                        long work, int iwork, int info );
    native void dgtsvx( String fact, String trans, int n, int nrhs,
                        long dl,  long d,  long du,
                        long dlf, long df, long duf, long du2,int ipiv,  long b, int ldb,
                        long x, int ldx, long rcond, long ferr,
                        long berr, long work, int iwork,int info );
    native void cgtsvx( String fact, String trans, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du, IComplexNumber dlf,
                        IComplexNumber df, IComplexNumber duf,
                        IComplexNumber du2, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zgtsvx( String fact, String trans, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du, IComplexNumber dlf,
                        IComplexNumber df, IComplexNumber duf,
                        IComplexNumber du2, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void sposv( String uplo, int n, int nrhs, long a,int lda, long b, int ldb,int info );
    native void dposv( String uplo, int n, int nrhs, long a,int lda, long b, int ldb,int info );
    native void cposv( String uplo, int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb, int info );
    native void zposv( String uplo, int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,int info );
    native void dsposv( String uplo, int n, int nrhs, long a,int lda, long b, int ldb, long x,int ldx, long work, long swork,int iter, int info );
    native void zcposv( String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        IComplexNumber work, IComplexNumber swork,
                        long rwork, int iter, int info );
    native void sposvx( String fact, String uplo, int n, int nrhs,
                        long a, int lda, long af, int ldaf,
                        String equed, long s, long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                        long work, int iwork, int info );
    native void dposvx( String fact, String uplo, int n, int nrhs,
                        long a, int lda, long af, int ldaf,
                        String equed, long s, long b, int ldb,
                        long x, int ldx, long rcond, long ferr,
                        long berr, long work, int iwork,int info );
    native void cposvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf, String equed,
                        long s, IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zposvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf, String equed,
                        long s, IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void dposvxx( String fact, String uplo, int n, int nrhs,
                         long a, int lda, long af, int ldaf,
                         String equed, long s, long b, int ldb,
                         long x, int ldx, long rcond, long rpvgrw,
                         long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
    native void sposvxx( String fact, String uplo, int n, int nrhs,
                         long a, int lda, long af, int ldaf,
                         String equed, long s, long b, int ldb, long x,int ldx, long rcond, long rpvgrw, long berr,int int_bnds, long erint_norm,
                         long erint_comp, int nparams, long params,
                         long work, int iwork, int info );
    native void zposvxx( String fact, String uplo, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf, String equed,
                         long s, IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void cposvxx( String fact, String uplo, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf, String equed,
                         long s, IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void sppsv( String uplo, int n, int nrhs, long ap,
                       long b, int ldb, int info );
    native void dppsv( String uplo, int n, int nrhs, long ap,
                       long b, int ldb, int info );
    native void cppsv( String uplo, int n, int nrhs,
                       IComplexNumber ap, IComplexNumber b,int ldb, int info );
    native void zppsv( String uplo, int n, int nrhs,
                       IComplexNumber ap, IComplexNumber b,int ldb, int info );
    native void sppsvx( String fact, String uplo, int n, int nrhs,
                        long ap, long afp, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int iwork,int info );
    native void dppsvx( String fact, String uplo, int n, int nrhs,
                        long ap, long afp, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int iwork,int info );
    native void cppsvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber afp,
                        String equed, long s, IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void zppsvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber afp,
                        String equed, long s, IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void spbsv( String uplo, int n, int kd, int nrhs,
                       long ab, int ldab, long b, int ldb,int info );
    native void dpbsv( String uplo, int n, int kd, int nrhs,
                       long ab, int ldab, long b, int ldb,int info );
    native void cpbsv( String uplo, int n, int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb, int info );
    native void zpbsv( String uplo, int n, int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb,int info );
    native void spbsvx( String fact, String uplo, int n, int kd,int nrhs, long ab, int ldab, long afb,int ldafb, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int iwork,int info );
    native void dpbsvx( String fact, String uplo, int n, int kd,int nrhs, long ab, int ldab, long afb,int ldafb, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int iwork,int info );
    native void cpbsvx( String fact, String uplo, int n, int kd,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, String equed, long s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void zpbsvx( String fact, String uplo, int n, int kd,int nrhs, IComplexNumber ab,int ldab, IComplexNumber afb,int ldafb, String equed, long s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx, long rcond,
                        long ferr, long berr, IComplexNumber work,
                        long rwork, int info );
    native void sptsv(int n, int nrhs, long d, long e,
                      long b, int ldb, int info );
    native void dptsv(int n, int nrhs, long d, long e,
                      long b, int ldb, int info );
    native void cptsv(int n, int nrhs, long d,
                      IComplexNumber e, IComplexNumber b,int ldb, int info );
    native void zptsv(int n, int nrhs, long d,
                      IComplexNumber e, IComplexNumber b,int ldb, int info );
    native void sptsvx( String fact, int n, int nrhs,  long d,
                        long e, long df, long ef,  long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int info );
    native void dptsvx( String fact, int n, int nrhs,
                        long d,  long e, long df, long ef,
                        long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                        long work, int info );
    native void cptsvx( String fact, int n, int nrhs,  long d,
                        IComplexNumber e, long df,
                        IComplexNumber ef,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void zptsvx( String fact, int n, int nrhs,
                        long d,  IComplexNumber e, long df,
                        IComplexNumber ef,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void ssysv( String uplo, int n, int nrhs, long a,int lda, int ipiv, long b, int ldb,
                       long work, int lwork, int info );
    native void dsysv( String uplo, int n, int nrhs, long a,int lda, int ipiv, long b,int ldb, long work, int lwork,int info );
    native void csysv( String uplo, int n, int nrhs,
                       IComplexNumber a, int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork,int info );
    native void zsysv( String uplo, int n, int nrhs,
                       IComplexNumber a, int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork,int info );
    native void ssysvx( String fact, String uplo, int n, int nrhs,
                        long a, int lda, long af,int ldaf, int ipiv,  long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int lwork,int iwork, int info );
    native void dsysvx( String fact, String uplo, int n, int nrhs,
                        long a, int lda, long af,int ldaf, int ipiv,  long b,int ldb, long x, int ldx, long rcond,
                        long ferr, long berr, long work, int lwork,int iwork, int info );
    native void csysvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, int lwork, long rwork,int info );
    native void zsysvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, int lwork,
                        long rwork, int info );
    native void dsysvxx( String fact, String uplo, int n, int nrhs,
                         long a, int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
    native void ssysvxx( String fact, String uplo, int n, int nrhs,
                         long a, int lda, long af, int ldaf,int ipiv, String equed, long s, long b,int ldb, long x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params, long work,int iwork, int info );
    native void zsysvxx( String fact, String uplo, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void csysvxx( String fact, String uplo, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void chesv( String uplo, int n, int nrhs,
                       IComplexNumber a, int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork,int info );
    native void zhesv( String uplo, int n, int nrhs,
                       IComplexNumber a, int lda, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber work, int lwork,int info );
    native void chesvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, int lwork, long rwork,int info );
    native void zhesvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, int lwork,
                        long rwork, int info );
    native void zhesvxx( String fact, String uplo, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void chesvxx( String fact, String uplo, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,int ipiv, String equed, long s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx, long rcond,
                         long rpvgrw, long berr, int int_bnds,
                         long erint_norm, long erint_comp,int nparams, long params,
                         IComplexNumber work, long rwork,int info );
    native void sspsv( String uplo, int n, int nrhs, long ap,int ipiv, long b, int ldb,int info );
    native void dspsv( String uplo, int n, int nrhs, long ap,int ipiv, long b, int ldb,int info );
    native void cspsv( String uplo, int n, int nrhs,
                       IComplexNumber ap, int ipiv,
                       IComplexNumber b, int ldb, int info );
    native void zspsv( String uplo, int n, int nrhs,
                       IComplexNumber ap, int ipiv,
                       IComplexNumber b, int ldb,int info );
    native void sspsvx( String fact, String uplo, int n, int nrhs,
                        long ap, long afp, int ipiv,
                        long b, int ldb, long x, int ldx,
                        long rcond, long ferr, long berr, long work,int iwork, int info );
    native void dspsvx( String fact, String uplo, int n, int nrhs,
                        long ap, long afp, int ipiv,
                        long b, int ldb, long x,int ldx, long rcond, long ferr, long berr,
                        long work, int iwork, int info );
    native void cspsvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void zspsvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void chpsv( String uplo, int n, int nrhs,
                       IComplexNumber ap, int ipiv,
                       IComplexNumber b, int ldb, int info );
    native void zhpsv( String uplo, int n, int nrhs,
                       IComplexNumber ap, int ipiv,
                       IComplexNumber b, int ldb,int info );
    native void chpsvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void zhpsvx( String fact, String uplo, int n, int nrhs,
                        IComplexNumber ap, IComplexNumber afp,int ipiv,  IComplexNumber b,int ldb, IComplexNumber x, int ldx,
                        long rcond, long ferr, long berr,
                        IComplexNumber work, long rwork,int info );
    native void sgeqrf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void dgeqrf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void cgeqrf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void zgeqrf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sgeqpf(int m, int n, long a, int lda,int jpvt, long tau, long work,int info );
    native void dgeqpf(int m, int n, long a, int lda,int jpvt, long tau, long work,int info );
    native void cgeqpf(int m, int n, IComplexNumber a,int lda, int jpvt,
                       IComplexNumber tau, IComplexNumber work,
                       long rwork, int info );
    native void zgeqpf(int m, int n, IComplexNumber a,int lda, int jpvt,
                       IComplexNumber tau, IComplexNumber work,
                       long rwork, int info );
    native void sgeqp3(int m, int n, long a, int lda,int jpvt, long tau, long work,int lwork, int info );
    native void dgeqp3(int m, int n, long a, int lda,int jpvt, long tau, long work,int lwork, int info );
    native void cgeqp3(int m, int n, IComplexNumber a,int lda, int jpvt,
                       IComplexNumber tau, IComplexNumber work,int lwork, long rwork, int info );
    native void zgeqp3(int m, int n, IComplexNumber a,int lda, int jpvt,
                       IComplexNumber tau, IComplexNumber work,int lwork, long rwork, int info );
    native void sorgqr(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void dorgqr(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void sormqr( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void dormqr( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void cungqr(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau, IComplexNumber work,int lwork, int info );
    native void zungqr(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void cunmqr( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void zunmqr( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void sgelqf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void dgelqf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void cgelqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void zgelqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sorglq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void dorglq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void sormlq( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void dormlq( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void cunglq(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau, IComplexNumber work,int lwork, int info );
    native void zunglq(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void cunmlq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void zunmlq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void sgeqlf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void dgeqlf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void cgeqlf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void zgeqlf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sorgql(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void dorgql(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void cungql(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau, IComplexNumber work,int lwork, int info );
    native void zungql(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sormql( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void dormql( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void cunmql( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void zunmql( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void sgerqf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void dgerqf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void cgerqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void zgerqf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sorgrq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void dorgrq(int m, int n, int k, long a,int lda,  long tau, long work,int lwork, int info );
    native void cungrq(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau, IComplexNumber work,int lwork, int info );
    native void zungrq(int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sormrq( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void dormrq( String side, String trans, int m, int n,int k,  long a, int lda,
                        long tau, long c, int ldc, long work,int lwork, int info );
    native void cunmrq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void zunmrq( String side, String trans, int m, int n,int k,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void stzrzf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void dtzrzf(int m, int n, long a, int lda,
                       long tau, long work, int lwork,int info );
    native void ctzrzf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void ztzrzf(int m, int n, IComplexNumber a,int lda, IComplexNumber tau,
                       IComplexNumber work, int lwork,int info );
    native void sormrz( String side, String trans, int m, int n,int k, int l,  long a,int lda,  long tau, long c,int ldc, long work, int lwork,int info );
    native void dormrz( String side, String trans, int m, int n,int k, int l,  long a,int lda,  long tau, long c,int ldc, long work, int lwork,int info );
    native void cunmrz( String side, String trans, int m, int n,int k, int l,  IComplexNumber a,int lda,  IComplexNumber tau,
                        IComplexNumber c, int ldc,
                        IComplexNumber work, int lwork,int info );
    native void zunmrz( String side, String trans, int m, int n,int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber tau, IComplexNumber c,int ldc, IComplexNumber work,int lwork, int info );
    native void sggqrf(int n, int m, int p, long a,int lda, long taua, long b, int ldb,
                       long taub, long work, int lwork,int info );
    native void dggqrf(int n, int m, int p, long a,int lda, long taua, long b, int ldb,
                       long taub, long work, int lwork,int info );
    native void cggqrf(int n, int m, int p,
                       IComplexNumber a, int lda,
                       IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                       IComplexNumber work, int lwork,int info );
    native void zggqrf(int n, int m, int p,
                       IComplexNumber a, int lda,
                       IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                       IComplexNumber work, int lwork,int info );
    native void sggrqf(int m, int p, int n, long a,int lda, long taua, long b, int ldb,
                       long taub, long work, int lwork,int info );
    native void dggrqf(int m, int p, int n, long a,int lda, long taua, long b, int ldb,
                       long taub, long work, int lwork,int info );
    native void cggrqf(int m, int p, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                       IComplexNumber work, int lwork,int info );
    native void zggrqf(int m, int p, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber taua, IComplexNumber b,int ldb, IComplexNumber taub,
                       IComplexNumber work, int lwork,int info );
    native void sgebrd(int m, int n, long a, int lda,
                       long d, long e, long tauq, long taup, long work,int lwork, int info );
    native void dgebrd(int m, int n, long a, int lda,
                       long d, long e, long tauq, long taup,
                       long work, int lwork, int info );
    native void cgebrd(int m, int n, IComplexNumber a,int lda, long d, long e,
                       IComplexNumber tauq, IComplexNumber taup,
                       IComplexNumber work, int lwork,int info );
    native void zgebrd(int m, int n, IComplexNumber a,int lda, long d, long e,
                       IComplexNumber tauq, IComplexNumber taup,
                       IComplexNumber work, int lwork,int info );
    native void sgbbrd( String vect, int m, int n, int ncc,int kl, int ku, long ab, int ldab,
                        long d, long e, long q, int ldq, long pt,int ldpt, long c, int ldc, long work,int info );
}