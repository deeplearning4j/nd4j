package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

public  interface Lapack {

/* C-LAPACK function prototypes */

    int sbdsdc( int matriint, char uplo, char compq,
                       int n, INDArray d, INDArray e, INDArray u,
                       int ldu, INDArray vt, int ldvt, INDArray q,
                       int iq );
    int dbdsdc( int matriint, char uplo, char compq,
                       int n, INDArray d, INDArray e, INDArray u,
                       int ldu, INDArray vt, int ldvt,
                       INDArray q, int iq );

    int sbdsqr( int matriint, char uplo, int n,
                       int ncvt, int nru, int ncc,
                       INDArray d, INDArray e, INDArray vt, int ldvt,
                       INDArray u, int ldu, INDArray c, int ldc );
    int dbdsqr( int matriint, char uplo, int n,
                       int ncvt, int nru, int ncc,
                       INDArray d, INDArray e, INDArray vt, int ldvt,
                       INDArray u, int ldu, INDArray c,
                       int ldc );
    int cbdsqr( int matriint, char uplo, int n,
                       int ncvt, int nru, int ncc,
                       INDArray d, INDArray e, IComplexNumber vt,
                       int ldvt, IComplexNumber u,
                       int ldu, IComplexNumber c,
                       int ldc );
    int zbdsqr( int matriint, char uplo, int n,
                       int ncvt, int nru, int ncc,
                       INDArray d, INDArray e, IComplexNumber vt,
                       int ldvt, IComplexNumber u,
                       int ldu, IComplexNumber c,
                       int ldc );
    int sbdsvdx( int matriint, char uplo, char jobz, char range,
                        int n, INDArray d, INDArray e,
                        int vl, int vu,
                        int il, int iu, int ns,
                        INDArray s, INDArray z, int ldz,
                        int superb );
    int dbdsvdx( int matriint, char uplo, char jobz, char range,
                        int n, INDArray d, INDArray e,
                        int vl, int vu,
                        int il, int iu, int ns,
                        INDArray s, INDArray z, int ldz,
                        int superb );
    int sdisna( char job, int m, int n,  INDArray d,
                       INDArray sep );
    int ddisna( char job, int m, int n,
                       INDArray d, INDArray sep );

    int sgbbrd( int matriint, char vect, int m,
                       int n, int ncc, int kl,
                       int ku, INDArray ab, int ldab, INDArray d,
                       INDArray e, INDArray q, int ldq, INDArray pt,
                       int ldpt, INDArray c, int ldc );
    int dgbbrd( int matriint, char vect, int m,
                       int n, int ncc, int kl,
                       int ku, INDArray ab, int ldab,
                       INDArray d, INDArray e, INDArray q, int ldq,
                       INDArray pt, int ldpt, INDArray c,
                       int ldc );
    int cgbbrd( int matriint, char vect, int m,
                       int n, int ncc, int kl,
                       int ku, IComplexNumber ab,
                       int ldab, INDArray d, INDArray e,
                       IComplexNumber q, int ldq,
                       IComplexNumber pt, int ldpt,
                       IComplexNumber c, int ldc );
    int zgbbrd( int matriint, char vect, int m,
                       int n, int ncc, int kl,
                       int ku, IComplexNumber ab,
                       int ldab, INDArray d, INDArray e,
                       IComplexNumber q, int ldq,
                       IComplexNumber pt, int ldpt,
                       IComplexNumber c, int ldc );

    int sgbcon( int matriint, char norm, int n,
                       int kl, int ku,  INDArray ab,
                       int ldab,  int ipiv, float anorm,
                       INDArray rcond );
    int dgbcon( int matriint, char norm, int n,
                       int kl, int ku,  INDArray ab,
                       int ldab,  int ipiv,
                       double anorm, INDArray rcond );
    int cgbcon( int matriint, char norm, int n,
                       int kl, int ku,
                       IComplexNumber ab, int ldab,
                       int ipiv, float anorm, INDArray rcond );
    int zgbcon( int matriint, char norm, int n,
                       int kl, int ku,
                       IComplexNumber ab, int ldab,
                       int ipiv, double anorm,
                       INDArray rcond );

    int sgbequ( int matriint, int m, int n,
                       int kl, int ku,  INDArray ab,
                       int ldab, INDArray r, INDArray c, INDArray rowcnd,
                       INDArray colcnd, INDArray amax );
    int dgbequ( int matriint, int m, int n,
                       int kl, int ku,  INDArray ab,
                       int ldab, INDArray r, INDArray c,
                       INDArray rowcnd, INDArray colcnd, INDArray amax );
    int cgbequ( int matriint, int m, int n,
                       int kl, int ku,
                       IComplexNumber ab, int ldab,
                       INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                       INDArray amax );
    int zgbequ( int matriint, int m, int n,
                       int kl, int ku,
                       IComplexNumber ab, int ldab,
                       INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                       INDArray amax );

    int sgbequb( int matriint, int m, int n,
                        int kl, int ku,  INDArray ab,
                        int ldab, INDArray r, INDArray c, INDArray rowcnd,
                        INDArray colcnd, INDArray amax );
    int dgbequb( int matriint, int m, int n,
                        int kl, int ku,  INDArray ab,
                        int ldab, INDArray r, INDArray c,
                        INDArray rowcnd, INDArray colcnd, INDArray amax );
    int cgbequb( int matriint, int m, int n,
                        int kl, int ku,
                        IComplexNumber ab, int ldab,
                        INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                        INDArray amax );
    int zgbequb( int matriint, int m, int n,
                        int kl, int ku,
                        IComplexNumber ab, int ldab,
                        INDArray r, INDArray c, INDArray rowcnd,
                        INDArray colcnd, INDArray amax );

    int sgbrfs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       INDArray ab, int ldab,  INDArray afb,
                       int ldafb,  int ipiv,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int dgbrfs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       INDArray ab, int ldab,  INDArray afb,
                       int ldafb,  int ipiv,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int cgbrfs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber afb, int ldafb,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zgbrfs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber afb, int ldafb,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int sgbrfsx( int matriint, char trans, char equed,
                        int n, int kl, int ku,
                        int nrhs,  INDArray ab, int ldab,
                        INDArray afb, int ldafb,
                        int ipiv,  INDArray r,
                        INDArray c,  INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dgbrfsx( int matriint, char trans, char equed,
                        int n, int kl, int ku,
                        int nrhs,  INDArray ab, int ldab,
                        INDArray afb, int ldafb,
                        int ipiv,  INDArray r,
                        INDArray c,  INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int cgbrfsx( int matriint, char trans, char equed,
                        int n, int kl, int ku,
                        int nrhs,  IComplexNumber ab,
                        int ldab,  IComplexNumber afb,
                        int ldafb,  int ipiv,
                        INDArray r,  INDArray c,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int zgbrfsx( int matriint, char trans, char equed,
                        int n, int kl, int ku,
                        int nrhs,  IComplexNumber ab,
                        int ldab,  IComplexNumber afb,
                        int ldafb,  int ipiv,
                        INDArray r,  INDArray c,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );

    int sgbsv( int matriint, int n, int kl,
                      int ku, int nrhs, INDArray ab,
                      int ldab, int ipiv, INDArray b,
                      int ldb );
    int dgbsv( int matriint, int n, int kl,
                      int ku, int nrhs, INDArray ab,
                      int ldab, int ipiv, INDArray b,
                      int ldb );
    int cgbsv( int matriint, int n, int kl,
                      int ku, int nrhs,
                      IComplexNumber ab, int ldab,
                      int ipiv, IComplexNumber b,
                      int ldb );
    int zgbsv( int matriint, int n, int kl,
                      int ku, int nrhs,
                      IComplexNumber ab, int ldab,
                      int ipiv, IComplexNumber b,
                      int ldb );

    int sgbsvx( int matriint, char fact, char trans,
                       int n, int kl, int ku,
                       int nrhs, INDArray ab, int ldab,
                       INDArray afb, int ldafb, int ipiv,
                       String equed, INDArray r, INDArray c, INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr,
                       INDArray rpivot );
    int dgbsvx( int matriint, char fact, char trans,
                       int n, int kl, int ku,
                       int nrhs, INDArray ab, int ldab,
                       INDArray afb, int ldafb, int ipiv,
                       String equed, INDArray r, INDArray c, INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr,
                       INDArray rpivot );
    int cgbsvx( int matriint, char fact, char trans,
                       int n, int kl, int ku,
                       int nrhs, IComplexNumber ab,
                       int ldab, IComplexNumber afb,
                       int ldafb, int ipiv, String equed,
                       INDArray r, INDArray c, IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr, INDArray rpivot );
    int zgbsvx(int matriint, char fact, char trans,
                      int n, int kl, int ku,
                      int nrhs, IComplexNumber ab,
                      int ldab, IComplexNumber afb,
                      int ldafb, int ipiv, String equed,
                      INDArray r, INDArray c, IComplexNumber b,
                      int ldb, IComplexNumber x,
                      int ldx, INDArray rcond, INDArray ferr,
                      INDArray berr, INDArray rpivot );

    int sgbsvxx( int matriint, char fact, char trans,
                        int n, int kl, int ku,
                        int nrhs, INDArray ab, int ldab,
                        INDArray afb, int ldafb, int ipiv,
                        String equed, INDArray r, INDArray c, INDArray b,
                        int ldb, INDArray x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dgbsvxx( int matriint, char fact, char trans,
                        int n, int kl, int ku,
                        int nrhs, INDArray ab, int ldab,
                        INDArray afb, int ldafb, int ipiv,
                        String equed, INDArray r, INDArray c, INDArray b,
                        int ldb, INDArray x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int cgbsvxx( int matriint, char fact, char trans,
                        int n, int kl, int ku,
                        int nrhs, IComplexNumber ab,
                        int ldab, IComplexNumber afb,
                        int ldafb, int ipiv, String equed,
                        INDArray r, INDArray c, IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray rpvgrw,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int zgbsvxx( int matriint, char fact, char trans,
                        int n, int kl, int ku,
                        int nrhs, IComplexNumber ab,
                        int ldab, IComplexNumber afb,
                        int ldafb, int ipiv, String equed,
                        INDArray r, INDArray c, IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray rpvgrw,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );

    int sgbtrf( int matriint, int m, int n,
                       int kl, int ku, INDArray ab,
                       int ldab, int ipiv );
    int dgbtrf( int matriint, int m, int n,
                       int kl, int ku, INDArray ab,
                       int ldab, int ipiv );
    int cgbtrf( int matriint, int m, int n,
                       int kl, int ku,
                       IComplexNumber ab, int ldab,
                       int ipiv );
    int zgbtrf( int matriint, int m, int n,
                       int kl, int ku,
                       IComplexNumber ab, int ldab,
                       int ipiv );

    int sgbtrs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       INDArray ab, int ldab,
                       int ipiv, INDArray b, int ldb );
    int dgbtrs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       INDArray ab, int ldab,
                       int ipiv, INDArray b, int ldb );
    int cgbtrs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       IComplexNumber ab, int ldab,
                       int ipiv, IComplexNumber b,
                       int ldb );
    int zgbtrs( int matriint, char trans, int n,
                       int kl, int ku, int nrhs,
                       IComplexNumber ab, int ldab,
                       int ipiv, IComplexNumber b,
                       int ldb );

    int sgebak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray scale,
                       int m, INDArray v, int ldv );
    int dgebak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray scale,
                       int m, INDArray v, int ldv );
    int cgebak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray scale,
                       int m, IComplexNumber v,
                       int ldv );
    int zgebak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray scale,
                       int m, IComplexNumber v,
                       int ldv );

    int sgebal( int matriint, char job, int n, INDArray a,
                       int lda, int ilo, int ihi,
                       INDArray scale );
    int dgebal( int matriint, char job, int n, INDArray a,
                       int lda, int ilo, int ihi,
                       INDArray scale );
    int cgebal( int matriint, char job, int n,
                       IComplexNumber a, int lda,
                       int ilo, int ihi, INDArray scale );
    int zgebal( int matriint, char job, int n,
                       IComplexNumber a, int lda,
                       int ilo, int ihi, INDArray scale );

    int sgebrd( int matriint, int m, int n,
                       INDArray a, int lda, INDArray d, INDArray e,
                       INDArray tauq, INDArray taup );
    int dgebrd( int matriint, int m, int n,
                       INDArray a, int lda, INDArray d, INDArray e,
                       INDArray tauq, INDArray taup );
    int cgebrd( int matriint, int m, int n,
                       IComplexNumber a, int lda, INDArray d,
                       INDArray e, IComplexNumber tauq,
                       IComplexNumber taup );
    int zgebrd( int matriint, int m, int n,
                       IComplexNumber a, int lda, INDArray d,
                       INDArray e, IComplexNumber tauq,
                       IComplexNumber taup );

    int sgecon( int matriint, char norm, int n,
                       INDArray a, int lda, float anorm,
                       INDArray rcond );
    int dgecon( int matriint, char norm, int n,
                       INDArray a, int lda, double anorm,
                       INDArray rcond );
    int cgecon( int matriint, char norm, int n,
                       IComplexNumber a, int lda,
                       float anorm, INDArray rcond );
    int zgecon( int matriint, char norm, int n,
                       IComplexNumber a, int lda,
                       double anorm, INDArray rcond );

    int sgeequ( int matriint, int m, int n,
                       INDArray a, int lda, INDArray r, INDArray c,
                       INDArray rowcnd, INDArray colcnd, INDArray amax );
    int dgeequ( int matriint, int m, int n,
                       INDArray a, int lda, INDArray r,
                       INDArray c, INDArray rowcnd, INDArray colcnd,
                       INDArray amax );
    int cgeequ( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                       INDArray amax );
    int zgeequ( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                       INDArray amax );

    int sgeequb( int matriint, int m, int n,
                        INDArray a, int lda, INDArray r, INDArray c,
                        INDArray rowcnd, INDArray colcnd, INDArray amax );
    int dgeequb( int matriint, int m, int n,
                        INDArray a, int lda, INDArray r,
                        INDArray c, INDArray rowcnd, INDArray colcnd,
                        INDArray amax );
    int cgeequb( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                        INDArray amax );
    int zgeequb( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        INDArray r, INDArray c, INDArray rowcnd,
                        INDArray colcnd, INDArray amax );

    int sgees( int matriint, char jobvs, char sort,
                      int select, int n, INDArray a,
                      int lda, int sdim, INDArray wr,
                      INDArray wi, INDArray vs, int ldvs );
    int dgees( int matriint, char jobvs, char sort,
                      int select, int n, INDArray a,
                      int lda, int sdim, INDArray wr,
                      INDArray wi, INDArray vs, int ldvs );
    int cgees( int matriint, char jobvs, char sort,
                      int select, int n,
                      IComplexNumber a, int lda,
                      int sdim, IComplexNumber w,
                      IComplexNumber vs, int ldvs );
    int zgees( int matriint, char jobvs, char sort,
                      int select, int n,
                      IComplexNumber a, int lda,
                      int sdim, IComplexNumber w,
                      IComplexNumber vs, int ldvs );

    int sgeesx( int matriint, char jobvs, char sort,
                       int select, char sense, int n,
                       INDArray a, int lda, int sdim,
                       INDArray wr, INDArray wi, INDArray vs, int ldvs,
                       INDArray rconde, INDArray rcondv );
    int dgeesx( int matriint, char jobvs, char sort,
                       int select, char sense, int n,
                       INDArray a, int lda, int sdim,
                       INDArray wr, INDArray wi, INDArray vs, int ldvs,
                       INDArray rconde, INDArray rcondv );
    int cgeesx( int matriint, char jobvs, char sort,
                       int select, char sense, int n,
                       IComplexNumber a, int lda,
                       int sdim, IComplexNumber w,
                       IComplexNumber vs, int ldvs,
                       INDArray rconde, INDArray rcondv );
    int zgeesx( int matriint, char jobvs, char sort,
                       int select, char sense, int n,
                       IComplexNumber a, int lda,
                       int sdim, IComplexNumber w,
                       IComplexNumber vs, int ldvs,
                       INDArray rconde, INDArray rcondv );

    int sgeev( int matriint, char jobvl, char jobvr,
                      int n, INDArray a, int lda, INDArray wr,
                      INDArray wi, INDArray vl, int ldvl, INDArray vr,
                      int ldvr );
    int dgeev( int matriint, char jobvl, char jobvr,
                      int n, INDArray a, int lda, INDArray wr,
                      INDArray wi, INDArray vl, int ldvl, INDArray vr,
                      int ldvr );
    int cgeev( int matriint, char jobvl, char jobvr,
                      int n, IComplexNumber a, int lda,
                      IComplexNumber w, IComplexNumber vl,
                      int ldvl, IComplexNumber vr,
                      int ldvr );
    int zgeev( int matriint, char jobvl, char jobvr,
                      int n, IComplexNumber a,
                      int lda, IComplexNumber w,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr );

    int sgeevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n, INDArray a,
                       int lda, INDArray wr, INDArray wi, INDArray vl,
                       int ldvl, INDArray vr, int ldvr,
                       int ilo, int ihi, INDArray scale,
                       INDArray abnrm, INDArray rconde, INDArray rcondv );
    int dgeevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n, INDArray a,
                       int lda, INDArray wr, INDArray wi, INDArray vl,
                       int ldvl, INDArray vr, int ldvr,
                       int ilo, int ihi, INDArray scale,
                       INDArray abnrm, INDArray rconde, INDArray rcondv );
    int cgeevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber w, IComplexNumber vl,
                       int ldvl, IComplexNumber vr,
                       int ldvr, int ilo, int ihi,
                       INDArray scale, INDArray abnrm, INDArray rconde,
                       INDArray rcondv );
    int zgeevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber w, IComplexNumber vl,
                       int ldvl, IComplexNumber vr,
                       int ldvr, int ilo, int ihi,
                       INDArray scale, INDArray abnrm, INDArray rconde,
                       INDArray rcondv );

    int sgehrd( int matriint, int n, int ilo,
                       int ihi, INDArray a, int lda,
                       INDArray tau );
    int dgehrd( int matriint, int n, int ilo,
                       int ihi, INDArray a, int lda,
                       INDArray tau );
    int cgehrd( int matriint, int n, int ilo,
                       int ihi, IComplexNumber a,
                       int lda, IComplexNumber tau );
    int zgehrd( int matriint, int n, int ilo,
                       int ihi, IComplexNumber a,
                       int lda, IComplexNumber tau );

    int sgejsv( int matriint, char joba, char jobu, char jobv,
                       char jobr, char jobt, char jobp, int m,
                       int n, INDArray a, int lda, INDArray sva,
                       INDArray u, int ldu, INDArray v, int ldv,
                       INDArray stat, int istat );
    int dgejsv( int matriint, char joba, char jobu, char jobv,
                       char jobr, char jobt, char jobp, int m,
                       int n, INDArray a, int lda, INDArray sva,
                       INDArray u, int ldu, INDArray v, int ldv,
                       INDArray stat, int istat );
    int cgejsv( int matriint, char joba, char jobu, char jobv,
                       char jobr, char jobt, char jobp, int m,
                       int n, IComplexNumber a, int lda, INDArray sva,
                       IComplexNumber u, int ldu, IComplexNumber v, int ldv,
                       INDArray stat, int istat );
    int zgejsv( int matriint, char joba, char jobu, char jobv,
                       char jobr, char jobt, char jobp, int m,
                       int n, IComplexNumber a, int lda, INDArray sva,
                       IComplexNumber u, int ldu, IComplexNumber v, int ldv,
                       INDArray stat, int istat );

    int sgelq2( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dgelq2( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int cgelq2( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zgelq2( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int sgelqf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dgelqf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int cgelqf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zgelqf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int sgels( int matriint, char trans, int m,
                      int n, int nrhs, INDArray a,
                      int lda, INDArray b, int ldb );
    int dgels( int matriint, char trans, int m,
                      int n, int nrhs, INDArray a,
                      int lda, INDArray b, int ldb );
    int cgels( int matriint, char trans, int m,
                      int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb );
    int zgels( int matriint, char trans, int m,
                      int n, int nrhs,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb );

    int sgelsd( int matriint, int m, int n,
                       int nrhs, INDArray a, int lda, INDArray b,
                       int ldb, INDArray s, float rcond,
                       int rank );
    int dgelsd( int matriint, int m, int n,
                       int nrhs, INDArray a, int lda,
                       INDArray b, int ldb, INDArray s, double rcond,
                       int rank );
    int cgelsd( int matriint, int m, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, INDArray s, float rcond,
                       int rank );
    int zgelsd( int matriint, int m, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, INDArray s, double rcond,
                       int rank );

    int sgelss( int matriint, int m, int n,
                       int nrhs, INDArray a, int lda, INDArray b,
                       int ldb, INDArray s, float rcond,
                       int rank );
    int dgelss( int matriint, int m, int n,
                       int nrhs, INDArray a, int lda,
                       INDArray b, int ldb, INDArray s, double rcond,
                       int rank );
    int cgelss( int matriint, int m, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, INDArray s, float rcond,
                       int rank );
    int zgelss( int matriint, int m, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, INDArray s, double rcond,
                       int rank );

    int sgelsy( int matriint, int m, int n,
                       int nrhs, INDArray a, int lda, INDArray b,
                       int ldb, int jpvt, float rcond,
                       int rank );
    int dgelsy( int matriint, int m, int n,
                       int nrhs, INDArray a, int lda,
                       INDArray b, int ldb, int jpvt,
                       double rcond, int rank );
    int cgelsy( int matriint, int m, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, int jpvt, float rcond,
                       int rank );
    int zgelsy( int matriint, int m, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, int jpvt, double rcond,
                       int rank );

    int sgeqlf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dgeqlf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int cgeqlf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zgeqlf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int sgeqp3( int matriint, int m, int n,
                       INDArray a, int lda, int jpvt,
                       INDArray tau );
    int dgeqp3( int matriint, int m, int n,
                       INDArray a, int lda, int jpvt,
                       INDArray tau );
    int cgeqp3( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int jpvt, IComplexNumber tau );
    int zgeqp3( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int jpvt, IComplexNumber tau );

    int sgeqpf( int matriint, int m, int n,
                       INDArray a, int lda, int jpvt,
                       INDArray tau );
    int dgeqpf( int matriint, int m, int n,
                       INDArray a, int lda, int jpvt,
                       INDArray tau );
    int cgeqpf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int jpvt, IComplexNumber tau );
    int zgeqpf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int jpvt, IComplexNumber tau );

    int sgeqr2( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dgeqr2( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int cgeqr2( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zgeqr2( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int sgeqrf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dgeqrf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int cgeqrf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zgeqrf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int sgeqrfp( int matriint, int m, int n,
                        INDArray a, int lda, INDArray tau );
    int dgeqrfp( int matriint, int m, int n,
                        INDArray a, int lda, INDArray tau );
    int cgeqrfp( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau );
    int zgeqrfp( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber tau );

    int sgerfs( int matriint, char trans, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int dgerfs( int matriint, char trans, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int cgerfs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zgerfs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int sgerfsx( int matriint, char trans, char equed,
                        int n, int nrhs,  INDArray a,
                        int lda,  INDArray af, int ldaf,
                        int ipiv,  INDArray r,
                        INDArray c,  INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dgerfsx( int matriint, char trans, char equed,
                        int n, int nrhs,  INDArray a,
                        int lda,  INDArray af, int ldaf,
                        int ipiv,  INDArray r,
                        INDArray c,  INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int cgerfsx( int matriint, char trans, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv,  INDArray r,
                        INDArray c,  IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int zgerfsx( int matriint, char trans, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv,  INDArray r,
                        INDArray c,  IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );

    int sgerqf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dgerqf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int cgerqf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zgerqf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int sgesdd( int matriint, char jobz, int m,
                       int n, INDArray a, int lda, INDArray s,
                       INDArray u, int ldu, INDArray vt,
                       int ldvt );
    int dgesdd( int matriint, char jobz, int m,
                       int n, INDArray a, int lda, INDArray s,
                       INDArray u, int ldu, INDArray vt,
                       int ldvt );
    int cgesdd( int matriint, char jobz, int m,
                       int n, IComplexNumber a,
                       int lda, INDArray s, IComplexNumber u,
                       int ldu, IComplexNumber vt,
                       int ldvt );
    int zgesdd( int matriint, char jobz, int m,
                       int n, IComplexNumber a,
                       int lda, INDArray s, IComplexNumber u,
                       int ldu, IComplexNumber vt,
                       int ldvt );

    int sgesv( int matriint, int n, int nrhs,
                      INDArray a, int lda, int ipiv, INDArray b,
                      int ldb );
    int dgesv( int matriint, int n, int nrhs,
                      INDArray a, int lda, int ipiv,
                      INDArray b, int ldb );
    int cgesv( int matriint, int n, int nrhs,
                      IComplexNumber a, int lda,
                      int ipiv, IComplexNumber b,
                      int ldb );
    int zgesv( int matriint, int n, int nrhs,
                      IComplexNumber a, int lda,
                      int ipiv, IComplexNumber b,
                      int ldb );
    int dsgesv( int matriint, int n, int nrhs,
                       INDArray a, int lda, int ipiv,
                       INDArray b, int ldb, INDArray x, int ldx,
                       int iter );
    int zcgesv( int matriint, int n, int nrhs,
                       IComplexNumber a, int lda,
                       int ipiv, IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, int iter );

    int sgesvd( int matriint, char jobu, char jobvt,
                       int m, int n, INDArray a, int lda,
                       INDArray s, INDArray u, int ldu, INDArray vt,
                       int ldvt, INDArray superb );
    int dgesvd( int matriint, char jobu, char jobvt,
                       int m, int n, INDArray a,
                       int lda, INDArray s, INDArray u, int ldu,
                       INDArray vt, int ldvt, INDArray superb );
    int cgesvd( int matriint, char jobu, char jobvt,
                       int m, int n, IComplexNumber a,
                       int lda, INDArray s, IComplexNumber u,
                       int ldu, IComplexNumber vt,
                       int ldvt, INDArray superb );
    int zgesvd( int matriint, char jobu, char jobvt,
                       int m, int n, IComplexNumber a,
                       int lda, INDArray s, IComplexNumber u,
                       int ldu, IComplexNumber vt,
                       int ldvt, INDArray superb );

    int sgesvdx( int matriint, char jobu, char jobvt, char range,
                        int m, int n, INDArray a,
                        int lda, int vl, int vu,
                        int il, int iu, int ns,
                        INDArray s, INDArray u, int ldu,
                        INDArray vt, int ldvt,
                        int superb );
    int dgesvdx( int matriint, char jobu, char jobvt, char range,
                        int m, int n, INDArray a,
                        int lda, int vl, int vu,
                        int il, int iu, int ns,
                        INDArray s, INDArray u, int ldu,
                        INDArray vt, int ldvt,
                        int superb );
    int cgesvdx( int matriint, char jobu, char jobvt, char range,
                        int m, int n, IComplexNumber a,
                        int lda, int vl, int vu,
                        int il, int iu, int ns,
                        INDArray s, IComplexNumber u, int ldu,
                        IComplexNumber vt, int ldvt,
                        int superb );
    int zgesvdx( int matriint, char jobu, char jobvt, char range,
                        int m, int n, IComplexNumber a,
                        int lda, int vl, int vu,
                        int il, int iu, int ns,
                        INDArray s, IComplexNumber u, int ldu,
                        IComplexNumber vt, int ldvt,
                        int superb );

    int sgesvj( int matriint, char joba, char jobu, char jobv,
                       int m, int n, INDArray a, int lda,
                       INDArray sva, int mv, INDArray v, int ldv,
                       INDArray stat );
    int dgesvj( int matriint, char joba, char jobu, char jobv,
                       int m, int n, INDArray a,
                       int lda, INDArray sva, int mv,
                       INDArray v, int ldv, INDArray stat );
    int cgesvj( int matriint, char joba, char jobu, char jobv,
                       int m, int n, IComplexNumber a,
                       int lda, INDArray sva, int mv,
                       IComplexNumber v, int ldv, INDArray stat );
    int zgesvj( int matriint, char joba, char jobu, char jobv,
                       int m, int n, IComplexNumber a,
                       int lda, INDArray sva, int mv,
                       IComplexNumber v, int ldv, INDArray stat );

    int sgesvx( int matriint, char fact, char trans,
                       int n, int nrhs, INDArray a,
                       int lda, INDArray af, int ldaf,
                       int ipiv, String equed, INDArray r, INDArray c,
                       INDArray b, int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr,
                       INDArray rpivot );
    int dgesvx( int matriint, char fact, char trans,
                       int n, int nrhs, INDArray a,
                       int lda, INDArray af, int ldaf,
                       int ipiv, String equed, INDArray r, INDArray c,
                       INDArray b, int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr,
                       INDArray rpivot );
    int cgesvx( int matriint, char fact, char trans,
                       int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,
                       int ipiv, String equed, INDArray r, INDArray c,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr,
                       INDArray rpivot );
    int zgesvx( int matriint, char fact, char trans,
                       int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber af, int ldaf,
                       int ipiv, String equed, INDArray r, INDArray c,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr,
                       INDArray rpivot );

    int sgesvxx( int matriint, char fact, char trans,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray af, int ldaf,
                        int ipiv, String equed, INDArray r, INDArray c,
                        INDArray b, int ldb, INDArray x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dgesvxx( int matriint, char fact, char trans,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray af, int ldaf,
                        int ipiv, String equed, INDArray r, INDArray c,
                        INDArray b, int ldb, INDArray x,
                        int ldx, INDArray rcond, INDArray rpvgrw,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int cgesvxx( int matriint, char fact, char trans,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv, String equed, INDArray r, INDArray c,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int zgesvxx( int matriint, char fact, char trans,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv, String equed, INDArray r, INDArray c,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );

    int sgetf2( int matriint, int m, int n,
                       INDArray a, int lda, int ipiv );
    int dgetf2( int matriint, int m, int n,
                       INDArray a, int lda, int ipiv );
    int cgetf2( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zgetf2( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int sgetrf( int matriint, int m, int n,
                       INDArray a, int lda, int ipiv );
    int dgetrf( int matriint, int m, int n,
                       INDArray a, int lda, int ipiv );
    int cgetrf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zgetrf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int sgetrf2( int matriint, int m, int n,
                        INDArray a, int lda, int ipiv );
    int dgetrf2( int matriint, int m, int n,
                        INDArray a, int lda, int ipiv );
    int cgetrf2( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        int ipiv );
    int zgetrf2( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        int ipiv );

    int sgetri( int matriint, int n, INDArray a,
                       int lda,  int ipiv );
    int dgetri( int matriint, int n, INDArray a,
                       int lda,  int ipiv );
    int cgetri( int matriint, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zgetri( int matriint, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int sgetrs( int matriint, char trans, int n,
                       int nrhs,  INDArray a, int lda,
                       int ipiv, INDArray b, int ldb );
    int dgetrs( int matriint, char trans, int n,
                       int nrhs,  INDArray a, int lda,
                       int ipiv, INDArray b, int ldb );
    int cgetrs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  int ipiv,
                       IComplexNumber b, int ldb );
    int zgetrs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  int ipiv,
                       IComplexNumber b, int ldb );

    int sggbak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray lscale,
                       INDArray rscale, int m, INDArray v,
                       int ldv );
    int dggbak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray lscale,
                       INDArray rscale, int m, INDArray v,
                       int ldv );
    int cggbak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray lscale,
                       INDArray rscale, int m,
                       IComplexNumber v, int ldv );
    int zggbak( int matriint, char job, char side, int n,
                       int ilo, int ihi,  INDArray lscale,
                       INDArray rscale, int m,
                       IComplexNumber v, int ldv );

    int sggbal( int matriint, char job, int n, INDArray a,
                       int lda, INDArray b, int ldb,
                       int ilo, int ihi, INDArray lscale,
                       INDArray rscale );
    int dggbal( int matriint, char job, int n, INDArray a,
                       int lda, INDArray b, int ldb,
                       int ilo, int ihi, INDArray lscale,
                       INDArray rscale );
    int cggbal( int matriint, char job, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       int ilo, int ihi, INDArray lscale,
                       INDArray rscale );
    int zggbal( int matriint, char job, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       int ilo, int ihi, INDArray lscale,
                       INDArray rscale );

    int sgges( int matriint, char jobvsl, char jobvsr, char sort,
                      int selctg, int n, INDArray a,
                      int lda, INDArray b, int ldb,
                      int sdim, INDArray alphar, INDArray alphai,
                      INDArray beta, INDArray vsl, int ldvsl, INDArray vsr,
                      int ldvsr );
    int dgges( int matriint, char jobvsl, char jobvsr, char sort,
                      int selctg, int n, INDArray a,
                      int lda, INDArray b, int ldb,
                      int sdim, INDArray alphar, INDArray alphai,
                      INDArray beta, INDArray vsl, int ldvsl,
                      INDArray vsr, int ldvsr );
    int cgges( int matriint, char jobvsl, char jobvsr, char sort,
                      int selctg, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      int sdim, IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber vsl,
                      int ldvsl, IComplexNumber vsr,
                      int ldvsr );
    int zgges( int matriint, char jobvsl, char jobvsr, char sort,
                      int selctg, int n,
                      IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      int sdim, IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vsl, int ldvsl,
                      IComplexNumber vsr, int ldvsr );

    int sgges3( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       INDArray a, int lda, INDArray b, int ldb,
                       int sdim, INDArray alphar, INDArray alphai,
                       INDArray beta, INDArray vsl, int ldvsl,
                       INDArray vsr, int ldvsr );
    int dgges3( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       INDArray a, int lda, INDArray b, int ldb,
                       int sdim, INDArray alphar, INDArray alphai,
                       INDArray beta, INDArray vsl, int ldvsl,
                       INDArray vsr, int ldvsr );
    int cgges3( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       int sdim, IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vsl, int ldvsl,
                       IComplexNumber vsr, int ldvsr );
    int zgges3( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       int sdim, IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vsl, int ldvsl,
                       IComplexNumber vsr, int ldvsr );

    int sggesx( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, char sense,
                       int n, INDArray a, int lda, INDArray b,
                       int ldb, int sdim, INDArray alphar,
                       INDArray alphai, INDArray beta, INDArray vsl,
                       int ldvsl, INDArray vsr, int ldvsr,
                       INDArray rconde, INDArray rcondv );
    int dggesx( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, char sense,
                       int n, INDArray a, int lda, INDArray b,
                       int ldb, int sdim, INDArray alphar,
                       INDArray alphai, INDArray beta, INDArray vsl,
                       int ldvsl, INDArray vsr, int ldvsr,
                       INDArray rconde, INDArray rcondv );
    int cggesx( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, char sense,
                       int n, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, int sdim,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vsl, int ldvsl,
                       IComplexNumber vsr, int ldvsr,
                       INDArray rconde, INDArray rcondv );
    int zggesx( int matriint, char jobvsl, char jobvsr,
                       char sort, int selctg, char sense,
                       int n, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, int sdim,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vsl, int ldvsl,
                       IComplexNumber vsr, int ldvsr,
                       INDArray rconde, INDArray rcondv );

    int sggev( int matriint, char jobvl, char jobvr,
                      int n, INDArray a, int lda, INDArray b,
                      int ldb, INDArray alphar, INDArray alphai,
                      INDArray beta, INDArray vl, int ldvl, INDArray vr,
                      int ldvr );
    int dggev( int matriint, char jobvl, char jobvr,
                      int n, INDArray a, int lda, INDArray b,
                      int ldb, INDArray alphar, INDArray alphai,
                      INDArray beta, INDArray vl, int ldvl, INDArray vr,
                      int ldvr );
    int cggev( int matriint, char jobvl, char jobvr,
                      int n, IComplexNumber a, int lda,
                      IComplexNumber b, int ldb,
                      IComplexNumber alpha,
                      IComplexNumber beta, IComplexNumber vl,
                      int ldvl, IComplexNumber vr,
                      int ldvr );
    int zggev( int matriint, char jobvl, char jobvr,
                      int n, IComplexNumber a,
                      int lda, IComplexNumber b,
                      int ldb, IComplexNumber alpha,
                      IComplexNumber beta,
                      IComplexNumber vl, int ldvl,
                      IComplexNumber vr, int ldvr );

    int sggev3( int matriint, char jobvl, char jobvr,
                       int n, INDArray a, int lda,
                       INDArray b, int ldb,
                       INDArray alphar, INDArray alphai, INDArray beta,
                       INDArray vl, int ldvl,
                       INDArray vr, int ldvr );
    int dggev3( int matriint, char jobvl, char jobvr,
                       int n, INDArray a, int lda,
                       INDArray b, int ldb,
                       INDArray alphar, INDArray alphai, INDArray beta,
                       INDArray vl, int ldvl,
                       INDArray vr, int ldvr );
    int cggev3( int matriint, char jobvl, char jobvr,
                       int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr );
    int zggev3( int matriint, char jobvl, char jobvr,
                       int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr );

    int sggevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n, INDArray a,
                       int lda, INDArray b, int ldb,
                       INDArray alphar, INDArray alphai, INDArray beta, INDArray vl,
                       int ldvl, INDArray vr, int ldvr,
                       int ilo, int ihi, INDArray lscale,
                       INDArray rscale, INDArray abnrm, INDArray bbnrm,
                       INDArray rconde, INDArray rcondv );
    int dggevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n, INDArray a,
                       int lda, INDArray b, int ldb,
                       INDArray alphar, INDArray alphai, INDArray beta,
                       INDArray vl, int ldvl, INDArray vr,
                       int ldvr, int ilo, int ihi,
                       INDArray lscale, INDArray rscale, INDArray abnrm,
                       INDArray bbnrm, INDArray rconde, INDArray rcondv );
    int cggevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber alpha,
                       IComplexNumber beta, IComplexNumber vl,
                       int ldvl, IComplexNumber vr,
                       int ldvr, int ilo, int ihi,
                       INDArray lscale, INDArray rscale, INDArray abnrm,
                       INDArray bbnrm, INDArray rconde, INDArray rcondv );
    int zggevx( int matriint, char balanc, char jobvl,
                       char jobvr, char sense, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       int ilo, int ihi, INDArray lscale,
                       INDArray rscale, INDArray abnrm, INDArray bbnrm,
                       INDArray rconde, INDArray rcondv );

    int sggglm( int matriint, int n, int m,
                       int p, INDArray a, int lda, INDArray b,
                       int ldb, INDArray d, INDArray x, INDArray y );
    int dggglm( int matriint, int n, int m,
                       int p, INDArray a, int lda, INDArray b,
                       int ldb, INDArray d, INDArray x, INDArray y );
    int cggglm( int matriint, int n, int m,
                       int p, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, IComplexNumber d,
                       IComplexNumber x, IComplexNumber y );
    int zggglm( int matriint, int n, int m,
                       int p, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, IComplexNumber d,
                       IComplexNumber x, IComplexNumber y );

    int sgghrd( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       INDArray a, int lda, INDArray b, int ldb,
                       INDArray q, int ldq, INDArray z, int ldz );
    int dgghrd( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       INDArray a, int lda, INDArray b, int ldb,
                       INDArray q, int ldq, INDArray z,
                       int ldz );
    int cgghrd( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz );
    int zgghrd( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz );

    int sgghd3( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       INDArray a, int lda, INDArray b, int ldb,
                       INDArray q, int ldq, INDArray z, int ldz );
    int dgghd3( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       INDArray a, int lda, INDArray b, int ldb,
                       INDArray q, int ldq, INDArray z,
                       int ldz );
    int cgghd3( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz );
    int zgghd3( int matriint, char compq, char compz,
                       int n, int ilo, int ihi,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz );

    int sgglse( int matriint, int m, int n,
                       int p, INDArray a, int lda, INDArray b,
                       int ldb, INDArray c, INDArray d, INDArray x );
    int dgglse( int matriint, int m, int n,
                       int p, INDArray a, int lda, INDArray b,
                       int ldb, INDArray c, INDArray d, INDArray x );
    int cgglse( int matriint, int m, int n,
                       int p, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, IComplexNumber c,
                       IComplexNumber d, IComplexNumber x );
    int zgglse( int matriint, int m, int n,
                       int p, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, IComplexNumber c,
                       IComplexNumber d, IComplexNumber x );

    int sggqrf( int matriint, int n, int m,
                       int p, INDArray a, int lda, INDArray taua,
                       INDArray b, int ldb, INDArray taub );
    int dggqrf( int matriint, int n, int m,
                       int p, INDArray a, int lda,
                       INDArray taua, INDArray b, int ldb,
                       INDArray taub );
    int cggqrf( int matriint, int n, int m,
                       int p, IComplexNumber a,
                       int lda, IComplexNumber taua,
                       IComplexNumber b, int ldb,
                       IComplexNumber taub );
    int zggqrf( int matriint, int n, int m,
                       int p, IComplexNumber a,
                       int lda, IComplexNumber taua,
                       IComplexNumber b, int ldb,
                       IComplexNumber taub );

    int sggrqf( int matriint, int m, int p,
                       int n, INDArray a, int lda, INDArray taua,
                       INDArray b, int ldb, INDArray taub );
    int dggrqf( int matriint, int m, int p,
                       int n, INDArray a, int lda,
                       INDArray taua, INDArray b, int ldb,
                       INDArray taub );
    int cggrqf( int matriint, int m, int p,
                       int n, IComplexNumber a,
                       int lda, IComplexNumber taua,
                       IComplexNumber b, int ldb,
                       IComplexNumber taub );
    int zggrqf( int matriint, int m, int p,
                       int n, IComplexNumber a,
                       int lda, IComplexNumber taua,
                       IComplexNumber b, int ldb,
                       IComplexNumber taub );

    int sggsvd( int matriint, char jobu, char jobv, char jobq,
                       int m, int n, int p,
                       int k, int l, INDArray a,
                       int lda, INDArray b, int ldb,
                       INDArray alpha, INDArray beta, INDArray u, int ldu,
                       INDArray v, int ldv, INDArray q, int ldq,
                       int iwork );
    int dggsvd( int matriint, char jobu, char jobv, char jobq,
                       int m, int n, int p,
                       int k, int l, INDArray a,
                       int lda, INDArray b, int ldb,
                       INDArray alpha, INDArray beta, INDArray u,
                       int ldu, INDArray v, int ldv, INDArray q,
                       int ldq, int iwork );
    int cggsvd( int matriint, char jobu, char jobv, char jobq,
                       int m, int n, int p,
                       int k, int l,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       INDArray alpha, INDArray beta, IComplexNumber u,
                       int ldu, IComplexNumber v,
                       int ldv, IComplexNumber q,
                       int ldq, int iwork );
    int zggsvd( int matriint, char jobu, char jobv, char jobq,
                       int m, int n, int p,
                       int k, int l,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       INDArray alpha, INDArray beta,
                       IComplexNumber u, int ldu,
                       IComplexNumber v, int ldv,
                       IComplexNumber q, int ldq,
                       int iwork );

    int sggsvd3( int matriint, char jobu, char jobv, char jobq,
                        int m, int n, int p,
                        int k, int l, INDArray a,
                        int lda, INDArray b, int ldb,
                        INDArray alpha, INDArray beta, INDArray u, int ldu,
                        INDArray v, int ldv, INDArray q, int ldq,
                        int iwork );
    int dggsvd3( int matriint, char jobu, char jobv, char jobq,
                        int m, int n, int p,
                        int k, int l, INDArray a,
                        int lda, INDArray b, int ldb,
                        INDArray alpha, INDArray beta, INDArray u,
                        int ldu, INDArray v, int ldv, INDArray q,
                        int ldq, int iwork );
    int cggsvd3( int matriint, char jobu, char jobv, char jobq,
                        int m, int n, int p,
                        int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        INDArray alpha, INDArray beta, IComplexNumber u,
                        int ldu, IComplexNumber v,
                        int ldv, IComplexNumber q,
                        int ldq, int iwork );
    int zggsvd3( int matriint, char jobu, char jobv, char jobq,
                        int m, int n, int p,
                        int k, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        INDArray alpha, INDArray beta,
                        IComplexNumber u, int ldu,
                        IComplexNumber v, int ldv,
                        IComplexNumber q, int ldq,
                        int iwork );

    int sggsvp( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n, INDArray a,
                       int lda, INDArray b, int ldb, float tola,
                       float tolb, int k, int l, INDArray u,
                       int ldu, INDArray v, int ldv, INDArray q,
                       int ldq );
    int dggsvp( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n, INDArray a,
                       int lda, INDArray b, int ldb,
                       double tola, double tolb, int k,
                       int l, INDArray u, int ldu, INDArray v,
                       int ldv, INDArray q, int ldq );
    int cggsvp( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb, float tola,
                       float tolb, int k, int l,
                       IComplexNumber u, int ldu,
                       IComplexNumber v, int ldv,
                       IComplexNumber q, int ldq );
    int zggsvp( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       double tola, double tolb, int k,
                       int l, IComplexNumber u,
                       int ldu, IComplexNumber v,
                       int ldv, IComplexNumber q,
                       int ldq );

    int sggsvp3( int matriint, char jobu, char jobv, char jobq,
                        int m, int p, int n, INDArray a,
                        int lda, INDArray b, int ldb, float tola,
                        float tolb, int k, int l, INDArray u,
                        int ldu, INDArray v, int ldv, INDArray q,
                        int ldq );
    int dggsvp3( int matriint, char jobu, char jobv, char jobq,
                        int m, int p, int n, INDArray a,
                        int lda, INDArray b, int ldb,
                        double tola, double tolb, int k,
                        int l, INDArray u, int ldu, INDArray v,
                        int ldv, INDArray q, int ldq );
    int cggsvp3( int matriint, char jobu, char jobv, char jobq,
                        int m, int p, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb, float tola,
                        float tolb, int k, int l,
                        IComplexNumber u, int ldu,
                        IComplexNumber v, int ldv,
                        IComplexNumber q, int ldq );
    int zggsvp3( int matriint, char jobu, char jobv, char jobq,
                        int m, int p, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        double tola, double tolb, int k,
                        int l, IComplexNumber u,
                        int ldu, IComplexNumber v,
                        int ldv, IComplexNumber q,
                        int ldq );

    int sgtcon( char norm, int n,  INDArray dl,
                       INDArray d,  INDArray du,  INDArray du2,
                       int ipiv, float anorm, INDArray rcond );
    int dgtcon( char norm, int n,  INDArray dl,
                       INDArray d,  INDArray du,  INDArray du2,
                       int ipiv, double anorm,
                       INDArray rcond );
    int cgtcon( char norm, int n,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber du2,
                       int ipiv, float anorm, INDArray rcond );
    int zgtcon( char norm, int n,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber du2,
                       int ipiv, double anorm,
                       INDArray rcond );

    int sgtrfs( int matriint, char trans, int n,
                       int nrhs,  INDArray dl,  INDArray d,
                       INDArray du,  INDArray dlf,  INDArray df,
                       INDArray duf,  INDArray du2,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int dgtrfs( int matriint, char trans, int n,
                       int nrhs,  INDArray dl,  INDArray d,
                       INDArray du,  INDArray dlf,
                       INDArray df,  INDArray duf,
                       INDArray du2,  int ipiv,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int cgtrfs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber dlf,
                       IComplexNumber df,
                       IComplexNumber duf,
                       IComplexNumber du2,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zgtrfs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber dlf,
                       IComplexNumber df,
                       IComplexNumber duf,
                       IComplexNumber du2,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int sgtsv( int matriint, int n, int nrhs,
                      INDArray dl, INDArray d, INDArray du, INDArray b,
                      int ldb );
    int dgtsv( int matriint, int n, int nrhs,
                      INDArray dl, INDArray d, INDArray du, INDArray b,
                      int ldb );
    int cgtsv( int matriint, int n, int nrhs,
                      IComplexNumber dl, IComplexNumber d,
                      IComplexNumber du, IComplexNumber b,
                      int ldb );
    int zgtsv( int matriint, int n, int nrhs,
                      IComplexNumber dl, IComplexNumber d,
                      IComplexNumber du, IComplexNumber b,
                      int ldb );

    int sgtsvx( int matriint, char fact, char trans,
                       int n, int nrhs,  INDArray dl,
                       INDArray d,  INDArray du, INDArray dlf,
                       INDArray df, INDArray duf, INDArray du2, int ipiv,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int dgtsvx( int matriint, char fact, char trans,
                       int n, int nrhs,  INDArray dl,
                       INDArray d,  INDArray du, INDArray dlf,
                       INDArray df, INDArray duf, INDArray du2,
                       int ipiv,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond,
                       INDArray ferr, INDArray berr );
    int cgtsvx( int matriint, char fact, char trans,
                       int n, int nrhs,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber dlf, IComplexNumber df,
                       IComplexNumber duf, IComplexNumber du2,
                       int ipiv,  IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int zgtsvx( int matriint, char fact, char trans,
                       int n, int nrhs,
                       IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber dlf,
                       IComplexNumber df,
                       IComplexNumber duf,
                       IComplexNumber du2, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int sgttrf( int n, INDArray dl, INDArray d, INDArray du,
                       INDArray du2, int ipiv );
    int dgttrf( int n, INDArray dl, INDArray d, INDArray du,
                       INDArray du2, int ipiv );
    int cgttrf( int n, IComplexNumber dl,
                       IComplexNumber d, IComplexNumber du,
                       IComplexNumber du2, int ipiv );
    int zgttrf( int n, IComplexNumber dl,
                       IComplexNumber d, IComplexNumber du,
                       IComplexNumber du2, int ipiv );

    int sgttrs( int matriint, char trans, int n,
                       int nrhs,  INDArray dl,  INDArray d,
                       INDArray du,  INDArray du2,
                       int ipiv, INDArray b, int ldb );
    int dgttrs( int matriint, char trans, int n,
                       int nrhs,  INDArray dl,  INDArray d,
                       INDArray du,  INDArray du2,
                       int ipiv, INDArray b, int ldb );
    int cgttrs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber du2,
                       int ipiv, IComplexNumber b,
                       int ldb );
    int zgttrs( int matriint, char trans, int n,
                       int nrhs,  IComplexNumber dl,
                       IComplexNumber d,
                       IComplexNumber du,
                       IComplexNumber du2,
                       int ipiv, IComplexNumber b,
                       int ldb );

    int chbev( int matriint, char jobz, char uplo, int n,
                      int kd, IComplexNumber ab,
                      int ldab, INDArray w, IComplexNumber z,
                      int ldz );
    int zhbev( int matriint, char jobz, char uplo, int n,
                      int kd, IComplexNumber ab,
                      int ldab, INDArray w, IComplexNumber z,
                      int ldz );

    int chbevd( int matriint, char jobz, char uplo, int n,
                       int kd, IComplexNumber ab,
                       int ldab, INDArray w, IComplexNumber z,
                       int ldz );
    int zhbevd( int matriint, char jobz, char uplo, int n,
                       int kd, IComplexNumber ab,
                       int ldab, INDArray w, IComplexNumber z,
                       int ldz );

    int chbevx( int matriint, char jobz, char range, char uplo,
                       int n, int kd,
                       IComplexNumber ab, int ldab,
                       IComplexNumber q, int ldq, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, IComplexNumber z,
                       int ldz, int ifail );
    int zhbevx( int matriint, char jobz, char range, char uplo,
                       int n, int kd,
                       IComplexNumber ab, int ldab,
                       IComplexNumber q, int ldq, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );

    int chbgst( int matriint, char vect, char uplo, int n,
                       int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       IComplexNumber x, int ldx );
    int zhbgst( int matriint, char vect, char uplo, int n,
                       int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       IComplexNumber x, int ldx );

    int chbgv( int matriint, char jobz, char uplo, int n,
                      int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb, INDArray w,
                      IComplexNumber z, int ldz );
    int zhbgv( int matriint, char jobz, char uplo, int n,
                      int ka, int kb,
                      IComplexNumber ab, int ldab,
                      IComplexNumber bb, int ldbb, INDArray w,
                      IComplexNumber z, int ldz );

    int chbgvd( int matriint, char jobz, char uplo, int n,
                       int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb, INDArray w,
                       IComplexNumber z, int ldz );
    int zhbgvd( int matriint, char jobz, char uplo, int n,
                       int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       INDArray w, IComplexNumber z,
                       int ldz );

    int chbgvx( int matriint, char jobz, char range, char uplo,
                       int n, int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       IComplexNumber q, int ldq, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, IComplexNumber z,
                       int ldz, int ifail );
    int zhbgvx( int matriint, char jobz, char range, char uplo,
                       int n, int ka, int kb,
                       IComplexNumber ab, int ldab,
                       IComplexNumber bb, int ldbb,
                       IComplexNumber q, int ldq, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );

    int chbtrd( int matriint, char vect, char uplo, int n,
                       int kd, IComplexNumber ab,
                       int ldab, INDArray d, INDArray e,
                       IComplexNumber q, int ldq );
    int zhbtrd( int matriint, char vect, char uplo, int n,
                       int kd, IComplexNumber ab,
                       int ldab, INDArray d, INDArray e,
                       IComplexNumber q, int ldq );

    int checon( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv, float anorm, INDArray rcond );
    int zhecon( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv, double anorm,
                       INDArray rcond );

    int cheequb( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        INDArray s, INDArray scond, INDArray amax );
    int zheequb( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        INDArray s, INDArray scond, INDArray amax );

    int cheev( int matriint, char jobz, char uplo, int n,
                      IComplexNumber a, int lda, INDArray w );
    int zheev( int matriint, char jobz, char uplo, int n,
                      IComplexNumber a, int lda, INDArray w );

    int cheevd( int matriint, char jobz, char uplo, int n,
                       IComplexNumber a, int lda, INDArray w );
    int zheevd( int matriint, char jobz, char uplo, int n,
                       IComplexNumber a, int lda,
                       INDArray w );

    int cheevr( int matriint, char jobz, char range, char uplo,
                       int n, IComplexNumber a,
                       int lda, float vl, float vu, int il,
                       int iu, float abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int isuppz );
    int zheevr( int matriint, char jobz, char range, char uplo,
                       int n, IComplexNumber a,
                       int lda, double vl, double vu, int il,
                       int iu, double abstol, int m,
                       INDArray w, IComplexNumber z, int ldz,
                       int isuppz );

    int cheevx( int matriint, char jobz, char range, char uplo,
                       int n, IComplexNumber a,
                       int lda, float vl, float vu, int il,
                       int iu, float abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );
    int zheevx( int matriint, char jobz, char range, char uplo,
                       int n, IComplexNumber a,
                       int lda, double vl, double vu, int il,
                       int iu, double abstol, int m,
                       INDArray w, IComplexNumber z, int ldz,
                       int ifail );

    int chegst( int matriint, int itype, char uplo,
                       int n, IComplexNumber a,
                       int lda,  IComplexNumber b,
                       int ldb );
    int zhegst( int matriint, int itype, char uplo,
                       int n, IComplexNumber a,
                       int lda,  IComplexNumber b,
                       int ldb );

    int chegv( int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber a,
                      int lda, IComplexNumber b,
                      int ldb, INDArray w );
    int zhegv( int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber a,
                      int lda, IComplexNumber b,
                      int ldb, INDArray w );

    int chegvd( int matriint, int itype, char jobz,
                       char uplo, int n, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, INDArray w );
    int zhegvd( int matriint, int itype, char jobz,
                       char uplo, int n, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, INDArray w );

    int chegvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, IComplexNumber z,
                       int ldz, int ifail );
    int zhegvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );

    int cherfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zherfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int cherfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv,  INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int zherfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv,  INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );

    int chesv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber a,
                      int lda, int ipiv,
                      IComplexNumber b, int ldb );
    int zhesv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber a,
                      int lda, int ipiv,
                      IComplexNumber b, int ldb );

    int chesvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda, IComplexNumber af,
                       int ldaf, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zhesvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda, IComplexNumber af,
                       int ldaf, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int chesvxx( int matriint, char fact, char uplo,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv, String equed, INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int zhesvxx( int matriint, char fact, char uplo,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv, String equed, INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );

    int chetrd( int matriint, char uplo, int n,
                       IComplexNumber a, int lda, INDArray d,
                       INDArray e, IComplexNumber tau );
    int zhetrd( int matriint, char uplo, int n,
                       IComplexNumber a, int lda, INDArray d,
                       INDArray e, IComplexNumber tau );

    int chetrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zhetrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int chetri( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zhetri( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int chetrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  int ipiv,
                       IComplexNumber b, int ldb );
    int zhetrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  int ipiv,
                       IComplexNumber b, int ldb );

    int chfrk( int matriint, char transr, char uplo, char trans,
                      int n, int k, float alpha,
                      IComplexNumber a, int lda,
                      float beta, IComplexNumber c );
    int zhfrk( int matriint, char transr, char uplo, char trans,
                      int n, int k, double alpha,
                      IComplexNumber a, int lda,
                      double beta, IComplexNumber c );

    int shgeqz( int matriint, char job, char compq, char compz,
                       int n, int ilo, int ihi,
                       INDArray h, int ldh, INDArray t, int ldt,
                       INDArray alphar, INDArray alphai, INDArray beta, INDArray q,
                       int ldq, INDArray z, int ldz );
    int dhgeqz( int matriint, char job, char compq, char compz,
                       int n, int ilo, int ihi,
                       INDArray h, int ldh, INDArray t, int ldt,
                       INDArray alphar, INDArray alphai, INDArray beta,
                       INDArray q, int ldq, INDArray z,
                       int ldz );
    int chgeqz( int matriint, char job, char compq, char compz,
                       int n, int ilo, int ihi,
                       IComplexNumber h, int ldh,
                       IComplexNumber t, int ldt,
                       IComplexNumber alpha,
                       IComplexNumber beta, IComplexNumber q,
                       int ldq, IComplexNumber z,
                       int ldz );
    int zhgeqz( int matriint, char job, char compq, char compz,
                       int n, int ilo, int ihi,
                       IComplexNumber h, int ldh,
                       IComplexNumber t, int ldt,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz );

    int chpcon( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       int ipiv, float anorm, INDArray rcond );
    int zhpcon( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       int ipiv, double anorm,
                       INDArray rcond );

    int chpev( int matriint, char jobz, char uplo, int n,
                      IComplexNumber ap, INDArray w,
                      IComplexNumber z, int ldz );
    int zhpev( int matriint, char jobz, char uplo, int n,
                      IComplexNumber ap, INDArray w,
                      IComplexNumber z, int ldz );

    int chpevd( int matriint, char jobz, char uplo, int n,
                       IComplexNumber ap, INDArray w,
                       IComplexNumber z, int ldz );
    int zhpevd( int matriint, char jobz, char uplo, int n,
                       IComplexNumber ap, INDArray w,
                       IComplexNumber z, int ldz );

    int chpevx( int matriint, char jobz, char range, char uplo,
                       int n, IComplexNumber ap, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, IComplexNumber z,
                       int ldz, int ifail );
    int zhpevx( int matriint, char jobz, char range, char uplo,
                       int n, IComplexNumber ap, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );

    int chpgst( int matriint, int itype, char uplo,
                       int n, IComplexNumber ap,
                       IComplexNumber bp );
    int zhpgst( int matriint, int itype, char uplo,
                       int n, IComplexNumber ap,
                       IComplexNumber bp );

    int chpgv( int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber ap,
                      IComplexNumber bp, INDArray w,
                      IComplexNumber z, int ldz );
    int zhpgv( int matriint, int itype, char jobz,
                      char uplo, int n, IComplexNumber ap,
                      IComplexNumber bp, INDArray w,
                      IComplexNumber z, int ldz );

    int chpgvd( int matriint, int itype, char jobz,
                       char uplo, int n, IComplexNumber ap,
                       IComplexNumber bp, INDArray w,
                       IComplexNumber z, int ldz );
    int zhpgvd( int matriint, int itype, char jobz,
                       char uplo, int n, IComplexNumber ap,
                       IComplexNumber bp, INDArray w,
                       IComplexNumber z, int ldz );

    int chpgvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n,
                       IComplexNumber ap, IComplexNumber bp,
                       float vl, float vu, int il, int iu,
                       float abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );
    int zhpgvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n,
                       IComplexNumber ap, IComplexNumber bp,
                       double vl, double vu, int il, int iu,
                       double abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int ifail );

    int chprfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zhprfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int chpsv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber ap,
                      int ipiv, IComplexNumber b,
                      int ldb );
    int zhpsv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber ap,
                      int ipiv, IComplexNumber b,
                      int ldb );

    int chpsvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zhpsvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int chptrd( int matriint, char uplo, int n,
                       IComplexNumber ap, INDArray d, INDArray e,
                       IComplexNumber tau );
    int zhptrd( int matriint, char uplo, int n,
                       IComplexNumber ap, INDArray d, INDArray e,
                       IComplexNumber tau );

    int chptrf( int matriint, char uplo, int n,
                       IComplexNumber ap, int ipiv );
    int zhptrf( int matriint, char uplo, int n,
                       IComplexNumber ap, int ipiv );

    int chptri( int matriint, char uplo, int n,
                       IComplexNumber ap,  int ipiv );
    int zhptri( int matriint, char uplo, int n,
                       IComplexNumber ap,  int ipiv );

    int chptrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       int ipiv, IComplexNumber b,
                       int ldb );
    int zhptrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       int ipiv, IComplexNumber b,
                       int ldb );

    int shsein( int matriint, char job, char eigsrc, char initv,
                       int select, int n,  INDArray h,
                       int ldh, INDArray wr,  INDArray wi,
                       INDArray vl, int ldvl, INDArray vr,
                       int ldvr, int mm, int m,
                       int ifaill, int ifailr );
    int dhsein( int matriint, char job, char eigsrc, char initv,
                       int select, int n,
                       INDArray h, int ldh, INDArray wr,
                       INDArray wi, INDArray vl, int ldvl,
                       INDArray vr, int ldvr, int mm,
                       int m, int ifaill,
                       int ifailr );
    int chsein( int matriint, char job, char eigsrc, char initv,
                       int select, int n,
                       IComplexNumber h, int ldh,
                       IComplexNumber w, IComplexNumber vl,
                       int ldvl, IComplexNumber vr,
                       int ldvr, int mm, int m,
                       int ifaill, int ifailr );
    int zhsein( int matriint, char job, char eigsrc, char initv,
                       int select, int n,
                       IComplexNumber h, int ldh,
                       IComplexNumber w, IComplexNumber vl,
                       int ldvl, IComplexNumber vr,
                       int ldvr, int mm, int m,
                       int ifaill, int ifailr );

    int shseqr( int matriint, char job, char compz, int n,
                       int ilo, int ihi, INDArray h,
                       int ldh, INDArray wr, INDArray wi, INDArray z,
                       int ldz );
    int dhseqr( int matriint, char job, char compz, int n,
                       int ilo, int ihi, INDArray h,
                       int ldh, INDArray wr, INDArray wi, INDArray z,
                       int ldz );
    int chseqr( int matriint, char job, char compz, int n,
                       int ilo, int ihi,
                       IComplexNumber h, int ldh,
                       IComplexNumber w, IComplexNumber z,
                       int ldz );
    int zhseqr( int matriint, char job, char compz, int n,
                       int ilo, int ihi,
                       IComplexNumber h, int ldh,
                       IComplexNumber w, IComplexNumber z,
                       int ldz );


    int slacpy( int matriint, char uplo, int m,
                       int n,  INDArray a, int lda, INDArray b,
                       int ldb );
    int dlacpy( int matriint, char uplo, int m,
                       int n,  INDArray a, int lda, INDArray b,
                       int ldb );
    int clacpy( int matriint, char uplo, int m,
                       int n,  IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb );
    int zlacpy( int matriint, char uplo, int m,
                       int n,  IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb );

    int clacp2( int matriint, char uplo, int m,
                       int n,  INDArray a, int lda,
                       IComplexNumber b, int ldb );
    int zlacp2( int matriint, char uplo, int m,
                       int n,  INDArray a, int lda,
                       IComplexNumber b, int ldb );

    int zlag2c( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber sa, int ldsa );

    int slag2d( int matriint, int m, int n,
                       INDArray sa, int ldsa, INDArray a,
                       int lda );

    int dlag2s( int matriint, int m, int n,
                       INDArray a, int lda, INDArray sa,
                       int ldsa );

    int clag2z( int matriint, int m, int n,
                       IComplexNumber sa, int ldsa,
                       IComplexNumber a, int lda );

    int slagge( int matriint, int m, int n,
                       int kl, int ku,  INDArray d,
                       INDArray a, int lda, int iseed );
    int dlagge( int matriint, int m, int n,
                       int kl, int ku,  INDArray d,
                       INDArray a, int lda, int iseed );
    int clagge( int matriint, int m, int n,
                       int kl, int ku,  INDArray d,
                       IComplexNumber a, int lda,
                       int iseed );
    int zlagge( int matriint, int m, int n,
                       int kl, int ku,  INDArray d,
                       IComplexNumber a, int lda,
                       int iseed );

    float slamch( char cmach );
    double dlamch( char cmach );

    float slange( int matriint, char norm, int m,
                  int n,  INDArray a, int lda );
    double dlange( int matriint, char norm, int m,
                   int n,  INDArray a, int lda );
    float clange( int matriint, char norm, int m,
                  int n,  IComplexNumber a,
                  int lda );
    double zlange( int matriint, char norm, int m,
                   int n,  IComplexNumber a,
                   int lda );

    float clanhe( int matriint, char norm, char uplo, int n,
                  IComplexNumber a, int lda );
    double zlanhe( int matriint, char norm, char uplo, int n,
                   IComplexNumber a, int lda );

    float slansy( int matriint, char norm, char uplo, int n,
                  INDArray a, int lda );
    double dlansy( int matriint, char norm, char uplo, int n,
                   INDArray a, int lda );
    float clansy( int matriint, char norm, char uplo, int n,
                  IComplexNumber a, int lda );
    double zlansy( int matriint, char norm, char uplo, int n,
                   IComplexNumber a, int lda );

    float slantr( int matriint, char norm, char uplo, char diag,
                  int m, int n,  INDArray a,
                  int lda );
    double dlantr( int matriint, char norm, char uplo, char diag,
                   int m, int n,  INDArray a,
                   int lda );
    float clantr( int matriint, char norm, char uplo, char diag,
                  int m, int n,  IComplexNumber a,
                  int lda );
    double zlantr( int matriint, char norm, char uplo, char diag,
                   int m, int n,  IComplexNumber a,
                   int lda );


    int slarfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k,  INDArray v, int ldv,
                       INDArray t, int ldt, INDArray c,
                       int ldc );
    int dlarfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k,  INDArray v, int ldv,
                       INDArray t, int ldt, INDArray c,
                       int ldc );
    int clarfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k,  IComplexNumber v,
                       int ldv,  IComplexNumber t,
                       int ldt, IComplexNumber c,
                       int ldc );
    int zlarfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k,  IComplexNumber v,
                       int ldv,  IComplexNumber t,
                       int ldt, IComplexNumber c,
                       int ldc );



    int slarft( int matriint, char direct, char storev,
                       int n, int k,  INDArray v,
                       int ldv,  INDArray tau, INDArray t,
                       int ldt );
    int dlarft( int matriint, char direct, char storev,
                       int n, int k,  INDArray v,
                       int ldv,  INDArray tau, INDArray t,
                       int ldt );
    int clarft( int matriint, char direct, char storev,
                       int n, int k,
                       IComplexNumber v, int ldv,
                       IComplexNumber tau,
                       IComplexNumber t, int ldt );
    int zlarft( int matriint, char direct, char storev,
                       int n, int k,
                       IComplexNumber v, int ldv,
                       IComplexNumber tau,
                       IComplexNumber t, int ldt );

    int slarfx( int matriint, char side, int m,
                       int n,  INDArray v, float tau, INDArray c,
                       int ldc, INDArray work );
    int dlarfx( int matriint, char side, int m,
                       int n,  INDArray v, double tau, INDArray c,
                       int ldc, INDArray work );
    int clarfx( int matriint, char side, int m,
                       int n,  IComplexNumber v,
                       IComplexNumber tau, IComplexNumber c,
                       int ldc, IComplexNumber work );
    int zlarfx( int matriint, char side, int m,
                       int n,  IComplexNumber v,
                       IComplexNumber tau, IComplexNumber c,
                       int ldc, IComplexNumber work );



    int slascl( int matriint, char type, int kl,
                       int ku, float cfrom, float cto,
                       int m, int n, INDArray a,
                       int lda );
    int dlascl( int matriint, char type, int kl,
                       int ku, double cfrom, double cto,
                       int m, int n, INDArray a,
                       int lda );
    int clascl( int matriint, char type, int kl,
                       int ku, float cfrom, float cto,
                       int m, int n, IComplexNumber a,
                       int lda );
    int zlascl( int matriint, char type, int kl,
                       int ku, double cfrom, double cto,
                       int m, int n, IComplexNumber a,
                       int lda );

    int slaset( int matriint, char uplo, int m,
                       int n, float alpha, float beta, INDArray a,
                       int lda );
    int dlaset( int matriint, char uplo, int m,
                       int n, double alpha, double beta, INDArray a,
                       int lda );
    int claset( int matriint, char uplo, int m,
                       int n, IComplexNumber alpha,
                       IComplexNumber beta, IComplexNumber a,
                       int lda );
    int zlaset( int matriint, char uplo, int m,
                       int n, IComplexNumber alpha,
                       IComplexNumber beta, IComplexNumber a,
                       int lda );

    int slasrt( char id, int n, INDArray d );
    int dlasrt( char id, int n, INDArray d );

    int slaswp( int matriint, int n, INDArray a,
                       int lda, int k1, int k2,
                       int ipiv, int incx );
    int dlaswp( int matriint, int n, INDArray a,
                       int lda, int k1, int k2,
                       int ipiv, int incx );
    int claswp( int matriint, int n,
                       IComplexNumber a, int lda,
                       int k1, int k2,  int ipiv,
                       int incx );
    int zlaswp( int matriint, int n,
                       IComplexNumber a, int lda,
                       int k1, int k2,  int ipiv,
                       int incx );

    int slatms( int matriint, int m, int n,
                       char dist, int iseed, char sym, INDArray d,
                       int mode, float cond, float dmax,
                       int kl, int ku, char pack, INDArray a,
                       int lda );
    int dlatms( int matriint, int m, int n,
                       char dist, int iseed, char sym, INDArray d,
                       int mode, double cond, double dmax,
                       int kl, int ku, char pack, INDArray a,
                       int lda );
    int clatms( int matriint, int m, int n,
                       char dist, int iseed, char sym, INDArray d,
                       int mode, float cond, float dmax,
                       int kl, int ku, char pack,
                       IComplexNumber a, int lda );
    int zlatms( int matriint, int m, int n,
                       char dist, int iseed, char sym, INDArray d,
                       int mode, double cond, double dmax,
                       int kl, int ku, char pack,
                       IComplexNumber a, int lda );

    int slauum( int matriint, char uplo, int n, INDArray a,
                       int lda );
    int dlauum( int matriint, char uplo, int n, INDArray a,
                       int lda );
    int clauum( int matriint, char uplo, int n,
                       IComplexNumber a, int lda );
    int zlauum( int matriint, char uplo, int n,
                       IComplexNumber a, int lda );

    int sopgtr( int matriint, char uplo, int n,
                       INDArray ap,  INDArray tau, INDArray q,
                       int ldq );
    int dopgtr( int matriint, char uplo, int n,
                       INDArray ap,  INDArray tau, INDArray q,
                       int ldq );

    int sopmtr( int matriint, char side, char uplo, char trans,
                       int m, int n,  INDArray ap,
                       INDArray tau, INDArray c, int ldc );
    int dopmtr( int matriint, char side, char uplo, char trans,
                       int m, int n,  INDArray ap,
                       INDArray tau, INDArray c, int ldc );

    int sorgbr( int matriint, char vect, int m,
                       int n, int k, INDArray a, int lda,
                       INDArray tau );
    int dorgbr( int matriint, char vect, int m,
                       int n, int k, INDArray a,
                       int lda,  INDArray tau );

    int sorghr( int matriint, int n, int ilo,
                       int ihi, INDArray a, int lda,
                       INDArray tau );
    int dorghr( int matriint, int n, int ilo,
                       int ihi, INDArray a, int lda,
                       INDArray tau );

    int sorglq( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );
    int dorglq( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );

    int sorgql( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );
    int dorgql( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );

    int sorgqr( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );
    int dorgqr( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );

    int sorgrq( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );
    int dorgrq( int matriint, int m, int n,
                       int k, INDArray a, int lda,
                       INDArray tau );

    int sorgtr( int matriint, char uplo, int n, INDArray a,
                       int lda,  INDArray tau );
    int dorgtr( int matriint, char uplo, int n, INDArray a,
                       int lda,  INDArray tau );

    int sormbr( int matriint, char vect, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );
    int dormbr( int matriint, char vect, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );

    int sormhr( int matriint, char side, char trans,
                       int m, int n, int ilo,
                       int ihi,  INDArray a, int lda,
                       INDArray tau, INDArray c, int ldc );
    int dormhr( int matriint, char side, char trans,
                       int m, int n, int ilo,
                       int ihi,  INDArray a, int lda,
                       INDArray tau, INDArray c, int ldc );

    int sormlq( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );
    int dormlq( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );

    int sormql( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );
    int dormql( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );

    int sormqr( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );
    int dormqr( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );

    int sormrq( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );
    int dormrq( int matriint, char side, char trans,
                       int m, int n, int k,
                       INDArray a, int lda,  INDArray tau,
                       INDArray c, int ldc );

    int sormrz( int matriint, char side, char trans,
                       int m, int n, int k,
                       int l,  INDArray a, int lda,
                       INDArray tau, INDArray c, int ldc );
    int dormrz( int matriint, char side, char trans,
                       int m, int n, int k,
                       int l,  INDArray a, int lda,
                       INDArray tau, INDArray c, int ldc );

    int sormtr( int matriint, char side, char uplo, char trans,
                       int m, int n,  INDArray a,
                       int lda,  INDArray tau, INDArray c,
                       int ldc );
    int dormtr( int matriint, char side, char uplo, char trans,
                       int m, int n,  INDArray a,
                       int lda,  INDArray tau, INDArray c,
                       int ldc );

    int spbcon( int matriint, char uplo, int n,
                       int kd,  INDArray ab, int ldab,
                       float anorm, INDArray rcond );
    int dpbcon( int matriint, char uplo, int n,
                       int kd,  INDArray ab, int ldab,
                       double anorm, INDArray rcond );
    int cpbcon( int matriint, char uplo, int n,
                       int kd,  IComplexNumber ab,
                       int ldab, float anorm, INDArray rcond );
    int zpbcon( int matriint, char uplo, int n,
                       int kd,  IComplexNumber ab,
                       int ldab, double anorm, INDArray rcond );

    int spbequ( int matriint, char uplo, int n,
                       int kd,  INDArray ab, int ldab,
                       INDArray s, INDArray scond, INDArray amax );
    int dpbequ( int matriint, char uplo, int n,
                       int kd,  INDArray ab, int ldab,
                       INDArray s, INDArray scond, INDArray amax );
    int cpbequ( int matriint, char uplo, int n,
                       int kd,  IComplexNumber ab,
                       int ldab, INDArray s, INDArray scond,
                       INDArray amax );
    int zpbequ( int matriint, char uplo, int n,
                       int kd,  IComplexNumber ab,
                       int ldab, INDArray s, INDArray scond,
                       INDArray amax );

    int spbrfs( int matriint, char uplo, int n,
                       int kd, int nrhs,  INDArray ab,
                       int ldab,  INDArray afb, int ldafb,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int dpbrfs( int matriint, char uplo, int n,
                       int kd, int nrhs,  INDArray ab,
                       int ldab,  INDArray afb, int ldafb,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int cpbrfs( int matriint, char uplo, int n,
                       int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber afb, int ldafb,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zpbrfs( int matriint, char uplo, int n,
                       int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber afb, int ldafb,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int spbstf( int matriint, char uplo, int n,
                       int kb, INDArray bb, int ldbb );
    int dpbstf( int matriint, char uplo, int n,
                       int kb, INDArray bb, int ldbb );
    int cpbstf( int matriint, char uplo, int n,
                       int kb, IComplexNumber bb,
                       int ldbb );
    int zpbstf( int matriint, char uplo, int n,
                       int kb, IComplexNumber bb,
                       int ldbb );

    int spbsv( int matriint, char uplo, int n,
                      int kd, int nrhs, INDArray ab,
                      int ldab, INDArray b, int ldb );
    int dpbsv( int matriint, char uplo, int n,
                      int kd, int nrhs, INDArray ab,
                      int ldab, INDArray b, int ldb );
    int cpbsv( int matriint, char uplo, int n,
                      int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb );
    int zpbsv( int matriint, char uplo, int n,
                      int kd, int nrhs,
                      IComplexNumber ab, int ldab,
                      IComplexNumber b, int ldb );

    int spbsvx( int matriint, char fact, char uplo, int n,
                       int kd, int nrhs, INDArray ab,
                       int ldab, INDArray afb, int ldafb,
                       String equed, INDArray s, INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int dpbsvx( int matriint, char fact, char uplo, int n,
                       int kd, int nrhs, INDArray ab,
                       int ldab, INDArray afb, int ldafb,
                       String equed, INDArray s, INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond,
                       INDArray ferr, INDArray berr );
    int cpbsvx( int matriint, char fact, char uplo, int n,
                       int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber afb, int ldafb,
                       String equed, INDArray s, IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int zpbsvx( int matriint, char fact, char uplo, int n,
                       int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber afb, int ldafb,
                       String equed, INDArray s, IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );

    int spbtrf( int matriint, char uplo, int n,
                       int kd, INDArray ab, int ldab );
    int dpbtrf( int matriint, char uplo, int n,
                       int kd, INDArray ab, int ldab );
    int cpbtrf( int matriint, char uplo, int n,
                       int kd, IComplexNumber ab,
                       int ldab );
    int zpbtrf( int matriint, char uplo, int n,
                       int kd, IComplexNumber ab,
                       int ldab );

    int spbtrs( int matriint, char uplo, int n,
                       int kd, int nrhs,  INDArray ab,
                       int ldab, INDArray b, int ldb );
    int dpbtrs( int matriint, char uplo, int n,
                       int kd, int nrhs,  INDArray ab,
                       int ldab, INDArray b, int ldb );
    int cpbtrs( int matriint, char uplo, int n,
                       int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb );
    int zpbtrs( int matriint, char uplo, int n,
                       int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb );

    int spftrf( int matriint, char transr, char uplo,
                       int n, INDArray a );
    int dpftrf( int matriint, char transr, char uplo,
                       int n, INDArray a );
    int cpftrf( int matriint, char transr, char uplo,
                       int n, IComplexNumber a );
    int zpftrf( int matriint, char transr, char uplo,
                       int n, IComplexNumber a );

    int spftri( int matriint, char transr, char uplo,
                       int n, INDArray a );
    int dpftri( int matriint, char transr, char uplo,
                       int n, INDArray a );
    int cpftri( int matriint, char transr, char uplo,
                       int n, IComplexNumber a );
    int zpftri( int matriint, char transr, char uplo,
                       int n, IComplexNumber a );

    int spftrs( int matriint, char transr, char uplo,
                       int n, int nrhs,  INDArray a,
                       INDArray b, int ldb );
    int dpftrs( int matriint, char transr, char uplo,
                       int n, int nrhs,  INDArray a,
                       INDArray b, int ldb );
    int cpftrs( int matriint, char transr, char uplo,
                       int n, int nrhs,
                       IComplexNumber a,
                       IComplexNumber b, int ldb );
    int zpftrs( int matriint, char transr, char uplo,
                       int n, int nrhs,
                       IComplexNumber a,
                       IComplexNumber b, int ldb );

    int spocon( int matriint, char uplo, int n,
                       INDArray a, int lda, float anorm,
                       INDArray rcond );
    int dpocon( int matriint, char uplo, int n,
                       INDArray a, int lda, double anorm,
                       INDArray rcond );
    int cpocon( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       float anorm, INDArray rcond );
    int zpocon( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       double anorm, INDArray rcond );

    int spoequ( int matriint, int n,  INDArray a,
                       int lda, INDArray s, INDArray scond,
                       INDArray amax );
    int dpoequ( int matriint, int n,  INDArray a,
                       int lda, INDArray s, INDArray scond,
                       INDArray amax );
    int cpoequ( int matriint, int n,
                       IComplexNumber a, int lda,
                       INDArray s, INDArray scond, INDArray amax );
    int zpoequ( int matriint, int n,
                       IComplexNumber a, int lda,
                       INDArray s, INDArray scond, INDArray amax );

    int spoequb( int matriint, int n,  INDArray a,
                        int lda, INDArray s, INDArray scond,
                        INDArray amax );
    int dpoequb( int matriint, int n,  INDArray a,
                        int lda, INDArray s, INDArray scond,
                        INDArray amax );
    int cpoequb( int matriint, int n,
                        IComplexNumber a, int lda,
                        INDArray s, INDArray scond, INDArray amax );
    int zpoequb( int matriint, int n,
                        IComplexNumber a, int lda,
                        INDArray s, INDArray scond, INDArray amax );

    int sporfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int dporfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int cporfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, INDArray ferr, INDArray berr );
    int zporfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, INDArray ferr, INDArray berr );

    int sporfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,  INDArray a,
                        int lda,  INDArray af, int ldaf,
                        INDArray s,  INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dporfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,  INDArray a,
                        int lda,  INDArray af, int ldaf,
                        INDArray s,  INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int cporfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        INDArray s,  IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int zporfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        INDArray s,  IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );

    int sposv( int matriint, char uplo, int n,
                      int nrhs, INDArray a, int lda, INDArray b,
                      int ldb );
    int dposv( int matriint, char uplo, int n,
                      int nrhs, INDArray a, int lda, INDArray b,
                      int ldb );
    int cposv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber a,
                      int lda, IComplexNumber b,
                      int ldb );
    int zposv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber a,
                      int lda, IComplexNumber b,
                      int ldb );
    int dsposv( int matriint, char uplo, int n,
                       int nrhs, INDArray a, int lda,
                       INDArray b, int ldb, INDArray x, int ldx,
                       int iter );
    int zcposv( int matriint, char uplo, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, IComplexNumber x,
                       int ldx, int iter );

    int sposvx( int matriint, char fact, char uplo, int n,
                       int nrhs, INDArray a, int lda, INDArray af,
                       int ldaf, String equed, INDArray s, INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int dposvx( int matriint, char fact, char uplo, int n,
                       int nrhs, INDArray a, int lda,
                       INDArray af, int ldaf, String equed, INDArray s,
                       INDArray b, int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int cposvx( int matriint, char fact, char uplo, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber af,
                       int ldaf, String equed, INDArray s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zposvx( int matriint, char fact, char uplo, int n,
                       int nrhs, IComplexNumber a,
                       int lda, IComplexNumber af,
                       int ldaf, String equed, INDArray s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int sposvxx( int matriint, char fact, char uplo,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray af, int ldaf,
                        String equed, INDArray s, INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond,
                        INDArray rpvgrw, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int dposvxx( int matriint, char fact, char uplo,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray af, int ldaf,
                        String equed, INDArray s, INDArray b, int ldb,
                        INDArray x, int ldx, INDArray rcond,
                        INDArray rpvgrw, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int cposvxx( int matriint, char fact, char uplo,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        String equed, INDArray s, IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray rpvgrw,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int zposvxx( int matriint, char fact, char uplo,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        String equed, INDArray s, IComplexNumber b,
                        int ldb, IComplexNumber x,
                        int ldx, INDArray rcond, INDArray rpvgrw,
                        INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );

    int spotrf2( int matriint, char uplo, int n, INDArray a,
                        int lda );
    int dpotrf2( int matriint, char uplo, int n, INDArray a,
                        int lda );
    int cpotrf2( int matriint, char uplo, int n,
                        IComplexNumber a, int lda );
    int zpotrf2( int matriint, char uplo, int n,
                        IComplexNumber a, int lda );

    int spotrf( int matriint, char uplo, int n, INDArray a,
                       int lda );
    int dpotrf( int matriint, char uplo, int n, INDArray a,
                       int lda );
    int cpotrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda );
    int zpotrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda );

    int spotri( int matriint, char uplo, int n, INDArray a,
                       int lda );
    int dpotri( int matriint, char uplo, int n, INDArray a,
                       int lda );
    int cpotri( int matriint, char uplo, int n,
                       IComplexNumber a, int lda );
    int zpotri( int matriint, char uplo, int n,
                       IComplexNumber a, int lda );

    int spotrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray b, int ldb );
    int dpotrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray b, int ldb );
    int cpotrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb );
    int zpotrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb );

    int sppcon( int matriint, char uplo, int n,
                       INDArray ap, float anorm, INDArray rcond );
    int dppcon( int matriint, char uplo, int n,
                       INDArray ap, double anorm, INDArray rcond );
    int cppcon( int matriint, char uplo, int n,
                       IComplexNumber ap, float anorm,
                       INDArray rcond );
    int zppcon( int matriint, char uplo, int n,
                       IComplexNumber ap, double anorm,
                       INDArray rcond );

    int sppequ( int matriint, char uplo, int n,
                       INDArray ap, INDArray s, INDArray scond,
                       INDArray amax );
    int dppequ( int matriint, char uplo, int n,
                       INDArray ap, INDArray s, INDArray scond,
                       INDArray amax );
    int cppequ( int matriint, char uplo, int n,
                       IComplexNumber ap, INDArray s,
                       INDArray scond, INDArray amax );
    int zppequ( int matriint, char uplo, int n,
                       IComplexNumber ap, INDArray s,
                       INDArray scond, INDArray amax );

    int spprfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap,  INDArray afp,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int dpprfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap,  INDArray afp,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int cpprfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zpprfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int sppsv( int matriint, char uplo, int n,
                      int nrhs, INDArray ap, INDArray b,
                      int ldb );
    int dppsv( int matriint, char uplo, int n,
                      int nrhs, INDArray ap, INDArray b,
                      int ldb );
    int cppsv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber ap,
                      IComplexNumber b, int ldb );
    int zppsv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber ap,
                      IComplexNumber b, int ldb );

    int sppsvx( int matriint, char fact, char uplo, int n,
                       int nrhs, INDArray ap, INDArray afp, String equed,
                       INDArray s, INDArray b, int ldb, INDArray x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int dppsvx( int matriint, char fact, char uplo, int n,
                       int nrhs, INDArray ap, INDArray afp,
                       String equed, INDArray s, INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond,
                       INDArray ferr, INDArray berr );
    int cppsvx( int matriint, char fact, char uplo, int n,
                       int nrhs, IComplexNumber ap,
                       IComplexNumber afp, String equed, INDArray s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zppsvx( int matriint, char fact, char uplo, int n,
                       int nrhs, IComplexNumber ap,
                       IComplexNumber afp, String equed, INDArray s,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int spptrf( int matriint, char uplo, int n,
                       INDArray ap );
    int dpptrf( int matriint, char uplo, int n,
                       INDArray ap );
    int cpptrf( int matriint, char uplo, int n,
                       IComplexNumber ap );
    int zpptrf( int matriint, char uplo, int n,
                       IComplexNumber ap );

    int spptri( int matriint, char uplo, int n,
                       INDArray ap );
    int dpptri( int matriint, char uplo, int n,
                       INDArray ap );
    int cpptri( int matriint, char uplo, int n,
                       IComplexNumber ap );
    int zpptri( int matriint, char uplo, int n,
                       IComplexNumber ap );

    int spptrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap, INDArray b,
                       int ldb );
    int dpptrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap, INDArray b,
                       int ldb );
    int cpptrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber b, int ldb );
    int zpptrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber b, int ldb );

    int spstrf( int matriint, char uplo, int n, INDArray a,
                       int lda, int piv, int rank,
                       float tol );
    int dpstrf( int matriint, char uplo, int n, INDArray a,
                       int lda, int piv, int rank,
                       double tol );
    int cpstrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int piv, int rank, float tol );
    int zpstrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int piv, int rank, double tol );

    int sptcon( int n,  INDArray d,  INDArray e,
                       float anorm, INDArray rcond );
    int dptcon( int n,  INDArray d,  INDArray e,
                       double anorm, INDArray rcond );
    int cptcon( int n,  INDArray d,
                       IComplexNumber e, float anorm,
                       INDArray rcond );
    int zptcon( int n,  INDArray d,
                       IComplexNumber e, double anorm,
                       INDArray rcond );

    int spteqr( int matriint, char compz, int n, INDArray d,
                       INDArray e, INDArray z, int ldz );
    int dpteqr( int matriint, char compz, int n,
                       INDArray d, INDArray e, INDArray z, int ldz );
    int cpteqr( int matriint, char compz, int n, INDArray d,
                       INDArray e, IComplexNumber z, int ldz );
    int zpteqr( int matriint, char compz, int n,
                       INDArray d, INDArray e, IComplexNumber z,
                       int ldz );

    int sptrfs( int matriint, int n, int nrhs,
                       INDArray d,  INDArray e,  INDArray df,
                       INDArray ef,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray ferr, INDArray berr );
    int dptrfs( int matriint, int n, int nrhs,
                       INDArray d,  INDArray e,  INDArray df,
                       INDArray ef,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray ferr,
                       INDArray berr );
    int cptrfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray d,
                       IComplexNumber e,  INDArray df,
                       IComplexNumber ef,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zptrfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray d,
                       IComplexNumber e,  INDArray df,
                       IComplexNumber ef,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int sptsv( int matriint, int n, int nrhs,
                      INDArray d, INDArray e, INDArray b, int ldb );
    int dptsv( int matriint, int n, int nrhs,
                      INDArray d, INDArray e, INDArray b, int ldb );
    int cptsv( int matriint, int n, int nrhs,
                      INDArray d, IComplexNumber e,
                      IComplexNumber b, int ldb );
    int zptsv( int matriint, int n, int nrhs,
                      INDArray d, IComplexNumber e,
                      IComplexNumber b, int ldb );

    int sptsvx( int matriint, char fact, int n,
                       int nrhs,  INDArray d,  INDArray e,
                       INDArray df, INDArray ef,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int dptsvx( int matriint, char fact, int n,
                       int nrhs,  INDArray d,  INDArray e,
                       INDArray df, INDArray ef,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int cptsvx( int matriint, char fact, int n,
                       int nrhs,  INDArray d,
                       IComplexNumber e, INDArray df,
                       IComplexNumber ef,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zptsvx( int matriint, char fact, int n,
                       int nrhs,  INDArray d,
                       IComplexNumber e, INDArray df,
                       IComplexNumber ef,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int spttrf( int n, INDArray d, INDArray e );
    int dpttrf( int n, INDArray d, INDArray e );
    int cpttrf( int n, INDArray d, IComplexNumber e );
    int zpttrf( int n, INDArray d, IComplexNumber e );

    int spttrs( int matriint, int n, int nrhs,
                       INDArray d,  INDArray e, INDArray b,
                       int ldb );
    int dpttrs( int matriint, int n, int nrhs,
                       INDArray d,  INDArray e, INDArray b,
                       int ldb );
    int cpttrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray d,
                       IComplexNumber e,
                       IComplexNumber b, int ldb );
    int zpttrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray d,
                       IComplexNumber e,
                       IComplexNumber b, int ldb );

    int ssbev( int matriint, char jobz, char uplo, int n,
                      int kd, INDArray ab, int ldab, INDArray w,
                      INDArray z, int ldz );
    int dsbev( int matriint, char jobz, char uplo, int n,
                      int kd, INDArray ab, int ldab, INDArray w,
                      INDArray z, int ldz );

    int ssbevd( int matriint, char jobz, char uplo, int n,
                       int kd, INDArray ab, int ldab, INDArray w,
                       INDArray z, int ldz );
    int dsbevd( int matriint, char jobz, char uplo, int n,
                       int kd, INDArray ab, int ldab,
                       INDArray w, INDArray z, int ldz );

    int ssbevx( int matriint, char jobz, char range, char uplo,
                       int n, int kd, INDArray ab,
                       int ldab, INDArray q, int ldq, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );
    int dsbevx( int matriint, char jobz, char range, char uplo,
                       int n, int kd, INDArray ab,
                       int ldab, INDArray q, int ldq,
                       double vl, double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int ifail );

    int ssbgst( int matriint, char vect, char uplo, int n,
                       int ka, int kb, INDArray ab,
                       int ldab,  INDArray bb, int ldbb,
                       INDArray x, int ldx );
    int dsbgst( int matriint, char vect, char uplo, int n,
                       int ka, int kb, INDArray ab,
                       int ldab,  INDArray bb, int ldbb,
                       INDArray x, int ldx );

    int ssbgv( int matriint, char jobz, char uplo, int n,
                      int ka, int kb, INDArray ab,
                      int ldab, INDArray bb, int ldbb, INDArray w,
                      INDArray z, int ldz );
    int dsbgv( int matriint, char jobz, char uplo, int n,
                      int ka, int kb, INDArray ab,
                      int ldab, INDArray bb, int ldbb,
                      INDArray w, INDArray z, int ldz );

    int ssbgvd( int matriint, char jobz, char uplo, int n,
                       int ka, int kb, INDArray ab,
                       int ldab, INDArray bb, int ldbb,
                       INDArray w, INDArray z, int ldz );
    int dsbgvd( int matriint, char jobz, char uplo, int n,
                       int ka, int kb, INDArray ab,
                       int ldab, INDArray bb, int ldbb,
                       INDArray w, INDArray z, int ldz );

    int ssbgvx( int matriint, char jobz, char range, char uplo,
                       int n, int ka, int kb,
                       INDArray ab, int ldab, INDArray bb,
                       int ldbb, INDArray q, int ldq, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );
    int dsbgvx( int matriint, char jobz, char range, char uplo,
                       int n, int ka, int kb,
                       INDArray ab, int ldab, INDArray bb,
                       int ldbb, INDArray q, int ldq,
                       double vl, double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int ifail );

    int ssbtrd( int matriint, char vect, char uplo, int n,
                       int kd, INDArray ab, int ldab, INDArray d,
                       INDArray e, INDArray q, int ldq );
    int dsbtrd( int matriint, char vect, char uplo, int n,
                       int kd, INDArray ab, int ldab,
                       INDArray d, INDArray e, INDArray q, int ldq );

    int ssfrk( int matriint, char transr, char uplo, char trans,
                      int n, int k, float alpha,
                      INDArray a, int lda, float beta,
                      INDArray c );
    int dsfrk( int matriint, char transr, char uplo, char trans,
                      int n, int k, double alpha,
                      INDArray a, int lda, double beta,
                      INDArray c );

    int sspcon( int matriint, char uplo, int n,
                       INDArray ap,  int ipiv, float anorm,
                       INDArray rcond );
    int dspcon( int matriint, char uplo, int n,
                       INDArray ap,  int ipiv,
                       double anorm, INDArray rcond );
    int cspcon( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       int ipiv, float anorm, INDArray rcond );
    int zspcon( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       int ipiv, double anorm,
                       INDArray rcond );

    int sspev( int matriint, char jobz, char uplo, int n,
                      INDArray ap, INDArray w, INDArray z, int ldz );
    int dspev( int matriint, char jobz, char uplo, int n,
                      INDArray ap, INDArray w, INDArray z, int ldz );

    int sspevd( int matriint, char jobz, char uplo, int n,
                       INDArray ap, INDArray w, INDArray z, int ldz );
    int dspevd( int matriint, char jobz, char uplo, int n,
                       INDArray ap, INDArray w, INDArray z, int ldz );

    int sspevx( int matriint, char jobz, char range, char uplo,
                       int n, INDArray ap, float vl, float vu,
                       int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );
    int dspevx( int matriint, char jobz, char range, char uplo,
                       int n, INDArray ap, double vl, double vu,
                       int il, int iu, double abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );

    int sspgst( int matriint, int itype, char uplo,
                       int n, INDArray ap,  INDArray bp );
    int dspgst( int matriint, int itype, char uplo,
                       int n, INDArray ap,  INDArray bp );

    int sspgv( int matriint, int itype, char jobz,
                      char uplo, int n, INDArray ap, INDArray bp,
                      INDArray w, INDArray z, int ldz );
    int dspgv( int matriint, int itype, char jobz,
                      char uplo, int n, INDArray ap, INDArray bp,
                      INDArray w, INDArray z, int ldz );

    int sspgvd( int matriint, int itype, char jobz,
                       char uplo, int n, INDArray ap, INDArray bp,
                       INDArray w, INDArray z, int ldz );
    int dspgvd( int matriint, int itype, char jobz,
                       char uplo, int n, INDArray ap, INDArray bp,
                       INDArray w, INDArray z, int ldz );

    int sspgvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n, INDArray ap,
                       INDArray bp, float vl, float vu, int il,
                       int iu, float abstol, int m, INDArray w,
                       INDArray z, int ldz, int ifail );
    int dspgvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n, INDArray ap,
                       INDArray bp, double vl, double vu, int il,
                       int iu, double abstol, int m,
                       INDArray w, INDArray z, int ldz,
                       int ifail );

    int ssprfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap,  INDArray afp,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int dsprfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap,  INDArray afp,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int csprfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zsprfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp,
                       int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int sspsv( int matriint, char uplo, int n,
                      int nrhs, INDArray ap, int ipiv,
                      INDArray b, int ldb );
    int dspsv( int matriint, char uplo, int n,
                      int nrhs, INDArray ap, int ipiv,
                      INDArray b, int ldb );
    int cspsv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber ap,
                      int ipiv, IComplexNumber b,
                      int ldb );
    int zspsv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber ap,
                      int ipiv, IComplexNumber b,
                      int ldb );

    int sspsvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  INDArray ap, INDArray afp,
                       int ipiv,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int dspsvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  INDArray ap, INDArray afp,
                       int ipiv,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray rcond,
                       INDArray ferr, INDArray berr );
    int cspsvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zspsvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       IComplexNumber afp, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int ssptrd( int matriint, char uplo, int n, INDArray ap,
                       INDArray d, INDArray e, INDArray tau );
    int dsptrd( int matriint, char uplo, int n,
                       INDArray ap, INDArray d, INDArray e, INDArray tau );

    int ssptrf( int matriint, char uplo, int n, INDArray ap,
                       int ipiv );
    int dsptrf( int matriint, char uplo, int n,
                       INDArray ap, int ipiv );
    int csptrf( int matriint, char uplo, int n,
                       IComplexNumber ap, int ipiv );
    int zsptrf( int matriint, char uplo, int n,
                       IComplexNumber ap, int ipiv );

    int ssptri( int matriint, char uplo, int n, INDArray ap,
                       int ipiv );
    int dsptri( int matriint, char uplo, int n,
                       INDArray ap,  int ipiv );
    int csptri( int matriint, char uplo, int n,
                       IComplexNumber ap,  int ipiv );
    int zsptri( int matriint, char uplo, int n,
                       IComplexNumber ap,  int ipiv );

    int ssptrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap,
                       int ipiv, INDArray b, int ldb );
    int dsptrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray ap,
                       int ipiv, INDArray b, int ldb );
    int csptrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       int ipiv, IComplexNumber b,
                       int ldb );
    int zsptrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber ap,
                       int ipiv, IComplexNumber b,
                       int ldb );

    int sstebz( char range, char order, int n, float vl,
                       float vu, int il, int iu, float abstol,
                       INDArray d,  INDArray e, int m,
                       int nsplit, INDArray w, int iblock,
                       int isplit );
    int dstebz( char range, char order, int n, double vl,
                       double vu, int il, int iu,
                       double abstol,  INDArray d,  INDArray e,
                       int m, int nsplit, INDArray w,
                       int iblock, int isplit );

    int sstedc( int matriint, char compz, int n, INDArray d,
                       INDArray e, INDArray z, int ldz );
    int dstedc( int matriint, char compz, int n,
                       INDArray d, INDArray e, INDArray z, int ldz );
    int cstedc( int matriint, char compz, int n, INDArray d,
                       INDArray e, IComplexNumber z, int ldz );
    int zstedc( int matriint, char compz, int n,
                       INDArray d, INDArray e, IComplexNumber z,
                       int ldz );

    int sstegr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, float vl, float vu,
                       int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int isuppz );
    int dstegr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int isuppz );
    int cstegr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, float vl, float vu,
                       int il, int iu, float abstol,
                       int m, INDArray w, IComplexNumber z,
                       int ldz, int isuppz );
    int zstegr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w,
                       IComplexNumber z, int ldz,
                       int isuppz );

    int sstein( int matriint, int n,  INDArray d,
                       INDArray e, int m,  INDArray w,
                       int iblock,  int isplit,
                       INDArray z, int ldz, int ifailv );
    int dstein( int matriint, int n,  INDArray d,
                       INDArray e, int m,  INDArray w,
                       int iblock,  int isplit,
                       INDArray z, int ldz, int ifailv );
    int cstein( int matriint, int n,  INDArray d,
                       INDArray e, int m,  INDArray w,
                       int iblock,  int isplit,
                       IComplexNumber z, int ldz,
                       int ifailv );
    int zstein( int matriint, int n,  INDArray d,
                       INDArray e, int m,  INDArray w,
                       int iblock,  int isplit,
                       IComplexNumber z, int ldz,
                       int ifailv );

    int sstemr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, float vl, float vu,
                       int il, int iu, int m,
                       INDArray w, INDArray z, int ldz, int nzc,
                       int isuppz, int tryrac );
    int dstemr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, double vl,
                       double vu, int il, int iu,
                       int m, INDArray w, INDArray z, int ldz,
                       int nzc, int isuppz,
                       int tryrac );
    int cstemr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, float vl, float vu,
                       int il, int iu, int m,
                       INDArray w, IComplexNumber z, int ldz,
                       int nzc, int isuppz,
                       int tryrac );
    int zstemr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, double vl,
                       double vu, int il, int iu,
                       int m, INDArray w, IComplexNumber z,
                       int ldz, int nzc, int isuppz,
                       int tryrac );

    int ssteqr( int matriint, char compz, int n, INDArray d,
                       INDArray e, INDArray z, int ldz );
    int dsteqr( int matriint, char compz, int n,
                       INDArray d, INDArray e, INDArray z, int ldz );
    int csteqr( int matriint, char compz, int n, INDArray d,
                       INDArray e, IComplexNumber z, int ldz );
    int zsteqr( int matriint, char compz, int n,
                       INDArray d, INDArray e, IComplexNumber z,
                       int ldz );

    int ssterf( int n, INDArray d, INDArray e );
    int dsterf( int n, INDArray d, INDArray e );

    int sstev( int matriint, char jobz, int n, INDArray d,
                      INDArray e, INDArray z, int ldz );
    int dstev( int matriint, char jobz, int n, INDArray d,
                      INDArray e, INDArray z, int ldz );

    int sstevd( int matriint, char jobz, int n, INDArray d,
                       INDArray e, INDArray z, int ldz );
    int dstevd( int matriint, char jobz, int n, INDArray d,
                       INDArray e, INDArray z, int ldz );

    int sstevr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, float vl, float vu,
                       int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int isuppz );
    int dstevr( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int isuppz );

    int sstevx( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, float vl, float vu,
                       int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );
    int dstevx( int matriint, char jobz, char range,
                       int n, INDArray d, INDArray e, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int ifail );

    int ssycon( int matriint, char uplo, int n,
                       INDArray a, int lda,
                       int ipiv, float anorm, INDArray rcond );
    int dsycon( int matriint, char uplo, int n,
                       INDArray a, int lda,
                       int ipiv, double anorm,
                       INDArray rcond );
    int csycon( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv, float anorm, INDArray rcond );
    int zsycon( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv, double anorm,
                       INDArray rcond );

    int ssyequb( int matriint, char uplo, int n,
                        INDArray a, int lda, INDArray s,
                        INDArray scond, INDArray amax );
    int dsyequb( int matriint, char uplo, int n,
                        INDArray a, int lda, INDArray s,
                        INDArray scond, INDArray amax );
    int csyequb( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        INDArray s, INDArray scond, INDArray amax );
    int zsyequb( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        INDArray s, INDArray scond, INDArray amax );

    int ssyev( int matriint, char jobz, char uplo, int n,
                      INDArray a, int lda, INDArray w );
    int dsyev( int matriint, char jobz, char uplo, int n,
                      INDArray a, int lda, INDArray w );

    int ssyevd( int matriint, char jobz, char uplo, int n,
                       INDArray a, int lda, INDArray w );
    int dsyevd( int matriint, char jobz, char uplo, int n,
                       INDArray a, int lda, INDArray w );

    int ssyevr( int matriint, char jobz, char range, char uplo,
                       int n, INDArray a, int lda, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int isuppz );
    int dsyevr( int matriint, char jobz, char range, char uplo,
                       int n, INDArray a, int lda, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int isuppz );

    int ssyevx( int matriint, char jobz, char range, char uplo,
                       int n, INDArray a, int lda, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );
    int dsyevx( int matriint, char jobz, char range, char uplo,
                       int n, INDArray a, int lda, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int ifail );

    int ssygst( int matriint, int itype, char uplo,
                       int n, INDArray a, int lda,
                       INDArray b, int ldb );
    int dsygst( int matriint, int itype, char uplo,
                       int n, INDArray a, int lda,
                       INDArray b, int ldb );

    int ssygv( int matriint, int itype, char jobz,
                      char uplo, int n, INDArray a, int lda,
                      INDArray b, int ldb, INDArray w );
    int dsygv( int matriint, int itype, char jobz,
                      char uplo, int n, INDArray a, int lda,
                      INDArray b, int ldb, INDArray w );

    int ssygvd( int matriint, int itype, char jobz,
                       char uplo, int n, INDArray a, int lda,
                       INDArray b, int ldb, INDArray w );
    int dsygvd( int matriint, int itype, char jobz,
                       char uplo, int n, INDArray a, int lda,
                       INDArray b, int ldb, INDArray w );

    int ssygvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n, INDArray a,
                       int lda, INDArray b, int ldb, float vl,
                       float vu, int il, int iu, float abstol,
                       int m, INDArray w, INDArray z, int ldz,
                       int ifail );
    int dsygvx( int matriint, int itype, char jobz,
                       char range, char uplo, int n, INDArray a,
                       int lda, INDArray b, int ldb, double vl,
                       double vu, int il, int iu,
                       double abstol, int m, INDArray w, INDArray z,
                       int ldz, int ifail );

    int ssyrfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int dsyrfs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf,
                       int ipiv,  INDArray b,
                       int ldb, INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int csyrfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx, INDArray ferr,
                       INDArray berr );
    int zsyrfs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  IComplexNumber af,
                       int ldaf,  int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int ssyrfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,  INDArray a,
                        int lda,  INDArray af, int ldaf,
                        int ipiv,  INDArray s,
                        INDArray b, int ldb, INDArray x,
                        int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dsyrfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,  INDArray a,
                        int lda,  INDArray af, int ldaf,
                        int ipiv,  INDArray s,
                        INDArray b, int ldb, INDArray x,
                        int ldx, INDArray rcond, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int csyrfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv,  INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );
    int zsyrfsx( int matriint, char uplo, char equed,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv,  INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray berr, int int_bnds,
                        INDArray erint_norm, INDArray erint_comp,
                        int nparams, INDArray params );

    int ssysv( int matriint, char uplo, int n,
                      int nrhs, INDArray a, int lda,
                      int ipiv, INDArray b, int ldb );
    int dsysv( int matriint, char uplo, int n,
                      int nrhs, INDArray a, int lda,
                      int ipiv, INDArray b, int ldb );
    int csysv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber a,
                      int lda, int ipiv,
                      IComplexNumber b, int ldb );
    int zsysv( int matriint, char uplo, int n,
                      int nrhs, IComplexNumber a,
                      int lda, int ipiv,
                      IComplexNumber b, int ldb );

    int ssysvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf, int ipiv,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int dsysvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       INDArray af, int ldaf, int ipiv,
                       INDArray b, int ldb, INDArray x,
                       int ldx, INDArray rcond, INDArray ferr,
                       INDArray berr );
    int csysvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda, IComplexNumber af,
                       int ldaf, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );
    int zsysvx( int matriint, char fact, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda, IComplexNumber af,
                       int ldaf, int ipiv,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray rcond, INDArray ferr, INDArray berr );

    int ssysvxx( int matriint, char fact, char uplo,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray af, int ldaf,
                        int ipiv, String equed, INDArray s, INDArray b,
                        int ldb, INDArray x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int dsysvxx( int matriint, char fact, char uplo,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray af, int ldaf,
                        int ipiv, String equed, INDArray s, INDArray b,
                        int ldb, INDArray x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int csysvxx( int matriint, char fact, char uplo,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv, String equed, INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );
    int zsysvxx( int matriint, char fact, char uplo,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber af, int ldaf,
                        int ipiv, String equed, INDArray s,
                        IComplexNumber b, int ldb,
                        IComplexNumber x, int ldx,
                        INDArray rcond, INDArray rpvgrw, INDArray berr,
                        int int_bnds, INDArray erint_norm,
                        INDArray erint_comp, int nparams,
                        INDArray params );

    int ssytrd( int matriint, char uplo, int n, INDArray a,
                       int lda, INDArray d, INDArray e, INDArray tau );
    int dsytrd( int matriint, char uplo, int n, INDArray a,
                       int lda, INDArray d, INDArray e, INDArray tau );

    int ssytrf( int matriint, char uplo, int n, INDArray a,
                       int lda, int ipiv );
    int dsytrf( int matriint, char uplo, int n, INDArray a,
                       int lda, int ipiv );
    int csytrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zsytrf( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int ssytri( int matriint, char uplo, int n, INDArray a,
                       int lda,  int ipiv );
    int dsytri( int matriint, char uplo, int n, INDArray a,
                       int lda,  int ipiv );
    int csytri( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );
    int zsytri( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       int ipiv );

    int ssytrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       int ipiv, INDArray b, int ldb );
    int dsytrs( int matriint, char uplo, int n,
                       int nrhs,  INDArray a, int lda,
                       int ipiv, INDArray b, int ldb );
    int csytrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  int ipiv,
                       IComplexNumber b, int ldb );
    int zsytrs( int matriint, char uplo, int n,
                       int nrhs,  IComplexNumber a,
                       int lda,  int ipiv,
                       IComplexNumber b, int ldb );

    int stbcon( int matriint, char norm, char uplo, char diag,
                       int n, int kd,  INDArray ab,
                       int ldab, INDArray rcond );
    int dtbcon( int matriint, char norm, char uplo, char diag,
                       int n, int kd,  INDArray ab,
                       int ldab, INDArray rcond );
    int ctbcon( int matriint, char norm, char uplo, char diag,
                       int n, int kd,
                       IComplexNumber ab, int ldab,
                       INDArray rcond );
    int ztbcon( int matriint, char norm, char uplo, char diag,
                       int n, int kd,
                       IComplexNumber ab, int ldab,
                       INDArray rcond );

    int stbrfs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       INDArray ab, int ldab,  INDArray b,
                       int ldb,  INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int dtbrfs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       INDArray ab, int ldab,  INDArray b,
                       int ldb,  INDArray x, int ldx,
                       INDArray ferr, INDArray berr );
    int ctbrfs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );
    int ztbrfs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int stbtrs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       INDArray ab, int ldab, INDArray b,
                       int ldb );
    int dtbtrs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       INDArray ab, int ldab, INDArray b,
                       int ldb );
    int ctbtrs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb );
    int ztbtrs( int matriint, char uplo, char trans, char diag,
                       int n, int kd, int nrhs,
                       IComplexNumber ab, int ldab,
                       IComplexNumber b, int ldb );

    int stfsm( int matriint, char transr, char side, char uplo,
                      char trans, char diag, int m, int n,
                      float alpha,  INDArray a, INDArray b,
                      int ldb );
    int dtfsm( int matriint, char transr, char side, char uplo,
                      char trans, char diag, int m, int n,
                      double alpha,  INDArray a, INDArray b,
                      int ldb );
    int ctfsm( int matriint, char transr, char side, char uplo,
                      char trans, char diag, int m, int n,
                      IComplexNumber alpha,
                      IComplexNumber a,
                      IComplexNumber b, int ldb );
    int ztfsm( int matriint, char transr, char side, char uplo,
                      char trans, char diag, int m, int n,
                      IComplexNumber alpha,
                      IComplexNumber a,
                      IComplexNumber b, int ldb );

    int stftri( int matriint, char transr, char uplo, char diag,
                       int n, INDArray a );
    int dtftri( int matriint, char transr, char uplo, char diag,
                       int n, INDArray a );
    int ctftri( int matriint, char transr, char uplo, char diag,
                       int n, IComplexNumber a );
    int ztftri( int matriint, char transr, char uplo, char diag,
                       int n, IComplexNumber a );

    int stfttp( int matriint, char transr, char uplo,
                       int n,  INDArray arf, INDArray ap );
    int dtfttp( int matriint, char transr, char uplo,
                       int n,  INDArray arf, INDArray ap );
    int ctfttp( int matriint, char transr, char uplo,
                       int n,  IComplexNumber arf,
                       IComplexNumber ap );
    int ztfttp( int matriint, char transr, char uplo,
                       int n,  IComplexNumber arf,
                       IComplexNumber ap );

    int stfttr( int matriint, char transr, char uplo,
                       int n,  INDArray arf, INDArray a,
                       int lda );
    int dtfttr( int matriint, char transr, char uplo,
                       int n,  INDArray arf, INDArray a,
                       int lda );
    int ctfttr( int matriint, char transr, char uplo,
                       int n,  IComplexNumber arf,
                       IComplexNumber a, int lda );
    int ztfttr( int matriint, char transr, char uplo,
                       int n,  IComplexNumber arf,
                       IComplexNumber a, int lda );

    int stgevc( int matriint, char side, char howmny,
                       int select, int n,
                       INDArray s, int lds,  INDArray p,
                       int ldp, INDArray vl, int ldvl,
                       INDArray vr, int ldvr, int mm,
                       int m );
    int dtgevc( int matriint, char side, char howmny,
                       int select, int n,
                       INDArray s, int lds,  INDArray p,
                       int ldp, INDArray vl, int ldvl,
                       INDArray vr, int ldvr, int mm,
                       int m );
    int ctgevc( int matriint, char side, char howmny,
                       int select, int n,
                       IComplexNumber s, int lds,
                       IComplexNumber p, int ldp,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       int mm, int m );
    int ztgevc( int matriint, char side, char howmny,
                       int select, int n,
                       IComplexNumber s, int lds,
                       IComplexNumber p, int ldp,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       int mm, int m );

    int stgexc( int matriint, int wantq,
                       int wantz, int n, INDArray a,
                       int lda, INDArray b, int ldb, INDArray q,
                       int ldq, INDArray z, int ldz,
                       int ifst, int ilst );
    int dtgexc( int matriint, int wantq,
                       int wantz, int n, INDArray a,
                       int lda, INDArray b, int ldb, INDArray q,
                       int ldq, INDArray z, int ldz,
                       int ifst, int ilst );
    int ctgexc( int matriint, int wantq,
                       int wantz, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz,
                       int ifst, int ilst );
    int ztgexc( int matriint, int wantq,
                       int wantz, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz,
                       int ifst, int ilst );

    int stgsen( int matriint, int ijob,
                       int wantq, int wantz,
                       int select, int n, INDArray a,
                       int lda, INDArray b, int ldb,
                       INDArray alphar, INDArray alphai, INDArray beta, INDArray q,
                       int ldq, INDArray z, int ldz,
                       int m, INDArray pl, INDArray pr, INDArray dif );
    int dtgsen( int matriint, int ijob,
                       int wantq, int wantz,
                       int select, int n,
                       INDArray a, int lda, INDArray b, int ldb,
                       INDArray alphar, INDArray alphai, INDArray beta,
                       INDArray q, int ldq, INDArray z, int ldz,
                       int m, INDArray pl, INDArray pr, INDArray dif );
    int ctgsen( int matriint, int ijob,
                       int wantq, int wantz,
                       int select, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber alpha,
                       IComplexNumber beta, IComplexNumber q,
                       int ldq, IComplexNumber z,
                       int ldz, int m, INDArray pl, INDArray pr,
                       INDArray dif );
    int ztgsen( int matriint, int ijob,
                       int wantq, int wantz,
                       int select, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber alpha,
                       IComplexNumber beta,
                       IComplexNumber q, int ldq,
                       IComplexNumber z, int ldz,
                       int m, INDArray pl, INDArray pr, INDArray dif );

    int stgsja( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n,
                       int k, int l, INDArray a, int lda,
                       INDArray b, int ldb, float tola, float tolb,
                       INDArray alpha, INDArray beta, INDArray u, int ldu,
                       INDArray v, int ldv, INDArray q, int ldq,
                       int ncycle );
    int dtgsja( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n,
                       int k, int l, INDArray a,
                       int lda, INDArray b, int ldb,
                       double tola, double tolb, INDArray alpha,
                       INDArray beta, INDArray u, int ldu, INDArray v,
                       int ldv, INDArray q, int ldq,
                       int ncycle );
    int ctgsja( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n,
                       int k, int l, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, float tola, float tolb, INDArray alpha,
                       INDArray beta, IComplexNumber u, int ldu,
                       IComplexNumber v, int ldv,
                       IComplexNumber q, int ldq,
                       int ncycle );
    int ztgsja( int matriint, char jobu, char jobv, char jobq,
                       int m, int p, int n,
                       int k, int l, IComplexNumber a,
                       int lda, IComplexNumber b,
                       int ldb, double tola, double tolb,
                       INDArray alpha, INDArray beta,
                       IComplexNumber u, int ldu,
                       IComplexNumber v, int ldv,
                       IComplexNumber q, int ldq,
                       int ncycle );

    int stgsna( int matriint, char job, char howmny,
                       int select, int n,
                       INDArray a, int lda,  INDArray b,
                       int ldb,  INDArray vl, int ldvl,
                       INDArray vr, int ldvr, INDArray s,
                       INDArray dif, int mm, int m );
    int dtgsna( int matriint, char job, char howmny,
                       int select, int n,
                       INDArray a, int lda,  INDArray b,
                       int ldb,  INDArray vl, int ldvl,
                       INDArray vr, int ldvr, INDArray s,
                       INDArray dif, int mm, int m );
    int ctgsna( int matriint, char job, char howmny,
                       int select, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       INDArray s, INDArray dif, int mm, int m );
    int ztgsna( int matriint, char job, char howmny,
                       int select, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       INDArray s, INDArray dif, int mm,
                       int m );

    int stgsyl( int matriint, char trans, int ijob,
                       int m, int n,  INDArray a,
                       int lda,  INDArray b, int ldb,
                       INDArray c, int ldc,  INDArray d,
                       int ldd,  INDArray e, int lde,
                       INDArray f, int ldf, INDArray scale, INDArray dif );
    int dtgsyl( int matriint, char trans, int ijob,
                       int m, int n,  INDArray a,
                       int lda,  INDArray b, int ldb,
                       INDArray c, int ldc,  INDArray d,
                       int ldd,  INDArray e, int lde,
                       INDArray f, int ldf, INDArray scale,
                       INDArray dif );
    int ctgsyl( int matriint, char trans, int ijob,
                       int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber c, int ldc,
                       IComplexNumber d, int ldd,
                       IComplexNumber e, int lde,
                       IComplexNumber f, int ldf,
                       INDArray scale, INDArray dif );
    int ztgsyl( int matriint, char trans, int ijob,
                       int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber c, int ldc,
                       IComplexNumber d, int ldd,
                       IComplexNumber e, int lde,
                       IComplexNumber f, int ldf,
                       INDArray scale, INDArray dif );

    int stpcon( int matriint, char norm, char uplo, char diag,
                       int n,  INDArray ap, INDArray rcond );
    int dtpcon( int matriint, char norm, char uplo, char diag,
                       int n,  INDArray ap, INDArray rcond );
    int ctpcon( int matriint, char norm, char uplo, char diag,
                       int n,  IComplexNumber ap,
                       INDArray rcond );
    int ztpcon( int matriint, char norm, char uplo, char diag,
                       int n,  IComplexNumber ap,
                       INDArray rcond );

    int stprfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray ap,
                       INDArray b, int ldb,  INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int dtprfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray ap,
                       INDArray b, int ldb,  INDArray x,
                       int ldx, INDArray ferr, INDArray berr );
    int ctprfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber ap,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );
    int ztprfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber ap,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int stptri( int matriint, char uplo, char diag, int n,
                       INDArray ap );
    int dtptri( int matriint, char uplo, char diag, int n,
                       INDArray ap );
    int ctptri( int matriint, char uplo, char diag, int n,
                       IComplexNumber ap );
    int ztptri( int matriint, char uplo, char diag, int n,
                       IComplexNumber ap );

    int stptrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray ap,
                       INDArray b, int ldb );
    int dtptrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray ap,
                       INDArray b, int ldb );
    int ctptrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber ap,
                       IComplexNumber b, int ldb );
    int ztptrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber ap,
                       IComplexNumber b, int ldb );

    int stpttf( int matriint, char transr, char uplo,
                       int n,  INDArray ap, INDArray arf );
    int dtpttf( int matriint, char transr, char uplo,
                       int n,  INDArray ap, INDArray arf );
    int ctpttf( int matriint, char transr, char uplo,
                       int n,  IComplexNumber ap,
                       IComplexNumber arf );
    int ztpttf( int matriint, char transr, char uplo,
                       int n,  IComplexNumber ap,
                       IComplexNumber arf );

    int stpttr( int matriint, char uplo, int n,
                       INDArray ap, INDArray a, int lda );
    int dtpttr( int matriint, char uplo, int n,
                       INDArray ap, INDArray a, int lda );
    int ctpttr( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       IComplexNumber a, int lda );
    int ztpttr( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       IComplexNumber a, int lda );

    int strcon( int matriint, char norm, char uplo, char diag,
                       int n,  INDArray a, int lda,
                       INDArray rcond );
    int dtrcon( int matriint, char norm, char uplo, char diag,
                       int n,  INDArray a, int lda,
                       INDArray rcond );
    int ctrcon( int matriint, char norm, char uplo, char diag,
                       int n,  IComplexNumber a,
                       int lda, INDArray rcond );
    int ztrcon( int matriint, char norm, char uplo, char diag,
                       int n,  IComplexNumber a,
                       int lda, INDArray rcond );

    int strevc( int matriint, char side, char howmny,
                       int select, int n,  INDArray t,
                       int ldt, INDArray vl, int ldvl,
                       INDArray vr, int ldvr, int mm,
                       int m );
    int dtrevc( int matriint, char side, char howmny,
                       int select, int n,
                       INDArray t, int ldt, INDArray vl,
                       int ldvl, INDArray vr, int ldvr,
                       int mm, int m );
    int ctrevc( int matriint, char side, char howmny,
                       int select, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       int mm, int m );
    int ztrevc( int matriint, char side, char howmny,
                       int select, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       int mm, int m );

    int strexc( int matriint, char compq, int n, INDArray t,
                       int ldt, INDArray q, int ldq,
                       int ifst, int ilst );
    int dtrexc( int matriint, char compq, int n,
                       INDArray t, int ldt, INDArray q, int ldq,
                       int ifst, int ilst );
    int ctrexc( int matriint, char compq, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber q, int ldq,
                       int ifst, int ilst );
    int ztrexc( int matriint, char compq, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber q, int ldq,
                       int ifst, int ilst );

    int strrfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray a,
                       int lda,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray ferr,
                       INDArray berr );
    int dtrrfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray a,
                       int lda,  INDArray b, int ldb,
                       INDArray x, int ldx, INDArray ferr,
                       INDArray berr );
    int ctrrfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );
    int ztrrfs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber x, int ldx,
                       INDArray ferr, INDArray berr );

    int strsen( int matriint, char job, char compq,
                       int select, int n, INDArray t,
                       int ldt, INDArray q, int ldq, INDArray wr,
                       INDArray wi, int m, INDArray s, INDArray sep );
    int dtrsen( int matriint, char job, char compq,
                       int select, int n,
                       INDArray t, int ldt, INDArray q, int ldq,
                       INDArray wr, INDArray wi, int m, INDArray s,
                       INDArray sep );
    int ctrsen( int matriint, char job, char compq,
                       int select, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber q, int ldq,
                       IComplexNumber w, int m, INDArray s,
                       INDArray sep );
    int ztrsen( int matriint, char job, char compq,
                       int select, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber q, int ldq,
                       IComplexNumber w, int m, INDArray s,
                       INDArray sep );

    int strsna( int matriint, char job, char howmny,
                       int select, int n,
                       INDArray t, int ldt,  INDArray vl,
                       int ldvl,  INDArray vr, int ldvr,
                       INDArray s, INDArray sep, int mm, int m );
    int dtrsna( int matriint, char job, char howmny,
                       int select, int n,
                       INDArray t, int ldt,  INDArray vl,
                       int ldvl,  INDArray vr, int ldvr,
                       INDArray s, INDArray sep, int mm,
                       int m );
    int ctrsna( int matriint, char job, char howmny,
                       int select, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       INDArray s, INDArray sep, int mm, int m );
    int ztrsna( int matriint, char job, char howmny,
                       int select, int n,
                       IComplexNumber t, int ldt,
                       IComplexNumber vl, int ldvl,
                       IComplexNumber vr, int ldvr,
                       INDArray s, INDArray sep, int mm,
                       int m );

    int strsyl( int matriint, char trana, char tranb,
                       int isgn, int m, int n,
                       INDArray a, int lda,  INDArray b,
                       int ldb, INDArray c, int ldc,
                       INDArray scale );
    int dtrsyl( int matriint, char trana, char tranb,
                       int isgn, int m, int n,
                       INDArray a, int lda,  INDArray b,
                       int ldb, INDArray c, int ldc,
                       INDArray scale );
    int ctrsyl( int matriint, char trana, char tranb,
                       int isgn, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber c, int ldc,
                       INDArray scale );
    int ztrsyl( int matriint, char trana, char tranb,
                       int isgn, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber c, int ldc,
                       INDArray scale );

    int strtri( int matriint, char uplo, char diag, int n,
                       INDArray a, int lda );
    int dtrtri( int matriint, char uplo, char diag, int n,
                       INDArray a, int lda );
    int ctrtri( int matriint, char uplo, char diag, int n,
                       IComplexNumber a, int lda );
    int ztrtri( int matriint, char uplo, char diag, int n,
                       IComplexNumber a, int lda );

    int strtrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray a,
                       int lda, INDArray b, int ldb );
    int dtrtrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,  INDArray a,
                       int lda, INDArray b, int ldb );
    int ctrtrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb );
    int ztrtrs( int matriint, char uplo, char trans, char diag,
                       int n, int nrhs,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb );

    int strttf( int matriint, char transr, char uplo,
                       int n,  INDArray a, int lda,
                       INDArray arf );
    int dtrttf( int matriint, char transr, char uplo,
                       int n,  INDArray a, int lda,
                       INDArray arf );
    int ctrttf( int matriint, char transr, char uplo,
                       int n,  IComplexNumber a,
                       int lda, IComplexNumber arf );
    int ztrttf( int matriint, char transr, char uplo,
                       int n,  IComplexNumber a,
                       int lda, IComplexNumber arf );

    int strttp( int matriint, char uplo, int n,
                       INDArray a, int lda, INDArray ap );
    int dtrttp( int matriint, char uplo, int n,
                       INDArray a, int lda, INDArray ap );
    int ctrttp( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber ap );
    int ztrttp( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber ap );

    int stzrzf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int dtzrzf( int matriint, int m, int n,
                       INDArray a, int lda, INDArray tau );
    int ctzrzf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int ztzrzf( int matriint, int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int cungbr( int matriint, char vect, int m,
                       int n, int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );
    int zungbr( int matriint, char vect, int m,
                       int n, int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );

    int cunghr( int matriint, int n, int ilo,
                       int ihi, IComplexNumber a,
                       int lda,  IComplexNumber tau );
    int zunghr( int matriint, int n, int ilo,
                       int ihi, IComplexNumber a,
                       int lda,  IComplexNumber tau );

    int cunglq( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );
    int zunglq( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );

    int cungql( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );
    int zungql( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );

    int cungqr( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );
    int zungqr( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );

    int cungrq( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );
    int zungrq( int matriint, int m, int n,
                       int k, IComplexNumber a,
                       int lda,  IComplexNumber tau );

    int cungtr( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );
    int zungtr( int matriint, char uplo, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau );

    int cunmbr( int matriint, char vect, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmbr( int matriint, char vect, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmhr( int matriint, char side, char trans,
                       int m, int n, int ilo,
                       int ihi,  IComplexNumber a,
                       int lda,  IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmhr( int matriint, char side, char trans,
                       int m, int n, int ilo,
                       int ihi,  IComplexNumber a,
                       int lda,  IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmlq( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmlq( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmql( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmql( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmqr( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmqr( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmrq( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmrq( int matriint, char side, char trans,
                       int m, int n, int k,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmrz( int matriint, char side, char trans,
                       int m, int n, int k,
                       int l,  IComplexNumber a,
                       int lda,  IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmrz( int matriint, char side, char trans,
                       int m, int n, int k,
                       int l,  IComplexNumber a,
                       int lda,  IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cunmtr( int matriint, char side, char uplo, char trans,
                       int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zunmtr( int matriint, char side, char uplo, char trans,
                       int m, int n,
                       IComplexNumber a, int lda,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int cupgtr( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       IComplexNumber tau,
                       IComplexNumber q, int ldq );
    int zupgtr( int matriint, char uplo, int n,
                       IComplexNumber ap,
                       IComplexNumber tau,
                       IComplexNumber q, int ldq );

    int cupmtr( int matriint, char side, char uplo, char trans,
                       int m, int n,
                       IComplexNumber ap,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );
    int zupmtr( int matriint, char side, char uplo, char trans,
                       int m, int n,
                       IComplexNumber ap,
                       IComplexNumber tau,
                       IComplexNumber c, int ldc );

    int sbdsdint( int matriint, char uplo, char compq,
                         int n, INDArray d, INDArray e, INDArray u,
                         int ldu, INDArray vt, int ldvt,
                         INDArray q, int iq, INDArray work,
                         int iwork );
    int dbdsdint( int matriint, char uplo, char compq,
                         int n, INDArray d, INDArray e, INDArray u,
                         int ldu, INDArray vt, int ldvt,
                         INDArray q, int iq, INDArray work,
                         int iwork );

    int sbdsvdint( int matriint, char uplo, char jobz, char range,
                          int n, INDArray d, INDArray e,
                          int vl, int vu,
                          int il, int iu, int ns,
                          INDArray s, INDArray z, int ldz,
                          INDArray work, int iwork );
    int dbdsvdint( int matriint, char uplo, char jobz, char range,
                          int n, INDArray d, INDArray e,
                          int vl, int vu,
                          int il, int iu, int ns,
                          INDArray s, INDArray z, int ldz,
                          INDArray work, int iwork );

    int sbdsqint( int matriint, char uplo, int n,
                         int ncvt, int nru, int ncc,
                         INDArray d, INDArray e, INDArray vt, int ldvt,
                         INDArray u, int ldu, INDArray c,
                         int ldc, INDArray work );
    int dbdsqint( int matriint, char uplo, int n,
                         int ncvt, int nru, int ncc,
                         INDArray d, INDArray e, INDArray vt,
                         int ldvt, INDArray u, int ldu,
                         INDArray c, int ldc, INDArray work );
    int cbdsqint( int matriint, char uplo, int n,
                         int ncvt, int nru, int ncc,
                         INDArray d, INDArray e, IComplexNumber vt,
                         int ldvt, IComplexNumber u,
                         int ldu, IComplexNumber c,
                         int ldc, INDArray work );
    int zbdsqint( int matriint, char uplo, int n,
                         int ncvt, int nru, int ncc,
                         INDArray d, INDArray e, IComplexNumber vt,
                         int ldvt, IComplexNumber u,
                         int ldu, IComplexNumber c,
                         int ldc, INDArray work );

    int sdisnint( char job, int m, int n,
                         INDArray d, INDArray sep );
    int ddisnint( char job, int m, int n,
                         INDArray d, INDArray sep );

    int sgbbrint( int matriint, char vect, int m,
                         int n, int ncc, int kl,
                         int ku, INDArray ab, int ldab,
                         INDArray d, INDArray e, INDArray q, int ldq,
                         INDArray pt, int ldpt, INDArray c,
                         int ldc, INDArray work );
    int dgbbrint( int matriint, char vect, int m,
                         int n, int ncc, int kl,
                         int ku, INDArray ab, int ldab,
                         INDArray d, INDArray e, INDArray q, int ldq,
                         INDArray pt, int ldpt, INDArray c,
                         int ldc, INDArray work );
    int cgbbrint( int matriint, char vect, int m,
                         int n, int ncc, int kl,
                         int ku, IComplexNumber ab,
                         int ldab, INDArray d, INDArray e,
                         IComplexNumber q, int ldq,
                         IComplexNumber pt, int ldpt,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, INDArray rwork );
    int zgbbrint( int matriint, char vect, int m,
                         int n, int ncc, int kl,
                         int ku, IComplexNumber ab,
                         int ldab, INDArray d, INDArray e,
                         IComplexNumber q, int ldq,
                         IComplexNumber pt, int ldpt,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, INDArray rwork );

    int sgbcoint( int matriint, char norm, int n,
                         int kl, int ku,  INDArray ab,
                         int ldab,  int ipiv,
                         float anorm, INDArray rcond, INDArray work,
                         int iwork );
    int dgbcoint( int matriint, char norm, int n,
                         int kl, int ku,  INDArray ab,
                         int ldab,  int ipiv,
                         double anorm, INDArray rcond, INDArray work,
                         int iwork );
    int cgbcoint( int matriint, char norm, int n,
                         int kl, int ku,
                         IComplexNumber ab, int ldab,
                         int ipiv, float anorm,
                         INDArray rcond, IComplexNumber work,
                         INDArray rwork );
    int zgbcoint( int matriint, char norm, int n,
                         int kl, int ku,
                         IComplexNumber ab,
                         int ldab,  int ipiv,
                         double anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );

    int sgbeqint( int matriint, int m, int n,
                         int kl, int ku,  INDArray ab,
                         int ldab, INDArray r, INDArray c,
                         INDArray rowcnd, INDArray colcnd, INDArray amax );
    int dgbeqint( int matriint, int m, int n,
                         int kl, int ku,  INDArray ab,
                         int ldab, INDArray r, INDArray c,
                         INDArray rowcnd, INDArray colcnd, INDArray amax );
    int cgbeqint( int matriint, int m, int n,
                         int kl, int ku,
                         IComplexNumber ab, int ldab,
                         INDArray r, INDArray c, INDArray rowcnd,
                         INDArray colcnd, INDArray amax );
    int zgbeqint( int matriint, int m, int n,
                         int kl, int ku,
                         IComplexNumber ab,
                         int ldab, INDArray r, INDArray c,
                         INDArray rowcnd, INDArray colcnd, INDArray amax );

    int sgbequint( int matriint, int m, int n,
                          int kl, int ku,  INDArray ab,
                          int ldab, INDArray r, INDArray c,
                          INDArray rowcnd, INDArray colcnd, INDArray amax );
    int dgbequint( int matriint, int m, int n,
                          int kl, int ku,  INDArray ab,
                          int ldab, INDArray r, INDArray c,
                          INDArray rowcnd, INDArray colcnd, INDArray amax );
    int cgbequint( int matriint, int m, int n,
                          int kl, int ku,
                          IComplexNumber ab,
                          int ldab, INDArray r, INDArray c,
                          INDArray rowcnd, INDArray colcnd, INDArray amax );
    int zgbequint( int matriint, int m, int n,
                          int kl, int ku,
                          IComplexNumber ab,
                          int ldab, INDArray r, INDArray c,
                          INDArray rowcnd, INDArray colcnd, INDArray amax );

    int sgbrfint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         INDArray ab, int ldab,
                         INDArray afb, int ldafb,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dgbrfint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         INDArray ab, int ldab,
                         INDArray afb, int ldafb,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int cgbrfint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb,
                         int ldafb,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zgbrfint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         IComplexNumber ab,
                         int ldab,
                         IComplexNumber afb,
                         int ldafb,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sgbrfsint( int matriint, char trans, char equed,
                          int n, int kl, int ku,
                          int nrhs,  INDArray ab,
                          int ldab,  INDArray afb,
                          int ldafb,  int ipiv,
                          INDArray r,  INDArray c,  INDArray b,
                          int ldb, INDArray x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int dgbrfsint( int matriint, char trans, char equed,
                          int n, int kl, int ku,
                          int nrhs,  INDArray ab,
                          int ldab,  INDArray afb,
                          int ldafb,  int ipiv,
                          INDArray r,  INDArray c,
                          INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int cgbrfsint( int matriint, char trans, char equed,
                          int n, int kl, int ku,
                          int nrhs,
                          IComplexNumber ab,
                          int ldab,
                          IComplexNumber afb,
                          int ldafb,  int ipiv,
                          INDArray r,  INDArray c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zgbrfsint( int matriint, char trans, char equed,
                          int n, int kl, int ku,
                          int nrhs,
                          IComplexNumber ab,
                          int ldab,
                          IComplexNumber afb,
                          int ldafb,  int ipiv,
                          INDArray r,  INDArray c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int sgbsint( int matriint, int n, int kl,
                        int ku, int nrhs, INDArray ab,
                        int ldab, int ipiv, INDArray b,
                        int ldb );
    int dgbsint( int matriint, int n, int kl,
                        int ku, int nrhs, INDArray ab,
                        int ldab, int ipiv, INDArray b,
                        int ldb );
    int cgbsint( int matriint, int n, int kl,
                        int ku, int nrhs,
                        IComplexNumber ab, int ldab,
                        int ipiv, IComplexNumber b,
                        int ldb );
    int zgbsint( int matriint, int n, int kl,
                        int ku, int nrhs,
                        IComplexNumber ab, int ldab,
                        int ipiv, IComplexNumber b,
                        int ldb );

    int sgbsvint( int matriint, char fact, char trans,
                         int n, int kl, int ku,
                         int nrhs, INDArray ab, int ldab,
                         INDArray afb, int ldafb, int ipiv,
                         String equed, INDArray r, INDArray c, INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int dgbsvint( int matriint, char fact, char trans,
                         int n, int kl, int ku,
                         int nrhs, INDArray ab, int ldab,
                         INDArray afb, int ldafb, int ipiv,
                         String equed, INDArray r, INDArray c, INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cgbsvint( int matriint, char fact, char trans,
                         int n, int kl, int ku,
                         int nrhs, IComplexNumber ab,
                         int ldab, IComplexNumber afb,
                         int ldafb, int ipiv, String equed,
                         INDArray r, INDArray c, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );
    int zgbsvint( int matriint, char fact, char trans,
                         int n, int kl, int ku,
                         int nrhs, IComplexNumber ab,
                         int ldab, IComplexNumber afb,
                         int ldafb, int ipiv, String equed,
                         INDArray r, INDArray c, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );

    int sgbsvxint( int matriint, char fact, char trans,
                          int n, int kl, int ku,
                          int nrhs, INDArray ab, int ldab,
                          INDArray afb, int ldafb, int ipiv,
                          String equed, INDArray r, INDArray c, INDArray b,
                          int ldb, INDArray x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int dgbsvxint( int matriint, char fact, char trans,
                          int n, int kl, int ku,
                          int nrhs, INDArray ab, int ldab,
                          INDArray afb, int ldafb,
                          int ipiv, String equed, INDArray r,
                          INDArray c, INDArray b, int ldb,
                          INDArray x, int ldx, INDArray rcond,
                          INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int cgbsvxint( int matriint, char fact, char trans,
                          int n, int kl, int ku,
                          int nrhs, IComplexNumber ab,
                          int ldab, IComplexNumber afb,
                          int ldafb, int ipiv,
                          String equed, INDArray r, INDArray c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zgbsvxint( int matriint, char fact, char trans,
                          int n, int kl, int ku,
                          int nrhs, IComplexNumber ab,
                          int ldab, IComplexNumber afb,
                          int ldafb, int ipiv,
                          String equed, INDArray r, INDArray c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int sgbtrint( int matriint, int m, int n,
                         int kl, int ku, INDArray ab,
                         int ldab, int ipiv );
    int dgbtrint( int matriint, int m, int n,
                         int kl, int ku, INDArray ab,
                         int ldab, int ipiv );
    int cgbtrint( int matriint, int m, int n,
                         int kl, int ku,
                         IComplexNumber ab, int ldab,
                         int ipiv );
    int zgbtrint( int matriint, int m, int n,
                         int kl, int ku,
                         IComplexNumber ab, int ldab,
                         int ipiv );

    int sgbtrint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         INDArray ab, int ldab,
                         int ipiv, INDArray b,
                         int ldb );
    int dgbtrint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         INDArray ab, int ldab,
                         int ipiv, INDArray b,
                         int ldb );
    int cgbtrint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         IComplexNumber ab, int ldab,
                         int ipiv, IComplexNumber b,
                         int ldb );
    int zgbtrint( int matriint, char trans, int n,
                         int kl, int ku, int nrhs,
                         IComplexNumber ab,
                         int ldab,  int ipiv,
                         IComplexNumber b, int ldb );

    int sgebaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray scale, int m, INDArray v,
                         int ldv );
    int dgebaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray scale, int m, INDArray v,
                         int ldv );
    int cgebaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray scale, int m,
                         IComplexNumber v, int ldv );
    int zgebaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray scale, int m,
                         IComplexNumber v, int ldv );

    int sgebaint( int matriint, char job, int n,
                         INDArray a, int lda, int ilo,
                         int ihi, INDArray scale );
    int dgebaint( int matriint, char job, int n,
                         INDArray a, int lda, int ilo,
                         int ihi, INDArray scale );
    int cgebaint( int matriint, char job, int n,
                         IComplexNumber a, int lda,
                         int ilo, int ihi,
                         INDArray scale );
    int zgebaint( int matriint, char job, int n,
                         IComplexNumber a, int lda,
                         int ilo, int ihi,
                         INDArray scale );

    int sgebrint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray d, INDArray e,
                         INDArray tauq, INDArray taup, INDArray work,
                         int lwork );
    int dgebrint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray d, INDArray e,
                         INDArray tauq, INDArray taup, INDArray work,
                         int lwork );
    int cgebrint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         INDArray d, INDArray e, IComplexNumber tauq,
                         IComplexNumber taup,
                         IComplexNumber work, int lwork );
    int zgebrint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         INDArray d, INDArray e,
                         IComplexNumber tauq,
                         IComplexNumber taup,
                         IComplexNumber work, int lwork );

    int sgecoint( int matriint, char norm, int n,
                         INDArray a, int lda, float anorm,
                         INDArray rcond, INDArray work, int iwork );
    int dgecoint( int matriint, char norm, int n,
                         INDArray a, int lda, double anorm,
                         INDArray rcond, INDArray work,
                         int iwork );
    int cgecoint( int matriint, char norm, int n,
                         IComplexNumber a, int lda,
                         float anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );
    int zgecoint( int matriint, char norm, int n,
                         IComplexNumber a, int lda,
                         double anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );

    int sgeeqint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray r,
                         INDArray c, INDArray rowcnd, INDArray colcnd,
                         INDArray amax );
    int dgeeqint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray r,
                         INDArray c, INDArray rowcnd, INDArray colcnd,
                         INDArray amax );
    int cgeeqint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         INDArray r, INDArray c, INDArray rowcnd,
                         INDArray colcnd, INDArray amax );
    int zgeeqint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         INDArray r, INDArray c, INDArray rowcnd,
                         INDArray colcnd, INDArray amax );

    int sgeequint( int matriint, int m, int n,
                          INDArray a, int lda, INDArray r,
                          INDArray c, INDArray rowcnd, INDArray colcnd,
                          INDArray amax );
    int dgeequint( int matriint, int m, int n,
                          INDArray a, int lda, INDArray r,
                          INDArray c, INDArray rowcnd, INDArray colcnd,
                          INDArray amax );
    int cgeequint( int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          INDArray r, INDArray c, INDArray rowcnd,
                          INDArray colcnd, INDArray amax );
    int zgeequint( int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          INDArray r, INDArray c, INDArray rowcnd,
                          INDArray colcnd, INDArray amax );

    int sgeeint( int matriint, char jobvs, char sort,
                        int select, int n, INDArray a,
                        int lda, int sdim, INDArray wr,
                        INDArray wi, INDArray vs, int ldvs,
                        INDArray work, int lwork,
                        int bwork );
    int dgeeint( int matriint, char jobvs, char sort,
                        int select, int n, INDArray a,
                        int lda, int sdim, INDArray wr,
                        INDArray wi, INDArray vs, int ldvs,
                        INDArray work, int lwork,
                        int bwork );
    int cgeeint( int matriint, char jobvs, char sort,
                        int select, int n,
                        IComplexNumber a, int lda,
                        int sdim, IComplexNumber w,
                        IComplexNumber vs, int ldvs,
                        IComplexNumber work, int lwork,
                        INDArray rwork, int bwork );
    int zgeeint( int matriint, char jobvs, char sort,
                        int select, int n,
                        IComplexNumber a, int lda,
                        int sdim, IComplexNumber w,
                        IComplexNumber vs, int ldvs,
                        IComplexNumber work, int lwork,
                        INDArray rwork, int bwork );

    int sgeesint( int matriint, char jobvs, char sort,
                         int select, char sense,
                         int n, INDArray a, int lda,
                         int sdim, INDArray wr, INDArray wi,
                         INDArray vs, int ldvs, INDArray rconde,
                         INDArray rcondv, INDArray work, int lwork,
                         int iwork, int liwork,
                         int bwork );
    int dgeesint( int matriint, char jobvs, char sort,
                         int select, char sense,
                         int n, INDArray a, int lda,
                         int sdim, INDArray wr, INDArray wi,
                         INDArray vs, int ldvs, INDArray rconde,
                         INDArray rcondv, INDArray work, int lwork,
                         int iwork, int liwork,
                         int bwork );
    int cgeesint( int matriint, char jobvs, char sort,
                         int select, char sense,
                         int n, IComplexNumber a,
                         int lda, int sdim,
                         IComplexNumber w,
                         IComplexNumber vs, int ldvs,
                         INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int bwork );
    int zgeesint( int matriint, char jobvs, char sort,
                         int select, char sense,
                         int n, IComplexNumber a,
                         int lda, int sdim,
                         IComplexNumber w,
                         IComplexNumber vs, int ldvs,
                         INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int bwork );

    int sgeeint( int matriint, char jobvl, char jobvr,
                        int n, INDArray a, int lda,
                        INDArray wr, INDArray wi, INDArray vl, int ldvl,
                        INDArray vr, int ldvr, INDArray work,
                        int lwork );
    int dgeeint( int matriint, char jobvl, char jobvr,
                        int n, INDArray a, int lda,
                        INDArray wr, INDArray wi, INDArray vl,
                        int ldvl, INDArray vr, int ldvr,
                        INDArray work, int lwork );
    int cgeeint( int matriint, char jobvl, char jobvr,
                        int n, IComplexNumber a,
                        int lda, IComplexNumber w,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,
                        IComplexNumber work, int lwork,
                        INDArray rwork );
    int zgeeint( int matriint, char jobvl, char jobvr,
                        int n, IComplexNumber a,
                        int lda, IComplexNumber w,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,
                        IComplexNumber work, int lwork,
                        INDArray rwork );

    int sgeevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n, INDArray a,
                         int lda, INDArray wr, INDArray wi, INDArray vl,
                         int ldvl, INDArray vr, int ldvr,
                         int ilo, int ihi, INDArray scale,
                         INDArray abnrm, INDArray rconde, INDArray rcondv,
                         INDArray work, int lwork,
                         int iwork );
    int dgeevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n, INDArray a,
                         int lda, INDArray wr, INDArray wi,
                         INDArray vl, int ldvl, INDArray vr,
                         int ldvr, int ilo,
                         int ihi, INDArray scale, INDArray abnrm,
                         INDArray rconde, INDArray rcondv, INDArray work,
                         int lwork, int iwork );
    int cgeevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber w,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int ilo, int ihi, INDArray scale,
                         INDArray abnrm, INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork );
    int zgeevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber w,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int ilo, int ihi, INDArray scale,
                         INDArray abnrm, INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork );

    int sgehrint( int matriint, int n, int ilo,
                         int ihi, INDArray a, int lda,
                         INDArray tau, INDArray work, int lwork );
    int dgehrint( int matriint, int n, int ilo,
                         int ihi, INDArray a, int lda,
                         INDArray tau, INDArray work, int lwork );
    int cgehrint( int matriint, int n, int ilo,
                         int ihi, IComplexNumber a,
                         int lda, IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zgehrint( int matriint, int n, int ilo,
                         int ihi, IComplexNumber a,
                         int lda, IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int sgejsint( int matriint, char joba, char jobu,
                         char jobv, char jobr, char jobt, char jobp,
                         int m, int n, INDArray a,
                         int lda, INDArray sva, INDArray u,
                         int ldu, INDArray v, int ldv,
                         INDArray work, int lwork,
                         int iwork );
    int dgejsint( int matriint, char joba, char jobu,
                         char jobv, char jobr, char jobt, char jobp,
                         int m, int n, INDArray a,
                         int lda, INDArray sva, INDArray u,
                         int ldu, INDArray v, int ldv,
                         INDArray work, int lwork,
                         int iwork );
    int cgejsint( int matriint, char joba, char jobu,
                         char jobv, char jobr, char jobt, char jobp,
                         int m, int n, IComplexNumber a,
                         int lda, INDArray sva, IComplexNumber u,
                         int ldu, IComplexNumber v, int ldv,
                         IComplexNumber cwork, int lwork,
                         INDArray work, int lrwork,
                         int iwork );
    int zgejsint( int matriint, char joba, char jobu,
                         char jobv, char jobr, char jobt, char jobp,
                         int m, int n, IComplexNumber a,
                         int lda, INDArray sva, IComplexNumber u,
                         int ldu, IComplexNumber v, int ldv,
                         IComplexNumber cwork, int lwork,
                         INDArray work, int lrwork,
                         int iwork );

    int sgelq2_work( int matriint, int m, int n,
                            INDArray a, int lda, INDArray tau,
                            INDArray work );
    int dgelq2_work( int matriint, int m, int n,
                            INDArray a, int lda, INDArray tau,
                            INDArray work );
    int cgelq2_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work );
    int zgelq2_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work );

    int sgelqint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int dgelqint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int cgelqint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zgelqint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int sgelint( int matriint, char trans, int m,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray b, int ldb,
                        INDArray work, int lwork );
    int dgelint( int matriint, char trans, int m,
                        int n, int nrhs, INDArray a,
                        int lda, INDArray b, int ldb,
                        INDArray work, int lwork );
    int cgelint( int matriint, char trans, int m,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int lwork );
    int zgelint( int matriint, char trans, int m,
                        int n, int nrhs,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int lwork );

    int sgelsint( int matriint, int m, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, INDArray s, float rcond,
                         int rank, INDArray work, int lwork,
                         int iwork );
    int dgelsint( int matriint, int m, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, INDArray s,
                         double rcond, int rank, INDArray work,
                         int lwork, int iwork );
    int cgelsint( int matriint, int m, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, INDArray s, float rcond,
                         int rank, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int iwork );
    int zgelsint( int matriint, int m, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, INDArray s, double rcond,
                         int rank, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int iwork );

    int sgelsint( int matriint, int m, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, INDArray s, float rcond,
                         int rank, INDArray work,
                         int lwork );
    int dgelsint( int matriint, int m, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, INDArray s,
                         double rcond, int rank, INDArray work,
                         int lwork );
    int cgelsint( int matriint, int m, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, INDArray s, float rcond,
                         int rank, IComplexNumber work,
                         int lwork, INDArray rwork );
    int zgelsint( int matriint, int m, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, INDArray s, double rcond,
                         int rank, IComplexNumber work,
                         int lwork, INDArray rwork );

    int sgelsint( int matriint, int m, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, int jpvt,
                         float rcond, int rank, INDArray work,
                         int lwork );
    int dgelsint( int matriint, int m, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, int jpvt,
                         double rcond, int rank, INDArray work,
                         int lwork );
    int cgelsint( int matriint, int m, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, int jpvt, float rcond,
                         int rank, IComplexNumber work,
                         int lwork, INDArray rwork );
    int zgelsint( int matriint, int m, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, int jpvt, double rcond,
                         int rank, IComplexNumber work,
                         int lwork, INDArray rwork );

    int sgeqlint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int dgeqlint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int cgeqlint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zgeqlint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int sgeqp3_work( int matriint, int m, int n,
                            INDArray a, int lda, int jpvt,
                            INDArray tau, INDArray work, int lwork );
    int dgeqp3_work( int matriint, int m, int n,
                            INDArray a, int lda, int jpvt,
                            INDArray tau, INDArray work, int lwork );
    int cgeqp3_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            int jpvt, IComplexNumber tau,
                            IComplexNumber work, int lwork,
                            INDArray rwork );
    int zgeqp3_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            int jpvt, IComplexNumber tau,
                            IComplexNumber work, int lwork,
                            INDArray rwork );

    int sgeqpint( int matriint, int m, int n,
                         INDArray a, int lda, int jpvt,
                         INDArray tau, INDArray work );
    int dgeqpint( int matriint, int m, int n,
                         INDArray a, int lda, int jpvt,
                         INDArray tau, INDArray work );
    int cgeqpint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         int jpvt, IComplexNumber tau,
                         IComplexNumber work, INDArray rwork );
    int zgeqpint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         int jpvt, IComplexNumber tau,
                         IComplexNumber work, INDArray rwork );

    int sgeqr2_work( int matriint, int m, int n,
                            INDArray a, int lda, INDArray tau,
                            INDArray work );
    int dgeqr2_work( int matriint, int m, int n,
                            INDArray a, int lda, INDArray tau,
                            INDArray work );
    int cgeqr2_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work );
    int zgeqr2_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            IComplexNumber tau,
                            IComplexNumber work );

    int sgeqrint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int dgeqrint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int cgeqrint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zgeqrint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int sgeqrfint( int matriint, int m, int n,
                          INDArray a, int lda, INDArray tau,
                          INDArray work, int lwork );
    int dgeqrfint( int matriint, int m, int n,
                          INDArray a, int lda, INDArray tau,
                          INDArray work, int lwork );
    int cgeqrfint( int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber work, int lwork );
    int zgeqrfint( int matriint, int m, int n,
                          IComplexNumber a, int lda,
                          IComplexNumber tau,
                          IComplexNumber work,
                          int lwork );

    int sgerfint( int matriint, char trans, int n,
                         int nrhs,  INDArray a, int lda,
                         INDArray af, int ldaf,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dgerfint( int matriint, char trans, int n,
                         int nrhs,  INDArray a,
                         int lda,  INDArray af,
                         int ldaf,  int ipiv,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cgerfint( int matriint, char trans, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zgerfint( int matriint, char trans, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sgerfsint( int matriint, char trans, char equed,
                          int n, int nrhs,  INDArray a,
                          int lda,  INDArray af,
                          int ldaf,  int ipiv,
                          INDArray r,  INDArray c,  INDArray b,
                          int ldb, INDArray x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int dgerfsint( int matriint, char trans, char equed,
                          int n, int nrhs,  INDArray a,
                          int lda,  INDArray af,
                          int ldaf,  int ipiv,
                          INDArray r,  INDArray c,
                          INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int cgerfsint( int matriint, char trans, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  int ipiv,
                          INDArray r,  INDArray c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zgerfsint( int matriint, char trans, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  int ipiv,
                          INDArray r,  INDArray c,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int sgerqint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int dgerqint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int cgerqint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zgerqint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int sgesdint( int matriint, char jobz, int m,
                         int n, INDArray a, int lda,
                         INDArray s, INDArray u, int ldu, INDArray vt,
                         int ldvt, INDArray work, int lwork,
                         int iwork );
    int dgesdint( int matriint, char jobz, int m,
                         int n, INDArray a, int lda,
                         INDArray s, INDArray u, int ldu,
                         INDArray vt, int ldvt, INDArray work,
                         int lwork, int iwork );
    int cgesdint( int matriint, char jobz, int m,
                         int n, IComplexNumber a,
                         int lda, INDArray s,
                         IComplexNumber u, int ldu,
                         IComplexNumber vt, int ldvt,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int iwork );
    int zgesdint( int matriint, char jobz, int m,
                         int n, IComplexNumber a,
                         int lda, INDArray s,
                         IComplexNumber u, int ldu,
                         IComplexNumber vt, int ldvt,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int iwork );

    int sgesint( int matriint, int n, int nrhs,
                        INDArray a, int lda, int ipiv,
                        INDArray b, int ldb );
    int dgesint( int matriint, int n, int nrhs,
                        INDArray a, int lda, int ipiv,
                        INDArray b, int ldb );
    int cgesint( int matriint, int n, int nrhs,
                        IComplexNumber a, int lda,
                        int ipiv, IComplexNumber b,
                        int ldb );
    int zgesint( int matriint, int n, int nrhs,
                        IComplexNumber a, int lda,
                        int ipiv, IComplexNumber b,
                        int ldb );
    int dsgesint( int matriint, int n, int nrhs,
                         INDArray a, int lda, int ipiv,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray work, INDArray swork,
                         int iter );
    int zcgesint( int matriint, int n, int nrhs,
                         IComplexNumber a, int lda,
                         int ipiv, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, IComplexNumber work,
                         IComplexNumber swork, INDArray rwork,
                         int iter );

    int sgesvint( int matriint, char jobu, char jobvt,
                         int m, int n, INDArray a,
                         int lda, INDArray s, INDArray u,
                         int ldu, INDArray vt, int ldvt,
                         INDArray work, int lwork );
    int dgesvint( int matriint, char jobu, char jobvt,
                         int m, int n, INDArray a,
                         int lda, INDArray s, INDArray u,
                         int ldu, INDArray vt, int ldvt,
                         INDArray work, int lwork );
    int cgesvint( int matriint, char jobu, char jobvt,
                         int m, int n,
                         IComplexNumber a, int lda,
                         INDArray s, IComplexNumber u,
                         int ldu, IComplexNumber vt,
                         int ldvt, IComplexNumber work,
                         int lwork, INDArray rwork );
    int zgesvint( int matriint, char jobu, char jobvt,
                         int m, int n,
                         IComplexNumber a, int lda,
                         INDArray s, IComplexNumber u,
                         int ldu, IComplexNumber vt,
                         int ldvt, IComplexNumber work,
                         int lwork, INDArray rwork );

    int sgesvdint( int matriint, char jobu, char jobvt, char range,
                          int m, int n, INDArray a,
                          int lda, int vl, int vu,
                          int il, int iu, int ns,
                          INDArray s, INDArray u, int ldu,
                          INDArray vt, int ldvt,
                          INDArray work, int lwork, int iwork );
    int dgesvdint( int matriint, char jobu, char jobvt, char range,
                          int m, int n, INDArray a,
                          int lda, int vl, int vu,
                          int il, int iu, int ns,
                          INDArray s, INDArray u, int ldu,
                          INDArray vt, int ldvt,
                          INDArray work, int lwork, int iwork );
    int cgesvdint( int matriint, char jobu, char jobvt, char range,
                          int m, int n, IComplexNumber a,
                          int lda, int vl, int vu,
                          int il, int iu, int ns,
                          INDArray s, IComplexNumber u, int ldu,
                          IComplexNumber vt, int ldvt,
                          IComplexNumber work, int lwork,
                          INDArray rwork, int iwork );
    int zgesvdint( int matriint, char jobu, char jobvt, char range,
                          int m, int n, IComplexNumber a,
                          int lda, int vl, int vu,
                          int il, int iu, int ns,
                          INDArray s, IComplexNumber u, int ldu,
                          IComplexNumber vt, int ldvt,
                          IComplexNumber work, int lwork,
                          INDArray rwork, int iwork );

    int sgesvint( int matriint, char joba, char jobu,
                         char jobv, int m, int n, INDArray a,
                         int lda, INDArray sva, int mv,
                         INDArray v, int ldv, INDArray work,
                         int lwork );
    int dgesvint( int matriint, char joba, char jobu,
                         char jobv, int m, int n,
                         INDArray a, int lda, INDArray sva,
                         int mv, INDArray v, int ldv,
                         INDArray work, int lwork );
    int cgesvint( int matriint, char joba, char jobu,
                         char jobv, int m, int n, IComplexNumber a,
                         int lda, INDArray sva, int mv,
                         IComplexNumber v, int ldv,
                         IComplexNumber cwork, int lwork,
                         INDArray rwork,int lrwork );
    int zgesvint( int matriint, char joba, char jobu,
                         char jobv, int m, int n,
                         IComplexNumber a, int lda, INDArray sva,
                         int mv, IComplexNumber v, int ldv,
                         IComplexNumber cwork, int lwork,
                         INDArray rwork, int lrwork );

    int sgesvint( int matriint, char fact, char trans,
                         int n, int nrhs, INDArray a,
                         int lda, INDArray af, int ldaf,
                         int ipiv, String equed, INDArray r,
                         INDArray c, INDArray b, int ldb, INDArray x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int dgesvint( int matriint, char fact, char trans,
                         int n, int nrhs, INDArray a,
                         int lda, INDArray af, int ldaf,
                         int ipiv, String equed, INDArray r,
                         INDArray c, INDArray b, int ldb, INDArray x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int cgesvint( int matriint, char fact, char trans,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         int ipiv, String equed, INDArray r,
                         INDArray c, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );
    int zgesvint( int matriint, char fact, char trans,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         int ipiv, String equed, INDArray r,
                         INDArray c, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );

    int sgesvxint( int matriint, char fact, char trans,
                          int n, int nrhs, INDArray a,
                          int lda, INDArray af, int ldaf,
                          int ipiv, String equed, INDArray r,
                          INDArray c, INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray rpvgrw,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params, INDArray work,
                          int iwork );
    int dgesvxint( int matriint, char fact, char trans,
                          int n, int nrhs, INDArray a,
                          int lda, INDArray af, int ldaf,
                          int ipiv, String equed, INDArray r,
                          INDArray c, INDArray b, int ldb,
                          INDArray x, int ldx, INDArray rcond,
                          INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int cgesvxint( int matriint, char fact, char trans,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          int ipiv, String equed, INDArray r,
                          INDArray c, IComplexNumber b,
                          int ldb, IComplexNumber x,
                          int ldx, INDArray rcond, INDArray rpvgrw,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params,
                          IComplexNumber work, INDArray rwork );
    int zgesvxint( int matriint, char fact, char trans,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          int ipiv, String equed, INDArray r,
                          INDArray c, IComplexNumber b,
                          int ldb, IComplexNumber x,
                          int ldx, INDArray rcond, INDArray rpvgrw,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params,
                          IComplexNumber work, INDArray rwork );

    int sgetf2_work( int matriint, int m, int n,
                            INDArray a, int lda, int ipiv );
    int dgetf2_work( int matriint, int m, int n,
                            INDArray a, int lda, int ipiv );
    int cgetf2_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            int ipiv );
    int zgetf2_work( int matriint, int m, int n,
                            IComplexNumber a, int lda,
                            int ipiv );

    int sgetrint( int matriint, int m, int n,
                         INDArray a, int lda, int ipiv );
    int dgetrint( int matriint, int m, int n,
                         INDArray a, int lda, int ipiv );
    int cgetrint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         int ipiv );
    int zgetrint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         int ipiv );

    int sgetrf2_work( int matriint, int m, int n,
                             INDArray a, int lda, int ipiv );
    int dgetrf2_work( int matriint, int m, int n,
                             INDArray a, int lda, int ipiv );
    int cgetrf2_work( int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             int ipiv );
    int zgetrf2_work( int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             int ipiv );

    int sgetrint( int matriint, int n, INDArray a,
                         int lda,  int ipiv,
                         INDArray work, int lwork );
    int dgetrint( int matriint, int n, INDArray a,
                         int lda,  int ipiv,
                         INDArray work, int lwork );
    int cgetrint( int matriint, int n,
                         IComplexNumber a, int lda,
                         int ipiv,
                         IComplexNumber work, int lwork );
    int zgetrint( int matriint, int n,
                         IComplexNumber a, int lda,
                         int ipiv,
                         IComplexNumber work, int lwork );

    int sgetrint( int matriint, char trans, int n,
                         int nrhs,  INDArray a, int lda,
                         int ipiv, INDArray b,
                         int ldb );
    int dgetrint( int matriint, char trans, int n,
                         int nrhs,  INDArray a,
                         int lda,  int ipiv,
                         INDArray b, int ldb );
    int cgetrint( int matriint, char trans, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  int ipiv,
                         IComplexNumber b, int ldb );
    int zgetrint( int matriint, char trans, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  int ipiv,
                         IComplexNumber b, int ldb );

    int sggbaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray lscale,  INDArray rscale,
                         int m, INDArray v, int ldv );
    int dggbaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray lscale,  INDArray rscale,
                         int m, INDArray v, int ldv );
    int cggbaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray lscale,  INDArray rscale,
                         int m, IComplexNumber v,
                         int ldv );
    int zggbaint( int matriint, char job, char side,
                         int n, int ilo, int ihi,
                         INDArray lscale,  INDArray rscale,
                         int m, IComplexNumber v,
                         int ldv );

    int sggbaint( int matriint, char job, int n,
                         INDArray a, int lda, INDArray b,
                         int ldb, int ilo,
                         int ihi, INDArray lscale, INDArray rscale,
                         INDArray work );
    int dggbaint( int matriint, char job, int n,
                         INDArray a, int lda, INDArray b,
                         int ldb, int ilo,
                         int ihi, INDArray lscale, INDArray rscale,
                         INDArray work );
    int cggbaint( int matriint, char job, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         int ilo, int ihi, INDArray lscale,
                         INDArray rscale, INDArray work );
    int zggbaint( int matriint, char job, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         int ilo, int ihi,
                         INDArray lscale, INDArray rscale, INDArray work );

    int sggeint( int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, int n,
                        INDArray a, int lda, INDArray b,
                        int ldb, int sdim, INDArray alphar,
                        INDArray alphai, INDArray beta, INDArray vsl,
                        int ldvsl, INDArray vsr, int ldvsr,
                        INDArray work, int lwork,
                        int bwork );
    int dggeint( int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, int n,
                        INDArray a, int lda, INDArray b,
                        int ldb, int sdim, INDArray alphar,
                        INDArray alphai, INDArray beta, INDArray vsl,
                        int ldvsl, INDArray vsr, int ldvsr,
                        INDArray work, int lwork,
                        int bwork );
    int cggeint( int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        int sdim, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vsl, int ldvsl,
                        IComplexNumber vsr, int ldvsr,
                        IComplexNumber work, int lwork,
                        INDArray rwork, int bwork );
    int zggeint( int matriint, char jobvsl, char jobvsr,
                        char sort, int selctg, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        int sdim, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vsl, int ldvsl,
                        IComplexNumber vsr, int ldvsr,
                        IComplexNumber work, int lwork,
                        INDArray rwork, int bwork );

    int sgges3_work( int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg,
                            int n,
                            INDArray a, int lda,
                            INDArray b, int ldb, int sdim,
                            INDArray alphar, INDArray alphai, INDArray beta,
                            INDArray vsl, int ldvsl,
                            INDArray vsr, int ldvsr,
                            INDArray work, int lwork,
                            int bwork );
    int dgges3_work( int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg,
                            int n,
                            INDArray a, int lda,
                            INDArray b, int ldb, int sdim,
                            INDArray alphar, INDArray alphai, INDArray beta,
                            INDArray vsl, int ldvsl,
                            INDArray vsr, int ldvsr,
                            INDArray work, int lwork,
                            int bwork );
    int cgges3_work( int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg,
                            int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            int sdim, IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vsl, int ldvsl,
                            IComplexNumber vsr, int ldvsr,
                            IComplexNumber work, int lwork,
                            INDArray rwork, int bwork );
    int zgges3_work( int matriint, char jobvsl, char jobvsr,
                            char sort, int selctg,
                            int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            int sdim, IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vsl, int ldvsl,
                            IComplexNumber vsr, int ldvsr,
                            IComplexNumber work, int lwork,
                            INDArray rwork, int bwork );

    int sggesint( int matriint, char jobvsl, char jobvsr,
                         char sort, int selctg, char sense,
                         int n, INDArray a, int lda,
                         INDArray b, int ldb, int sdim,
                         INDArray alphar, INDArray alphai, INDArray beta,
                         INDArray vsl, int ldvsl, INDArray vsr,
                         int ldvsr, INDArray rconde, INDArray rcondv,
                         INDArray work, int lwork,
                         int iwork, int liwork,
                         int bwork );
    int dggesint( int matriint, char jobvsl, char jobvsr,
                         char sort, int selctg, char sense,
                         int n, INDArray a, int lda,
                         INDArray b, int ldb, int sdim,
                         INDArray alphar, INDArray alphai, INDArray beta,
                         INDArray vsl, int ldvsl, INDArray vsr,
                         int ldvsr, INDArray rconde,
                         INDArray rcondv, INDArray work, int lwork,
                         int iwork, int liwork,
                         int bwork );
    int cggesint( int matriint, char jobvsl, char jobvsr,
                         char sort, int selctg, char sense,
                         int n, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, int sdim,
                         IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber vsl, int ldvsl,
                         IComplexNumber vsr, int ldvsr,
                         INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int iwork,
                         int liwork, int bwork );
    int zggesint( int matriint, char jobvsl, char jobvsr,
                         char sort, int selctg, char sense,
                         int n, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, int sdim,
                         IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber vsl, int ldvsl,
                         IComplexNumber vsr, int ldvsr,
                         INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int iwork,
                         int liwork, int bwork );

    int sggeint( int matriint, char jobvl, char jobvr,
                        int n, INDArray a, int lda, INDArray b,
                        int ldb, INDArray alphar, INDArray alphai,
                        INDArray beta, INDArray vl, int ldvl,
                        INDArray vr, int ldvr, INDArray work,
                        int lwork );
    int dggeint( int matriint, char jobvl, char jobvr,
                        int n, INDArray a, int lda,
                        INDArray b, int ldb, INDArray alphar,
                        INDArray alphai, INDArray beta, INDArray vl,
                        int ldvl, INDArray vr, int ldvr,
                        INDArray work, int lwork );
    int cggeint( int matriint, char jobvl, char jobvr,
                        int n, IComplexNumber a,
                        int lda, IComplexNumber b,
                        int ldb, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,
                        IComplexNumber work, int lwork,
                        INDArray rwork );
    int zggeint( int matriint, char jobvl, char jobvr,
                        int n, IComplexNumber a,
                        int lda, IComplexNumber b,
                        int ldb, IComplexNumber alpha,
                        IComplexNumber beta,
                        IComplexNumber vl, int ldvl,
                        IComplexNumber vr, int ldvr,
                        IComplexNumber work, int lwork,
                        INDArray rwork );

    int sggev3_work( int matriint, char jobvl, char jobvr,
                            int n,
                            INDArray a, int lda,
                            INDArray b, int ldb,
                            INDArray alphar, INDArray alphai, INDArray beta,
                            INDArray vl, int ldvl,
                            INDArray vr, int ldvr,
                            INDArray work, int lwork );
    int dggev3_work( int matriint, char jobvl, char jobvr,
                            int n,
                            INDArray a, int lda,
                            INDArray b, int ldb,
                            INDArray alphar, INDArray alphai, INDArray beta,
                            INDArray vl, int ldvl,
                            INDArray vr, int ldvr,
                            INDArray work, int lwork );
    int cggev3_work( int matriint, char jobvl, char jobvr,
                            int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,
                            IComplexNumber work, int lwork,
                            INDArray rwork );
    int zggev3_work( int matriint, char jobvl, char jobvr,
                            int n,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber alpha,
                            IComplexNumber beta,
                            IComplexNumber vl, int ldvl,
                            IComplexNumber vr, int ldvr,
                            IComplexNumber work, int lwork,
                            INDArray rwork );

    int sggevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray alphar, INDArray alphai, INDArray beta,
                         INDArray vl, int ldvl, INDArray vr,
                         int ldvr, int ilo,
                         int ihi, INDArray lscale, INDArray rscale,
                         INDArray abnrm, INDArray bbnrm, INDArray rconde,
                         INDArray rcondv, INDArray work, int lwork,
                         int iwork, int bwork );
    int dggevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray alphar, INDArray alphai, INDArray beta,
                         INDArray vl, int ldvl, INDArray vr,
                         int ldvr, int ilo,
                         int ihi, INDArray lscale, INDArray rscale,
                         INDArray abnrm, INDArray bbnrm, INDArray rconde,
                         INDArray rcondv, INDArray work, int lwork,
                         int iwork, int bwork );
    int cggevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int ilo, int ihi, INDArray lscale,
                         INDArray rscale, INDArray abnrm, INDArray bbnrm,
                         INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int iwork,
                         int bwork );
    int zggevint( int matriint, char balanc, char jobvl,
                         char jobvr, char sense, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int ilo, int ihi,
                         INDArray lscale, INDArray rscale, INDArray abnrm,
                         INDArray bbnrm, INDArray rconde, INDArray rcondv,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int iwork,
                         int bwork );

    int sggglint( int matriint, int n, int m,
                         int p, INDArray a, int lda,
                         INDArray b, int ldb, INDArray d, INDArray x,
                         INDArray y, INDArray work, int lwork );
    int dggglint( int matriint, int n, int m,
                         int p, INDArray a, int lda,
                         INDArray b, int ldb, INDArray d, INDArray x,
                         INDArray y, INDArray work, int lwork );
    int cggglint( int matriint, int n, int m,
                         int p, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, IComplexNumber d,
                         IComplexNumber x,
                         IComplexNumber y,
                         IComplexNumber work, int lwork );
    int zggglint( int matriint, int n, int m,
                         int p, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, IComplexNumber d,
                         IComplexNumber x,
                         IComplexNumber y,
                         IComplexNumber work, int lwork );

    int sgghrint( int matriint, char compq, char compz,
                         int n, int ilo, int ihi,
                         INDArray a, int lda, INDArray b,
                         int ldb, INDArray q, int ldq,
                         INDArray z, int ldz );
    int dgghrint( int matriint, char compq, char compz,
                         int n, int ilo, int ihi,
                         INDArray a, int lda, INDArray b,
                         int ldb, INDArray q, int ldq,
                         INDArray z, int ldz );
    int cgghrint( int matriint, char compq, char compz,
                         int n, int ilo, int ihi,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz );
    int zgghrint( int matriint, char compq, char compz,
                         int n, int ilo, int ihi,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz );

    int sgghd3_work( int matriint, char compq, char compz,
                            int n, int ilo, int ihi,
                            INDArray a, int lda,
                            INDArray b, int ldb,
                            INDArray q, int ldq,
                            INDArray z, int ldz,
                            INDArray work, int lwork );
    int dgghd3_work( int matriint, char compq, char compz,
                            int n, int ilo, int ihi,
                            INDArray a, int lda,
                            INDArray b, int ldb,
                            INDArray q, int ldq,
                            INDArray z, int ldz,
                            INDArray work, int lwork );
    int cgghd3_work( int matriint, char compq, char compz,
                            int n, int ilo, int ihi,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,
                            IComplexNumber work, int lwork );
    int zgghd3_work( int matriint, char compq, char compz,
                            int n, int ilo, int ihi,
                            IComplexNumber a, int lda,
                            IComplexNumber b, int ldb,
                            IComplexNumber q, int ldq,
                            IComplexNumber z, int ldz,
                            IComplexNumber work,
                            int lwork );

    int sgglsint( int matriint, int m, int n,
                         int p, INDArray a, int lda,
                         INDArray b, int ldb, INDArray c, INDArray d,
                         INDArray x, INDArray work, int lwork );
    int dgglsint( int matriint, int m, int n,
                         int p, INDArray a, int lda,
                         INDArray b, int ldb, INDArray c, INDArray d,
                         INDArray x, INDArray work, int lwork );
    int cgglsint( int matriint, int m, int n,
                         int p, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, IComplexNumber c,
                         IComplexNumber d,
                         IComplexNumber x,
                         IComplexNumber work, int lwork );
    int zgglsint( int matriint, int m, int n,
                         int p, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, IComplexNumber c,
                         IComplexNumber d,
                         IComplexNumber x,
                         IComplexNumber work, int lwork );

    int sggqrint( int matriint, int n, int m,
                         int p, INDArray a, int lda,
                         INDArray taua, INDArray b, int ldb,
                         INDArray taub, INDArray work, int lwork );
    int dggqrint( int matriint, int n, int m,
                         int p, INDArray a, int lda,
                         INDArray taua, INDArray b, int ldb,
                         INDArray taub, INDArray work, int lwork );
    int cggqrint( int matriint, int n, int m,
                         int p, IComplexNumber a,
                         int lda, IComplexNumber taua,
                         IComplexNumber b, int ldb,
                         IComplexNumber taub,
                         IComplexNumber work, int lwork );
    int zggqrint( int matriint, int n, int m,
                         int p, IComplexNumber a,
                         int lda, IComplexNumber taua,
                         IComplexNumber b, int ldb,
                         IComplexNumber taub,
                         IComplexNumber work, int lwork );

    int sggrqint( int matriint, int m, int p,
                         int n, INDArray a, int lda,
                         INDArray taua, INDArray b, int ldb,
                         INDArray taub, INDArray work, int lwork );
    int dggrqint( int matriint, int m, int p,
                         int n, INDArray a, int lda,
                         INDArray taua, INDArray b, int ldb,
                         INDArray taub, INDArray work, int lwork );
    int cggrqint( int matriint, int m, int p,
                         int n, IComplexNumber a,
                         int lda, IComplexNumber taua,
                         IComplexNumber b, int ldb,
                         IComplexNumber taub,
                         IComplexNumber work, int lwork );
    int zggrqint( int matriint, int m, int p,
                         int n, IComplexNumber a,
                         int lda, IComplexNumber taua,
                         IComplexNumber b, int ldb,
                         IComplexNumber taub,
                         IComplexNumber work, int lwork );

    int sggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int n,
                         int p, int k, int l,
                         INDArray a, int lda, INDArray b,
                         int ldb, INDArray alpha, INDArray beta,
                         INDArray u, int ldu, INDArray v,
                         int ldv, INDArray q, int ldq,
                         INDArray work, int iwork );
    int dggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int n,
                         int p, int k, int l,
                         INDArray a, int lda, INDArray b,
                         int ldb, INDArray alpha, INDArray beta,
                         INDArray u, int ldu, INDArray v,
                         int ldv, INDArray q, int ldq,
                         INDArray work, int iwork );
    int cggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int n,
                         int p, int k, int l,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         INDArray alpha, INDArray beta,
                         IComplexNumber u, int ldu,
                         IComplexNumber v, int ldv,
                         IComplexNumber q, int ldq,
                         IComplexNumber work, INDArray rwork,
                         int iwork );
    int zggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int n,
                         int p, int k, int l,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         INDArray alpha, INDArray beta,
                         IComplexNumber u, int ldu,
                         IComplexNumber v, int ldv,
                         IComplexNumber q, int ldq,
                         IComplexNumber work, INDArray rwork,
                         int iwork );

    int sggsvd3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int n,
                             int p, int k, int l,
                             INDArray a, int lda, INDArray b,
                             int ldb, INDArray alpha, INDArray beta,
                             INDArray u, int ldu, INDArray v,
                             int ldv, INDArray q, int ldq,
                             INDArray work, int lwork,
                             int iwork );
    int dggsvd3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int n,
                             int p, int k, int l,
                             INDArray a, int lda, INDArray b,
                             int ldb, INDArray alpha, INDArray beta,
                             INDArray u, int ldu, INDArray v,
                             int ldv, INDArray q, int ldq,
                             INDArray work, int lwork,
                             int iwork );
    int cggsvd3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int n,
                             int p, int k, int l,
                             IComplexNumber a, int lda,
                             IComplexNumber b, int ldb,
                             INDArray alpha, INDArray beta,
                             IComplexNumber u, int ldu,
                             IComplexNumber v, int ldv,
                             IComplexNumber q, int ldq,
                             IComplexNumber work, int lwork,
                             INDArray rwork, int iwork );
    int zggsvd3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int n,
                             int p, int k, int l,
                             IComplexNumber a, int lda,
                             IComplexNumber b, int ldb,
                             INDArray alpha, INDArray beta,
                             IComplexNumber u, int ldu,
                             IComplexNumber v, int ldv,
                             IComplexNumber q, int ldq,
                             IComplexNumber work, int lwork,
                             INDArray rwork, int iwork );

    int sggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, INDArray a, int lda,
                         INDArray b, int ldb, float tola,
                         float tolb, int k, int l,
                         INDArray u, int ldu, INDArray v,
                         int ldv, INDArray q, int ldq,
                         int iwork, INDArray tau, INDArray work );
    int dggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, INDArray a, int lda,
                         INDArray b, int ldb, double tola,
                         double tolb, int k, int l,
                         INDArray u, int ldu, INDArray v,
                         int ldv, INDArray q, int ldq,
                         int iwork, INDArray tau, INDArray work );
    int cggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, float tola, float tolb,
                         int k, int l,
                         IComplexNumber u, int ldu,
                         IComplexNumber v, int ldv,
                         IComplexNumber q, int ldq,
                         int iwork, INDArray rwork,
                         IComplexNumber tau,
                         IComplexNumber work );
    int zggsvint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, double tola, double tolb,
                         int k, int l,
                         IComplexNumber u, int ldu,
                         IComplexNumber v, int ldv,
                         IComplexNumber q, int ldq,
                         int iwork, INDArray rwork,
                         IComplexNumber tau,
                         IComplexNumber work );

    int sggsvp3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int p,
                             int n, INDArray a, int lda,
                             INDArray b, int ldb, float tola,
                             float tolb, int k, int l,
                             INDArray u, int ldu, INDArray v,
                             int ldv, INDArray q, int ldq,
                             int iwork, INDArray tau,
                             INDArray work, int lwork );
    int dggsvp3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int p,
                             int n, INDArray a, int lda,
                             INDArray b, int ldb, double tola,
                             double tolb, int k, int l,
                             INDArray u, int ldu, INDArray v,
                             int ldv, INDArray q, int ldq,
                             int iwork, INDArray tau, INDArray work,
                             int lwork );
    int cggsvp3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int p,
                             int n, IComplexNumber a,
                             int lda, IComplexNumber b,
                             int ldb, float tola, float tolb,
                             int k, int l,
                             IComplexNumber u, int ldu,
                             IComplexNumber v, int ldv,
                             IComplexNumber q, int ldq,
                             int iwork, INDArray rwork,
                             IComplexNumber tau,
                             IComplexNumber work, int lwork );
    int zggsvp3_work( int matriint, char jobu, char jobv,
                             char jobq, int m, int p,
                             int n, IComplexNumber a,
                             int lda, IComplexNumber b,
                             int ldb, double tola, double tolb,
                             int k, int l,
                             IComplexNumber u, int ldu,
                             IComplexNumber v, int ldv,
                             IComplexNumber q, int ldq,
                             int iwork, INDArray rwork,
                             IComplexNumber tau,
                             IComplexNumber work, int lwork );

    int sgtcoint( char norm, int n,  INDArray dl,
                         INDArray d,  INDArray du,
                         INDArray du2,  int ipiv,
                         float anorm, INDArray rcond, INDArray work,
                         int iwork );
    int dgtcoint( char norm, int n,  INDArray dl,
                         INDArray d,  INDArray du,
                         INDArray du2,  int ipiv,
                         double anorm, INDArray rcond, INDArray work,
                         int iwork );
    int cgtcoint( char norm, int n,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2,
                         int ipiv, float anorm,
                         INDArray rcond, IComplexNumber work );
    int zgtcoint( char norm, int n,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2,
                         int ipiv, double anorm,
                         INDArray rcond, IComplexNumber work );

    int sgtrfint( int matriint, char trans, int n,
                         int nrhs,  INDArray dl,
                         INDArray d,  INDArray du,
                         INDArray dlf,  INDArray df,
                         INDArray duf,  INDArray du2,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dgtrfint( int matriint, char trans, int n,
                         int nrhs,  INDArray dl,
                         INDArray d,  INDArray du,
                         INDArray dlf,  INDArray df,
                         INDArray duf,  INDArray du2,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int cgtrfint( int matriint, char trans, int n,
                         int nrhs,  IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber dlf,
                         IComplexNumber df,
                         IComplexNumber duf,
                         IComplexNumber du2,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zgtrfint( int matriint, char trans, int n,
                         int nrhs,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber dlf,
                         IComplexNumber df,
                         IComplexNumber duf,
                         IComplexNumber du2,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sgtsint( int matriint, int n, int nrhs,
                        INDArray dl, INDArray d, INDArray du, INDArray b,
                        int ldb );
    int dgtsint( int matriint, int n, int nrhs,
                        INDArray dl, INDArray d, INDArray du, INDArray b,
                        int ldb );
    int cgtsint( int matriint, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber b, int ldb );
    int zgtsint( int matriint, int n, int nrhs,
                        IComplexNumber dl,
                        IComplexNumber d,
                        IComplexNumber du,
                        IComplexNumber b, int ldb );

    int sgtsvint( int matriint, char fact, char trans,
                         int n, int nrhs,  INDArray dl,
                         INDArray d,  INDArray du, INDArray dlf,
                         INDArray df, INDArray duf, INDArray du2,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int dgtsvint( int matriint, char fact, char trans,
                         int n, int nrhs,  INDArray dl,
                         INDArray d,  INDArray du, INDArray dlf,
                         INDArray df, INDArray duf, INDArray du2,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cgtsvint( int matriint, char fact, char trans,
                         int n, int nrhs,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber dlf,
                         IComplexNumber df,
                         IComplexNumber duf,
                         IComplexNumber du2, int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zgtsvint( int matriint, char fact, char trans,
                         int n, int nrhs,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber dlf,
                         IComplexNumber df,
                         IComplexNumber duf,
                         IComplexNumber du2, int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sgttrint( int n, INDArray dl, INDArray d, INDArray du,
                         INDArray du2, int ipiv );
    int dgttrint( int n, INDArray dl, INDArray d, INDArray du,
                         INDArray du2, int ipiv );
    int cgttrint( int n, IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2, int ipiv );
    int zgttrint( int n, IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2, int ipiv );

    int sgttrint( int matriint, char trans, int n,
                         int nrhs,  INDArray dl,
                         INDArray d,  INDArray du,
                         INDArray du2,  int ipiv,
                         INDArray b, int ldb );
    int dgttrint( int matriint, char trans, int n,
                         int nrhs,  INDArray dl,
                         INDArray d,  INDArray du,
                         INDArray du2,  int ipiv,
                         INDArray b, int ldb );
    int cgttrint( int matriint, char trans, int n,
                         int nrhs,  IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2,
                         int ipiv, IComplexNumber b,
                         int ldb );
    int zgttrint( int matriint, char trans, int n,
                         int nrhs,
                         IComplexNumber dl,
                         IComplexNumber d,
                         IComplexNumber du,
                         IComplexNumber du2,
                         int ipiv,
                         IComplexNumber b, int ldb );

    int chbeint( int matriint, char jobz, char uplo,
                        int n, int kd,
                        IComplexNumber ab, int ldab,
                        INDArray w, IComplexNumber z,
                        int ldz, IComplexNumber work,
                        INDArray rwork );
    int zhbeint( int matriint, char jobz, char uplo,
                        int n, int kd,
                        IComplexNumber ab, int ldab,
                        INDArray w, IComplexNumber z,
                        int ldz, IComplexNumber work,
                        INDArray rwork );

    int chbevint( int matriint, char jobz, char uplo,
                         int n, int kd,
                         IComplexNumber ab, int ldab,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );
    int zhbevint( int matriint, char jobz, char uplo,
                         int n, int kd,
                         IComplexNumber ab, int ldab,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );

    int chbevint( int matriint, char jobz, char range,
                         char uplo, int n, int kd,
                         IComplexNumber ab, int ldab,
                         IComplexNumber q, int ldq,
                         float vl, float vu, int il,
                         int iu, float abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         INDArray rwork, int iwork,
                         int ifail );
    int zhbevint( int matriint, char jobz, char range,
                         char uplo, int n, int kd,
                         IComplexNumber ab, int ldab,
                         IComplexNumber q, int ldq,
                         double vl, double vu, int il,
                         int iu, double abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         INDArray rwork, int iwork,
                         int ifail );

    int chbgsint( int matriint, char vect, char uplo,
                         int n, int ka, int kb,
                         IComplexNumber ab, int ldab,
                         IComplexNumber bb, int ldbb,
                         IComplexNumber x, int ldx,
                         IComplexNumber work, INDArray rwork );
    int zhbgsint( int matriint, char vect, char uplo,
                         int n, int ka, int kb,
                         IComplexNumber ab, int ldab,
                         IComplexNumber bb,
                         int ldbb, IComplexNumber x,
                         int ldx, IComplexNumber work,
                         INDArray rwork );

    int chbgint( int matriint, char jobz, char uplo,
                        int n, int ka, int kb,
                        IComplexNumber ab, int ldab,
                        IComplexNumber bb, int ldbb,
                        INDArray w, IComplexNumber z,
                        int ldz, IComplexNumber work,
                        INDArray rwork );
    int zhbgint( int matriint, char jobz, char uplo,
                        int n, int ka, int kb,
                        IComplexNumber ab, int ldab,
                        IComplexNumber bb, int ldbb,
                        INDArray w, IComplexNumber z,
                        int ldz, IComplexNumber work,
                        INDArray rwork );

    int chbgvint( int matriint, char jobz, char uplo,
                         int n, int ka, int kb,
                         IComplexNumber ab, int ldab,
                         IComplexNumber bb, int ldbb,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );
    int zhbgvint( int matriint, char jobz, char uplo,
                         int n, int ka, int kb,
                         IComplexNumber ab, int ldab,
                         IComplexNumber bb, int ldbb,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );

    int chbgvint( int matriint, char jobz, char range,
                         char uplo, int n, int ka,
                         int kb, IComplexNumber ab,
                         int ldab, IComplexNumber bb,
                         int ldbb, IComplexNumber q,
                         int ldq, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, INDArray rwork,
                         int iwork, int ifail );
    int zhbgvint( int matriint, char jobz, char range,
                         char uplo, int n, int ka,
                         int kb, IComplexNumber ab,
                         int ldab, IComplexNumber bb,
                         int ldbb, IComplexNumber q,
                         int ldq, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, INDArray rwork,
                         int iwork, int ifail );

    int chbtrint( int matriint, char vect, char uplo,
                         int n, int kd,
                         IComplexNumber ab, int ldab,
                         INDArray d, INDArray e, IComplexNumber q,
                         int ldq, IComplexNumber work );
    int zhbtrint( int matriint, char vect, char uplo,
                         int n, int kd,
                         IComplexNumber ab, int ldab,
                         INDArray d, INDArray e, IComplexNumber q,
                         int ldq, IComplexNumber work );

    int checoint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, float anorm,
                         INDArray rcond, IComplexNumber work );
    int zhecoint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, double anorm,
                         INDArray rcond, IComplexNumber work );

    int cheequint( int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          INDArray s, INDArray scond, INDArray amax,
                          IComplexNumber work );
    int zheequint( int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          INDArray s, INDArray scond, INDArray amax,
                          IComplexNumber work );

    int cheeint( int matriint, char jobz, char uplo,
                        int n, IComplexNumber a,
                        int lda, INDArray w,
                        IComplexNumber work, int lwork,
                        INDArray rwork );
    int zheeint( int matriint, char jobz, char uplo,
                        int n, IComplexNumber a,
                        int lda, INDArray w,
                        IComplexNumber work, int lwork,
                        INDArray rwork );

    int cheevint( int matriint, char jobz, char uplo,
                         int n, IComplexNumber a,
                         int lda, INDArray w,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork, int liwork );
    int zheevint( int matriint, char jobz, char uplo,
                         int n, IComplexNumber a,
                         int lda, INDArray w,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork, int liwork );

    int cheevint( int matriint, char jobz, char range,
                         char uplo, int n,
                         IComplexNumber a, int lda,
                         float vl, float vu, int il,
                         int iu, float abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, int isuppz,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork, int liwork );
    int zheevint( int matriint, char jobz, char range,
                         char uplo, int n,
                         IComplexNumber a, int lda,
                         double vl, double vu, int il,
                         int iu, double abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, int isuppz,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork, int liwork );

    int cheevint( int matriint, char jobz, char range,
                         char uplo, int n,
                         IComplexNumber a, int lda,
                         float vl, float vu, int il,
                         int iu, float abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int iwork, int ifail );
    int zheevint( int matriint, char jobz, char range,
                         char uplo, int n,
                         IComplexNumber a, int lda,
                         double vl, double vu, int il,
                         int iu, double abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int iwork, int ifail );

    int chegsint( int matriint, int itype, char uplo,
                         int n, IComplexNumber a,
                         int lda,  IComplexNumber b,
                         int ldb );
    int zhegsint( int matriint, int itype, char uplo,
                         int n, IComplexNumber a,
                         int lda,  IComplexNumber b,
                         int ldb );

    int chegint( int matriint, int itype, char jobz,
                        char uplo, int n, IComplexNumber a,
                        int lda, IComplexNumber b,
                        int ldb, INDArray w,
                        IComplexNumber work, int lwork,
                        INDArray rwork );
    int zhegint( int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        INDArray w, IComplexNumber work,
                        int lwork, INDArray rwork );

    int chegvint( int matriint, int itype, char jobz,
                         char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         INDArray w, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );
    int zhegvint( int matriint, int itype, char jobz,
                         char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         INDArray w, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );

    int chegvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         float vl, float vu, int il,
                         int iu, float abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int iwork, int ifail );
    int zhegvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         double vl, double vu, int il,
                         int iu, double abstol, int m,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int iwork, int ifail );

    int cherfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zherfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int cherfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  int ipiv,
                          INDArray s,  IComplexNumber b,
                          int ldb, IComplexNumber x,
                          int ldx, INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zherfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  int ipiv,
                          INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int chesint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int lwork );
    int zhesint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int lwork );

    int chesvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         int ipiv,  IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         int lwork, INDArray rwork );
    int zhesvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, int lwork,
                         INDArray rwork );

    int chesvxint( int matriint, char fact, char uplo,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          int ipiv, String equed, INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zhesvxint( int matriint, char fact, char uplo,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          int ipiv, String equed, INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int chetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         INDArray d, INDArray e, IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zhetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         INDArray d, INDArray e,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int chetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, IComplexNumber work,
                         int lwork );
    int zhetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, IComplexNumber work,
                         int lwork );

    int chetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv,
                         IComplexNumber work );
    int zhetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv,
                         IComplexNumber work );


    int chfrint( int matriint, char transr, char uplo,
                        char trans, int n, int k,
                        float alpha,  IComplexNumber a,
                        int lda, float beta,
                        IComplexNumber c );
    int zhfrint( int matriint, char transr, char uplo,
                        char trans, int n, int k,
                        double alpha,  IComplexNumber a,
                        int lda, double beta,
                        IComplexNumber c );

    int shgeqint( int matriint, char job, char compq,
                         char compz, int n, int ilo,
                         int ihi, INDArray h, int ldh,
                         INDArray t, int ldt, INDArray alphar,
                         INDArray alphai, INDArray beta, INDArray q,
                         int ldq, INDArray z, int ldz,
                         INDArray work, int lwork );
    int dhgeqint( int matriint, char job, char compq,
                         char compz, int n, int ilo,
                         int ihi, INDArray h, int ldh,
                         INDArray t, int ldt, INDArray alphar,
                         INDArray alphai, INDArray beta, INDArray q,
                         int ldq, INDArray z, int ldz,
                         INDArray work, int lwork );
    int chgeqint( int matriint, char job, char compq,
                         char compz, int n, int ilo,
                         int ihi, IComplexNumber h,
                         int ldh, IComplexNumber t,
                         int ldt, IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, int lwork,
                         INDArray rwork );
    int zhgeqint( int matriint, char job, char compq,
                         char compz, int n, int ilo,
                         int ihi, IComplexNumber h,
                         int ldh, IComplexNumber t,
                         int ldt, IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, int lwork,
                         INDArray rwork );

    int chpcoint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv, float anorm,
                         INDArray rcond, IComplexNumber work );
    int zhpcoint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv, double anorm,
                         INDArray rcond, IComplexNumber work );

    int chpeint( int matriint, char jobz, char uplo,
                        int n, IComplexNumber ap, INDArray w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, INDArray rwork );
    int zhpeint( int matriint, char jobz, char uplo,
                        int n, IComplexNumber ap,
                        INDArray w, IComplexNumber z,
                        int ldz, IComplexNumber work,
                        INDArray rwork );

    int chpevint( int matriint, char jobz, char uplo,
                         int n, IComplexNumber ap,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );
    int zhpevint( int matriint, char jobz, char uplo,
                         int n, IComplexNumber ap,
                         INDArray w, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );

    int chpevint( int matriint, char jobz, char range,
                         char uplo, int n,
                         IComplexNumber ap, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, INDArray rwork,
                         int iwork, int ifail );
    int zhpevint( int matriint, char jobz, char range,
                         char uplo, int n,
                         IComplexNumber ap, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, INDArray rwork,
                         int iwork, int ifail );

    int chpgsint( int matriint, int itype, char uplo,
                         int n, IComplexNumber ap,
                         IComplexNumber bp );
    int zhpgsint( int matriint, int itype, char uplo,
                         int n, IComplexNumber ap,
                         IComplexNumber bp );

    int chpgint( int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber bp, INDArray w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, INDArray rwork );
    int zhpgint( int matriint, int itype, char jobz,
                        char uplo, int n,
                        IComplexNumber ap,
                        IComplexNumber bp, INDArray w,
                        IComplexNumber z, int ldz,
                        IComplexNumber work, INDArray rwork );

    int chpgvint( int matriint, int itype, char jobz,
                         char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber bp, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork, int liwork );
    int zhpgvint( int matriint, int itype, char jobz,
                         char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber bp, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork, int liwork );

    int chpgvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber bp, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, INDArray rwork,
                         int iwork, int ifail );
    int zhpgvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber bp, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, INDArray rwork,
                         int iwork, int ifail );

    int chprfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber ap,
                         IComplexNumber afp,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zhprfint( int matriint, char uplo, int n,
                         int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int chpsint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber ap,
                        int ipiv, IComplexNumber b,
                        int ldb );
    int zhpsint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber ap,
                        int ipiv, IComplexNumber b,
                        int ldb );

    int chpsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp, int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zhpsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp, int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int chptrint( int matriint, char uplo, int n,
                         IComplexNumber ap, INDArray d, INDArray e,
                         IComplexNumber tau );
    int zhptrint( int matriint, char uplo, int n,
                         IComplexNumber ap, INDArray d, INDArray e,
                         IComplexNumber tau );

    int chptrint( int matriint, char uplo, int n,
                         IComplexNumber ap, int ipiv );
    int zhptrint( int matriint, char uplo, int n,
                         IComplexNumber ap, int ipiv );

    int chptrint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv,
                         IComplexNumber work );
    int zhptrint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv,
                         IComplexNumber work );

    int chptrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber ap,
                         int ipiv, IComplexNumber b,
                         int ldb );
    int zhptrint( int matriint, char uplo, int n,
                         int nrhs,
                         IComplexNumber ap,
                         int ipiv,
                         IComplexNumber b, int ldb );

    int shseiint( int matriint, char job, char eigsrc,
                         char initv, int select,
                         int n,  INDArray h, int ldh,
                         INDArray wr,  INDArray wi, INDArray vl,
                         int ldvl, INDArray vr, int ldvr,
                         int mm, int m, INDArray work,
                         int ifaill, int ifailr );
    int dhseiint( int matriint, char job, char eigsrc,
                         char initv, int select,
                         int n,  INDArray h, int ldh,
                         INDArray wr,  INDArray wi, INDArray vl,
                         int ldvl, INDArray vr, int ldvr,
                         int mm, int m, INDArray work,
                         int ifaill, int ifailr );
    int chseiint( int matriint, char job, char eigsrc,
                         char initv,  int select,
                         int n,  IComplexNumber h,
                         int ldh, IComplexNumber w,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int mm, int m,
                         IComplexNumber work, INDArray rwork,
                         int ifaill, int ifailr );
    int zhseiint( int matriint, char job, char eigsrc,
                         char initv,  int select,
                         int n,  IComplexNumber h,
                         int ldh, IComplexNumber w,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int mm, int m,
                         IComplexNumber work, INDArray rwork,
                         int ifaill, int ifailr );

    int shseqint( int matriint, char job, char compz,
                         int n, int ilo, int ihi,
                         INDArray h, int ldh, INDArray wr, INDArray wi,
                         INDArray z, int ldz, INDArray work,
                         int lwork );
    int dhseqint( int matriint, char job, char compz,
                         int n, int ilo, int ihi,
                         INDArray h, int ldh, INDArray wr,
                         INDArray wi, INDArray z, int ldz,
                         INDArray work, int lwork );
    int chseqint( int matriint, char job, char compz,
                         int n, int ilo, int ihi,
                         IComplexNumber h, int ldh,
                         IComplexNumber w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, int lwork );
    int zhseqint( int matriint, char job, char compz,
                         int n, int ilo, int ihi,
                         IComplexNumber h, int ldh,
                         IComplexNumber w,
                         IComplexNumber z, int ldz,
                         IComplexNumber work, int lwork );

    int clacgint( int n, IComplexNumber x,
                         int incx );
    int zlacgint( int n, IComplexNumber x,
                         int incx );

    int slacn2_work( int n, INDArray v, INDArray x,
                            int isgn, INDArray est, int kase,
                            int isave );
    int dlacn2_work( int n, INDArray v, INDArray x,
                            int isgn, INDArray est, int kase,
                            int isave );
    int clacn2_work( int n, IComplexNumber v,
                            IComplexNumber x,
                            INDArray est, int kase,
                            int isave );
    int zlacn2_work( int n, IComplexNumber v,
                            IComplexNumber x,
                            INDArray est, int kase,
                            int isave );

    int slacpint( int matriint, char uplo, int m,
                         int n,  INDArray a, int lda,
                         INDArray b, int ldb );
    int dlacpint( int matriint, char uplo, int m,
                         int n,  INDArray a, int lda,
                         INDArray b, int ldb );
    int clacpint( int matriint, char uplo, int m,
                         int n,  IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb );
    int zlacpint( int matriint, char uplo, int m,
                         int n,  IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb );

    int clacp2_work( int matriint, char uplo, int m,
                            int n,  INDArray a, int lda,
                            IComplexNumber b, int ldb );
    int zlacp2_work( int matriint, char uplo, int m,
                            int n,  INDArray a, int lda,
                            IComplexNumber b, int ldb );

    int zlag2int( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber sa, int ldsa );

    int slag2int( int matriint, int m, int n,
                         INDArray sa, int ldsa, INDArray a,
                         int lda );

    int dlag2int( int matriint, int m, int n,
                         INDArray a, int lda, INDArray sa,
                         int ldsa );

    int clag2int( int matriint, int m, int n,
                         IComplexNumber sa, int ldsa,
                         IComplexNumber a, int lda );

    int slaggint( int matriint, int m, int n,
                         int kl, int ku,  INDArray d,
                         INDArray a, int lda, int iseed,
                         INDArray work );
    int dlaggint( int matriint, int m, int n,
                         int kl, int ku,  INDArray d,
                         INDArray a, int lda, int iseed,
                         INDArray work );
    int claggint( int matriint, int m, int n,
                         int kl, int ku,  INDArray d,
                         IComplexNumber a, int lda,
                         int iseed, IComplexNumber work );
    int zlaggint( int matriint, int m, int n,
                         int kl, int ku,  INDArray d,
                         IComplexNumber a, int lda,
                         int iseed,
                         IComplexNumber work );

    int claghint( int matriint, int n, int k,
                         INDArray d, IComplexNumber a,
                         int lda, int iseed,
                         IComplexNumber work );
    int zlaghint( int matriint, int n, int k,
                         INDArray d, IComplexNumber a,
                         int lda, int iseed,
                         IComplexNumber work );

    int slagsint( int matriint, int n, int k,
                         INDArray d, INDArray a, int lda,
                         int iseed, INDArray work );
    int dlagsint( int matriint, int n, int k,
                         INDArray d, INDArray a, int lda,
                         int iseed, INDArray work );
    int clagsint( int matriint, int n, int k,
                         INDArray d, IComplexNumber a,
                         int lda, int iseed,
                         IComplexNumber work );
    int zlagsint( int matriint, int n, int k,
                         INDArray d, IComplexNumber a,
                         int lda, int iseed,
                         IComplexNumber work );

    int slapmint( int matriint, int forwrd,
                         int m, int n, INDArray x,
                         int ldx, int k );
    int dlapmint( int matriint, int forwrd,
                         int m, int n, INDArray x,
                         int ldx, int k );
    int clapmint( int matriint, int forwrd,
                         int m, int n,
                         IComplexNumber x, int ldx,
                         int k );
    int zlapmint( int matriint, int forwrd,
                         int m, int n,
                         IComplexNumber x, int ldx,
                         int k );


    int slartgint( float f, float g, INDArray cs, INDArray sn,
                          INDArray r );
    int dlartgint( double f, double g, INDArray cs, INDArray sn,
                          INDArray r );

    int slartgint( float x, float y, float sigma, INDArray cs,
                          INDArray sn );
    int dlartgint( double x, double y, double sigma, INDArray cs,
                          INDArray sn );

    float slapy2_work( float x, float y );
    double dlapy2_work( double x, double y );

    float slapy3_work( float x, float y, float z );
    double dlapy3_work( double x, double y, double z );

    float slamcint( char cmach );
    double dlamcint( char cmach );

    float slangint( int matriint, char norm, int m,
                    int n,  INDArray a, int lda,
                    INDArray work );
    double dlangint( int matriint, char norm, int m,
                     int n,  INDArray a, int lda,
                     INDArray work );
    float clangint( int matriint, char norm, int m,
                    int n,  IComplexNumber a,
                    int lda, INDArray work );
    double zlangint( int matriint, char norm, int m,
                     int n,  IComplexNumber a,
                     int lda, INDArray work );

    float clanhint( int matriint, char norm, char uplo,
                    int n,  IComplexNumber a,
                    int lda, INDArray work );
    double zlanhint( int matriint, char norm, char uplo,
                     int n,  IComplexNumber a,
                     int lda, INDArray work );

    float slansint( int matriint, char norm, char uplo,
                    int n,  INDArray a, int lda,
                    INDArray work );
    double dlansint( int matriint, char norm, char uplo,
                     int n,  INDArray a, int lda,
                     INDArray work );
    float clansint( int matriint, char norm, char uplo,
                    int n,  IComplexNumber a,
                    int lda, INDArray work );
    double zlansint( int matriint, char norm, char uplo,
                     int n,  IComplexNumber a,
                     int lda, INDArray work );

    float slantint( int matriint, char norm, char uplo,
                    char diag, int m, int n,  INDArray a,
                    int lda, INDArray work );
    double dlantint( int matriint, char norm, char uplo,
                     char diag, int m, int n,
                     INDArray a, int lda, INDArray work );
    float clantint( int matriint, char norm, char uplo,
                    char diag, int m, int n,
                    IComplexNumber a, int lda,
                    INDArray work );
    double zlantint( int matriint, char norm, char uplo,
                     char diag, int m, int n,
                     IComplexNumber a, int lda,
                     INDArray work );

    int slarfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k,  INDArray v,
                         int ldv,  INDArray t, int ldt,
                         INDArray c, int ldc, INDArray work,
                         int ldwork );
    int dlarfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k,  INDArray v,
                         int ldv,  INDArray t, int ldt,
                         INDArray c, int ldc, INDArray work,
                         int ldwork );
    int clarfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k,
                         IComplexNumber v, int ldv,
                         IComplexNumber t, int ldt,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int ldwork );
    int zlarfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k,
                         IComplexNumber v, int ldv,
                         IComplexNumber t, int ldt,
                         IComplexNumber c, int ldc,
                         IComplexNumber work,
                         int ldwork );

    int slarfint( int n, INDArray alpha, INDArray x,
                         int incx, INDArray tau );
    int dlarfint( int n, INDArray alpha, INDArray x,
                         int incx, INDArray tau );
    int clarfint( int n, IComplexNumber alpha,
                         IComplexNumber x, int incx,
                         IComplexNumber tau );
    int zlarfint( int n, IComplexNumber alpha,
                         IComplexNumber x, int incx,
                         IComplexNumber tau );

    int slarfint( int matriint, char direct, char storev,
                         int n, int k,  INDArray v,
                         int ldv,  INDArray tau, INDArray t,
                         int ldt );
    int dlarfint( int matriint, char direct, char storev,
                         int n, int k,  INDArray v,
                         int ldv,  INDArray tau, INDArray t,
                         int ldt );
    int clarfint( int matriint, char direct, char storev,
                         int n, int k,
                         IComplexNumber v, int ldv,
                         IComplexNumber tau,
                         IComplexNumber t, int ldt );
    int zlarfint( int matriint, char direct, char storev,
                         int n, int k,
                         IComplexNumber v, int ldv,
                         IComplexNumber tau,
                         IComplexNumber t, int ldt );

    int slarfint( int matriint, char side, int m,
                         int n,  INDArray v, float tau,
                         INDArray c, int ldc, INDArray work );
    int dlarfint( int matriint, char side, int m,
                         int n,  INDArray v, double tau,
                         INDArray c, int ldc, INDArray work );
    int clarfint( int matriint, char side, int m,
                         int n,  IComplexNumber v,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work );
    int zlarfint( int matriint, char side, int m,
                         int n,  IComplexNumber v,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work );

    int slarnint( int idist, int iseed,
                         int n, INDArray x );
    int dlarnint( int idist, int iseed,
                         int n, INDArray x );
    int clarnint( int idist, int iseed,
                         int n, IComplexNumber x );
    int zlarnint( int idist, int iseed,
                         int n, IComplexNumber x );


    int slascint( int matriint, char type, int kl,
                         int ku, float cfrom, float cto,
                         int m, int n, INDArray a,
                         int lda );
    int dlascint( int matriint, char type, int kl,
                         int ku, double cfrom, double cto,
                         int m, int n, INDArray a,
                         int lda );
    int clascint( int matriint, char type, int kl,
                         int ku, float cfrom, float cto,
                         int m, int n, IComplexNumber a,
                         int lda );
    int zlascint( int matriint, char type, int kl,
                         int ku, double cfrom, double cto,
                         int m, int n, IComplexNumber a,
                         int lda );

    int slaseint( int matriint, char uplo, int m,
                         int n, float alpha, float beta, INDArray a,
                         int lda );
    int dlaseint( int matriint, char uplo, int m,
                         int n, double alpha, double beta,
                         INDArray a, int lda );
    int claseint( int matriint, char uplo, int m,
                         int n, IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber a, int lda );
    int zlaseint( int matriint, char uplo, int m,
                         int n, IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber a, int lda );

    int slasrint( char id, int n, INDArray d );
    int dlasrint( char id, int n, INDArray d );

    int slaswint( int matriint, int n, INDArray a,
                         int lda, int k1, int k2,
                         int ipiv, int incx );
    int dlaswint( int matriint, int n, INDArray a,
                         int lda, int k1, int k2,
                         int ipiv, int incx );
    int claswint( int matriint, int n,
                         IComplexNumber a, int lda,
                         int k1, int k2,
                         int ipiv, int incx );
    int zlaswint( int matriint, int n,
                         IComplexNumber a, int lda,
                         int k1, int k2,
                         int ipiv, int incx );

    int slatmint( int matriint, int m, int n,
                         char dist, int iseed, char sym,
                         INDArray d, int mode, float cond,
                         float dmax, int kl, int ku,
                         char pack, INDArray a, int lda,
                         INDArray work );
    int dlatmint( int matriint, int m, int n,
                         char dist, int iseed, char sym,
                         INDArray d, int mode, double cond,
                         double dmax, int kl, int ku,
                         char pack, INDArray a, int lda,
                         INDArray work );
    int clatmint( int matriint, int m, int n,
                         char dist, int iseed, char sym,
                         INDArray d, int mode, float cond,
                         float dmax, int kl, int ku,
                         char pack, IComplexNumber a,
                         int lda, IComplexNumber work );
    int zlatmint( int matriint, int m, int n,
                         char dist, int iseed, char sym,
                         INDArray d, int mode, double cond,
                         double dmax, int kl, int ku,
                         char pack, IComplexNumber a,
                         int lda, IComplexNumber work );

    int slauuint( int matriint, char uplo, int n,
                         INDArray a, int lda );
    int dlauuint( int matriint, char uplo, int n,
                         INDArray a, int lda );
    int clauuint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda );
    int zlauuint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda );

    int sopgtint( int matriint, char uplo, int n,
                         INDArray ap,  INDArray tau, INDArray q,
                         int ldq, INDArray work );
    int dopgtint( int matriint, char uplo, int n,
                         INDArray ap,  INDArray tau, INDArray q,
                         int ldq, INDArray work );

    int sopmtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         INDArray ap,  INDArray tau, INDArray c,
                         int ldc, INDArray work );
    int dopmtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         INDArray ap,  INDArray tau, INDArray c,
                         int ldc, INDArray work );

    int sorgbint( int matriint, char vect, int m,
                         int n, int k, INDArray a,
                         int lda,  INDArray tau, INDArray work,
                         int lwork );
    int dorgbint( int matriint, char vect, int m,
                         int n, int k, INDArray a,
                         int lda,  INDArray tau, INDArray work,
                         int lwork );

    int sorghint( int matriint, int n, int ilo,
                         int ihi, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );
    int dorghint( int matriint, int n, int ilo,
                         int ihi, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );

    int sorglint( int matriint, int m, int n,
                         int k, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );
    int dorglint( int matriint, int m, int n,
                         int k, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );

    int sorgqint( int matriint, int m, int n,
                         int k, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );
    int dorgqint( int matriint, int m, int n,
                         int k, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );



    int sorgrint( int matriint, int m, int n,
                         int k, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );
    int dorgrint( int matriint, int m, int n,
                         int k, INDArray a, int lda,
                         INDArray tau, INDArray work,
                         int lwork );

    int sorgtint( int matriint, char uplo, int n,
                         INDArray a, int lda,  INDArray tau,
                         INDArray work, int lwork );
    int dorgtint( int matriint, char uplo, int n,
                         INDArray a, int lda,  INDArray tau,
                         INDArray work, int lwork );

    int sormbint( int matriint, char vect, char side,
                         char trans, int m, int n,
                         int k,  INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormbint( int matriint, char vect, char side,
                         char trans, int m, int n,
                         int k,  INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );

    int sormhint( int matriint, char side, char trans,
                         int m, int n, int ilo,
                         int ihi,  INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormhint( int matriint, char side, char trans,
                         int m, int n, int ilo,
                         int ihi,  INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );

    int sormlint( int matriint, char side, char trans,
                         int m, int n, int k,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormlint( int matriint, char side, char trans,
                         int m, int n, int k,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );

    int sormqint( int matriint, char side, char trans,
                         int m, int n, int k,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormqint( int matriint, char side, char trans,
                         int m, int n, int k,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );



    int sormrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );

    int sormrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         int l,  INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         int l,  INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );

    int sormtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );
    int dormtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         INDArray a, int lda,
                         INDArray tau, INDArray c, int ldc,
                         INDArray work, int lwork );

    int spbcoint( int matriint, char uplo, int n,
                         int kd,  INDArray ab, int ldab,
                         float anorm, INDArray rcond, INDArray work,
                         int iwork );
    int dpbcoint( int matriint, char uplo, int n,
                         int kd,  INDArray ab,
                         int ldab, double anorm, INDArray rcond,
                         INDArray work, int iwork );
    int cpbcoint( int matriint, char uplo, int n,
                         int kd,  IComplexNumber ab,
                         int ldab, float anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );
    int zpbcoint( int matriint, char uplo, int n,
                         int kd,  IComplexNumber ab,
                         int ldab, double anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );

    int spbeqint( int matriint, char uplo, int n,
                         int kd,  INDArray ab, int ldab,
                         INDArray s, INDArray scond, INDArray amax );
    int dpbeqint( int matriint, char uplo, int n,
                         int kd,  INDArray ab,
                         int ldab, INDArray s, INDArray scond,
                         INDArray amax );
    int cpbeqint( int matriint, char uplo, int n,
                         int kd,  IComplexNumber ab,
                         int ldab, INDArray s, INDArray scond,
                         INDArray amax );
    int zpbeqint( int matriint, char uplo, int n,
                         int kd,  IComplexNumber ab,
                         int ldab, INDArray s, INDArray scond,
                         INDArray amax );

    int spbrfint( int matriint, char uplo, int n,
                         int kd, int nrhs,  INDArray ab,
                         int ldab,  INDArray afb,
                         int ldafb,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dpbrfint( int matriint, char uplo, int n,
                         int kd, int nrhs,
                         INDArray ab, int ldab,
                         INDArray afb, int ldafb,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cpbrfint( int matriint, char uplo, int n,
                         int kd, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb,
                         int ldafb,  IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zpbrfint( int matriint, char uplo, int n,
                         int kd, int nrhs,
                         IComplexNumber ab,
                         int ldab,
                         IComplexNumber afb,
                         int ldafb,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int spbstint( int matriint, char uplo, int n,
                         int kb, INDArray bb, int ldbb );
    int dpbstint( int matriint, char uplo, int n,
                         int kb, INDArray bb, int ldbb );
    int cpbstint( int matriint, char uplo, int n,
                         int kb, IComplexNumber bb,
                         int ldbb );
    int zpbstint( int matriint, char uplo, int n,
                         int kb, IComplexNumber bb,
                         int ldbb );

    int spbsint( int matriint, char uplo, int n,
                        int kd, int nrhs, INDArray ab,
                        int ldab, INDArray b, int ldb );
    int dpbsint( int matriint, char uplo, int n,
                        int kd, int nrhs, INDArray ab,
                        int ldab, INDArray b, int ldb );
    int cpbsint( int matriint, char uplo, int n,
                        int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb );
    int zpbsint( int matriint, char uplo, int n,
                        int kd, int nrhs,
                        IComplexNumber ab, int ldab,
                        IComplexNumber b, int ldb );

    int spbsvint( int matriint, char fact, char uplo,
                         int n, int kd, int nrhs,
                         INDArray ab, int ldab, INDArray afb,
                         int ldafb, String equed, INDArray s,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int dpbsvint( int matriint, char fact, char uplo,
                         int n, int kd, int nrhs,
                         INDArray ab, int ldab, INDArray afb,
                         int ldafb, String equed, INDArray s,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int cpbsvint( int matriint, char fact, char uplo,
                         int n, int kd, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb, int ldafb,
                         String equed, INDArray s, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );
    int zpbsvint( int matriint, char fact, char uplo,
                         int n, int kd, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber afb, int ldafb,
                         String equed, INDArray s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int spbtrint( int matriint, char uplo, int n,
                         int kd, INDArray ab, int ldab );
    int dpbtrint( int matriint, char uplo, int n,
                         int kd, INDArray ab, int ldab );
    int cpbtrint( int matriint, char uplo, int n,
                         int kd, IComplexNumber ab,
                         int ldab );
    int zpbtrint( int matriint, char uplo, int n,
                         int kd, IComplexNumber ab,
                         int ldab );

    int spbtrint( int matriint, char uplo, int n,
                         int kd, int nrhs,  INDArray ab,
                         int ldab, INDArray b, int ldb );
    int dpbtrint( int matriint, char uplo, int n,
                         int kd, int nrhs,
                         INDArray ab, int ldab, INDArray b,
                         int ldb );
    int cpbtrint( int matriint, char uplo, int n,
                         int kd, int nrhs,
                         IComplexNumber ab, int ldab,
                         IComplexNumber b, int ldb );
    int zpbtrint( int matriint, char uplo, int n,
                         int kd, int nrhs,
                         IComplexNumber ab,
                         int ldab, IComplexNumber b,
                         int ldb );

    int spftrint( int matriint, char transr, char uplo,
                         int n, INDArray a );
    int dpftrint( int matriint, char transr, char uplo,
                         int n, INDArray a );
    int cpftrint( int matriint, char transr, char uplo,
                         int n, IComplexNumber a );
    int zpftrint( int matriint, char transr, char uplo,
                         int n, IComplexNumber a );



    int spftrint( int matriint, char transr, char uplo,
                         int n, int nrhs,  INDArray a,
                         INDArray b, int ldb );
    int dpftrint( int matriint, char transr, char uplo,
                         int n, int nrhs,  INDArray a,
                         INDArray b, int ldb );
    int cpftrint( int matriint, char transr, char uplo,
                         int n, int nrhs,
                         IComplexNumber a,
                         IComplexNumber b, int ldb );
    int zpftrint( int matriint, char transr, char uplo,
                         int n, int nrhs,
                         IComplexNumber a,
                         IComplexNumber b, int ldb );

    int spocoint( int matriint, char uplo, int n,
                         INDArray a, int lda, float anorm,
                         INDArray rcond, INDArray work, int iwork );
    int dpocoint( int matriint, char uplo, int n,
                         INDArray a, int lda, double anorm,
                         INDArray rcond, INDArray work,
                         int iwork );
    int cpocoint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         float anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );
    int zpocoint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         double anorm, INDArray rcond,
                         IComplexNumber work, INDArray rwork );

    int spoeqint( int matriint, int n,  INDArray a,
                         int lda, INDArray s, INDArray scond,
                         INDArray amax );
    int dpoeqint( int matriint, int n,  INDArray a,
                         int lda, INDArray s, INDArray scond,
                         INDArray amax );
    int cpoeqint( int matriint, int n,
                         IComplexNumber a, int lda,
                         INDArray s, INDArray scond, INDArray amax );
    int zpoeqint( int matriint, int n,
                         IComplexNumber a, int lda,
                         INDArray s, INDArray scond, INDArray amax );

    int spoequint( int matriint, int n,  INDArray a,
                          int lda, INDArray s, INDArray scond,
                          INDArray amax );
    int dpoequint( int matriint, int n,
                          INDArray a, int lda, INDArray s,
                          INDArray scond, INDArray amax );
    int cpoequint( int matriint, int n,
                          IComplexNumber a, int lda,
                          INDArray s, INDArray scond, INDArray amax );
    int zpoequint( int matriint, int n,
                          IComplexNumber a, int lda,
                          INDArray s, INDArray scond, INDArray amax );

    int sporfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a, int lda,
                         INDArray af, int ldaf,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int dporfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a,
                         int lda,  INDArray af,
                         int ldaf,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int cporfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zporfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sporfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,  INDArray a,
                          int lda,  INDArray af,
                          int ldaf,  INDArray s,
                          INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int dporfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,  INDArray a,
                          int lda,  INDArray af,
                          int ldaf,  INDArray s,
                          INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int cporfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zporfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int sposint( int matriint, char uplo, int n,
                        int nrhs, INDArray a, int lda,
                        INDArray b, int ldb );
    int dposint( int matriint, char uplo, int n,
                        int nrhs, INDArray a, int lda,
                        INDArray b, int ldb );
    int cposint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, IComplexNumber b,
                        int ldb );
    int zposint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, IComplexNumber b,
                        int ldb );
    int dsposint( int matriint, char uplo, int n,
                         int nrhs, INDArray a, int lda,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray work, INDArray swork,
                         int iter );
    int zcposint( int matriint, char uplo, int n,
                         int nrhs, IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, IComplexNumber work,
                         IComplexNumber swork, INDArray rwork,
                         int iter );

    int sposvint( int matriint, char fact, char uplo,
                         int n, int nrhs, INDArray a,
                         int lda, INDArray af, int ldaf,
                         String equed, INDArray s, INDArray b, int ldb,
                         INDArray x, int ldx, INDArray rcond,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dposvint( int matriint, char fact, char uplo,
                         int n, int nrhs, INDArray a,
                         int lda, INDArray af, int ldaf,
                         String equed, INDArray s, INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cposvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         String equed, INDArray s, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );
    int zposvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         String equed, INDArray s,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sposvxint( int matriint, char fact, char uplo,
                          int n, int nrhs, INDArray a,
                          int lda, INDArray af, int ldaf,
                          String equed, INDArray s, INDArray b,
                          int ldb, INDArray x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int dposvxint( int matriint, char fact, char uplo,
                          int n, int nrhs, INDArray a,
                          int lda, INDArray af, int ldaf,
                          String equed, INDArray s, INDArray b,
                          int ldb, INDArray x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int cposvxint( int matriint, char fact, char uplo,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          String equed, INDArray s, IComplexNumber b,
                          int ldb, IComplexNumber x,
                          int ldx, INDArray rcond, INDArray rpvgrw,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params,
                          IComplexNumber work, INDArray rwork );
    int zposvxint( int matriint, char fact, char uplo,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          String equed, INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int spotrf2_work( int matriint, char uplo, int n,
                             INDArray a, int lda );
    int dpotrf2_work( int matriint, char uplo, int n,
                             INDArray a, int lda );
    int cpotrf2_work( int matriint, char uplo, int n,
                             IComplexNumber a, int lda );
    int zpotrf2_work( int matriint, char uplo, int n,
                             IComplexNumber a, int lda );

    int spotrint( int matriint, char uplo, int n,
                         INDArray a, int lda );
    int dpotrint( int matriint, char uplo, int n,
                         INDArray a, int lda );
    int cpotrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda );
    int zpotrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda );



    int spotrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a, int lda,
                         INDArray b, int ldb );
    int dpotrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a,
                         int lda, INDArray b, int ldb );
    int cpotrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb );
    int zpotrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda, IComplexNumber b,
                         int ldb );

    int sppcoint( int matriint, char uplo, int n,
                         INDArray ap, float anorm, INDArray rcond,
                         INDArray work, int iwork );
    int dppcoint( int matriint, char uplo, int n,
                         INDArray ap, double anorm, INDArray rcond,
                         INDArray work, int iwork );
    int cppcoint( int matriint, char uplo, int n,
                         IComplexNumber ap, float anorm,
                         INDArray rcond, IComplexNumber work,
                         INDArray rwork );
    int zppcoint( int matriint, char uplo, int n,
                         IComplexNumber ap, double anorm,
                         INDArray rcond, IComplexNumber work,
                         INDArray rwork );

    int sppeqint( int matriint, char uplo, int n,
                         INDArray ap, INDArray s, INDArray scond,
                         INDArray amax );
    int dppeqint( int matriint, char uplo, int n,
                         INDArray ap, INDArray s, INDArray scond,
                         INDArray amax );
    int cppeqint( int matriint, char uplo, int n,
                         IComplexNumber ap, INDArray s,
                         INDArray scond, INDArray amax );
    int zppeqint( int matriint, char uplo, int n,
                         IComplexNumber ap, INDArray s,
                         INDArray scond, INDArray amax );

    int spprfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap,
                         INDArray afp,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dpprfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap,
                         INDArray afp,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int cpprfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber ap,
                         IComplexNumber afp,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zpprfint( int matriint, char uplo, int n,
                         int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sppsint( int matriint, char uplo, int n,
                        int nrhs, INDArray ap, INDArray b,
                        int ldb );
    int dppsint( int matriint, char uplo, int n,
                        int nrhs, INDArray ap, INDArray b,
                        int ldb );
    int cppsint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber ap,
                        IComplexNumber b, int ldb );
    int zppsint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber ap,
                        IComplexNumber b, int ldb );

    int sppsvint( int matriint, char fact, char uplo,
                         int n, int nrhs, INDArray ap,
                         INDArray afp, String equed, INDArray s, INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int dppsvint( int matriint, char fact, char uplo,
                         int n, int nrhs, INDArray ap,
                         INDArray afp, String equed, INDArray s, INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cppsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp, String equed,
                         INDArray s, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );
    int zppsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp, String equed,
                         INDArray s, IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         INDArray rwork );

    int spptrint( int matriint, char uplo, int n,
                         INDArray ap );
    int dpptrint( int matriint, char uplo, int n,
                         INDArray ap );
    int cpptrint( int matriint, char uplo, int n,
                         IComplexNumber ap );
    int zpptrint( int matriint, char uplo, int n,
                         IComplexNumber ap );



    int spptrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap, INDArray b,
                         int ldb );
    int dpptrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap, INDArray b,
                         int ldb );
    int cpptrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber ap,
                         IComplexNumber b, int ldb );
    int zpptrint( int matriint, char uplo, int n,
                         int nrhs,
                         IComplexNumber ap,
                         IComplexNumber b, int ldb );

    int spstrint( int matriint, char uplo, int n,
                         INDArray a, int lda, int piv,
                         int rank, float tol, INDArray work );
    int dpstrint( int matriint, char uplo, int n,
                         INDArray a, int lda, int piv,
                         int rank, double tol, INDArray work );
    int cpstrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int piv, int rank, float tol,
                         INDArray work );
    int zpstrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int piv, int rank, double tol,
                         INDArray work );

    int sptcoint( int n,  INDArray d,  INDArray e,
                         float anorm, INDArray rcond, INDArray work );
    int dptcoint( int n,  INDArray d,  INDArray e,
                         double anorm, INDArray rcond, INDArray work );
    int cptcoint( int n,  INDArray d,
                         IComplexNumber e, float anorm,
                         INDArray rcond, INDArray work );
    int zptcoint( int n,  INDArray d,
                         IComplexNumber e, double anorm,
                         INDArray rcond, INDArray work );

    int spteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work );
    int dpteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work );
    int cpteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, IComplexNumber z,
                         int ldz, INDArray work );
    int zpteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, IComplexNumber z,
                         int ldz, INDArray work );

    int sptrfint( int matriint, int n, int nrhs,
                         INDArray d,  INDArray e,  INDArray df,
                         INDArray ef,  INDArray b, int ldb,
                         INDArray x, int ldx, INDArray ferr,
                         INDArray berr, INDArray work );
    int dptrfint( int matriint, int n, int nrhs,
                         INDArray d,  INDArray e,
                         INDArray df,  INDArray ef,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work );
    int cptrfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray d,
                         IComplexNumber e,  INDArray df,
                         IComplexNumber ef,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zptrfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray d,
                         IComplexNumber e,
                         INDArray df,
                         IComplexNumber ef,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sptsint( int matriint, int n, int nrhs,
                        INDArray d, INDArray e, INDArray b, int ldb );
    int dptsint( int matriint, int n, int nrhs,
                        INDArray d, INDArray e, INDArray b,
                        int ldb );
    int cptsint( int matriint, int n, int nrhs,
                        INDArray d, IComplexNumber e,
                        IComplexNumber b, int ldb );
    int zptsint( int matriint, int n, int nrhs,
                        INDArray d, IComplexNumber e,
                        IComplexNumber b, int ldb );

    int sptsvint( int matriint, char fact, int n,
                         int nrhs,  INDArray d,  INDArray e,
                         INDArray df, INDArray ef,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work );
    int dptsvint( int matriint, char fact, int n,
                         int nrhs,  INDArray d,
                         INDArray e, INDArray df, INDArray ef,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, INDArray work );
    int cptsvint( int matriint, char fact, int n,
                         int nrhs,  INDArray d,
                         IComplexNumber e, INDArray df,
                         IComplexNumber ef,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zptsvint( int matriint, char fact, int n,
                         int nrhs,  INDArray d,
                         IComplexNumber e, INDArray df,
                         IComplexNumber ef,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int spttrint( int n, INDArray d, INDArray e );
    int dpttrint( int n, INDArray d, INDArray e );
    int cpttrint( int n, INDArray d,
                         IComplexNumber e );
    int zpttrint( int n, INDArray d,
                         IComplexNumber e );

    int spttrint( int matriint, int n, int nrhs,
                         INDArray d,  INDArray e, INDArray b,
                         int ldb );
    int dpttrint( int matriint, int n, int nrhs,
                         INDArray d,  INDArray e, INDArray b,
                         int ldb );
    int cpttrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray d,
                         IComplexNumber e,
                         IComplexNumber b, int ldb );
    int zpttrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray d,
                         IComplexNumber e,
                         IComplexNumber b, int ldb );

    int ssbeint( int matriint, char jobz, char uplo,
                        int n, int kd, INDArray ab,
                        int ldab, INDArray w, INDArray z,
                        int ldz, INDArray work );
    int dsbeint( int matriint, char jobz, char uplo,
                        int n, int kd, INDArray ab,
                        int ldab, INDArray w, INDArray z,
                        int ldz, INDArray work );

    int ssbevint( int matriint, char jobz, char uplo,
                         int n, int kd, INDArray ab,
                         int ldab, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int liwork );
    int dsbevint( int matriint, char jobz, char uplo,
                         int n, int kd, INDArray ab,
                         int ldab, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int liwork );

    int ssbevint( int matriint, char jobz, char range,
                         char uplo, int n, int kd,
                         INDArray ab, int ldab, INDArray q,
                         int ldq, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifail );
    int dsbevint( int matriint, char jobz, char range,
                         char uplo, int n, int kd,
                         INDArray ab, int ldab, INDArray q,
                         int ldq, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifail );

    int ssbgsint( int matriint, char vect, char uplo,
                         int n, int ka, int kb,
                         INDArray ab, int ldab,  INDArray bb,
                         int ldbb, INDArray x, int ldx,
                         INDArray work );
    int dsbgsint( int matriint, char vect, char uplo,
                         int n, int ka, int kb,
                         INDArray ab, int ldab,  INDArray bb,
                         int ldbb, INDArray x, int ldx,
                         INDArray work );

    int ssbgint( int matriint, char jobz, char uplo,
                        int n, int ka, int kb,
                        INDArray ab, int ldab, INDArray bb,
                        int ldbb, INDArray w, INDArray z,
                        int ldz, INDArray work );
    int dsbgint( int matriint, char jobz, char uplo,
                        int n, int ka, int kb,
                        INDArray ab, int ldab, INDArray bb,
                        int ldbb, INDArray w, INDArray z,
                        int ldz, INDArray work );

    int ssbgvint( int matriint, char jobz, char uplo,
                         int n, int ka, int kb,
                         INDArray ab, int ldab, INDArray bb,
                         int ldbb, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int liwork );
    int dsbgvint( int matriint, char jobz, char uplo,
                         int n, int ka, int kb,
                         INDArray ab, int ldab, INDArray bb,
                         int ldbb, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int liwork );

    int ssbgvint( int matriint, char jobz, char range,
                         char uplo, int n, int ka,
                         int kb, INDArray ab, int ldab,
                         INDArray bb, int ldbb, INDArray q,
                         int ldq, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifail );
    int dsbgvint( int matriint, char jobz, char range,
                         char uplo, int n, int ka,
                         int kb, INDArray ab, int ldab,
                         INDArray bb, int ldbb, INDArray q,
                         int ldq, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifail );

    int ssbtrint( int matriint, char vect, char uplo,
                         int n, int kd, INDArray ab,
                         int ldab, INDArray d, INDArray e, INDArray q,
                         int ldq, INDArray work );
    int dsbtrint( int matriint, char vect, char uplo,
                         int n, int kd, INDArray ab,
                         int ldab, INDArray d, INDArray e,
                         INDArray q, int ldq, INDArray work );

    int ssfrint( int matriint, char transr, char uplo,
                        char trans, int n, int k,
                        float alpha,  INDArray a, int lda,
                        float beta, INDArray c );
    int dsfrint( int matriint, char transr, char uplo,
                        char trans, int n, int k,
                        double alpha,  INDArray a, int lda,
                        double beta, INDArray c );

    int sspcoint( int matriint, char uplo, int n,
                         INDArray ap,  int ipiv,
                         float anorm, INDArray rcond, INDArray work,
                         int iwork );
    int dspcoint( int matriint, char uplo, int n,
                         INDArray ap,  int ipiv,
                         double anorm, INDArray rcond, INDArray work,
                         int iwork );
    int cspcoint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv, float anorm,
                         INDArray rcond, IComplexNumber work );
    int zspcoint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv, double anorm,
                         INDArray rcond, IComplexNumber work );

    int sspeint( int matriint, char jobz, char uplo,
                        int n, INDArray ap, INDArray w, INDArray z,
                        int ldz, INDArray work );
    int dspeint( int matriint, char jobz, char uplo,
                        int n, INDArray ap, INDArray w, INDArray z,
                        int ldz, INDArray work );

    int sspevint( int matriint, char jobz, char uplo,
                         int n, INDArray ap, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int liwork );
    int dspevint( int matriint, char jobz, char uplo,
                         int n, INDArray ap, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int liwork );

    int sspevint( int matriint, char jobz, char range,
                         char uplo, int n, INDArray ap, float vl,
                         float vu, int il, int iu,
                         float abstol, int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifail );
    int dspevint( int matriint, char jobz, char range,
                         char uplo, int n, INDArray ap, double vl,
                         double vu, int il, int iu,
                         double abstol, int m, INDArray w,
                         INDArray z, int ldz, INDArray work,
                         int iwork, int ifail );

    int sspgsint( int matriint, int itype, char uplo,
                         int n, INDArray ap,  INDArray bp );
    int dspgsint( int matriint, int itype, char uplo,
                         int n, INDArray ap,  INDArray bp );

    int sspgint( int matriint, int itype, char jobz,
                        char uplo, int n, INDArray ap, INDArray bp,
                        INDArray w, INDArray z, int ldz,
                        INDArray work );
    int dspgint( int matriint, int itype, char jobz,
                        char uplo, int n, INDArray ap, INDArray bp,
                        INDArray w, INDArray z, int ldz,
                        INDArray work );

    int sspgvint( int matriint, int itype, char jobz,
                         char uplo, int n, INDArray ap, INDArray bp,
                         INDArray w, INDArray z, int ldz, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int dspgvint( int matriint, int itype, char jobz,
                         char uplo, int n, INDArray ap, INDArray bp,
                         INDArray w, INDArray z, int ldz,
                         INDArray work, int lwork,
                         int iwork, int liwork );

    int sspgvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n, INDArray ap,
                         INDArray bp, float vl, float vu, int il,
                         int iu, float abstol, int m,
                         INDArray w, INDArray z, int ldz, INDArray work,
                         int iwork, int ifail );
    int dspgvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n, INDArray ap,
                         INDArray bp, double vl, double vu, int il,
                         int iu, double abstol, int m,
                         INDArray w, INDArray z, int ldz,
                         INDArray work, int iwork,
                         int ifail );

    int ssprfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap,
                         INDArray afp,  int ipiv,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int dsprfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap,
                         INDArray afp,  int ipiv,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int csprfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber ap,
                         IComplexNumber afp,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zsprfint( int matriint, char uplo, int n,
                         int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int sspsint( int matriint, char uplo, int n,
                        int nrhs, INDArray ap, int ipiv,
                        INDArray b, int ldb );
    int dspsint( int matriint, char uplo, int n,
                        int nrhs, INDArray ap, int ipiv,
                        INDArray b, int ldb );
    int cspsint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber ap,
                        int ipiv, IComplexNumber b,
                        int ldb );
    int zspsint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber ap,
                        int ipiv, IComplexNumber b,
                        int ldb );

    int sspsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,  INDArray ap,
                         INDArray afp, int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int dspsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,  INDArray ap,
                         INDArray afp, int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int cspsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp, int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zspsvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber afp, int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int ssptrint( int matriint, char uplo, int n,
                         INDArray ap, INDArray d, INDArray e, INDArray tau );
    int dsptrint( int matriint, char uplo, int n,
                         INDArray ap, INDArray d, INDArray e, INDArray tau );

    int ssptrint( int matriint, char uplo, int n,
                         INDArray ap, int ipiv );
    int dsptrint( int matriint, char uplo, int n,
                         INDArray ap, int ipiv );
    int csptrint( int matriint, char uplo, int n,
                         IComplexNumber ap, int ipiv );
    int zsptrint( int matriint, char uplo, int n,
                         IComplexNumber ap, int ipiv );

    int ssptrint( int matriint, char uplo, int n,
                         INDArray ap,  int ipiv,
                         INDArray work );
    int dsptrint( int matriint, char uplo, int n,
                         INDArray ap,  int ipiv,
                         INDArray work );
    int csptrint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv,
                         IComplexNumber work );
    int zsptrint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         int ipiv,
                         IComplexNumber work );

    int ssptrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap,
                         int ipiv, INDArray b,
                         int ldb );
    int dsptrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray ap,
                         int ipiv, INDArray b,
                         int ldb );
    int csptrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber ap,
                         int ipiv, IComplexNumber b,
                         int ldb );
    int zsptrint( int matriint, char uplo, int n,
                         int nrhs,
                         IComplexNumber ap,
                         int ipiv,
                         IComplexNumber b, int ldb );

    int sstebint( char range, char order, int n, float vl,
                         float vu, int il, int iu,
                         float abstol,  INDArray d,  INDArray e,
                         int m, int nsplit, INDArray w,
                         int iblock, int isplit,
                         INDArray work, int iwork );
    int dstebint( char range, char order, int n, double vl,
                         double vu, int il, int iu,
                         double abstol,  INDArray d,  INDArray e,
                         int m, int nsplit, INDArray w,
                         int iblock, int isplit,
                         INDArray work, int iwork );

    int sstedint( int matriint, char compz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int dstedint( int matriint, char compz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int cstedint( int matriint, char compz, int n,
                         INDArray d, INDArray e, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );
    int zstedint( int matriint, char compz, int n,
                         INDArray d, INDArray e, IComplexNumber z,
                         int ldz, IComplexNumber work,
                         int lwork, INDArray rwork,
                         int lrwork, int iwork,
                         int liwork );

    int sstegint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, float vl,
                         float vu, int il, int iu,
                         float abstol, int m, INDArray w, INDArray z,
                         int ldz, int isuppz, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int dstegint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, double vl,
                         double vu, int il, int iu,
                         double abstol, int m, INDArray w,
                         INDArray z, int ldz, int isuppz,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int cstegint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, float vl,
                         float vu, int il, int iu,
                         float abstol, int m, INDArray w,
                         IComplexNumber z, int ldz,
                         int isuppz, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int zstegint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, double vl,
                         double vu, int il, int iu,
                         double abstol, int m, INDArray w,
                         IComplexNumber z, int ldz,
                         int isuppz, INDArray work,
                         int lwork, int iwork,
                         int liwork );

    int ssteiint( int matriint, int n,  INDArray d,
                         INDArray e, int m,  INDArray w,
                         int iblock,
                         int isplit, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifailv );
    int dsteiint( int matriint, int n,  INDArray d,
                         INDArray e, int m,  INDArray w,
                         int iblock,
                         int isplit, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifailv );
    int csteiint( int matriint, int n,  INDArray d,
                         INDArray e, int m,  INDArray w,
                         int iblock,
                         int isplit,
                         IComplexNumber z, int ldz,
                         INDArray work, int iwork,
                         int ifailv );
    int zsteiint( int matriint, int n,  INDArray d,
                         INDArray e, int m,  INDArray w,
                         int iblock,
                         int isplit,
                         IComplexNumber z, int ldz,
                         INDArray work, int iwork,
                         int ifailv );

    int sstemint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, float vl,
                         float vu, int il, int iu,
                         int m, INDArray w, INDArray z,
                         int ldz, int nzc,
                         int isuppz, int tryrac,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int dstemint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, double vl,
                         double vu, int il, int iu,
                         int m, INDArray w, INDArray z,
                         int ldz, int nzc,
                         int isuppz, int tryrac,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int cstemint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, float vl,
                         float vu, int il, int iu,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         int nzc, int isuppz,
                         int tryrac, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int zstemint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, double vl,
                         double vu, int il, int iu,
                         int m, INDArray w,
                         IComplexNumber z, int ldz,
                         int nzc, int isuppz,
                         int tryrac, INDArray work,
                         int lwork, int iwork,
                         int liwork );

    int ssteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work );
    int dsteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work );
    int csteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, IComplexNumber z,
                         int ldz, INDArray work );
    int zsteqint( int matriint, char compz, int n,
                         INDArray d, INDArray e, IComplexNumber z,
                         int ldz, INDArray work );

    int ssterint( int n, INDArray d, INDArray e );
    int dsterint( int n, INDArray d, INDArray e );

    int ssteint( int matriint, char jobz, int n,
                        INDArray d, INDArray e, INDArray z, int ldz,
                        INDArray work );
    int dsteint( int matriint, char jobz, int n,
                        INDArray d, INDArray e, INDArray z, int ldz,
                        INDArray work );

    int sstevint( int matriint, char jobz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int dstevint( int matriint, char jobz, int n,
                         INDArray d, INDArray e, INDArray z, int ldz,
                         INDArray work, int lwork,
                         int iwork, int liwork );

    int sstevint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, float vl,
                         float vu, int il, int iu,
                         float abstol, int m, INDArray w, INDArray z,
                         int ldz, int isuppz, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int dstevint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, double vl,
                         double vu, int il, int iu,
                         double abstol, int m, INDArray w,
                         INDArray z, int ldz, int isuppz,
                         INDArray work, int lwork,
                         int iwork, int liwork );

    int sstevint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, float vl,
                         float vu, int il, int iu,
                         float abstol, int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int iwork,
                         int ifail );
    int dstevint( int matriint, char jobz, char range,
                         int n, INDArray d, INDArray e, double vl,
                         double vu, int il, int iu,
                         double abstol, int m, INDArray w,
                         INDArray z, int ldz, INDArray work,
                         int iwork, int ifail );

    int ssycoint( int matriint, char uplo, int n,
                         INDArray a, int lda,
                         int ipiv, float anorm,
                         INDArray rcond, INDArray work, int iwork );
    int dsycoint( int matriint, char uplo, int n,
                         INDArray a, int lda,
                         int ipiv, double anorm,
                         INDArray rcond, INDArray work,
                         int iwork );
    int csycoint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, float anorm,
                         INDArray rcond, IComplexNumber work );
    int zsycoint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, double anorm,
                         INDArray rcond, IComplexNumber work );

    int ssyequint( int matriint, char uplo, int n,
                          INDArray a, int lda, INDArray s,
                          INDArray scond, INDArray amax, INDArray work );
    int dsyequint( int matriint, char uplo, int n,
                          INDArray a, int lda, INDArray s,
                          INDArray scond, INDArray amax, INDArray work );
    int csyequint( int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          INDArray s, INDArray scond, INDArray amax,
                          IComplexNumber work );
    int zsyequint( int matriint, char uplo, int n,
                          IComplexNumber a, int lda,
                          INDArray s, INDArray scond, INDArray amax,
                          IComplexNumber work );

    int ssyeint( int matriint, char jobz, char uplo,
                        int n, INDArray a, int lda, INDArray w,
                        INDArray work, int lwork );
    int dsyeint( int matriint, char jobz, char uplo,
                        int n, INDArray a, int lda,
                        INDArray w, INDArray work, int lwork );

    int ssyevint( int matriint, char jobz, char uplo,
                         int n, INDArray a, int lda,
                         INDArray w, INDArray work, int lwork,
                         int iwork, int liwork );
    int dsyevint( int matriint, char jobz, char uplo,
                         int n, INDArray a, int lda,
                         INDArray w, INDArray work, int lwork,
                         int iwork, int liwork );

    int ssyevint( int matriint, char jobz, char range,
                         char uplo, int n, INDArray a,
                         int lda, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, int isuppz, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int dsyevint( int matriint, char jobz, char range,
                         char uplo, int n, INDArray a,
                         int lda, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, int isuppz,
                         INDArray work, int lwork,
                         int iwork, int liwork );

    int ssyevint( int matriint, char jobz, char range,
                         char uplo, int n, INDArray a,
                         int lda, float vl, float vu,
                         int il, int iu, float abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int ifail );
    int dsyevint( int matriint, char jobz, char range,
                         char uplo, int n, INDArray a,
                         int lda, double vl, double vu,
                         int il, int iu, double abstol,
                         int m, INDArray w, INDArray z,
                         int ldz, INDArray work, int lwork,
                         int iwork, int ifail );

    int ssygsint( int matriint, int itype, char uplo,
                         int n, INDArray a, int lda,
                         INDArray b, int ldb );
    int dsygsint( int matriint, int itype, char uplo,
                         int n, INDArray a, int lda,
                         INDArray b, int ldb );

    int ssygint( int matriint, int itype, char jobz,
                        char uplo, int n, INDArray a,
                        int lda, INDArray b, int ldb,
                        INDArray w, INDArray work, int lwork );
    int dsygint( int matriint, int itype, char jobz,
                        char uplo, int n, INDArray a,
                        int lda, INDArray b, int ldb,
                        INDArray w, INDArray work, int lwork );

    int ssygvint( int matriint, int itype, char jobz,
                         char uplo, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray w, INDArray work, int lwork,
                         int iwork, int liwork );
    int dsygvint( int matriint, int itype, char jobz,
                         char uplo, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray w, INDArray work, int lwork,
                         int iwork, int liwork );

    int ssygvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         float vl, float vu, int il,
                         int iu, float abstol, int m,
                         INDArray w, INDArray z, int ldz, INDArray work,
                         int lwork, int iwork,
                         int ifail );
    int dsygvint( int matriint, int itype, char jobz,
                         char range, char uplo, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         double vl, double vu, int il,
                         int iu, double abstol, int m,
                         INDArray w, INDArray z, int ldz,
                         INDArray work, int lwork,
                         int iwork, int ifail );

    int ssyrfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a, int lda,
                         INDArray af, int ldaf,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dsyrfint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a,
                         int lda,  INDArray af,
                         int ldaf,  int ipiv,
                         INDArray b, int ldb, INDArray x,
                         int ldx, INDArray ferr, INDArray berr,
                         INDArray work, int iwork );
    int csyrfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int zsyrfint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  IComplexNumber af,
                         int ldaf,  int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int ssyrfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,  INDArray a,
                          int lda,  INDArray af,
                          int ldaf,  int ipiv,
                          INDArray s,  INDArray b, int ldb,
                          INDArray x, int ldx, INDArray rcond,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params, INDArray work,
                          int iwork );
    int dsyrfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,  INDArray a,
                          int lda,  INDArray af,
                          int ldaf,  int ipiv,
                          INDArray s,  INDArray b,
                          int ldb, INDArray x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, INDArray work,
                          int iwork );
    int csyrfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  int ipiv,
                          INDArray s,  IComplexNumber b,
                          int ldb, IComplexNumber x,
                          int ldx, INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zsyrfsint( int matriint, char uplo, char equed,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af,
                          int ldaf,  int ipiv,
                          INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int ssysint( int matriint, char uplo, int n,
                        int nrhs, INDArray a, int lda,
                        int ipiv, INDArray b, int ldb,
                        INDArray work, int lwork );
    int dsysint( int matriint, char uplo, int n,
                        int nrhs, INDArray a, int lda,
                        int ipiv, INDArray b, int ldb,
                        INDArray work, int lwork );
    int csysint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int lwork );
    int zsysint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, int ipiv,
                        IComplexNumber b, int ldb,
                        IComplexNumber work, int lwork );

    int ssysvint( int matriint, char fact, char uplo,
                         int n, int nrhs,  INDArray a,
                         int lda, INDArray af, int ldaf,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int lwork,
                         int iwork );
    int dsysvint( int matriint, char fact, char uplo,
                         int n, int nrhs,  INDArray a,
                         int lda, INDArray af, int ldaf,
                         int ipiv,  INDArray b,
                         int ldb, INDArray x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         INDArray work, int lwork,
                         int iwork );
    int csysvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         int ipiv,  IComplexNumber b,
                         int ldb, IComplexNumber x,
                         int ldx, INDArray rcond, INDArray ferr,
                         INDArray berr, IComplexNumber work,
                         int lwork, INDArray rwork );
    int zsysvint( int matriint, char fact, char uplo,
                         int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber af, int ldaf,
                         int ipiv,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray rcond, INDArray ferr, INDArray berr,
                         IComplexNumber work, int lwork,
                         INDArray rwork );

    int ssysvxint( int matriint, char fact, char uplo,
                          int n, int nrhs, INDArray a,
                          int lda, INDArray af, int ldaf,
                          int ipiv, String equed, INDArray s,
                          INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray rpvgrw,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params, INDArray work,
                          int iwork );
    int dsysvxint( int matriint, char fact, char uplo,
                          int n, int nrhs, INDArray a,
                          int lda, INDArray af, int ldaf,
                          int ipiv, String equed, INDArray s,
                          INDArray b, int ldb, INDArray x,
                          int ldx, INDArray rcond, INDArray rpvgrw,
                          INDArray berr, int int_bnds,
                          INDArray erint_norm, INDArray erint_comp,
                          int nparams, INDArray params,
                          INDArray work, int iwork );
    int csysvxint( int matriint, char fact, char uplo,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          int ipiv, String equed, INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );
    int zsysvxint( int matriint, char fact, char uplo,
                          int n, int nrhs,
                          IComplexNumber a, int lda,
                          IComplexNumber af, int ldaf,
                          int ipiv, String equed, INDArray s,
                          IComplexNumber b, int ldb,
                          IComplexNumber x, int ldx,
                          INDArray rcond, INDArray rpvgrw, INDArray berr,
                          int int_bnds, INDArray erint_norm,
                          INDArray erint_comp, int nparams,
                          INDArray params, IComplexNumber work,
                          INDArray rwork );

    int ssytrint( int matriint, char uplo, int n,
                         INDArray a, int lda, INDArray d, INDArray e,
                         INDArray tau, INDArray work, int lwork );
    int dsytrint( int matriint, char uplo, int n,
                         INDArray a, int lda, INDArray d, INDArray e,
                         INDArray tau, INDArray work, int lwork );

    int ssytrint( int matriint, char uplo, int n,
                         INDArray a, int lda, int ipiv,
                         INDArray work, int lwork );
    int dsytrint( int matriint, char uplo, int n,
                         INDArray a, int lda, int ipiv,
                         INDArray work, int lwork );
    int csytrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, IComplexNumber work,
                         int lwork );
    int zsytrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, IComplexNumber work,
                         int lwork );

    int ssytrint( int matriint, char uplo, int n,
                         INDArray a, int lda,
                         int ipiv, INDArray work );
    int dsytrint( int matriint, char uplo, int n,
                         INDArray a, int lda,
                         int ipiv, INDArray work );
    int csytrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv,
                         IComplexNumber work );
    int zsytrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv,
                         IComplexNumber work );



    int stbcoint( int matriint, char norm, char uplo,
                         char diag, int n, int kd,
                         INDArray ab, int ldab, INDArray rcond,
                         INDArray work, int iwork );
    int dtbcoint( int matriint, char norm, char uplo,
                         char diag, int n, int kd,
                         INDArray ab, int ldab,
                         INDArray rcond, INDArray work,
                         int iwork );
    int ctbcoint( int matriint, char norm, char uplo,
                         char diag, int n, int kd,
                         IComplexNumber ab, int ldab,
                         INDArray rcond, IComplexNumber work,
                         INDArray rwork );
    int ztbcoint( int matriint, char norm, char uplo,
                         char diag, int n, int kd,
                         IComplexNumber ab,
                         int ldab, INDArray rcond,
                         IComplexNumber work, INDArray rwork );

    int stbrfint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,  INDArray ab,
                         int ldab,  INDArray b, int ldb,
                         INDArray x, int ldx, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int dtbrfint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,  INDArray ab,
                         int ldab,  INDArray b,
                         int ldb,  INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int ctbrfint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,  IComplexNumber ab,
                         int ldab,  IComplexNumber b,
                         int ldb,  IComplexNumber x,
                         int ldx, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int ztbrfint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,
                         IComplexNumber ab,
                         int ldab,  IComplexNumber b,
                         int ldb,  IComplexNumber x,
                         int ldx, INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int stbtrint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,  INDArray ab,
                         int ldab, INDArray b, int ldb );
    int dtbtrint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,  INDArray ab,
                         int ldab, INDArray b, int ldb );
    int ctbtrint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,  IComplexNumber ab,
                         int ldab, IComplexNumber b,
                         int ldb );
    int ztbtrint( int matriint, char uplo, char trans,
                         char diag, int n, int kd,
                         int nrhs,
                         IComplexNumber ab,
                         int ldab, IComplexNumber b,
                         int ldb );

    int stfsint( int matriint, char transr, char side,
                        char uplo, char trans, char diag, int m,
                        int n, float alpha,  INDArray a,
                        INDArray b, int ldb );
    int dtfsint( int matriint, char transr, char side,
                        char uplo, char trans, char diag, int m,
                        int n, double alpha,  INDArray a,
                        INDArray b, int ldb );
    int ctfsint( int matriint, char transr, char side,
                        char uplo, char trans, char diag, int m,
                        int n, IComplexNumber alpha,
                        IComplexNumber a,
                        IComplexNumber b, int ldb );
    int ztfsint( int matriint, char transr, char side,
                        char uplo, char trans, char diag, int m,
                        int n, IComplexNumber alpha,
                        IComplexNumber a,
                        IComplexNumber b, int ldb );

    int stftrint( int matriint, char transr, char uplo,
                         char diag, int n, INDArray a );
    int dtftrint( int matriint, char transr, char uplo,
                         char diag, int n, INDArray a );
    int ctftrint( int matriint, char transr, char uplo,
                         char diag, int n,
                         IComplexNumber a );
    int ztftrint( int matriint, char transr, char uplo,
                         char diag, int n,
                         IComplexNumber a );

    int stfttint( int matriint, char transr, char uplo,
                         int n,  INDArray arf, INDArray ap );
    int dtfttint( int matriint, char transr, char uplo,
                         int n,  INDArray arf, INDArray ap );
    int ctfttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber arf,
                         IComplexNumber ap );
    int ztfttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber arf,
                         IComplexNumber ap );

    int stfttint( int matriint, char transr, char uplo,
                         int n,  INDArray arf, INDArray a,
                         int lda );
    int dtfttint( int matriint, char transr, char uplo,
                         int n,  INDArray arf, INDArray a,
                         int lda );
    int ctfttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber arf,
                         IComplexNumber a, int lda );
    int ztfttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber arf,
                         IComplexNumber a, int lda );

    int stgevint( int matriint, char side, char howmny,
                         int select, int n,
                         INDArray s, int lds,  INDArray p,
                         int ldp, INDArray vl, int ldvl,
                         INDArray vr, int ldvr, int mm,
                         int m, INDArray work );
    int dtgevint( int matriint, char side, char howmny,
                         int select, int n,
                         INDArray s, int lds,
                         INDArray p, int ldp, INDArray vl,
                         int ldvl, INDArray vr, int ldvr,
                         int mm, int m, INDArray work );
    int ctgevint( int matriint, char side, char howmny,
                         int select, int n,
                         IComplexNumber s, int lds,
                         IComplexNumber p, int ldp,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int mm, int m,
                         IComplexNumber work, INDArray rwork );
    int ztgevint( int matriint, char side, char howmny,
                         int select, int n,
                         IComplexNumber s, int lds,
                         IComplexNumber p, int ldp,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int mm, int m,
                         IComplexNumber work, INDArray rwork );

    int stgexint( int matriint, int wantq,
                         int wantz, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray q, int ldq, INDArray z,
                         int ldz, int ifst,
                         int ilst, INDArray work,
                         int lwork );
    int dtgexint( int matriint, int wantq,
                         int wantz, int n, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray q, int ldq, INDArray z,
                         int ldz, int ifst,
                         int ilst, INDArray work,
                         int lwork );
    int ctgexint( int matriint, int wantq,
                         int wantz, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz,
                         int ifst, int ilst );
    int ztgexint( int matriint, int wantq,
                         int wantz, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz,
                         int ifst, int ilst );

    int stgseint( int matriint, int ijob,
                         int wantq, int wantz,
                         int select, int n,
                         INDArray a, int lda, INDArray b,
                         int ldb, INDArray alphar, INDArray alphai,
                         INDArray beta, INDArray q, int ldq, INDArray z,
                         int ldz, int m, INDArray pl,
                         INDArray pr, INDArray dif, INDArray work,
                         int lwork, int iwork,
                         int liwork );
    int dtgseint( int matriint, int ijob,
                         int wantq, int wantz,
                         int select, int n,
                         INDArray a, int lda, INDArray b,
                         int ldb, INDArray alphar, INDArray alphai,
                         INDArray beta, INDArray q, int ldq,
                         INDArray z, int ldz, int m,
                         INDArray pl, INDArray pr, INDArray dif,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int ctgseint( int matriint, int ijob,
                         int wantq, int wantz,
                         int select, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz,
                         int m, INDArray pl, INDArray pr, INDArray dif,
                         IComplexNumber work, int lwork,
                         int iwork, int liwork );
    int ztgseint( int matriint, int ijob,
                         int wantq, int wantz,
                         int select, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber alpha,
                         IComplexNumber beta,
                         IComplexNumber q, int ldq,
                         IComplexNumber z, int ldz,
                         int m, INDArray pl, INDArray pr,
                         INDArray dif, IComplexNumber work,
                         int lwork, int iwork,
                         int liwork );

    int stgsjint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, int k, int l,
                         INDArray a, int lda, INDArray b,
                         int ldb, float tola, float tolb,
                         INDArray alpha, INDArray beta, INDArray u,
                         int ldu, INDArray v, int ldv,
                         INDArray q, int ldq, INDArray work,
                         int ncycle );
    int dtgsjint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, int k, int l,
                         INDArray a, int lda, INDArray b,
                         int ldb, double tola, double tolb,
                         INDArray alpha, INDArray beta, INDArray u,
                         int ldu, INDArray v, int ldv,
                         INDArray q, int ldq, INDArray work,
                         int ncycle );
    int ctgsjint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, int k, int l,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         float tola, float tolb, INDArray alpha,
                         INDArray beta, IComplexNumber u,
                         int ldu, IComplexNumber v,
                         int ldv, IComplexNumber q,
                         int ldq, IComplexNumber work,
                         int ncycle );
    int ztgsjint( int matriint, char jobu, char jobv,
                         char jobq, int m, int p,
                         int n, int k, int l,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         double tola, double tolb, INDArray alpha,
                         INDArray beta, IComplexNumber u,
                         int ldu, IComplexNumber v,
                         int ldv, IComplexNumber q,
                         int ldq, IComplexNumber work,
                         int ncycle );

    int stgsnint( int matriint, char job, char howmny,
                         int select, int n,
                         INDArray a, int lda,  INDArray b,
                         int ldb,  INDArray vl,
                         int ldvl,  INDArray vr,
                         int ldvr, INDArray s, INDArray dif,
                         int mm, int m, INDArray work,
                         int lwork, int iwork );
    int dtgsnint( int matriint, char job, char howmny,
                         int select, int n,
                         INDArray a, int lda,
                         INDArray b, int ldb,
                         INDArray vl, int ldvl,
                         INDArray vr, int ldvr, INDArray s,
                         INDArray dif, int mm, int m,
                         INDArray work, int lwork,
                         int iwork );
    int ctgsnint( int matriint, char job, char howmny,
                         int select, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         INDArray s, INDArray dif, int mm,
                         int m, IComplexNumber work,
                         int lwork, int iwork );
    int ztgsnint( int matriint, char job, char howmny,
                         int select, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber vl,
                         int ldvl,
                         IComplexNumber vr,
                         int ldvr, INDArray s, INDArray dif,
                         int mm, int m,
                         IComplexNumber work, int lwork,
                         int iwork );

    int stgsyint( int matriint, char trans, int ijob,
                         int m, int n,  INDArray a,
                         int lda,  INDArray b, int ldb,
                         INDArray c, int ldc,  INDArray d,
                         int ldd,  INDArray e, int lde,
                         INDArray f, int ldf, INDArray scale,
                         INDArray dif, INDArray work, int lwork,
                         int iwork );
    int dtgsyint( int matriint, char trans, int ijob,
                         int m, int n,  INDArray a,
                         int lda,  INDArray b, int ldb,
                         INDArray c, int ldc,  INDArray d,
                         int ldd,  INDArray e, int lde,
                         INDArray f, int ldf, INDArray scale,
                         INDArray dif, INDArray work, int lwork,
                         int iwork );
    int ctgsyint( int matriint, char trans, int ijob,
                         int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber c, int ldc,
                         IComplexNumber d, int ldd,
                         IComplexNumber e, int lde,
                         IComplexNumber f, int ldf,
                         INDArray scale, INDArray dif,
                         IComplexNumber work, int lwork,
                         int iwork );
    int ztgsyint( int matriint, char trans, int ijob,
                         int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber c, int ldc,
                         IComplexNumber d, int ldd,
                         IComplexNumber e, int lde,
                         IComplexNumber f, int ldf,
                         INDArray scale, INDArray dif,
                         IComplexNumber work, int lwork,
                         int iwork );

    int stpcoint( int matriint, char norm, char uplo,
                         char diag, int n,  INDArray ap,
                         INDArray rcond, INDArray work, int iwork );
    int dtpcoint( int matriint, char norm, char uplo,
                         char diag, int n,  INDArray ap,
                         INDArray rcond, INDArray work,
                         int iwork );
    int ctpcoint( int matriint, char norm, char uplo,
                         char diag, int n,
                         IComplexNumber ap, INDArray rcond,
                         IComplexNumber work, INDArray rwork );
    int ztpcoint( int matriint, char norm, char uplo,
                         char diag, int n,
                         IComplexNumber ap, INDArray rcond,
                         IComplexNumber work, INDArray rwork );

    int stprfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray ap,  INDArray b, int ldb,
                         INDArray x, int ldx, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int dtprfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray ap,  INDArray b,
                         int ldb,  INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int ctprfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int ztprfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int stptrint( int matriint, char uplo, char diag,
                         int n, INDArray ap );
    int dtptrint( int matriint, char uplo, char diag,
                         int n, INDArray ap );
    int ctptrint( int matriint, char uplo, char diag,
                         int n, IComplexNumber ap );
    int ztptrint( int matriint, char uplo, char diag,
                         int n, IComplexNumber ap );

    int stptrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray ap, INDArray b, int ldb );
    int dtptrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray ap, INDArray b, int ldb );
    int ctptrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber b, int ldb );
    int ztptrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber ap,
                         IComplexNumber b, int ldb );

    int stpttint( int matriint, char transr, char uplo,
                         int n,  INDArray ap, INDArray arf );
    int dtpttint( int matriint, char transr, char uplo,
                         int n,  INDArray ap, INDArray arf );
    int ctpttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber ap,
                         IComplexNumber arf );
    int ztpttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber ap,
                         IComplexNumber arf );

    int stpttint( int matriint, char uplo, int n,
                         INDArray ap, INDArray a, int lda );
    int dtpttint( int matriint, char uplo, int n,
                         INDArray ap, INDArray a, int lda );
    int ctpttint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber a, int lda );
    int ztpttint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber a, int lda );

    int strcoint( int matriint, char norm, char uplo,
                         char diag, int n,  INDArray a,
                         int lda, INDArray rcond, INDArray work,
                         int iwork );
    int dtrcoint( int matriint, char norm, char uplo,
                         char diag, int n,  INDArray a,
                         int lda, INDArray rcond, INDArray work,
                         int iwork );
    int ctrcoint( int matriint, char norm, char uplo,
                         char diag, int n,
                         IComplexNumber a, int lda,
                         INDArray rcond, IComplexNumber work,
                         INDArray rwork );
    int ztrcoint( int matriint, char norm, char uplo,
                         char diag, int n,
                         IComplexNumber a, int lda,
                         INDArray rcond, IComplexNumber work,
                         INDArray rwork );

    int strevint( int matriint, char side, char howmny,
                         int select, int n,
                         INDArray t, int ldt, INDArray vl,
                         int ldvl, INDArray vr, int ldvr,
                         int mm, int m, INDArray work );
    int dtrevint( int matriint, char side, char howmny,
                         int select, int n,
                         INDArray t, int ldt, INDArray vl,
                         int ldvl, INDArray vr, int ldvr,
                         int mm, int m, INDArray work );
    int ctrevint( int matriint, char side, char howmny,
                         int select, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int mm, int m,
                         IComplexNumber work, INDArray rwork );
    int ztrevint( int matriint, char side, char howmny,
                         int select, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         int mm, int m,
                         IComplexNumber work, INDArray rwork );

    int strexint( int matriint, char compq, int n,
                         INDArray t, int ldt, INDArray q,
                         int ldq, int ifst,
                         int ilst, INDArray work );
    int dtrexint( int matriint, char compq, int n,
                         INDArray t, int ldt, INDArray q,
                         int ldq, int ifst,
                         int ilst, INDArray work );
    int ctrexint( int matriint, char compq, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber q, int ldq,
                         int ifst, int ilst );
    int ztrexint( int matriint, char compq, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber q, int ldq,
                         int ifst, int ilst );

    int strrfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray a, int lda,  INDArray b,
                         int ldb,  INDArray x, int ldx,
                         INDArray ferr, INDArray berr, INDArray work,
                         int iwork );
    int dtrrfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray a, int lda,
                         INDArray b, int ldb,
                         INDArray x, int ldx, INDArray ferr,
                         INDArray berr, INDArray work, int iwork );
    int ctrrfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );
    int ztrrfint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber x, int ldx,
                         INDArray ferr, INDArray berr,
                         IComplexNumber work, INDArray rwork );

    int strseint( int matriint, char job, char compq,
                         int select, int n,
                         INDArray t, int ldt, INDArray q,
                         int ldq, INDArray wr, INDArray wi,
                         int m, INDArray s, INDArray sep,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int dtrseint( int matriint, char job, char compq,
                         int select, int n,
                         INDArray t, int ldt, INDArray q,
                         int ldq, INDArray wr, INDArray wi,
                         int m, INDArray s, INDArray sep,
                         INDArray work, int lwork,
                         int iwork, int liwork );
    int ctrseint( int matriint, char job, char compq,
                         int select, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber q, int ldq,
                         IComplexNumber w, int m,
                         INDArray s, INDArray sep,
                         IComplexNumber work, int lwork );
    int ztrseint( int matriint, char job, char compq,
                         int select, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber q, int ldq,
                         IComplexNumber w, int m,
                         INDArray s, INDArray sep,
                         IComplexNumber work, int lwork );

    int strsnint( int matriint, char job, char howmny,
                         int select, int n,
                         INDArray t, int ldt,  INDArray vl,
                         int ldvl,  INDArray vr,
                         int ldvr, INDArray s, INDArray sep,
                         int mm, int m, INDArray work,
                         int ldwork, int iwork );
    int dtrsnint( int matriint, char job, char howmny,
                         int select, int n,
                         INDArray t, int ldt,
                         INDArray vl, int ldvl,
                         INDArray vr, int ldvr, INDArray s,
                         INDArray sep, int mm, int m,
                         INDArray work, int ldwork,
                         int iwork );
    int ctrsnint( int matriint, char job, char howmny,
                         int select, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber vl, int ldvl,
                         IComplexNumber vr, int ldvr,
                         INDArray s, INDArray sep, int mm,
                         int m, IComplexNumber work,
                         int ldwork, INDArray rwork );
    int ztrsnint( int matriint, char job, char howmny,
                         int select, int n,
                         IComplexNumber t, int ldt,
                         IComplexNumber vl,
                         int ldvl,
                         IComplexNumber vr,
                         int ldvr, INDArray s, INDArray sep,
                         int mm, int m,
                         IComplexNumber work, int ldwork,
                         INDArray rwork );

    int strsyint( int matriint, char trana, char tranb,
                         int isgn, int m, int n,
                         INDArray a, int lda,  INDArray b,
                         int ldb, INDArray c, int ldc,
                         INDArray scale );
    int dtrsyint( int matriint, char trana, char tranb,
                         int isgn, int m, int n,
                         INDArray a, int lda,
                         INDArray b, int ldb, INDArray c,
                         int ldc, INDArray scale );
    int ctrsyint( int matriint, char trana, char tranb,
                         int isgn, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber c, int ldc,
                         INDArray scale );
    int ztrsyint( int matriint, char trana, char tranb,
                         int isgn, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber c, int ldc,
                         INDArray scale );

    int strtrint( int matriint, char uplo, char diag,
                         int n, INDArray a, int lda );
    int dtrtrint( int matriint, char uplo, char diag,
                         int n, INDArray a, int lda );
    int ctrtrint( int matriint, char uplo, char diag,
                         int n, IComplexNumber a,
                         int lda );
    int ztrtrint( int matriint, char uplo, char diag,
                         int n, IComplexNumber a,
                         int lda );

    int strtrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray a, int lda, INDArray b,
                         int ldb );
    int dtrtrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         INDArray a, int lda, INDArray b,
                         int ldb );
    int ctrtrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb );
    int ztrtrint( int matriint, char uplo, char trans,
                         char diag, int n, int nrhs,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb );

    int strttint( int matriint, char transr, char uplo,
                         int n,  INDArray a, int lda,
                         INDArray arf );
    int dtrttint( int matriint, char transr, char uplo,
                         int n,  INDArray a, int lda,
                         INDArray arf );
    int ctrttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber a,
                         int lda, IComplexNumber arf );
    int ztrttint( int matriint, char transr, char uplo,
                         int n,  IComplexNumber a,
                         int lda, IComplexNumber arf );

    int strttint( int matriint, char uplo, int n,
                         INDArray a, int lda, INDArray ap );
    int dtrttint( int matriint, char uplo, int n,
                         INDArray a, int lda, INDArray ap );
    int ctrttint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber ap );
    int ztrttint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber ap );

    int stzrzint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int dtzrzint( int matriint, int m, int n,
                         INDArray a, int lda, INDArray tau,
                         INDArray work, int lwork );
    int ctzrzint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int ztzrzint( int matriint, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int cungbint( int matriint, char vect, int m,
                         int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zungbint( int matriint, char vect, int m,
                         int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int cunghint( int matriint, int n, int ilo,
                         int ihi, IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zunghint( int matriint, int n, int ilo,
                         int ihi, IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int cunglint( int matriint, int m, int n,
                         int k, IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zunglint( int matriint, int m, int n,
                         int k, IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int cungqint( int matriint, int m, int n,
                         int k, IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zungqint( int matriint, int m, int n,
                         int k, IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );


    int cungrint( int matriint, int m, int n,
                         int k, IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zungrint( int matriint, int m, int n,
                         int k, IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int cungtint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );
    int zungtint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber work, int lwork );

    int cunmbint( int matriint, char vect, char side,
                         char trans, int m, int n,
                         int k,  IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmbint( int matriint, char vect, char side,
                         char trans, int m, int n,
                         int k,  IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );

    int cunmhint( int matriint, char side, char trans,
                         int m, int n, int ilo,
                         int ihi,  IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmhint( int matriint, char side, char trans,
                         int m, int n, int ilo,
                         int ihi,  IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );

    int cunmlint( int matriint, char side, char trans,
                         int m, int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmlint( int matriint, char side, char trans,
                         int m, int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );

    int cunmqint( int matriint, char side, char trans,
                         int m, int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmqint( int matriint, char side, char trans,
                         int m, int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );



    int cunmrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );

    int cunmrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         int l,  IComplexNumber a,
                         int lda,  IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmrint( int matriint, char side, char trans,
                         int m, int n, int k,
                         int l,  IComplexNumber a,
                         int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );

    int cunmtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );
    int zunmtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         IComplexNumber a, int lda,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work, int lwork );

    int cupgtint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber tau,
                         IComplexNumber q, int ldq,
                         IComplexNumber work );
    int zupgtint( int matriint, char uplo, int n,
                         IComplexNumber ap,
                         IComplexNumber tau,
                         IComplexNumber q, int ldq,
                         IComplexNumber work );

    int cupmtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         IComplexNumber ap,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work );
    int zupmtint( int matriint, char side, char uplo,
                         char trans, int m, int n,
                         IComplexNumber ap,
                         IComplexNumber tau,
                         IComplexNumber c, int ldc,
                         IComplexNumber work );

    int claghe( int matriint, int n, int k,
                       INDArray d, IComplexNumber a,
                       int lda, int iseed );
    int zlaghe( int matriint, int n, int k,
                       INDArray d, IComplexNumber a,
                       int lda, int iseed );

    int slagsy( int matriint, int n, int k,
                       INDArray d, INDArray a, int lda,
                       int iseed );
    int dlagsy( int matriint, int n, int k,
                       INDArray d, INDArray a, int lda,
                       int iseed );
    int clagsy( int matriint, int n, int k,
                       INDArray d, IComplexNumber a,
                       int lda, int iseed );
    int zlagsy( int matriint, int n, int k,
                       INDArray d, IComplexNumber a,
                       int lda, int iseed );

    int slapmr( int matriint, int forwrd,
                       int m, int n, INDArray x, int ldx,
                       int k );
    int dlapmr( int matriint, int forwrd,
                       int m, int n, INDArray x,
                       int ldx, int k );
    int clapmr( int matriint, int forwrd,
                       int m, int n, IComplexNumber x,
                       int ldx, int k );
    int zlapmr( int matriint, int forwrd,
                       int m, int n, IComplexNumber x,
                       int ldx, int k );

    int slapmt( int matriint, int forwrd,
                       int m, int n, INDArray x, int ldx,
                       int k );
    int dlapmt( int matriint, int forwrd,
                       int m, int n, INDArray x,
                       int ldx, int k );
    int clapmt( int matriint, int forwrd,
                       int m, int n, IComplexNumber x,
                       int ldx, int k );
    int zlapmt( int matriint, int forwrd,
                       int m, int n, IComplexNumber x,
                       int ldx, int k );

    float slapy2( float x, float y );
    double dlapy2( double x, double y );

    float slapy3( float x, float y, float z );
    double dlapy3( double x, double y, double z );

    int slartgp( float f, float g, INDArray cs, INDArray sn, INDArray r );
    int dlartgp( double f, double g, INDArray cs, INDArray sn,
                        INDArray r );

    int slartgs( float x, float y, float sigma, INDArray cs,
                        INDArray sn );
    int dlartgs( double x, double y, double sigma, INDArray cs,
                        INDArray sn );


    //LAPACK 3.3.0
    int cbbcsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, int m,
                       int p, int q, INDArray theta, INDArray phi,
                       IComplexNumber u1, int ldu1,
                       IComplexNumber u2, int ldu2,
                       IComplexNumber v1t, int ldv1t,
                       IComplexNumber v2t, int ldv2t,
                       INDArray b11d, INDArray b11e, INDArray b12d, INDArray b12e,
                       INDArray b21d, INDArray b21e, INDArray b22d, INDArray b22e );
    int cbbcsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         int m, int p, int q,
                         INDArray theta, INDArray phi,
                         IComplexNumber u1, int ldu1,
                         IComplexNumber u2, int ldu2,
                         IComplexNumber v1t, int ldv1t,
                         IComplexNumber v2t, int ldv2t,
                         INDArray b11d, INDArray b11e, INDArray b12d,
                         INDArray b12e, INDArray b21d, INDArray b21e,
                         INDArray b22d, INDArray b22e, INDArray rwork,
                         int lrwork );
    int cheswapr( int matriint, char uplo, int n,
                         IComplexNumber a, int i1,
                         int i2 );
    int cheswapint( int matriint, char uplo, int n,
                           IComplexNumber a, int i1,
                           int i2 );
    int chetri2( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        int ipiv );
    int chetri2_work( int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             int ipiv,
                             IComplexNumber work, int lwork );
    int chetri2x( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, int nb );
    int chetri2int( int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           int ipiv,
                           IComplexNumber work, int nb );
    int chetrs2( int matriint, char uplo, int n,
                        int nrhs,  IComplexNumber a,
                        int lda,  int ipiv,
                        IComplexNumber b, int ldb );
    int chetrs2_work( int matriint, char uplo, int n,
                             int nrhs,  IComplexNumber a,
                             int lda,  int ipiv,
                             IComplexNumber b, int ldb,
                             IComplexNumber work );
    int csyconv( int matriint, char uplo, char way, int n,
                        IComplexNumber a, int lda,
                        int ipiv, IComplexNumber work  );
    int csyconint( int matriint, char uplo, char way,
                          int n, IComplexNumber a,
                          int lda,  int ipiv,
                          IComplexNumber work );
    int csyswapr( int matriint, char uplo, int n,
                         IComplexNumber a, int i1,
                         int i2 );
    int csyswapint( int matriint, char uplo, int n,
                           IComplexNumber a, int i1,
                           int i2 );
    int csytri2( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        int ipiv );
    int csytri2_work( int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             int ipiv,
                             IComplexNumber work, int lwork );
    int csytri2x( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, int nb );
    int csytri2int( int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           int ipiv,
                           IComplexNumber work, int nb );
    int csytrs2( int matriint, char uplo, int n,
                        int nrhs,  IComplexNumber a,
                        int lda,  int ipiv,
                        IComplexNumber b, int ldb );
    int csytrs2_work( int matriint, char uplo, int n,
                             int nrhs,  IComplexNumber a,
                             int lda,  int ipiv,
                             IComplexNumber b, int ldb,
                             IComplexNumber work );
    int cunbdb( int matriint, char trans, char signs,
                       int m, int p, int q,
                       IComplexNumber x11, int ldx11,
                       IComplexNumber x12, int ldx12,
                       IComplexNumber x21, int ldx21,
                       IComplexNumber x22, int ldx22,
                       INDArray theta, INDArray phi,
                       IComplexNumber taup1,
                       IComplexNumber taup2,
                       IComplexNumber tauq1,
                       IComplexNumber tauq2 );
    int cunbdint( int matriint, char trans, char signs,
                         int m, int p, int q,
                         IComplexNumber x11, int ldx11,
                         IComplexNumber x12, int ldx12,
                         IComplexNumber x21, int ldx21,
                         IComplexNumber x22, int ldx22,
                         INDArray theta, INDArray phi,
                         IComplexNumber taup1,
                         IComplexNumber taup2,
                         IComplexNumber tauq1,
                         IComplexNumber tauq2,
                         IComplexNumber work, int lwork );
    int cuncsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, char signs,
                       int m, int p, int q,
                       IComplexNumber x11, int ldx11,
                       IComplexNumber x12, int ldx12,
                       IComplexNumber x21, int ldx21,
                       IComplexNumber x22, int ldx22,
                       INDArray theta, IComplexNumber u1,
                       int ldu1, IComplexNumber u2,
                       int ldu2, IComplexNumber v1t,
                       int ldv1t, IComplexNumber v2t,
                       int ldv2t );
    int cuncsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         char signs, int m, int p,
                         int q, IComplexNumber x11,
                         int ldx11, IComplexNumber x12,
                         int ldx12, IComplexNumber x21,
                         int ldx21, IComplexNumber x22,
                         int ldx22, INDArray theta,
                         IComplexNumber u1, int ldu1,
                         IComplexNumber u2, int ldu2,
                         IComplexNumber v1t, int ldv1t,
                         IComplexNumber v2t, int ldv2t,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork );
    int cuncsd2by1( int matriint, char jobu1, char jobu2,
                           char jobv1t, int m, int p, int q,
                           IComplexNumber x11, int ldx11,
                           IComplexNumber x21, int ldx21,
                           IComplexNumber theta, IComplexNumber u1,
                           int ldu1, IComplexNumber u2,
                           int ldu2, IComplexNumber v1t, int ldv1t );
    int cuncsd2by1_work( int matriint, char jobu1, char jobu2,
                                char jobv1t, int m, int p,
                                int q, IComplexNumber x11, int ldx11,
                                IComplexNumber x21, int ldx21,
                                IComplexNumber theta, IComplexNumber u1,
                                int ldu1, IComplexNumber u2,
                                int ldu2, IComplexNumber v1t,
                                int ldv1t, IComplexNumber work,
                                int lwork, INDArray rwork, int lrwork,
                                int iwork );
    int dbbcsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, int m,
                       int p, int q, INDArray theta,
                       INDArray phi, INDArray u1, int ldu1, INDArray u2,
                       int ldu2, INDArray v1t, int ldv1t,
                       INDArray v2t, int ldv2t, INDArray b11d,
                       INDArray b11e, INDArray b12d, INDArray b12e,
                       INDArray b21d, INDArray b21e, INDArray b22d,
                       INDArray b22e );
    int dbbcsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         int m, int p, int q,
                         INDArray theta, INDArray phi, INDArray u1,
                         int ldu1, INDArray u2, int ldu2,
                         INDArray v1t, int ldv1t, INDArray v2t,
                         int ldv2t, INDArray b11d, INDArray b11e,
                         INDArray b12d, INDArray b12e, INDArray b21d,
                         INDArray b21e, INDArray b22d, INDArray b22e,
                         INDArray work, int lwork );
    int dorbdb( int matriint, char trans, char signs,
                       int m, int p, int q,
                       INDArray x11, int ldx11, INDArray x12,
                       int ldx12, INDArray x21, int ldx21,
                       INDArray x22, int ldx22, INDArray theta,
                       INDArray phi, INDArray taup1, INDArray taup2,
                       INDArray tauq1, INDArray tauq2 );
    int dorbdint( int matriint, char trans, char signs,
                         int m, int p, int q,
                         INDArray x11, int ldx11, INDArray x12,
                         int ldx12, INDArray x21, int ldx21,
                         INDArray x22, int ldx22, INDArray theta,
                         INDArray phi, INDArray taup1, INDArray taup2,
                         INDArray tauq1, INDArray tauq2, INDArray work,
                         int lwork );
    int dorcsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, char signs,
                       int m, int p, int q,
                       INDArray x11, int ldx11, INDArray x12,
                       int ldx12, INDArray x21, int ldx21,
                       INDArray x22, int ldx22, INDArray theta,
                       INDArray u1, int ldu1, INDArray u2,
                       int ldu2, INDArray v1t, int ldv1t,
                       INDArray v2t, int ldv2t );
    int dorcsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         char signs, int m, int p,
                         int q, INDArray x11, int ldx11,
                         INDArray x12, int ldx12, INDArray x21,
                         int ldx21, INDArray x22, int ldx22,
                         INDArray theta, INDArray u1, int ldu1,
                         INDArray u2, int ldu2, INDArray v1t,
                         int ldv1t, INDArray v2t, int ldv2t,
                         INDArray work, int lwork,
                         int iwork );
    int dorcsd2by1( int matriint, char jobu1, char jobu2,
                           char jobv1t, int m, int p, int q,
                           INDArray x11, int ldx11, INDArray x21, int ldx21,
                           INDArray theta, INDArray u1, int ldu1, INDArray u2,
                           int ldu2, INDArray v1t, int ldv1t);
    int dorcsd2by1_work( int matriint, char jobu1, char jobu2,
                                char jobv1t, int m, int p, int q,
                                INDArray x11, int ldx11, INDArray x21, int ldx21,
                                INDArray theta, INDArray u1, int ldu1, INDArray u2,
                                int ldu2, INDArray v1t, int ldv1t,
                                INDArray work, int lwork, int iwork );
    int dsyconv( int matriint, char uplo, char way, int n,
                        INDArray a, int lda,  int ipiv, INDArray work);
    int dsyconint( int matriint, char uplo, char way,
                          int n, INDArray a, int lda,
                          int ipiv, INDArray work );
    int dsyswapr( int matriint, char uplo, int n,
                         INDArray a, int i1, int i2 );
    int dsyswapint( int matriint, char uplo, int n,
                           INDArray a, int i1, int i2 );
    int dsytri2( int matriint, char uplo, int n,
                        INDArray a, int lda,  int ipiv );
    int dsytri2_work( int matriint, char uplo, int n,
                             INDArray a, int lda,
                             int ipiv,
                             IComplexNumber work, int lwork );
    int dsytri2x( int matriint, char uplo, int n,
                         INDArray a, int lda,  int ipiv,
                         int nb );
    int dsytri2int( int matriint, char uplo, int n,
                           INDArray a, int lda,
                           int ipiv, INDArray work,
                           int nb );
    int dsytrs2( int matriint, char uplo, int n,
                        int nrhs,  INDArray a, int lda,
                        int ipiv, INDArray b, int ldb );
    int dsytrs2_work( int matriint, char uplo, int n,
                             int nrhs,  INDArray a,
                             int lda,  int ipiv,
                             INDArray b, int ldb, INDArray work );
    int sbbcsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, int m,
                       int p, int q, INDArray theta, INDArray phi,
                       INDArray u1, int ldu1, INDArray u2,
                       int ldu2, INDArray v1t, int ldv1t,
                       INDArray v2t, int ldv2t, INDArray b11d,
                       INDArray b11e, INDArray b12d, INDArray b12e, INDArray b21d,
                       INDArray b21e, INDArray b22d, INDArray b22e );
    int sbbcsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         int m, int p, int q,
                         INDArray theta, INDArray phi, INDArray u1,
                         int ldu1, INDArray u2, int ldu2,
                         INDArray v1t, int ldv1t, INDArray v2t,
                         int ldv2t, INDArray b11d, INDArray b11e,
                         INDArray b12d, INDArray b12e, INDArray b21d,
                         INDArray b21e, INDArray b22d, INDArray b22e,
                         INDArray work, int lwork );
    int sorbdb( int matriint, char trans, char signs,
                       int m, int p, int q, INDArray x11,
                       int ldx11, INDArray x12, int ldx12,
                       INDArray x21, int ldx21, INDArray x22,
                       int ldx22, INDArray theta, INDArray phi,
                       INDArray taup1, INDArray taup2, INDArray tauq1,
                       INDArray tauq2 );
    int sorbdint( int matriint, char trans, char signs,
                         int m, int p, int q,
                         INDArray x11, int ldx11, INDArray x12,
                         int ldx12, INDArray x21, int ldx21,
                         INDArray x22, int ldx22, INDArray theta,
                         INDArray phi, INDArray taup1, INDArray taup2,
                         INDArray tauq1, INDArray tauq2, INDArray work,
                         int lwork );
    int sorcsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, char signs,
                       int m, int p, int q, INDArray x11,
                       int ldx11, INDArray x12, int ldx12,
                       INDArray x21, int ldx21, INDArray x22,
                       int ldx22, INDArray theta, INDArray u1,
                       int ldu1, INDArray u2, int ldu2,
                       INDArray v1t, int ldv1t, INDArray v2t,
                       int ldv2t );
    int sorcsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         char signs, int m, int p,
                         int q, INDArray x11, int ldx11,
                         INDArray x12, int ldx12, INDArray x21,
                         int ldx21, INDArray x22, int ldx22,
                         INDArray theta, INDArray u1, int ldu1,
                         INDArray u2, int ldu2, INDArray v1t,
                         int ldv1t, INDArray v2t, int ldv2t,
                         INDArray work, int lwork,
                         int iwork );
    int sorcsd2by1( int matriint, char jobu1, char jobu2,
                           char jobv1t, int m, int p, int q,
                           INDArray x11, int ldx11, INDArray x21, int ldx21,
                           INDArray theta, INDArray u1, int ldu1, INDArray u2,
                           int ldu2, INDArray v1t, int ldv1t);
    int sorcsd2by1_work( int matriint, char jobu1, char jobu2,
                                char jobv1t, int m, int p, int q,
                                INDArray x11, int ldx11, INDArray x21, int ldx21,
                                INDArray theta, INDArray u1, int ldu1, INDArray u2,
                                int ldu2, INDArray v1t, int ldv1t,
                                INDArray work, int lwork, int iwork );
    int ssyconv( int matriint, char uplo, char way, int n,
                        INDArray a, int lda,  int ipiv, INDArray work );
    int ssyconint( int matriint, char uplo, char way,
                          int n, INDArray a, int lda,
                          int ipiv, INDArray work );
    int ssyswapr( int matriint, char uplo, int n,
                         INDArray a, int i1, int i2 );
    int ssyswapint( int matriint, char uplo, int n,
                           INDArray a, int i1, int i2 );
    int ssytri2( int matriint, char uplo, int n, INDArray a,
                        int lda,  int ipiv );
    int ssytri2_work( int matriint, char uplo, int n,
                             INDArray a, int lda,
                             int ipiv,
                             IComplexNumber work, int lwork );
    int ssytri2x( int matriint, char uplo, int n,
                         INDArray a, int lda,  int ipiv,
                         int nb );
    int ssytri2int( int matriint, char uplo, int n,
                           INDArray a, int lda,
                           int ipiv, INDArray work,
                           int nb );
    int ssytrs2( int matriint, char uplo, int n,
                        int nrhs,  INDArray a, int lda,
                        int ipiv, INDArray b, int ldb );
    int ssytrs2_work( int matriint, char uplo, int n,
                             int nrhs,  INDArray a,
                             int lda,  int ipiv,
                             INDArray b, int ldb, INDArray work );
    int zbbcsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, int m,
                       int p, int q, INDArray theta,
                       INDArray phi, IComplexNumber u1,
                       int ldu1, IComplexNumber u2,
                       int ldu2, IComplexNumber v1t,
                       int ldv1t, IComplexNumber v2t,
                       int ldv2t, INDArray b11d, INDArray b11e,
                       INDArray b12d, INDArray b12e, INDArray b21d,
                       INDArray b21e, INDArray b22d, INDArray b22e );
    int zbbcsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         int m, int p, int q,
                         INDArray theta, INDArray phi,
                         IComplexNumber u1, int ldu1,
                         IComplexNumber u2, int ldu2,
                         IComplexNumber v1t, int ldv1t,
                         IComplexNumber v2t, int ldv2t,
                         INDArray b11d, INDArray b11e, INDArray b12d,
                         INDArray b12e, INDArray b21d, INDArray b21e,
                         INDArray b22d, INDArray b22e, INDArray rwork,
                         int lrwork );
    int zheswapr( int matriint, char uplo, int n,
                         IComplexNumber a, int i1,
                         int i2 );
    int zheswapint( int matriint, char uplo, int n,
                           IComplexNumber a, int i1,
                           int i2 );
    int zhetri2( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        int ipiv );
    int zhetri2_work( int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             int ipiv,
                             IComplexNumber work, int lwork );
    int zhetri2x( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, int nb );
    int zhetri2int( int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           int ipiv,
                           IComplexNumber work, int nb );
    int zhetrs2( int matriint, char uplo, int n,
                        int nrhs,  IComplexNumber a,
                        int lda,  int ipiv,
                        IComplexNumber b, int ldb );
    int zhetrs2_work( int matriint, char uplo, int n,
                             int nrhs,  IComplexNumber a,
                             int lda,  int ipiv,
                             IComplexNumber b, int ldb,
                             IComplexNumber work );
    int zsyconv( int matriint, char uplo, char way, int n,
                        IComplexNumber a, int lda,
                        int ipiv, IComplexNumber work );
    int zsyconint( int matriint, char uplo, char way,
                          int n, IComplexNumber a,
                          int lda,  int ipiv,
                          IComplexNumber work );
    int zsyswapr( int matriint, char uplo, int n,
                         IComplexNumber a, int i1,
                         int i2 );
    int zsyswapint( int matriint, char uplo, int n,
                           IComplexNumber a, int i1,
                           int i2 );
    int zsytri2( int matriint, char uplo, int n,
                        IComplexNumber a, int lda,
                        int ipiv );
    int zsytri2_work( int matriint, char uplo, int n,
                             IComplexNumber a, int lda,
                             int ipiv,
                             IComplexNumber work, int lwork );
    int zsytri2x( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv, int nb );
    int zsytri2int( int matriint, char uplo, int n,
                           IComplexNumber a, int lda,
                           int ipiv,
                           IComplexNumber work, int nb );
    int zsytrs2( int matriint, char uplo, int n,
                        int nrhs,  IComplexNumber a,
                        int lda,  int ipiv,
                        IComplexNumber b, int ldb );
    int zsytrs2_work( int matriint, char uplo, int n,
                             int nrhs,  IComplexNumber a,
                             int lda,  int ipiv,
                             IComplexNumber b, int ldb,
                             IComplexNumber work );
    int zunbdb( int matriint, char trans, char signs,
                       int m, int p, int q,
                       IComplexNumber x11, int ldx11,
                       IComplexNumber x12, int ldx12,
                       IComplexNumber x21, int ldx21,
                       IComplexNumber x22, int ldx22,
                       INDArray theta, INDArray phi,
                       IComplexNumber taup1,
                       IComplexNumber taup2,
                       IComplexNumber tauq1,
                       IComplexNumber tauq2 );
    int zunbdint( int matriint, char trans, char signs,
                         int m, int p, int q,
                         IComplexNumber x11, int ldx11,
                         IComplexNumber x12, int ldx12,
                         IComplexNumber x21, int ldx21,
                         IComplexNumber x22, int ldx22,
                         INDArray theta, INDArray phi,
                         IComplexNumber taup1,
                         IComplexNumber taup2,
                         IComplexNumber tauq1,
                         IComplexNumber tauq2,
                         IComplexNumber work, int lwork );
    int zuncsd( int matriint, char jobu1, char jobu2,
                       char jobv1t, char jobv2t, char trans, char signs,
                       int m, int p, int q,
                       IComplexNumber x11, int ldx11,
                       IComplexNumber x12, int ldx12,
                       IComplexNumber x21, int ldx21,
                       IComplexNumber x22, int ldx22,
                       INDArray theta, IComplexNumber u1,
                       int ldu1, IComplexNumber u2,
                       int ldu2, IComplexNumber v1t,
                       int ldv1t, IComplexNumber v2t,
                       int ldv2t );
    int zuncsint( int matriint, char jobu1, char jobu2,
                         char jobv1t, char jobv2t, char trans,
                         char signs, int m, int p,
                         int q, IComplexNumber x11,
                         int ldx11, IComplexNumber x12,
                         int ldx12, IComplexNumber x21,
                         int ldx21, IComplexNumber x22,
                         int ldx22, INDArray theta,
                         IComplexNumber u1, int ldu1,
                         IComplexNumber u2, int ldu2,
                         IComplexNumber v1t, int ldv1t,
                         IComplexNumber v2t, int ldv2t,
                         IComplexNumber work, int lwork,
                         INDArray rwork, int lrwork,
                         int iwork );
    int zuncsd2by1( int matriint, char jobu1, char jobu2,
                           char jobv1t, int m, int p, int q,
                           IComplexNumber x11, int ldx11,
                           IComplexNumber x21, int ldx21,
                           IComplexNumber theta, IComplexNumber u1,
                           int ldu1, IComplexNumber u2,
                           int ldu2, IComplexNumber v1t, int ldv1t );
    int zuncsd2by1_work( int matriint, char jobu1, char jobu2,
                                char jobv1t, int m, int p,
                                int q, IComplexNumber x11, int ldx11,
                                IComplexNumber x21, int ldx21,
                                IComplexNumber theta, IComplexNumber u1,
                                int ldu1, IComplexNumber u2,
                                int ldu2, IComplexNumber v1t,
                                int ldv1t, IComplexNumber work,
                                int lwork, INDArray rwork, int lrwork,
                                int iwork );

    //LAPACK 3.4.0
    int sgemqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int nb,  INDArray v, int ldv,
                        INDArray t, int ldt, INDArray c,
                        int ldc );
    int dgemqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int nb,  INDArray v, int ldv,
                        INDArray t, int ldt, INDArray c,
                        int ldc );
    int cgemqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int nb,  IComplexNumber v,
                        int ldv,  IComplexNumber t,
                        int ldt, IComplexNumber c,
                        int ldc );
    int zgemqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int nb,  IComplexNumber v,
                        int ldv,  IComplexNumber t,
                        int ldt, IComplexNumber c,
                        int ldc );

    int sgeqrt( int matriint, int m, int n,
                       int nb, INDArray a, int lda, INDArray t,
                       int ldt );
    int dgeqrt( int matriint, int m, int n,
                       int nb, INDArray a, int lda, INDArray t,
                       int ldt );
    int cgeqrt( int matriint, int m, int n,
                       int nb, IComplexNumber a,
                       int lda, IComplexNumber t,
                       int ldt );
    int zgeqrt( int matriint, int m, int n,
                       int nb, IComplexNumber a,
                       int lda, IComplexNumber t,
                       int ldt );

    int sgeqrt2( int matriint, int m, int n,
                        INDArray a, int lda, INDArray t,
                        int ldt );
    int dgeqrt2( int matriint, int m, int n,
                        INDArray a, int lda, INDArray t,
                        int ldt );
    int cgeqrt2( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber t, int ldt );
    int zgeqrt2( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber t, int ldt );

    int sgeqrt3( int matriint, int m, int n,
                        INDArray a, int lda, INDArray t,
                        int ldt );
    int dgeqrt3( int matriint, int m, int n,
                        INDArray a, int lda, INDArray t,
                        int ldt );
    int cgeqrt3( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber t, int ldt );
    int zgeqrt3( int matriint, int m, int n,
                        IComplexNumber a, int lda,
                        IComplexNumber t, int ldt );

    int stpmqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int l, int nb,  INDArray v,
                        int ldv,  INDArray t, int ldt,
                        INDArray a, int lda, INDArray b,
                        int ldb );
    int dtpmqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int l, int nb,  INDArray v,
                        int ldv,  INDArray t, int ldt,
                        INDArray a, int lda, INDArray b,
                        int ldb );
    int ctpmqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int l, int nb,
                        IComplexNumber v, int ldv,
                        IComplexNumber t, int ldt,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb );
    int ztpmqrt( int matriint, char side, char trans,
                        int m, int n, int k,
                        int l, int nb,
                        IComplexNumber v, int ldv,
                        IComplexNumber t, int ldt,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb );

    int stpqrt( int matriint, int m, int n,
                       int l, int nb, INDArray a,
                       int lda, INDArray b, int ldb, INDArray t,
                       int ldt );

    int dtpqrt( int matriint, int m, int n,
                       int l, int nb, INDArray a,
                       int lda, INDArray b, int ldb, INDArray t,
                       int ldt );
    int ctpqrt( int matriint, int m, int n,
                       int l, int nb,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber t, int ldt );
    int ztpqrt( int matriint, int m, int n,
                       int l, int nb,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb,
                       IComplexNumber t, int ldt );

    int stpqrt2( int matriint,
                        int m, int n, int l,
                        INDArray a, int lda,
                        INDArray b, int ldb,
                        INDArray t, int ldt );
    int dtpqrt2( int matriint,
                        int m, int n, int l,
                        INDArray a, int lda,
                        INDArray b, int ldb,
                        INDArray t, int ldt );
    int ctpqrt2( int matriint,
                        int m, int n, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber t, int ldt );
    int ztpqrt2( int matriint,
                        int m, int n, int l,
                        IComplexNumber a, int lda,
                        IComplexNumber b, int ldb,
                        IComplexNumber t, int ldt );

    int stprfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k, int l,  INDArray v,
                       int ldv,  INDArray t, int ldt,
                       INDArray a, int lda, INDArray b, int ldb );
    int dtprfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k, int l,  INDArray v,
                       int ldv,  INDArray t, int ldt,
                       INDArray a, int lda, INDArray b, int ldb );
    int ctprfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k, int l,
                       IComplexNumber v, int ldv,
                       IComplexNumber t, int ldt,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb );
    int ztprfb( int matriint, char side, char trans, char direct,
                       char storev, int m, int n,
                       int k, int l,
                       IComplexNumber v, int ldv,
                       IComplexNumber t, int ldt,
                       IComplexNumber a, int lda,
                       IComplexNumber b, int ldb );

    int sgemqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int nb,  INDArray v, int ldv,
                          INDArray t, int ldt, INDArray c,
                          int ldc, INDArray work );
    int dgemqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int nb,  INDArray v, int ldv,
                          INDArray t, int ldt, INDArray c,
                          int ldc, INDArray work );
    int cgemqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int nb,  IComplexNumber v,
                          int ldv,  IComplexNumber t,
                          int ldt, IComplexNumber c,
                          int ldc, IComplexNumber work );
    int zgemqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int nb,  IComplexNumber v,
                          int ldv,  IComplexNumber t,
                          int ldt, IComplexNumber c,
                          int ldc, IComplexNumber work );

    int sgeqrint( int matriint, int m, int n,
                         int nb, INDArray a, int lda,
                         INDArray t, int ldt, INDArray work );
    int dgeqrint( int matriint, int m, int n,
                         int nb, INDArray a, int lda,
                         INDArray t, int ldt, INDArray work );
    int cgeqrint( int matriint, int m, int n,
                         int nb, IComplexNumber a,
                         int lda, IComplexNumber t,
                         int ldt, IComplexNumber work );
    int zgeqrint( int matriint, int m, int n,
                         int nb, IComplexNumber a,
                         int lda, IComplexNumber t,
                         int ldt, IComplexNumber work );

    int sgeqrt2_work( int matriint, int m, int n,
                             INDArray a, int lda, INDArray t,
                             int ldt );
    int dgeqrt2_work( int matriint, int m, int n,
                             INDArray a, int lda, INDArray t,
                             int ldt );
    int cgeqrt2_work( int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             IComplexNumber t, int ldt );
    int zgeqrt2_work( int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             IComplexNumber t, int ldt );

    int sgeqrt3_work( int matriint, int m, int n,
                             INDArray a, int lda, INDArray t,
                             int ldt );
    int dgeqrt3_work( int matriint, int m, int n,
                             INDArray a, int lda, INDArray t,
                             int ldt );
    int cgeqrt3_work( int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             IComplexNumber t, int ldt );
    int zgeqrt3_work( int matriint, int m, int n,
                             IComplexNumber a, int lda,
                             IComplexNumber t, int ldt );

    int stpmqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int l, int nb,  INDArray v,
                          int ldv,  INDArray t, int ldt,
                          INDArray a, int lda, INDArray b,
                          int ldb, INDArray work );
    int dtpmqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int l, int nb,  INDArray v,
                          int ldv,  INDArray t,
                          int ldt, INDArray a, int lda,
                          INDArray b, int ldb, INDArray work );
    int ctpmqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int l, int nb,
                          IComplexNumber v, int ldv,
                          IComplexNumber t, int ldt,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber work );
    int ztpmqrint( int matriint, char side, char trans,
                          int m, int n, int k,
                          int l, int nb,
                          IComplexNumber v, int ldv,
                          IComplexNumber t, int ldt,
                          IComplexNumber a, int lda,
                          IComplexNumber b, int ldb,
                          IComplexNumber work );

    int stpqrint( int matriint, int m, int n,
                         int l, int nb, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray t, int ldt, INDArray work );
    int dtpqrint( int matriint, int m, int n,
                         int l, int nb, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray t, int ldt, INDArray work );
    int ctpqrint( int matriint, int m, int n,
                         int l, int nb,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber t, int ldt,
                         IComplexNumber work );
    int ztpqrint( int matriint, int m, int n,
                         int l, int nb,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber t, int ldt,
                         IComplexNumber work );

    int stpqrt2_work( int matriint,
                             int m, int n, int l,
                             INDArray a, int lda,
                             INDArray b, int ldb,
                             INDArray t, int ldt );
    int dtpqrt2_work( int matriint,
                             int m, int n, int l,
                             INDArray a, int lda,
                             INDArray b, int ldb,
                             INDArray t, int ldt );
    int ctpqrt2_work( int matriint,
                             int m, int n, int l,
                             IComplexNumber a, int lda,
                             IComplexNumber b, int ldb,
                             IComplexNumber t, int ldt );
    int ztpqrt2_work( int matriint,
                             int m, int n, int l,
                             IComplexNumber a, int lda,
                             IComplexNumber b, int ldb,
                             IComplexNumber t, int ldt );

    int stprfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k, int l,
                         INDArray v, int ldv,  INDArray t,
                         int ldt, INDArray a, int lda,
                         INDArray b, int ldb,  INDArray work,
                         int ldwork );
    int dtprfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k, int l,
                         INDArray v, int ldv,
                         INDArray t, int ldt, INDArray a,
                         int lda, INDArray b, int ldb,
                         INDArray work, int ldwork );
    int ctprfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k, int l,
                         IComplexNumber v, int ldv,
                         IComplexNumber t, int ldt,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber work, int ldwork );
    int ztprfint( int matriint, char side, char trans,
                         char direct, char storev, int m,
                         int n, int k, int l,
                         IComplexNumber v, int ldv,
                         IComplexNumber t, int ldt,
                         IComplexNumber a, int lda,
                         IComplexNumber b, int ldb,
                         IComplexNumber work, int ldwork );
    //LAPACK 3.X.X
    int ssysint( int matriint, char uplo, int n,
                        int nrhs, INDArray a, int lda,
                        int ipiv, INDArray b, int ldb );
    int dsysint( int matriint, char uplo, int n,
                        int nrhs, INDArray a, int lda,
                        int ipiv, INDArray b, int ldb );
    int csysint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, int ipiv,
                        IComplexNumber b, int ldb );
    int zsysint( int matriint, char uplo, int n,
                        int nrhs, IComplexNumber a,
                        int lda, int ipiv,
                        IComplexNumber b, int ldb );

    int ssytrint( int matriint, char uplo, int n, INDArray a,
                         int lda, int ipiv );
    int dsytrint( int matriint, char uplo, int n, INDArray a,
                         int lda, int ipiv );
    int csytrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv );
    int zsytrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv );

    int ssytrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a, int lda,
                         int ipiv, INDArray b, int ldb );
    int dsytrint( int matriint, char uplo, int n,
                         int nrhs,  INDArray a, int lda,
                         int ipiv, INDArray b, int ldb );
    int csytrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  int ipiv,
                         IComplexNumber b, int ldb );
    int zsytrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  int ipiv,
                         IComplexNumber b, int ldb );

    int chetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv );
    int zhetrint( int matriint, char uplo, int n,
                         IComplexNumber a, int lda,
                         int ipiv );

    int chetrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  int ipiv,
                         IComplexNumber b, int ldb );
    int zhetrint( int matriint, char uplo, int n,
                         int nrhs,  IComplexNumber a,
                         int lda,  int ipiv,
                         IComplexNumber b, int ldb );

    int csyr( int matriint, char uplo, int n,
                     IComplexNumber alpha,
                     IComplexNumber x, int incx,
                     IComplexNumber a, int lda );
    int zsyr( int matriint, char uplo, int n,
                     IComplexNumber alpha,
                     IComplexNumber x, int incx,
                     IComplexNumber a, int lda );

    int ssysint_work( int matriint, char uplo, int n,
                             int nrhs, INDArray a, int lda,
                             int ipiv, INDArray b, int ldb,
                             INDArray work, int lwork );
    int dsysint_work( int matriint, char uplo, int n,
                             int nrhs, INDArray a, int lda,
                             int ipiv, INDArray b, int ldb,
                             INDArray work, int lwork );
    int csysint_work( int matriint, char uplo, int n,
                             int nrhs, IComplexNumber a,
                             int lda, int ipiv,
                             IComplexNumber b, int ldb,
                             IComplexNumber work,
                             int lwork );
    int zsysint_work( int matriint, char uplo, int n,
                             int nrhs, IComplexNumber a,
                             int lda, int ipiv,
                             IComplexNumber b, int ldb,
                             IComplexNumber work,
                             int lwork );

    int ssytrint_work( int matriint, char uplo, int n,
                              INDArray a, int lda, int ipiv,
                              INDArray work, int lwork );
    int dsytrint_work( int matriint, char uplo, int n,
                              INDArray a, int lda, int ipiv,
                              INDArray work, int lwork );
    int csytrint_work( int matriint, char uplo, int n,
                              IComplexNumber a, int lda,
                              int ipiv, IComplexNumber work,
                              int lwork );
    int zsytrint_work( int matriint, char uplo, int n,
                              IComplexNumber a, int lda,
                              int ipiv, IComplexNumber work,
                              int lwork );

    int ssytrint_work( int matriint, char uplo, int n,
                              int nrhs,  INDArray a, int lda,
                              int ipiv, INDArray b,
                              int ldb );
    int dsytrint_work( int matriint, char uplo, int n,
                              int nrhs,  INDArray a,
                              int lda,  int ipiv,
                              INDArray b, int ldb );
    int csytrint_work( int matriint, char uplo, int n,
                              int nrhs,  IComplexNumber a,
                              int lda,  int ipiv,
                              IComplexNumber b, int ldb );
    int zsytrint_work( int matriint, char uplo, int n,
                              int nrhs,  IComplexNumber a,
                              int lda,  int ipiv,
                              IComplexNumber b, int ldb );

    int chetrint_work( int matriint, char uplo, int n,
                              IComplexNumber a, int lda,
                              int ipiv, IComplexNumber work,
                              int lwork );
    int zhetrint_work( int matriint, char uplo, int n,
                              IComplexNumber a, int lda,
                              int ipiv, IComplexNumber work,
                              int lwork );

    int chetrint_work( int matriint, char uplo, int n,
                              int nrhs,  IComplexNumber a,
                              int lda,  int ipiv,
                              IComplexNumber b, int ldb );
    int zhetrint_work( int matriint, char uplo, int n,
                              int nrhs,  IComplexNumber a,
                              int lda,  int ipiv,
                              IComplexNumber b, int ldb );


    int csyint( int matriint, char uplo, int n,
                       IComplexNumber alpha,
                       IComplexNumber x,
                       int incx, IComplexNumber a,
                       int lda );
    int zsyint( int matriint, char uplo, int n,
                       IComplexNumber alpha,
                       IComplexNumber x,
                       int incx, IComplexNumber a,
                       int lda );

    void dgetrf( int m, int n, INDArray a, int lda,
                 int ipiv, int info );
    void cgetrf( int m, int n, IComplexNumber a,
                 int lda, int ipiv, int info );
    void zgetrf( int m, int n, IComplexNumber a,
                 int lda, int ipiv, int info );
    void sgetrf2( int m, int n, INDArray a, int lda,
                  int ipiv, int info );
    void dgetrf2( int m, int n, INDArray a, int lda,
                  int ipiv, int info );
    void cgetrf2( int m, int n, IComplexNumber a,
                  int lda, int ipiv, int info );
    void zgetrf2( int m, int n, IComplexNumber a,
                  int lda, int ipiv, int info );
    void sgbtrf( int m, int n, int kl,
                 int ku, INDArray ab, int ldab,
                 int ipiv, int info );
    void dgbtrf( int m, int n, int kl,
                 int ku, INDArray ab, int ldab,
                 int ipiv, int info );
    void cgbtrf( int m, int n, int kl,
                 int ku, IComplexNumber ab, int ldab,
                 int ipiv, int info );
    void zgbtrf( int m, int n, int kl,
                 int ku, IComplexNumber ab, int ldab,
                 int ipiv, int info );
    void sgttrf( int n, INDArray dl, INDArray d, INDArray du, INDArray du2,
                 int ipiv, int info );
    void dgttrf( int n, INDArray dl, INDArray d, INDArray du,
                 INDArray du2, int ipiv, int info );
    void cgttrf( int n, IComplexNumber dl,
                 IComplexNumber d, IComplexNumber du,
                 IComplexNumber du2, int ipiv,
                 int info );
    void zgttrf( int n, IComplexNumber dl,
                 IComplexNumber d, IComplexNumber du,
                 IComplexNumber du2, int ipiv,
                 int info );
    void spotrf2( String uplo, int n, INDArray a, int lda,
                  int info );
    void dpotrf2( String uplo, int n, INDArray a, int lda,
                  int info );
    void cpotrf2( String uplo, int n, IComplexNumber a,
                  int lda, int info );
    void zpotrf2( String uplo, int n, IComplexNumber a,
                  int lda, int info );
    void spotrf( String uplo, int n, INDArray a, int lda,
                 int info );
    void dpotrf( String uplo, int n, INDArray a, int lda,
                 int info );
    void cpotrf( String uplo, int n, IComplexNumber a,
                 int lda, int info );
    void zpotrf( String uplo, int n, IComplexNumber a,
                 int lda, int info );
    void dpstrf( String uplo, int n, INDArray a, int lda,
                 int piv, int rank, INDArray tol,
                 INDArray work, int info );
    void spstrf( String uplo, int n, INDArray a, int lda,
                 int piv, int rank, INDArray tol, INDArray work,
                 int info );
    void zpstrf( String uplo, int n, IComplexNumber a,
                 int lda, int piv, int rank,
                 INDArray tol, INDArray work, int info );
    void cpstrf( String uplo, int n, IComplexNumber a,
                 int lda, int piv, int rank,
                 INDArray tol, INDArray work, int info );
    void dpftrf( String transr, String uplo, int n, INDArray a,
                 int info );
    void spftrf( String transr, String uplo, int n, INDArray a,
                 int info );
    void zpftrf( String transr, String uplo, int n,
                 IComplexNumber a, int info );
    void cpftrf( String transr, String uplo, int n,
                 IComplexNumber a, int info );
    void spptrf( String uplo, int n, INDArray ap, int info );
    void dpptrf( String uplo, int n, INDArray ap, int info );
    void cpptrf( String uplo, int n, IComplexNumber ap,
                 int info );
    void zpptrf( String uplo, int n, IComplexNumber ap,
                 int info );
    void spbtrf( String uplo, int n, int kd, INDArray ab,
                 int ldab, int info );
    void dpbtrf( String uplo, int n, int kd, INDArray ab,
                 int ldab, int info );
    void cpbtrf( String uplo, int n, int kd,
                 IComplexNumber ab, int ldab,
                 int info );
    void zpbtrf( String uplo, int n, int kd,
                 IComplexNumber ab, int ldab,
                 int info );
    void spttrf( int n, INDArray d, INDArray e, int info );
    void dpttrf( int n, INDArray d, INDArray e, int info );
    void cpttrf( int n, INDArray d, IComplexNumber e,
                 int info );
    void zpttrf( int n, INDArray d, IComplexNumber e,
                 int info );
    void ssytrf( String uplo, int n, INDArray a, int lda,
                 int ipiv, INDArray work, int lwork,
                 int info );
    void dsytrf( String uplo, int n, INDArray a, int lda,
                 int ipiv, INDArray work, int lwork,
                 int info );
    void csytrf( String uplo, int n, IComplexNumber a,
                 int lda, int ipiv,
                 IComplexNumber work, int lwork,
                 int info );
    void zsytrf( String uplo, int n, IComplexNumber a,
                 int lda, int ipiv,
                 IComplexNumber work, int lwork,
                 int info );
    void chetrf( String uplo, int n, IComplexNumber a,
                 int lda, int ipiv,
                 IComplexNumber work, int lwork,
                 int info );
    void zhetrf( String uplo, int n, IComplexNumber a,
                 int lda, int ipiv,
                 IComplexNumber work, int lwork,
                 int info );
    void ssptrf( String uplo, int n, INDArray ap, int ipiv,
                 int info );
    void dsptrf( String uplo, int n, INDArray ap, int ipiv,
                 int info );
    void csptrf( String uplo, int n, IComplexNumber ap,
                 int ipiv, int info );
    void zsptrf( String uplo, int n, IComplexNumber ap,
                 int ipiv, int info );
    void chptrf( String uplo, int n, IComplexNumber ap,
                 int ipiv, int info );
    void zhptrf( String uplo, int n, IComplexNumber ap,
                 int ipiv, int info );
    void sgetrs( String trans, int n, int nrhs,
                 INDArray a, int lda,  int ipiv,
                 INDArray b, int ldb, int info );
    void dgetrs( String trans, int n, int nrhs,
                 INDArray a, int lda,  int ipiv,
                 INDArray b, int ldb, int info );
    void cgetrs( String trans, int n, int nrhs,
                 IComplexNumber a, int lda,
                 int ipiv, IComplexNumber b,
                 int ldb, int info );
    void zgetrs( String trans, int n, int nrhs,
                 IComplexNumber a, int lda,
                 int ipiv, IComplexNumber b,
                 int ldb, int info );
    void sgbtrs( String trans, int n, int kl, int ku,
                 int nrhs,  INDArray ab, int ldab,
                 int ipiv, INDArray b, int ldb,
                 int info );
    void dgbtrs( String trans, int n, int kl, int ku,
                 int nrhs,  INDArray ab, int ldab,
                 int ipiv, INDArray b, int ldb,
                 int info );
    void cgbtrs( String trans, int n, int kl, int ku,
                 int nrhs,  IComplexNumber ab,
                 int ldab,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void zgbtrs( String trans, int n, int kl, int ku,
                 int nrhs,  IComplexNumber ab,
                 int ldab,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void sgttrs( String trans, int n, int nrhs,
                 INDArray dl,  INDArray d,  INDArray du,
                 INDArray du2,  int ipiv, INDArray b,
                 int ldb, int info );
    void dgttrs( String trans, int n, int nrhs,
                 INDArray dl,  INDArray d,  INDArray du,
                 INDArray du2,  int ipiv, INDArray b,
                 int ldb, int info );
    void cgttrs( String trans, int n, int nrhs,
                 IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du,
                 IComplexNumber du2,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void zgttrs( String trans, int n, int nrhs,
                 IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du,
                 IComplexNumber du2,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void spotrs( String uplo, int n, int nrhs,  INDArray a,
                 int lda, INDArray b, int ldb,
                 int info );
    void dpotrs( String uplo, int n, int nrhs,
                 INDArray a, int lda, INDArray b,
                 int ldb, int info );
    void cpotrs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 int info );
    void zpotrs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 int info );
    void dpftrs( String transr, String uplo, int n, int nrhs,
                 INDArray a, INDArray b, int ldb,
                 int info );
    void spftrs( String transr, String uplo, int n, int nrhs,
                 INDArray a, INDArray b, int ldb,
                 int info );
    void zpftrs( String transr, String uplo, int n, int nrhs,
                 IComplexNumber a, IComplexNumber b,
                 int ldb, int info );
    void cpftrs( String transr, String uplo, int n, int nrhs,
                 IComplexNumber a, IComplexNumber b,
                 int ldb, int info );
    void spptrs( String uplo, int n, int nrhs,
                 INDArray ap, INDArray b, int ldb,
                 int info );
    void dpptrs( String uplo, int n, int nrhs,
                 INDArray ap, INDArray b, int ldb,
                 int info );
    void cpptrs( String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber b,
                 int ldb, int info );
    void zpptrs( String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber b,
                 int ldb, int info );
    void spbtrs( String uplo, int n, int kd, int nrhs,
                 INDArray ab, int ldab, INDArray b,
                 int ldb, int info );
    void dpbtrs( String uplo, int n, int kd, int nrhs,
                 INDArray ab, int ldab, INDArray b,
                 int ldb, int info );
    void cpbtrs( String uplo, int n, int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber b, int ldb,
                 int info );
    void zpbtrs( String uplo, int n, int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber b, int ldb,
                 int info );
    void spttrs( int n, int nrhs,  INDArray d,
                 INDArray e, INDArray b, int ldb,
                 int info );
    void dpttrs( int n, int nrhs,  INDArray d,
                 INDArray e, INDArray b, int ldb,
                 int info );
    void cpttrs( String uplo, int n, int nrhs,  INDArray d,
                 IComplexNumber e, IComplexNumber b,
                 int ldb, int info );
    void zpttrs( String uplo, int n, int nrhs,
                 INDArray d,  IComplexNumber e,
                 IComplexNumber b, int ldb,
                 int info );
    void ssytrs( String uplo, int n, int nrhs,  INDArray a,
                 int lda,  int ipiv, INDArray b,
                 int ldb, int info );
    void dsytrs( String uplo, int n, int nrhs,
                 INDArray a, int lda,  int ipiv,
                 INDArray b, int ldb, int info );
    void csytrs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 int ipiv, IComplexNumber b,
                 int ldb, int info );
    void zsytrs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 int ipiv, IComplexNumber b,
                 int ldb, int info );
    void chetrs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 int ipiv, IComplexNumber b,
                 int ldb, int info );
    void zhetrs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 int ipiv, IComplexNumber b,
                 int ldb, int info );
    void ssptrs( String uplo, int n, int nrhs,
                 INDArray ap,  int ipiv, INDArray b,
                 int ldb, int info );
    void dsptrs( String uplo, int n, int nrhs,
                 INDArray ap,  int ipiv, INDArray b,
                 int ldb, int info );
    void csptrs( String uplo, int n, int nrhs,
                 IComplexNumber ap,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void zsptrs( String uplo, int n, int nrhs,
                 IComplexNumber ap,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void chptrs( String uplo, int n, int nrhs,
                 IComplexNumber ap,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void zhptrs( String uplo, int n, int nrhs,
                 IComplexNumber ap,  int ipiv,
                 IComplexNumber b, int ldb,
                 int info );
    void strtrs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray a, int lda, INDArray b,
                 int ldb, int info );
    void dtrtrs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray a, int lda,
                 INDArray b, int ldb, int info );
    void ctrtrs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber a,
                 int lda, IComplexNumber b, int ldb,
                 int info );
    void ztrtrs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber a,
                 int lda, IComplexNumber b, int ldb,
                 int info );
    void stptrs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray ap, INDArray b,
                 int ldb, int info );
    void dtptrs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray ap, INDArray b,
                 int ldb, int info );
    void ctptrs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber ap,
                 IComplexNumber b, int ldb,
                 int info );
    void ztptrs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber ap,
                 IComplexNumber b, int ldb,
                 int info );
    void stbtrs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,  INDArray ab,
                 int ldab, INDArray b, int ldb,
                 int info );
    void dtbtrs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,  INDArray ab,
                 int ldab, INDArray b, int ldb,
                 int info );
    void ctbtrs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber b, int ldb,
                 int info );
    void ztbtrs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber b, int ldb,
                 int info );
    void sgecon( String norm, int n,  INDArray a, int lda,
                 INDArray anorm, INDArray rcond, INDArray work, int iwork,
                 int info );
    void dgecon( String norm, int n,  INDArray a, int lda,
                 INDArray anorm, INDArray rcond, INDArray work,
                 int iwork, int info );
    void cgecon( String norm, int n,  IComplexNumber a,
                 int lda, INDArray anorm, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zgecon( String norm, int n,  IComplexNumber a,
                 int lda, INDArray anorm, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void sgbcon( String norm, int n, int kl, int ku,
                 INDArray ab, int ldab,  int ipiv,
                 INDArray anorm, INDArray rcond, INDArray work, int iwork,
                 int info );
    void dgbcon( String norm, int n, int kl, int ku,
                 INDArray ab, int ldab,  int ipiv,
                 INDArray anorm, INDArray rcond, INDArray work,
                 int iwork, int info );
    void cgbcon( String norm, int n, int kl, int ku,
                 IComplexNumber ab, int ldab,
                 int ipiv, INDArray anorm, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zgbcon( String norm, int n, int kl, int ku,
                 IComplexNumber ab, int ldab,
                 int ipiv, INDArray anorm, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void sgtcon( String norm, int n,  INDArray dl,  INDArray d,
                 INDArray du,  INDArray du2,  int ipiv,
                 INDArray anorm, INDArray rcond, INDArray work, int iwork,
                 int info );
    void dgtcon( String norm, int n,  INDArray dl,
                 INDArray d,  INDArray du,  INDArray du2,
                 int ipiv, INDArray anorm, INDArray rcond,
                 INDArray work, int iwork, int info );
    void cgtcon( String norm, int n,  IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du,
                 IComplexNumber du2,  int ipiv,
                 INDArray anorm, INDArray rcond, IComplexNumber work,
                 int info );
    void zgtcon( String norm, int n,  IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du,
                 IComplexNumber du2,  int ipiv,
                 INDArray anorm, INDArray rcond, IComplexNumber work,
                 int info );
    void spocon( String uplo, int n,  INDArray a, int lda,
                 INDArray anorm, INDArray rcond, INDArray work, int iwork,
                 int info );
    void dpocon( String uplo, int n,  INDArray a, int lda,
                 INDArray anorm, INDArray rcond, INDArray work,
                 int iwork, int info );
    void cpocon( String uplo, int n,  IComplexNumber a,
                 int lda, INDArray anorm, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zpocon( String uplo, int n,  IComplexNumber a,
                 int lda, INDArray anorm, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void sppcon( String uplo, int n,  INDArray ap, INDArray anorm,
                 INDArray rcond, INDArray work, int iwork,
                 int info );
    void dppcon( String uplo, int n,  INDArray ap, INDArray anorm,
                 INDArray rcond, INDArray work, int iwork,
                 int info );
    void cppcon( String uplo, int n,  IComplexNumber ap,
                 INDArray anorm, INDArray rcond, IComplexNumber work,
                 INDArray rwork, int info );
    void zppcon( String uplo, int n,  IComplexNumber ap,
                 INDArray anorm, INDArray rcond, IComplexNumber work,
                 INDArray rwork, int info );
    void spbcon( String uplo, int n, int kd,  INDArray ab,
                 int ldab, INDArray anorm, INDArray rcond, INDArray work,
                 int iwork, int info );
    void dpbcon( String uplo, int n, int kd,  INDArray ab,
                 int ldab, INDArray anorm, INDArray rcond,
                 INDArray work, int iwork, int info );
    void cpbcon( String uplo, int n, int kd,
                 IComplexNumber ab, int ldab,
                 INDArray anorm, INDArray rcond, IComplexNumber work,
                 INDArray rwork, int info );
    void zpbcon( String uplo, int n, int kd,
                 IComplexNumber ab, int ldab,
                 INDArray anorm, INDArray rcond, IComplexNumber work,
                 INDArray rwork, int info );
    void sptcon( int n,  INDArray d,  INDArray e, INDArray anorm,
                 INDArray rcond, INDArray work, int info );
    void dptcon( int n,  INDArray d,  INDArray e,
                 INDArray anorm, INDArray rcond, INDArray work,
                 int info );
    void cptcon( int n,  INDArray d,
                 IComplexNumber e, INDArray anorm, INDArray rcond,
                 INDArray work, int info );
    void zptcon( int n,  INDArray d,
                 IComplexNumber e, INDArray anorm,
                 INDArray rcond, INDArray work, int info );
    void ssycon( String uplo, int n,  INDArray a, int lda,
                 int ipiv, INDArray anorm, INDArray rcond,
                 INDArray work, int iwork, int info );
    void dsycon( String uplo, int n,  INDArray a, int lda,
                 int ipiv, INDArray anorm, INDArray rcond,
                 INDArray work, int iwork, int info );
    void csycon( String uplo, int n,  IComplexNumber a,
                 int lda,  int ipiv, INDArray anorm,
                 INDArray rcond, IComplexNumber work,
                 int info );
    void zsycon( String uplo, int n,  IComplexNumber a,
                 int lda,  int ipiv, INDArray anorm,
                 INDArray rcond, IComplexNumber work,
                 int info );
    void checon( String uplo, int n,  IComplexNumber a,
                 int lda,  int ipiv, INDArray anorm,
                 INDArray rcond, IComplexNumber work,
                 int info );
    void zhecon( String uplo, int n,  IComplexNumber a,
                 int lda,  int ipiv, INDArray anorm,
                 INDArray rcond, IComplexNumber work,
                 int info );
    void sspcon( String uplo, int n,  INDArray ap,
                 int ipiv, INDArray anorm, INDArray rcond,
                 INDArray work, int iwork, int info );
    void dspcon( String uplo, int n,  INDArray ap,
                 int ipiv, INDArray anorm, INDArray rcond,
                 INDArray work, int iwork, int info );
    void cspcon( String uplo, int n,  IComplexNumber ap,
                 int ipiv, INDArray anorm, INDArray rcond,
                 IComplexNumber work, int info );
    void zspcon( String uplo, int n,  IComplexNumber ap,
                 int ipiv, INDArray anorm, INDArray rcond,
                 IComplexNumber work, int info );
    void chpcon( String uplo, int n,  IComplexNumber ap,
                 int ipiv, INDArray anorm, INDArray rcond,
                 IComplexNumber work, int info );
    void zhpcon( String uplo, int n,  IComplexNumber ap,
                 int ipiv, INDArray anorm, INDArray rcond,
                 IComplexNumber work, int info );
    void strcon( String norm, String uplo, String diag, int n,
                 INDArray a, int lda, INDArray rcond, INDArray work,
                 int iwork, int info );
    void dtrcon( String norm, String uplo, String diag, int n,
                 INDArray a, int lda, INDArray rcond,
                 INDArray work, int iwork, int info );
    void ctrcon( String norm, String uplo, String diag, int n,
                 IComplexNumber a, int lda,
                 INDArray rcond, IComplexNumber work, INDArray rwork,
                 int info );
    void ztrcon( String norm, String uplo, String diag, int n,
                 IComplexNumber a, int lda,
                 INDArray rcond, IComplexNumber work, INDArray rwork,
                 int info );
    void stpcon( String norm, String uplo, String diag, int n,
                 INDArray ap, INDArray rcond, INDArray work,
                 int iwork, int info );
    void dtpcon( String norm, String uplo, String diag, int n,
                 INDArray ap, INDArray rcond, INDArray work,
                 int iwork, int info );
    void ctpcon( String norm, String uplo, String diag, int n,
                 IComplexNumber ap, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void ztpcon( String norm, String uplo, String diag, int n,
                 IComplexNumber ap, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void stbcon( String norm, String uplo, String diag, int n,
                 int kd,  INDArray ab, int ldab,
                 INDArray rcond, INDArray work, int iwork,
                 int info );
    void dtbcon( String norm, String uplo, String diag, int n,
                 int kd,  INDArray ab, int ldab,
                 INDArray rcond, INDArray work, int iwork,
                 int info );
    void ctbcon( String norm, String uplo, String diag, int n,
                 int kd,  IComplexNumber ab,
                 int ldab, INDArray rcond, IComplexNumber work,
                 INDArray rwork, int info );
    void ztbcon( String norm, String uplo, String diag, int n,
                 int kd,  IComplexNumber ab,
                 int ldab, INDArray rcond,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void sgerfs( String trans, int n, int nrhs,
                 INDArray a, int lda,  INDArray af,
                 int ldaf,  int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void dgerfs( String trans, int n, int nrhs,
                 INDArray a, int lda,  INDArray af,
                 int ldaf,  int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void cgerfs( String trans, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zgerfs( String trans, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void dgerfsx( String trans, String equed, int n, int nrhs,
                  INDArray a, int lda,  INDArray af,
                  int ldaf,  int ipiv,  INDArray r,
                  INDArray c,  INDArray b, int ldb,
                  INDArray x, int ldx, INDArray rcond, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void sgerfsx( String trans, String equed, int n, int nrhs,
                  INDArray a, int lda,  INDArray af,
                  int ldaf,  int ipiv,  INDArray r,
                  INDArray c,  INDArray b, int ldb, INDArray x,
                  int ldx, INDArray rcond, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void zgerfsx( String trans, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv,  INDArray r,  INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cgerfsx( String trans, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv,  INDArray r,  INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void sgbrfs( String trans, int n, int kl, int ku,
                 int nrhs,  INDArray ab, int ldab,
                 INDArray afb, int ldafb,  int ipiv,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dgbrfs( String trans, int n, int kl, int ku,
                 int nrhs,  INDArray ab, int ldab,
                 INDArray afb, int ldafb,
                 int ipiv,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void cgbrfs( String trans, int n, int kl, int ku,
                 int nrhs,  IComplexNumber ab,
                 int ldab,  IComplexNumber afb,
                 int ldafb,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zgbrfs( String trans, int n, int kl, int ku,
                 int nrhs,  IComplexNumber ab,
                 int ldab,  IComplexNumber afb,
                 int ldafb,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void dgbrfsx( String trans, String equed, int n, int kl,
                  int ku, int nrhs,  INDArray ab,
                  int ldab,  INDArray afb, int ldafb,
                  int ipiv,  INDArray r,  INDArray c,
                  INDArray b, int ldb, INDArray x,
                  int ldx, INDArray rcond, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void sgbrfsx( String trans, String equed, int n, int kl,
                  int ku, int nrhs,  INDArray ab,
                  int ldab,  INDArray afb, int ldafb,
                  int ipiv,  INDArray r,  INDArray c,
                  INDArray b, int ldb, INDArray x, int ldx,
                  INDArray rcond, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params, INDArray work,
                  int iwork, int info );
    void zgbrfsx( String trans, String equed, int n, int kl,
                  int ku, int nrhs,
                  IComplexNumber ab, int ldab,
                  IComplexNumber afb, int ldafb,
                  int ipiv,  INDArray r,  INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cgbrfsx( String trans, String equed, int n, int kl,
                  int ku, int nrhs,
                  IComplexNumber ab, int ldab,
                  IComplexNumber afb, int ldafb,
                  int ipiv,  INDArray r,  INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void sgtrfs( String trans, int n, int nrhs,
                 INDArray dl,  INDArray d,  INDArray du,
                 INDArray dlf,  INDArray df,  INDArray duf,
                 INDArray du2,  int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void dgtrfs( String trans, int n, int nrhs,
                 INDArray dl,  INDArray d,  INDArray du,
                 INDArray dlf,  INDArray df,  INDArray duf,
                 INDArray du2,  int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void cgtrfs( String trans, int n, int nrhs,
                 IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du,
                 IComplexNumber dlf,
                 IComplexNumber df,
                 IComplexNumber duf,
                 IComplexNumber du2,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zgtrfs( String trans, int n, int nrhs,
                 IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du,
                 IComplexNumber dlf,
                 IComplexNumber df,
                 IComplexNumber duf,
                 IComplexNumber du2,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void sporfs( String uplo, int n, int nrhs,  INDArray a,
                 int lda,  INDArray af, int ldaf,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dporfs( String uplo, int n, int nrhs,
                 INDArray a, int lda,  INDArray af,
                 int ldaf,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void cporfs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zporfs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void dporfsx( String uplo, String equed, int n, int nrhs,
                  INDArray a, int lda,  INDArray af,
                  int ldaf,  INDArray s,  INDArray b,
                  int ldb, INDArray x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params, INDArray work,
                  int iwork, int info );
    void sporfsx( String uplo, String equed, int n, int nrhs,
                  INDArray a, int lda,  INDArray af,
                  int ldaf,  INDArray s,  INDArray b,
                  int ldb, INDArray x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void zporfsx( String uplo, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  INDArray s,  IComplexNumber b,
                  int ldb, IComplexNumber x, int ldx,
                  INDArray rcond, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cporfsx( String uplo, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  INDArray s,  IComplexNumber b,
                  int ldb, IComplexNumber x, int ldx,
                  INDArray rcond, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void spprfs( String uplo, int n, int nrhs,
                 INDArray ap,  INDArray afp,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void dpprfs( String uplo, int n, int nrhs,
                 INDArray ap,  INDArray afp,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void cpprfs( String uplo, int n, int nrhs,
                 IComplexNumber ap,
                 IComplexNumber afp,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zpprfs( String uplo, int n, int nrhs,
                 IComplexNumber ap,
                 IComplexNumber afp,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void spbrfs( String uplo, int n, int kd, int nrhs,
                 INDArray ab, int ldab,  INDArray afb,
                 int ldafb,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void dpbrfs( String uplo, int n, int kd, int nrhs,
                 INDArray ab, int ldab,  INDArray afb,
                 int ldafb,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void cpbrfs( String uplo, int n, int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber afb, int ldafb,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zpbrfs( String uplo, int n, int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber afb, int ldafb,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void sptrfs( int n, int nrhs,  INDArray d,
                 INDArray e,  INDArray df,  INDArray ef,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray ferr, INDArray berr, INDArray work, int info );
    void dptrfs( int n, int nrhs,  INDArray d,
                 INDArray e,  INDArray df,  INDArray ef,
                 INDArray b, int ldb, INDArray x,
                 int ldx, INDArray ferr, INDArray berr, INDArray work,
                 int info );
    void cptrfs( String uplo, int n, int nrhs,  INDArray d,
                 IComplexNumber e,  INDArray df,
                 IComplexNumber ef,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zptrfs( String uplo, int n, int nrhs,
                 INDArray d,  IComplexNumber e,
                 INDArray df,  IComplexNumber ef,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void ssyrfs( String uplo, int n, int nrhs,  INDArray a,
                 int lda,  INDArray af, int ldaf,
                 int ipiv,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void dsyrfs( String uplo, int n, int nrhs,
                 INDArray a, int lda,  INDArray af,
                 int ldaf,  int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void csyrfs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zsyrfs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void dsyrfsx( String uplo, String equed, int n, int nrhs,
                  INDArray a, int lda,  INDArray af,
                  int ldaf,  int ipiv,  INDArray s,
                  INDArray b, int ldb, INDArray x,
                  int ldx, INDArray rcond, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void ssyrfsx( String uplo, String equed, int n, int nrhs,
                  INDArray a, int lda,  INDArray af,
                  int ldaf,  int ipiv,  INDArray s,
                  INDArray b, int ldb, INDArray x, int ldx,
                  INDArray rcond, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params, INDArray work,
                  int iwork, int info );
    void zsyrfsx( String uplo, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv,  INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void csyrfsx( String uplo, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv,  INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cherfs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zherfs( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zherfsx( String uplo, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv,  INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cherfsx( String uplo, String equed, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv,  INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray berr, int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void ssprfs( String uplo, int n, int nrhs,
                 INDArray ap,  INDArray afp,  int ipiv,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dsprfs( String uplo, int n, int nrhs,
                 INDArray ap,  INDArray afp,  int ipiv,
                 INDArray b, int ldb, INDArray x,
                 int ldx, INDArray ferr, INDArray berr, INDArray work,
                 int iwork, int info );
    void csprfs( String uplo, int n, int nrhs,
                 IComplexNumber ap,
                 IComplexNumber afp,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zsprfs( String uplo, int n, int nrhs,
                 IComplexNumber ap,
                 IComplexNumber afp,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void chprfs( String uplo, int n, int nrhs,
                 IComplexNumber ap,
                 IComplexNumber afp,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void zhprfs( String uplo, int n, int nrhs,
                 IComplexNumber ap,
                 IComplexNumber afp,  int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void strrfs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray a, int lda,
                 INDArray b, int ldb,  INDArray x,
                 int ldx, INDArray ferr, INDArray berr, INDArray work,
                 int iwork, int info );
    void dtrrfs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray a, int lda,
                 INDArray b, int ldb,  INDArray x,
                 int ldx, INDArray ferr, INDArray berr, INDArray work,
                 int iwork, int info );
    void ctrrfs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber a,
                 int lda,  IComplexNumber b,
                 int ldb,  IComplexNumber x,
                 int ldx, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void ztrrfs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber a,
                 int lda,  IComplexNumber b,
                 int ldb,  IComplexNumber x,
                 int ldx, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void stprfs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray ap,  INDArray b,
                 int ldb,  INDArray x, int ldx,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dtprfs( String uplo, String trans, String diag, int n,
                 int nrhs,  INDArray ap,  INDArray b,
                 int ldb,  INDArray x, int ldx,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void ctprfs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber ap,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void ztprfs( String uplo, String trans, String diag, int n,
                 int nrhs,  IComplexNumber ap,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void stbrfs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,  INDArray ab,
                 int ldab,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void dtbrfs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,  INDArray ab,
                 int ldab,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void ctbrfs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray ferr,
                 INDArray berr, IComplexNumber work, INDArray rwork,
                 int info );
    void ztbrfs( String uplo, String trans, String diag, int n,
                 int kd, int nrhs,
                 IComplexNumber ab, int ldab,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void sgetri( int n, INDArray a, int lda,
                 int ipiv, INDArray work, int lwork,
                 int info );
    void dgetri( int n, INDArray a, int lda,
                 int ipiv, INDArray work, int lwork,
                 int info );
    void cgetri( int n, IComplexNumber a, int lda,
                 int ipiv, IComplexNumber work,
                 int lwork, int info );
    void zgetri( int n, IComplexNumber a, int lda,
                 int ipiv, IComplexNumber work,
                 int lwork, int info );
    void spotri( String uplo, int n, INDArray a, int lda,
                 int info );
    void dpotri( String uplo, int n, INDArray a, int lda,
                 int info );
    void cpotri( String uplo, int n, IComplexNumber a,
                 int lda, int info );
    void zpotri( String uplo, int n, IComplexNumber a,
                 int lda, int info );
    void dpftri( String transr, String uplo, int n, INDArray a,
                 int info );
    void spftri( String transr, String uplo, int n, INDArray a,
                 int info );
    void zpftri( String transr, String uplo, int n,
                 IComplexNumber a, int info );
    void cpftri( String transr, String uplo, int n,
                 IComplexNumber a, int info );
    void spptri( String uplo, int n, INDArray ap, int info );
    void dpptri( String uplo, int n, INDArray ap, int info );
    void cpptri( String uplo, int n, IComplexNumber ap,
                 int info );
    void zpptri( String uplo, int n, IComplexNumber ap,
                 int info );
    void ssytri( String uplo, int n, INDArray a, int lda,
                 int ipiv, INDArray work, int info );
    void dsytri( String uplo, int n, INDArray a, int lda,
                 int ipiv, INDArray work, int info );
    void csytri( String uplo, int n, IComplexNumber a,
                 int lda,  int ipiv,
                 IComplexNumber work, int info );
    void zsytri( String uplo, int n, IComplexNumber a,
                 int lda,  int ipiv,
                 IComplexNumber work, int info );
    void chetri( String uplo, int n, IComplexNumber a,
                 int lda,  int ipiv,
                 IComplexNumber work, int info );
    void zhetri( String uplo, int n, IComplexNumber a,
                 int lda,  int ipiv,
                 IComplexNumber work, int info );
    void ssptri( String uplo, int n, INDArray ap,
                 int ipiv, INDArray work, int info );
    void dsptri( String uplo, int n, INDArray ap,
                 int ipiv, INDArray work, int info );
    void csptri( String uplo, int n, IComplexNumber ap,
                 int ipiv, IComplexNumber work,
                 int info );
    void zsptri( String uplo, int n, IComplexNumber ap,
                 int ipiv, IComplexNumber work,
                 int info );
    void chptri( String uplo, int n, IComplexNumber ap,
                 int ipiv, IComplexNumber work,
                 int info );
    void zhptri( String uplo, int n, IComplexNumber ap,
                 int ipiv, IComplexNumber work,
                 int info );
    void strtri( String uplo, String diag, int n, INDArray a,
                 int lda, int info );
    void dtrtri( String uplo, String diag, int n, INDArray a,
                 int lda, int info );
    void ctrtri( String uplo, String diag, int n,
                 IComplexNumber a, int lda,
                 int info );
    void ztrtri( String uplo, String diag, int n,
                 IComplexNumber a, int lda,
                 int info );
    void dtftri( String transr, String uplo, String diag, int n,
                 INDArray a, int info );
    void stftri( String transr, String uplo, String diag, int n,
                 INDArray a, int info );
    void ztftri( String transr, String uplo, String diag, int n,
                 IComplexNumber a, int info );
    void ctftri( String transr, String uplo, String diag, int n,
                 IComplexNumber a, int info );
    void stptri( String uplo, String diag, int n, INDArray ap,
                 int info );
    void dtptri( String uplo, String diag, int n, INDArray ap,
                 int info );
    void ctptri( String uplo, String diag, int n,
                 IComplexNumber ap, int info );
    void ztptri( String uplo, String diag, int n,
                 IComplexNumber ap, int info );
    void sgeequ( int m, int n,  INDArray a,
                 int lda, INDArray r, INDArray c, INDArray rowcnd,
                 INDArray colcnd, INDArray amax, int info );
    void dgeequ( int m, int n,  INDArray a,
                 int lda, INDArray r, INDArray c, INDArray rowcnd,
                 INDArray colcnd, INDArray amax, int info );
    void cgeequ( int m, int n,  IComplexNumber a,
                 int lda, INDArray r, INDArray c, INDArray rowcnd,
                 INDArray colcnd, INDArray amax, int info );
    void zgeequ( int m, int n,
                 IComplexNumber a, int lda, INDArray r,
                 INDArray c, INDArray rowcnd, INDArray colcnd, INDArray amax,
                 int info );
    void dgeequb( int m, int n,  INDArray a,
                  int lda, INDArray r, INDArray c, INDArray rowcnd,
                  INDArray colcnd, INDArray amax, int info );
    void sgeequb( int m, int n,  INDArray a,
                  int lda, INDArray r, INDArray c, INDArray rowcnd,
                  INDArray colcnd, INDArray amax, int info );
    void zgeequb( int m, int n,
                  IComplexNumber a, int lda, INDArray r,
                  INDArray c, INDArray rowcnd, INDArray colcnd, INDArray amax,
                  int info );
    void cgeequb( int m, int n,
                  IComplexNumber a, int lda, INDArray r,
                  INDArray c, INDArray rowcnd, INDArray colcnd, INDArray amax,
                  int info );
    void sgbequ( int m, int n, int kl,
                 int ku,  INDArray ab, int ldab, INDArray r,
                 INDArray c, INDArray rowcnd, INDArray colcnd, INDArray amax,
                 int info );
    void dgbequ( int m, int n, int kl,
                 int ku,  INDArray ab, int ldab,
                 INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                 INDArray amax, int info );
    void cgbequ( int m, int n, int kl,
                 int ku,  IComplexNumber ab,
                 int ldab, INDArray r, INDArray c, INDArray rowcnd,
                 INDArray colcnd, INDArray amax, int info );
    void zgbequ( int m, int n, int kl,
                 int ku,  IComplexNumber ab,
                 int ldab, INDArray r, INDArray c, INDArray rowcnd,
                 INDArray colcnd, INDArray amax, int info );
    void dgbequb( int m, int n, int kl,
                  int ku,  INDArray ab, int ldab,
                  INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                  INDArray amax, int info );
    void sgbequb( int m, int n, int kl,
                  int ku,  INDArray ab, int ldab,
                  INDArray r, INDArray c, INDArray rowcnd, INDArray colcnd,
                  INDArray amax, int info );
    void zgbequb( int m, int n, int kl,
                  int ku,  IComplexNumber ab,
                  int ldab, INDArray r, INDArray c, INDArray rowcnd,
                  INDArray colcnd, INDArray amax, int info );
    void cgbequb( int m, int n, int kl,
                  int ku,  IComplexNumber ab,
                  int ldab, INDArray r, INDArray c, INDArray rowcnd,
                  INDArray colcnd, INDArray amax, int info );
    void spoequ( int n,  INDArray a, int lda, INDArray s,
                 INDArray scond, INDArray amax, int info );
    void dpoequ( int n,  INDArray a, int lda, INDArray s,
                 INDArray scond, INDArray amax, int info );
    void cpoequ( int n,  IComplexNumber a,
                 int lda, INDArray s, INDArray scond, INDArray amax,
                 int info );
    void zpoequ( int n,  IComplexNumber a,
                 int lda, INDArray s, INDArray scond, INDArray amax,
                 int info );
    void dpoequb( int n,  INDArray a, int lda, INDArray s,
                  INDArray scond, INDArray amax, int info );
    void spoequb( int n,  INDArray a, int lda, INDArray s,
                  INDArray scond, INDArray amax, int info );
    void zpoequb( int n,  IComplexNumber a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  int info );
    void cpoequb( int n,  IComplexNumber a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  int info );
    void sppequ( String uplo, int n,  INDArray ap, INDArray s,
                 INDArray scond, INDArray amax, int info );
    void dppequ( String uplo, int n,  INDArray ap, INDArray s,
                 INDArray scond, INDArray amax, int info );
    void cppequ( String uplo, int n,  IComplexNumber ap,
                 INDArray s, INDArray scond, INDArray amax, int info );
    void zppequ( String uplo, int n,  IComplexNumber ap,
                 INDArray s, INDArray scond, INDArray amax, int info );
    void spbequ( String uplo, int n, int kd,  INDArray ab,
                 int ldab, INDArray s, INDArray scond, INDArray amax,
                 int info );
    void dpbequ( String uplo, int n, int kd,  INDArray ab,
                 int ldab, INDArray s, INDArray scond, INDArray amax,
                 int info );
    void cpbequ( String uplo, int n, int kd,
                 IComplexNumber ab, int ldab, INDArray s,
                 INDArray scond, INDArray amax, int info );
    void zpbequ( String uplo, int n, int kd,
                 IComplexNumber ab, int ldab,
                 INDArray s, INDArray scond, INDArray amax, int info );
    void dsyequb( String uplo, int n,  INDArray a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  INDArray work, int info );
    void ssyequb( String uplo, int n,  INDArray a, int lda,
                  INDArray s, INDArray scond, INDArray amax, INDArray work,
                  int info );
    void zsyequb( String uplo, int n,  IComplexNumber a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  IComplexNumber work, int info );
    void csyequb( String uplo, int n,  IComplexNumber a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  IComplexNumber work, int info );
    void zheequb( String uplo, int n,  IComplexNumber a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  IComplexNumber work, int info );
    void cheequb( String uplo, int n,  IComplexNumber a,
                  int lda, INDArray s, INDArray scond, INDArray amax,
                  IComplexNumber work, int info );
    void sgesv( int n, int nrhs, INDArray a, int lda,
                int ipiv, INDArray b, int ldb,
                int info );
    void dgesv( int n, int nrhs, INDArray a, int lda,
                int ipiv, INDArray b, int ldb,
                int info );
    void cgesv( int n, int nrhs, IComplexNumber a,
                int lda, int ipiv, IComplexNumber b,
                int ldb, int info );
    void zgesv( int n, int nrhs, IComplexNumber a,
                int lda, int ipiv, IComplexNumber b,
                int ldb, int info );
    void dsgesv( int n, int nrhs, INDArray a, int lda,
                 int ipiv, INDArray b, int ldb, INDArray x,
                 int ldx, INDArray work, INDArray swork,
                 int iter, int info );
    void zcgesv( int n, int nrhs, IComplexNumber a,
                 int lda, int ipiv, IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 IComplexNumber work, IComplexNumber swork,
                 INDArray rwork, int iter, int info );
    void sgesvx( String fact, String trans, int n, int nrhs,
                 INDArray a, int lda, INDArray af, int ldaf,
                 int ipiv, String equed, INDArray r, INDArray c, INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dgesvx( String fact, String trans, int n, int nrhs,
                 INDArray a, int lda, INDArray af, int ldaf,
                 int ipiv, String equed, INDArray r, INDArray c,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr, INDArray work,
                 int iwork, int info );
    void cgesvx( String fact, String trans, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv, String equed, INDArray r, INDArray c,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zgesvx( String fact, String trans, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv, String equed, INDArray r, INDArray c,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void dgesvxx( String fact, String trans, int n, int nrhs,
                  INDArray a, int lda, INDArray af, int ldaf,
                  int ipiv, String equed, INDArray r, INDArray c,
                  INDArray b, int ldb, INDArray x, int ldx,
                  INDArray rcond, INDArray rpvgrw, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void sgesvxx( String fact, String trans, int n, int nrhs,
                  INDArray a, int lda, INDArray af, int ldaf,
                  int ipiv, String equed, INDArray r, INDArray c,
                  INDArray b, int ldb, INDArray x, int ldx,
                  INDArray rcond, INDArray rpvgrw, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void zgesvxx( String fact, String trans, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv, String equed, INDArray r, INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cgesvxx( String fact, String trans, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv, String equed, INDArray r, INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void sgbsv( int n, int kl, int ku,
                int nrhs, INDArray ab, int ldab,
                int ipiv, INDArray b, int ldb,
                int info );
    void dgbsv( int n, int kl, int ku,
                int nrhs, INDArray ab, int ldab,
                int ipiv, INDArray b, int ldb,
                int info );
    void cgbsv( int n, int kl, int ku,
                int nrhs, IComplexNumber ab, int ldab,
                int ipiv, IComplexNumber b, int ldb,
                int info );
    void zgbsv( int n, int kl, int ku,
                int nrhs, IComplexNumber ab,
                int ldab, int ipiv, IComplexNumber b,
                int ldb, int info );
    void sgbsvx( String fact, String trans, int n, int kl,
                 int ku, int nrhs, INDArray ab,
                 int ldab, INDArray afb, int ldafb,
                 int ipiv, String equed, INDArray r, INDArray c, INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dgbsvx( String fact, String trans, int n, int kl,
                 int ku, int nrhs, INDArray ab,
                 int ldab, INDArray afb, int ldafb,
                 int ipiv, String equed, INDArray r, INDArray c,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr, INDArray work,
                 int iwork, int info );
    void cgbsvx( String fact, String trans, int n, int kl,
                 int ku, int nrhs, IComplexNumber ab,
                 int ldab, IComplexNumber afb,
                 int ldafb, int ipiv, String equed, INDArray r,
                 INDArray c, IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zgbsvx( String fact, String trans, int n, int kl,
                 int ku, int nrhs, IComplexNumber ab,
                 int ldab, IComplexNumber afb,
                 int ldafb, int ipiv, String equed, INDArray r,
                 INDArray c, IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void dgbsvxx( String fact, String trans, int n, int kl,
                  int ku, int nrhs, INDArray ab,
                  int ldab, INDArray afb, int ldafb,
                  int ipiv, String equed, INDArray r, INDArray c,
                  INDArray b, int ldb, INDArray x, int ldx,
                  INDArray rcond, INDArray rpvgrw, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void sgbsvxx( String fact, String trans, int n, int kl,
                  int ku, int nrhs, INDArray ab,
                  int ldab, INDArray afb, int ldafb,
                  int ipiv, String equed, INDArray r, INDArray c,
                  INDArray b, int ldb, INDArray x, int ldx,
                  INDArray rcond, INDArray rpvgrw, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void zgbsvxx( String fact, String trans, int n, int kl,
                  int ku, int nrhs,
                  IComplexNumber ab, int ldab,
                  IComplexNumber afb, int ldafb,
                  int ipiv, String equed, INDArray r, INDArray c,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cgbsvxx( String fact, String trans, int n, int kl,
                  int ku, int nrhs, IComplexNumber ab,
                  int ldab, IComplexNumber afb,
                  int ldafb, int ipiv, String equed, INDArray r,
                  INDArray c, IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void sgtsv( int n, int nrhs, INDArray dl, INDArray d,
                INDArray du, INDArray b, int ldb, int info );
    void dgtsv( int n, int nrhs, INDArray dl, INDArray d,
                INDArray du, INDArray b, int ldb, int info );
    void cgtsv( int n, int nrhs, IComplexNumber dl,
                IComplexNumber d, IComplexNumber du,
                IComplexNumber b, int ldb, int info );
    void zgtsv( int n, int nrhs, IComplexNumber dl,
                IComplexNumber d, IComplexNumber du,
                IComplexNumber b, int ldb,
                int info );
    void sgtsvx( String fact, String trans, int n, int nrhs,
                 INDArray dl,  INDArray d,  INDArray du,
                 INDArray dlf, INDArray df, INDArray duf, INDArray du2,
                 int ipiv,  INDArray b, int ldb, INDArray x,
                 int ldx, INDArray rcond, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void dgtsvx( String fact, String trans, int n, int nrhs,
                 INDArray dl,  INDArray d,  INDArray du,
                 INDArray dlf, INDArray df, INDArray duf, INDArray du2,
                 int ipiv,  INDArray b, int ldb,
                 INDArray x, int ldx, INDArray rcond, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void cgtsvx( String fact, String trans, int n, int nrhs,
                 IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du, IComplexNumber dlf,
                 IComplexNumber df, IComplexNumber duf,
                 IComplexNumber du2, int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zgtsvx( String fact, String trans, int n, int nrhs,
                 IComplexNumber dl,
                 IComplexNumber d,
                 IComplexNumber du, IComplexNumber dlf,
                 IComplexNumber df, IComplexNumber duf,
                 IComplexNumber du2, int ipiv,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void sposv( String uplo, int n, int nrhs, INDArray a,
                int lda, INDArray b, int ldb,
                int info );
    void dposv( String uplo, int n, int nrhs, INDArray a,
                int lda, INDArray b, int ldb,
                int info );
    void cposv( String uplo, int n, int nrhs,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb, int info );
    void zposv( String uplo, int n, int nrhs,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb,
                int info );
    void dsposv( String uplo, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb, INDArray x,
                 int ldx, INDArray work, INDArray swork,
                 int iter, int info );
    void zcposv( String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx,
                 IComplexNumber work, IComplexNumber swork,
                 INDArray rwork, int iter, int info );
    void sposvx( String fact, String uplo, int n, int nrhs,
                 INDArray a, int lda, INDArray af, int ldaf,
                 String equed, INDArray s, INDArray b, int ldb, INDArray x,
                 int ldx, INDArray rcond, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void dposvx( String fact, String uplo, int n, int nrhs,
                 INDArray a, int lda, INDArray af, int ldaf,
                 String equed, INDArray s, INDArray b, int ldb,
                 INDArray x, int ldx, INDArray rcond, INDArray ferr,
                 INDArray berr, INDArray work, int iwork,
                 int info );
    void cposvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf, String equed,
                 INDArray s, IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zposvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf, String equed,
                 INDArray s, IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void dposvxx( String fact, String uplo, int n, int nrhs,
                  INDArray a, int lda, INDArray af, int ldaf,
                  String equed, INDArray s, INDArray b, int ldb,
                  INDArray x, int ldx, INDArray rcond, INDArray rpvgrw,
                  INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params, INDArray work,
                  int iwork, int info );
    void sposvxx( String fact, String uplo, int n, int nrhs,
                  INDArray a, int lda, INDArray af, int ldaf,
                  String equed, INDArray s, INDArray b, int ldb, INDArray x,
                  int ldx, INDArray rcond, INDArray rpvgrw, INDArray berr,
                  int int_bnds, INDArray erint_norm,
                  INDArray erint_comp, int nparams, INDArray params,
                  INDArray work, int iwork, int info );
    void zposvxx( String fact, String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf, String equed,
                  INDArray s, IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void cposvxx( String fact, String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf, String equed,
                  INDArray s, IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void sppsv( String uplo, int n, int nrhs, INDArray ap,
                INDArray b, int ldb, int info );
    void dppsv( String uplo, int n, int nrhs, INDArray ap,
                INDArray b, int ldb, int info );
    void cppsv( String uplo, int n, int nrhs,
                IComplexNumber ap, IComplexNumber b,
                int ldb, int info );
    void zppsv( String uplo, int n, int nrhs,
                IComplexNumber ap, IComplexNumber b,
                int ldb, int info );
    void sppsvx( String fact, String uplo, int n, int nrhs,
                 INDArray ap, INDArray afp, String equed, INDArray s, INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dppsvx( String fact, String uplo, int n, int nrhs,
                 INDArray ap, INDArray afp, String equed, INDArray s, INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void cppsvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber afp,
                 String equed, INDArray s, IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zppsvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber afp,
                 String equed, INDArray s, IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void spbsv( String uplo, int n, int kd, int nrhs,
                INDArray ab, int ldab, INDArray b, int ldb,
                int info );
    void dpbsv( String uplo, int n, int kd, int nrhs,
                INDArray ab, int ldab, INDArray b, int ldb,
                int info );
    void cpbsv( String uplo, int n, int kd, int nrhs,
                IComplexNumber ab, int ldab,
                IComplexNumber b, int ldb, int info );
    void zpbsv( String uplo, int n, int kd, int nrhs,
                IComplexNumber ab, int ldab,
                IComplexNumber b, int ldb,
                int info );
    void spbsvx( String fact, String uplo, int n, int kd,
                 int nrhs, INDArray ab, int ldab, INDArray afb,
                 int ldafb, String equed, INDArray s, INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void dpbsvx( String fact, String uplo, int n, int kd,
                 int nrhs, INDArray ab, int ldab, INDArray afb,
                 int ldafb, String equed, INDArray s, INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int iwork,
                 int info );
    void cpbsvx( String fact, String uplo, int n, int kd,
                 int nrhs, IComplexNumber ab,
                 int ldab, IComplexNumber afb,
                 int ldafb, String equed, INDArray s,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void zpbsvx( String fact, String uplo, int n, int kd,
                 int nrhs, IComplexNumber ab,
                 int ldab, IComplexNumber afb,
                 int ldafb, String equed, INDArray s,
                 IComplexNumber b, int ldb,
                 IComplexNumber x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, IComplexNumber work,
                 INDArray rwork, int info );
    void sptsv( int n, int nrhs, INDArray d, INDArray e,
                INDArray b, int ldb, int info );
    void dptsv( int n, int nrhs, INDArray d, INDArray e,
                INDArray b, int ldb, int info );
    void cptsv( int n, int nrhs, INDArray d,
                IComplexNumber e, IComplexNumber b,
                int ldb, int info );
    void zptsv( int n, int nrhs, INDArray d,
                IComplexNumber e, IComplexNumber b,
                int ldb, int info );
    void sptsvx( String fact, int n, int nrhs,  INDArray d,
                 INDArray e, INDArray df, INDArray ef,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int info );
    void dptsvx( String fact, int n, int nrhs,
                 INDArray d,  INDArray e, INDArray df, INDArray ef,
                 INDArray b, int ldb, INDArray x,
                 int ldx, INDArray rcond, INDArray ferr, INDArray berr,
                 INDArray work, int info );
    void cptsvx( String fact, int n, int nrhs,  INDArray d,
                 IComplexNumber e, INDArray df,
                 IComplexNumber ef,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zptsvx( String fact, int n, int nrhs,
                 INDArray d,  IComplexNumber e, INDArray df,
                 IComplexNumber ef,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void ssysv( String uplo, int n, int nrhs, INDArray a,
                int lda, int ipiv, INDArray b, int ldb,
                INDArray work, int lwork, int info );
    void dsysv( String uplo, int n, int nrhs, INDArray a,
                int lda, int ipiv, INDArray b,
                int ldb, INDArray work, int lwork,
                int info );
    void csysv( String uplo, int n, int nrhs,
                IComplexNumber a, int lda, int ipiv,
                IComplexNumber b, int ldb,
                IComplexNumber work, int lwork,
                int info );
    void zsysv( String uplo, int n, int nrhs,
                IComplexNumber a, int lda, int ipiv,
                IComplexNumber b, int ldb,
                IComplexNumber work, int lwork,
                int info );
    void ssysvx( String fact, String uplo, int n, int nrhs,
                 INDArray a, int lda, INDArray af,
                 int ldaf, int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int lwork,
                 int iwork, int info );
    void dsysvx( String fact, String uplo, int n, int nrhs,
                 INDArray a, int lda, INDArray af,
                 int ldaf, int ipiv,  INDArray b,
                 int ldb, INDArray x, int ldx, INDArray rcond,
                 INDArray ferr, INDArray berr, INDArray work, int lwork,
                 int iwork, int info );
    void csysvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int info );
    void zsysvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void dsysvxx( String fact, String uplo, int n, int nrhs,
                  INDArray a, int lda, INDArray af, int ldaf,
                  int ipiv, String equed, INDArray s, INDArray b,
                  int ldb, INDArray x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params, INDArray work,
                  int iwork, int info );
    void ssysvxx( String fact, String uplo, int n, int nrhs,
                  INDArray a, int lda, INDArray af, int ldaf,
                  int ipiv, String equed, INDArray s, INDArray b,
                  int ldb, INDArray x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params, INDArray work,
                  int iwork, int info );
    void zsysvxx( String fact, String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv, String equed, INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void csysvxx( String fact, String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv, String equed, INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void chesv( String uplo, int n, int nrhs,
                IComplexNumber a, int lda, int ipiv,
                IComplexNumber b, int ldb,
                IComplexNumber work, int lwork,
                int info );
    void zhesv( String uplo, int n, int nrhs,
                IComplexNumber a, int lda, int ipiv,
                IComplexNumber b, int ldb,
                IComplexNumber work, int lwork,
                int info );
    void chesvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int info );
    void zhesvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber af, int ldaf,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void zhesvxx( String fact, String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv, String equed, INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void chesvxx( String fact, String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  IComplexNumber af, int ldaf,
                  int ipiv, String equed, INDArray s,
                  IComplexNumber b, int ldb,
                  IComplexNumber x, int ldx, INDArray rcond,
                  INDArray rpvgrw, INDArray berr, int int_bnds,
                  INDArray erint_norm, INDArray erint_comp,
                  int nparams, INDArray params,
                  IComplexNumber work, INDArray rwork,
                  int info );
    void sspsv( String uplo, int n, int nrhs, INDArray ap,
                int ipiv, INDArray b, int ldb,
                int info );
    void dspsv( String uplo, int n, int nrhs, INDArray ap,
                int ipiv, INDArray b, int ldb,
                int info );
    void cspsv( String uplo, int n, int nrhs,
                IComplexNumber ap, int ipiv,
                IComplexNumber b, int ldb, int info );
    void zspsv( String uplo, int n, int nrhs,
                IComplexNumber ap, int ipiv,
                IComplexNumber b, int ldb,
                int info );
    void sspsvx( String fact, String uplo, int n, int nrhs,
                 INDArray ap, INDArray afp, int ipiv,
                 INDArray b, int ldb, INDArray x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr, INDArray work,
                 int iwork, int info );
    void dspsvx( String fact, String uplo, int n, int nrhs,
                 INDArray ap, INDArray afp, int ipiv,
                 INDArray b, int ldb, INDArray x,
                 int ldx, INDArray rcond, INDArray ferr, INDArray berr,
                 INDArray work, int iwork, int info );
    void cspsvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber afp,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zspsvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber afp,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void chpsv( String uplo, int n, int nrhs,
                IComplexNumber ap, int ipiv,
                IComplexNumber b, int ldb, int info );
    void zhpsv( String uplo, int n, int nrhs,
                IComplexNumber ap, int ipiv,
                IComplexNumber b, int ldb,
                int info );
    void chpsvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber afp,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zhpsvx( String fact, String uplo, int n, int nrhs,
                 IComplexNumber ap, IComplexNumber afp,
                 int ipiv,  IComplexNumber b,
                 int ldb, IComplexNumber x, int ldx,
                 INDArray rcond, INDArray ferr, INDArray berr,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void sgeqrf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void dgeqrf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void cgeqrf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void zgeqrf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sgeqpf( int m, int n, INDArray a, int lda,
                 int jpvt, INDArray tau, INDArray work,
                 int info );
    void dgeqpf( int m, int n, INDArray a, int lda,
                 int jpvt, INDArray tau, INDArray work,
                 int info );
    void cgeqpf( int m, int n, IComplexNumber a,
                 int lda, int jpvt,
                 IComplexNumber tau, IComplexNumber work,
                 INDArray rwork, int info );
    void zgeqpf( int m, int n, IComplexNumber a,
                 int lda, int jpvt,
                 IComplexNumber tau, IComplexNumber work,
                 INDArray rwork, int info );
    void sgeqp3( int m, int n, INDArray a, int lda,
                 int jpvt, INDArray tau, INDArray work,
                 int lwork, int info );
    void dgeqp3( int m, int n, INDArray a, int lda,
                 int jpvt, INDArray tau, INDArray work,
                 int lwork, int info );
    void cgeqp3( int m, int n, IComplexNumber a,
                 int lda, int jpvt,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, INDArray rwork, int info );
    void zgeqp3( int m, int n, IComplexNumber a,
                 int lda, int jpvt,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, INDArray rwork, int info );
    void sorgqr( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void dorgqr( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void sormqr( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void dormqr( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void cungqr( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zungqr( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void cunmqr( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmqr( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void sgelqf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void dgelqf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void cgelqf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void zgelqf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sorglq( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void dorglq( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void sormlq( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void dormlq( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void cunglq( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zunglq( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void cunmlq( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmlq( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void sgeqlf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void dgeqlf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void cgeqlf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void zgeqlf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sorgql( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void dorgql( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void cungql( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zungql( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sormql( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void dormql( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void cunmql( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmql( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void sgerqf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void dgerqf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void cgerqf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void zgerqf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sorgrq( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void dorgrq( int m, int n, int k, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void cungrq( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zungrq( int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sormrq( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void dormrq( String side, String trans, int m, int n,
                 int k,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void cunmrq( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmrq( String side, String trans, int m, int n,
                 int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void stzrzf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void dtzrzf( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void ctzrzf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void ztzrzf( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void sormrz( String side, String trans, int m, int n,
                 int k, int l,  INDArray a,
                 int lda,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int lwork,
                 int info );
    void dormrz( String side, String trans, int m, int n,
                 int k, int l,  INDArray a,
                 int lda,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int lwork,
                 int info );
    void cunmrz( String side, String trans, int m, int n,
                 int k, int l,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmrz( String side, String trans, int m, int n,
                 int k, int l,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber c,
                 int ldc, IComplexNumber work,
                 int lwork, int info );
    void sggqrf( int n, int m, int p, INDArray a,
                 int lda, INDArray taua, INDArray b, int ldb,
                 INDArray taub, INDArray work, int lwork,
                 int info );
    void dggqrf( int n, int m, int p, INDArray a,
                 int lda, INDArray taua, INDArray b, int ldb,
                 INDArray taub, INDArray work, int lwork,
                 int info );
    void cggqrf( int n, int m, int p,
                 IComplexNumber a, int lda,
                 IComplexNumber taua, IComplexNumber b,
                 int ldb, IComplexNumber taub,
                 IComplexNumber work, int lwork,
                 int info );
    void zggqrf( int n, int m, int p,
                 IComplexNumber a, int lda,
                 IComplexNumber taua, IComplexNumber b,
                 int ldb, IComplexNumber taub,
                 IComplexNumber work, int lwork,
                 int info );
    void sggrqf( int m, int p, int n, INDArray a,
                 int lda, INDArray taua, INDArray b, int ldb,
                 INDArray taub, INDArray work, int lwork,
                 int info );
    void dggrqf( int m, int p, int n, INDArray a,
                 int lda, INDArray taua, INDArray b, int ldb,
                 INDArray taub, INDArray work, int lwork,
                 int info );
    void cggrqf( int m, int p, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber taua, IComplexNumber b,
                 int ldb, IComplexNumber taub,
                 IComplexNumber work, int lwork,
                 int info );
    void zggrqf( int m, int p, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber taua, IComplexNumber b,
                 int ldb, IComplexNumber taub,
                 IComplexNumber work, int lwork,
                 int info );
    void sgebrd( int m, int n, INDArray a, int lda,
                 INDArray d, INDArray e, INDArray tauq, INDArray taup, INDArray work,
                 int lwork, int info );
    void dgebrd( int m, int n, INDArray a, int lda,
                 INDArray d, INDArray e, INDArray tauq, INDArray taup,
                 INDArray work, int lwork, int info );
    void cgebrd( int m, int n, IComplexNumber a,
                 int lda, INDArray d, INDArray e,
                 IComplexNumber tauq, IComplexNumber taup,
                 IComplexNumber work, int lwork,
                 int info );
    void zgebrd( int m, int n, IComplexNumber a,
                 int lda, INDArray d, INDArray e,
                 IComplexNumber tauq, IComplexNumber taup,
                 IComplexNumber work, int lwork,
                 int info );
    void sgbbrd( String vect, int m, int n, int ncc,
                 int kl, int ku, INDArray ab, int ldab,
                 INDArray d, INDArray e, INDArray q, int ldq, INDArray pt,
                 int ldpt, INDArray c, int ldc, INDArray work,
                 int info );
    void dgbbrd( String vect, int m, int n, int ncc,
                 int kl, int ku, INDArray ab,
                 int ldab, INDArray d, INDArray e, INDArray q,
                 int ldq, INDArray pt, int ldpt, INDArray c,
                 int ldc, INDArray work, int info );
    void cgbbrd( String vect, int m, int n, int ncc,
                 int kl, int ku, IComplexNumber ab,
                 int ldab, INDArray d, INDArray e,
                 IComplexNumber q, int ldq,
                 IComplexNumber pt, int ldpt,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zgbbrd( String vect, int m, int n, int ncc,
                 int kl, int ku, IComplexNumber ab,
                 int ldab, INDArray d, INDArray e,
                 IComplexNumber q, int ldq,
                 IComplexNumber pt, int ldpt,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void sorgbr( String vect, int m, int n, int k,
                 INDArray a, int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void dorgbr( String vect, int m, int n, int k,
                 INDArray a, int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void sormbr( String vect, String side, String trans, int m,
                 int n, int k,  INDArray a,
                 int lda,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int lwork,
                 int info );
    void dormbr( String vect, String side, String trans, int m,
                 int n, int k,  INDArray a,
                 int lda,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int lwork,
                 int info );
    void cungbr( String vect, int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zungbr( String vect, int m, int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void cunmbr( String vect, String side, String trans, int m,
                 int n, int k,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmbr( String vect, String side, String trans, int m,
                 int n, int k,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber c,
                 int ldc, IComplexNumber work,
                 int lwork, int info );
    void sbdsqr( String uplo, int n, int ncvt,
                 int nru, int ncc, INDArray d, INDArray e,
                 INDArray vt, int ldvt, INDArray u, int ldu,
                 INDArray c, int ldc, INDArray work, int info );
    void dbdsqr( String uplo, int n, int ncvt,
                 int nru, int ncc, INDArray d, INDArray e,
                 INDArray vt, int ldvt, INDArray u, int ldu,
                 INDArray c, int ldc, INDArray work,
                 int info );
    void cbdsqr( String uplo, int n, int ncvt,
                 int nru, int ncc, INDArray d, INDArray e,
                 IComplexNumber vt, int ldvt,
                 IComplexNumber u, int ldu,
                 IComplexNumber c, int ldc, INDArray work,
                 int info );
    void zbdsqr( String uplo, int n, int ncvt,
                 int nru, int ncc, INDArray d, INDArray e,
                 IComplexNumber vt, int ldvt,
                 IComplexNumber u, int ldu,
                 IComplexNumber c, int ldc, INDArray work,
                 int info );
    void sbdsdc( String uplo, String compq, int n, INDArray d, INDArray e,
                 INDArray u, int ldu, INDArray vt, int ldvt,
                 INDArray q, int iq, INDArray work, int iwork,
                 int info );
    void dbdsdc( String uplo, String compq, int n, INDArray d,
                 INDArray e, INDArray u, int ldu, INDArray vt,
                 int ldvt, INDArray q, int iq, INDArray work,
                 int iwork, int info );
    void sbdsvdx( String uplo, String jobz, String range,
                  int n, INDArray d, INDArray e,
                  int vl, int vu,
                  int il, int iu, int ns,
                  INDArray s, INDArray z, int ldz,
                  INDArray work, int iwork, int info );
    void dbdsvdx( String uplo, String jobz, String range,
                  int n, INDArray d, INDArray e,
                  int vl, int vu,
                  int il, int iu, int ns,
                  INDArray s, INDArray z, int ldz,
                  INDArray work, int iwork, int info );
    void ssytrd( String uplo, int n, INDArray a, int lda,
                 INDArray d, INDArray e, INDArray tau, INDArray work,
                 int lwork, int info );
    void dsytrd( String uplo, int n, INDArray a, int lda,
                 INDArray d, INDArray e, INDArray tau, INDArray work,
                 int lwork, int info );
    void sorgtr( String uplo, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void dorgtr( String uplo, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int lwork,
                 int info );
    void sormtr( String side, String uplo, String trans, int m,
                 int n,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void dormtr( String side, String uplo, String trans, int m,
                 int n,  INDArray a, int lda,
                 INDArray tau, INDArray c, int ldc, INDArray work,
                 int lwork, int info );
    void chetrd( String uplo, int n, IComplexNumber a,
                 int lda, INDArray d, INDArray e,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zhetrd( String uplo, int n, IComplexNumber a,
                 int lda, INDArray d, INDArray e,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void cungtr( String uplo, int n, IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void zungtr( String uplo, int n, IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void cunmtr( String side, String uplo, String trans, int m,
                 int n,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void zunmtr( String side, String uplo, String trans, int m,
                 int n,  IComplexNumber a,
                 int lda,  IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int lwork,
                 int info );
    void ssptrd( String uplo, int n, INDArray ap, INDArray d, INDArray e,
                 INDArray tau, int info );
    void dsptrd( String uplo, int n, INDArray ap, INDArray d, INDArray e,
                 INDArray tau, int info );
    void sopgtr( String uplo, int n,  INDArray ap,
                 INDArray tau, INDArray q, int ldq, INDArray work,
                 int info );
    void dopgtr( String uplo, int n,  INDArray ap,
                 INDArray tau, INDArray q, int ldq, INDArray work,
                 int info );
    void sopmtr( String side, String uplo, String trans, int m,
                 int n,  INDArray ap,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int info );
    void dopmtr( String side, String uplo, String trans, int m,
                 int n,  INDArray ap,  INDArray tau,
                 INDArray c, int ldc, INDArray work,
                 int info );
    void chptrd( String uplo, int n, IComplexNumber ap,
                 INDArray d, INDArray e, IComplexNumber tau,
                 int info );
    void zhptrd( String uplo, int n, IComplexNumber ap,
                 INDArray d, INDArray e, IComplexNumber tau,
                 int info );
    void cupgtr( String uplo, int n,  IComplexNumber ap,
                 IComplexNumber tau, IComplexNumber q,
                 int ldq, IComplexNumber work,
                 int info );
    void zupgtr( String uplo, int n,  IComplexNumber ap,
                 IComplexNumber tau, IComplexNumber q,
                 int ldq, IComplexNumber work,
                 int info );
    void cupmtr( String side, String uplo, String trans, int m,
                 int n,  IComplexNumber ap,
                 IComplexNumber tau, IComplexNumber c,
                 int ldc, IComplexNumber work,
                 int info );
    void zupmtr( String side, String uplo, String trans, int m,
                 int n,  IComplexNumber ap,
                 IComplexNumber tau, IComplexNumber c,
                 int ldc, IComplexNumber work,
                 int info );
    void ssbtrd( String vect, String uplo, int n, int kd,
                 INDArray ab, int ldab, INDArray d, INDArray e, INDArray q,
                 int ldq, INDArray work, int info );
    void dsbtrd( String vect, String uplo, int n, int kd,
                 INDArray ab, int ldab, INDArray d, INDArray e,
                 INDArray q, int ldq, INDArray work,
                 int info );
    void chbtrd( String vect, String uplo, int n, int kd,
                 IComplexNumber ab, int ldab, INDArray d,
                 INDArray e, IComplexNumber q, int ldq,
                 IComplexNumber work, int info );
    void zhbtrd( String vect, String uplo, int n, int kd,
                 IComplexNumber ab, int ldab, INDArray d,
                 INDArray e, IComplexNumber q, int ldq,
                 IComplexNumber work, int info );
    void ssterf( int n, INDArray d, INDArray e, int info );
    void dsterf( int n, INDArray d, INDArray e, int info );
    void ssteqr( String compz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int info );
    void dsteqr( String compz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int info );
    void csteqr( String compz, int n, INDArray d, INDArray e,
                 IComplexNumber z, int ldz, INDArray work,
                 int info );
    void zsteqr( String compz, int n, INDArray d, INDArray e,
                 IComplexNumber z, int ldz, INDArray work,
                 int info );
    void sstemr( String jobz, String range, int n, INDArray d, INDArray e,
                 INDArray vl, INDArray vu, int il, int iu,
                 int m, INDArray w, INDArray z, int ldz,
                 int nzc, int isuppz, int tryrac,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void dstemr( String jobz, String range, int n, INDArray d,
                 INDArray e, INDArray vl, INDArray vu, int il,
                 int iu, int m, INDArray w, INDArray z,
                 int ldz, int nzc, int isuppz,
                 int tryrac, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void cstemr( String jobz, String range, int n, INDArray d, INDArray e,
                 INDArray vl, INDArray vu, int il, int iu,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, int nzc, int isuppz,
                 int tryrac, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void zstemr( String jobz, String range, int n, INDArray d,
                 INDArray e, INDArray vl, INDArray vu, int il,
                 int iu, int m, INDArray w,
                 IComplexNumber z, int ldz, int nzc,
                 int isuppz, int tryrac, INDArray work,
                 int lwork, int iwork, int liwork,
                 int info );
    void sstedc( String compz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dstedc( String compz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void cstedc( String compz, int n, INDArray d, INDArray e,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zstedc( String compz, int n, INDArray d, INDArray e,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void sstegr( String jobz, String range, int n, INDArray d, INDArray e,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w, INDArray z,
                 int ldz, int isuppz, INDArray work,
                 int lwork, int iwork, int liwork,
                 int info );
    void dstegr( String jobz, String range, int n, INDArray d,
                 INDArray e, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, int isuppz,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void cstegr( String jobz, String range, int n, INDArray d, INDArray e,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w,
                 IComplexNumber z, int ldz,
                 int isuppz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void zstegr( String jobz, String range, int n, INDArray d,
                 INDArray e, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 IComplexNumber z, int ldz,
                 int isuppz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void spteqr( String compz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int info );
    void dpteqr( String compz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int info );
    void cpteqr( String compz, int n, INDArray d, INDArray e,
                 IComplexNumber z, int ldz, INDArray work,
                 int info );
    void zpteqr( String compz, int n, INDArray d, INDArray e,
                 IComplexNumber z, int ldz, INDArray work,
                 int info );
    void sstebz( String range, String order, int n, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 INDArray d,  INDArray e, int m,
                 int nsplit, INDArray w, int iblock,
                 int isplit, INDArray work, int iwork,
                 int info );
    void dstebz( String range, String order, int n, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 INDArray d,  INDArray e, int m,
                 int nsplit, INDArray w, int iblock,
                 int isplit, INDArray work, int iwork,
                 int info );
    void sstein( int n,  INDArray d,  INDArray e,
                 int m,  INDArray w,  int iblock,
                 int isplit, INDArray z, int ldz,
                 INDArray work, int iwork, int ifailv,
                 int info );
    void dstein( int n,  INDArray d,  INDArray e,
                 int m,  INDArray w,  int iblock,
                 int isplit, INDArray z, int ldz,
                 INDArray work, int iwork, int ifailv,
                 int info );
    void cstein( int n,  INDArray d,  INDArray e,
                 int m,  INDArray w,  int iblock,
                 int isplit, IComplexNumber z,
                 int ldz, INDArray work, int iwork,
                 int ifailv, int info );
    void zstein( int n,  INDArray d,  INDArray e,
                 int m,  INDArray w,  int iblock,
                 int isplit, IComplexNumber z,
                 int ldz, INDArray work, int iwork,
                 int ifailv, int info );
    void sdisna( String job, int m, int n,  INDArray d,
                 INDArray sep, int info );
    void ddisna( String job, int m, int n,  INDArray d,
                 INDArray sep, int info );
    void ssygst( int itype, String uplo, int n, INDArray a,
                 int lda,  INDArray b, int ldb,
                 int info );
    void dsygst( int itype, String uplo, int n, INDArray a,
                 int lda,  INDArray b, int ldb,
                 int info );
    void chegst( int itype, String uplo, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 int info );
    void zhegst( int itype, String uplo, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 int info );
    void sspgst( int itype, String uplo, int n, INDArray ap,
                 INDArray bp, int info );
    void dspgst( int itype, String uplo, int n, INDArray ap,
                 INDArray bp, int info );
    void chpgst( int itype, String uplo, int n,
                 IComplexNumber ap,  IComplexNumber bp,
                 int info );
    void zhpgst( int itype, String uplo, int n,
                 IComplexNumber ap,  IComplexNumber bp,
                 int info );
    void ssbgst( String vect, String uplo, int n, int ka,
                 int kb, INDArray ab, int ldab,
                 INDArray bb, int ldbb, INDArray x,
                 int ldx, INDArray work, int info );
    void dsbgst( String vect, String uplo, int n, int ka,
                 int kb, INDArray ab, int ldab,
                 INDArray bb, int ldbb, INDArray x,
                 int ldx, INDArray work, int info );
    void chbgst( String vect, String uplo, int n, int ka,
                 int kb, IComplexNumber ab, int ldab,
                 IComplexNumber bb, int ldbb,
                 IComplexNumber x, int ldx,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void zhbgst( String vect, String uplo, int n, int ka,
                 int kb, IComplexNumber ab, int ldab,
                 IComplexNumber bb, int ldbb,
                 IComplexNumber x, int ldx,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void spbstf( String uplo, int n, int kb, INDArray bb,
                 int ldbb, int info );
    void dpbstf( String uplo, int n, int kb, INDArray bb,
                 int ldbb, int info );
    void cpbstf( String uplo, int n, int kb,
                 IComplexNumber bb, int ldbb,
                 int info );
    void zpbstf( String uplo, int n, int kb,
                 IComplexNumber bb, int ldbb,
                 int info );
    void sgehrd( int n, int ilo, int ihi, INDArray a,
                 int lda, INDArray tau, INDArray work, int lwork,
                 int info );
    void dgehrd( int n, int ilo, int ihi, INDArray a,
                 int lda, INDArray tau, INDArray work,
                 int lwork, int info );
    void cgehrd( int n, int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zgehrd( int n, int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void sorghr( int n, int ilo, int ihi, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void dorghr( int n, int ilo, int ihi, INDArray a,
                 int lda,  INDArray tau, INDArray work,
                 int lwork, int info );
    void sormhr( String side, String trans, int m, int n,
                 int ilo, int ihi,  INDArray a,
                 int lda,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int lwork,
                 int info );
    void dormhr( String side, String trans, int m, int n,
                 int ilo, int ihi,  INDArray a,
                 int lda,  INDArray tau, INDArray c,
                 int ldc, INDArray work, int lwork,
                 int info );
    void cunghr( int n, int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber work,
                 int lwork, int info );
    void zunghr( int n, int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber tau,
                 IComplexNumber work, int lwork,
                 int info );
    void cunmhr( String side, String trans, int m, int n,
                 int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber c,
                 int ldc, IComplexNumber work,
                 int lwork, int info );
    void zunmhr( String side, String trans, int m, int n,
                 int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber tau, IComplexNumber c,
                 int ldc, IComplexNumber work,
                 int lwork, int info );
    void sgebal( String job, int n, INDArray a, int lda,
                 int ilo, int ihi, INDArray scale,
                 int info );
    void dgebal( String job, int n, INDArray a, int lda,
                 int ilo, int ihi, INDArray scale,
                 int info );
    void cgebal( String job, int n, IComplexNumber a,
                 int lda, int ilo, int ihi,
                 INDArray scale, int info );
    void zgebal( String job, int n, IComplexNumber a,
                 int lda, int ilo, int ihi,
                 INDArray scale, int info );
    void sgebak( String job, String side, int n, int ilo,
                 int ihi,  INDArray scale, int m,
                 INDArray v, int ldv, int info );
    void dgebak( String job, String side, int n, int ilo,
                 int ihi,  INDArray scale, int m,
                 INDArray v, int ldv, int info );
    void cgebak( String job, String side, int n, int ilo,
                 int ihi,  INDArray scale, int m,
                 IComplexNumber v, int ldv,
                 int info );
    void zgebak( String job, String side, int n, int ilo,
                 int ihi,  INDArray scale, int m,
                 IComplexNumber v, int ldv,
                 int info );
    void shseqr( String job, String compz, int n, int ilo,
                 int ihi, INDArray h, int ldh, INDArray wr,
                 INDArray wi, INDArray z, int ldz, INDArray work,
                 int lwork, int info );
    void dhseqr( String job, String compz, int n, int ilo,
                 int ihi, INDArray h, int ldh, INDArray wr,
                 INDArray wi, INDArray z, int ldz, INDArray work,
                 int lwork, int info );
    void chseqr( String job, String compz, int n, int ilo,
                 int ihi, IComplexNumber h, int ldh,
                 IComplexNumber w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, int info );
    void zhseqr( String job, String compz, int n, int ilo,
                 int ihi, IComplexNumber h, int ldh,
                 IComplexNumber w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, int info );
    void shsein( String job, String eigsrc, String initv,
                 int select, int n,  INDArray h,
                 int ldh, INDArray wr,  INDArray wi, INDArray vl,
                 int ldvl, INDArray vr, int ldvr,
                 int mm, int m, INDArray work,
                 int ifaill, int ifailr, int info );
    void dhsein( String job, String eigsrc, String initv,
                 int select, int n,  INDArray h,
                 int ldh, INDArray wr,  INDArray wi, INDArray vl,
                 int ldvl, INDArray vr, int ldvr,
                 int mm, int m, INDArray work,
                 int ifaill, int ifailr, int info );
    void chsein( String job, String eigsrc, String initv,
                 int select, int n,
                 IComplexNumber h, int ldh,
                 IComplexNumber w, IComplexNumber vl,
                 int ldvl, IComplexNumber vr,
                 int ldvr, int mm, int m,
                 IComplexNumber work, INDArray rwork,
                 int ifaill, int ifailr, int info );
    void zhsein( String job, String eigsrc, String initv,
                 int select, int n,
                 IComplexNumber h, int ldh,
                 IComplexNumber w, IComplexNumber vl,
                 int ldvl, IComplexNumber vr,
                 int ldvr, int mm, int m,
                 IComplexNumber work, INDArray rwork,
                 int ifaill, int ifailr, int info );
    void strevc( String side, String howmny, int select,
                 int n,  INDArray t, int ldt, INDArray vl,
                 int ldvl, INDArray vr, int ldvr,
                 int mm, int m, INDArray work,
                 int info );
    void dtrevc( String side, String howmny, int select,
                 int n,  INDArray t, int ldt, INDArray vl,
                 int ldvl, INDArray vr, int ldvr,
                 int mm, int m, INDArray work,
                 int info );
    void ctrevc( String side, String howmny,  int select,
                 int n, IComplexNumber t, int ldt,
                 IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr, int mm,
                 int m, IComplexNumber work, INDArray rwork,
                 int info );
    void ztrevc( String side, String howmny,  int select,
                 int n, IComplexNumber t, int ldt,
                 IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr, int mm,
                 int m, IComplexNumber work, INDArray rwork,
                 int info );
    void strsna( String job, String howmny,  int select,
                 int n,  INDArray t, int ldt,
                 INDArray vl, int ldvl,  INDArray vr,
                 int ldvr, INDArray s, INDArray sep, int mm,
                 int m, INDArray work, int ldwork,
                 int iwork, int info );
    void dtrsna( String job, String howmny,  int select,
                 int n,  INDArray t, int ldt,
                 INDArray vl, int ldvl,  INDArray vr,
                 int ldvr, INDArray s, INDArray sep, int mm,
                 int m, INDArray work, int ldwork,
                 int iwork, int info );
    void ctrsna( String job, String howmny,  int select,
                 int n,  IComplexNumber t,
                 int ldt,  IComplexNumber vl,
                 int ldvl,  IComplexNumber vr,
                 int ldvr, INDArray s, INDArray sep, int mm,
                 int m, IComplexNumber work,
                 int ldwork, INDArray rwork, int info );
    void ztrsna( String job, String howmny,  int select,
                 int n,  IComplexNumber t,
                 int ldt,  IComplexNumber vl,
                 int ldvl,  IComplexNumber vr,
                 int ldvr, INDArray s, INDArray sep, int mm,
                 int m, IComplexNumber work,
                 int ldwork, INDArray rwork, int info );
    void strexc( String compq, int n, INDArray t, int ldt,
                 INDArray q, int ldq, int ifst,
                 int ilst, INDArray work, int info );
    void dtrexc( String compq, int n, INDArray t, int ldt,
                 INDArray q, int ldq, int ifst,
                 int ilst, INDArray work, int info );
    void ctrexc( String compq, int n, IComplexNumber t,
                 int ldt, IComplexNumber q, int ldq,
                 int ifst, int ilst, int info );
    void ztrexc( String compq, int n, IComplexNumber t,
                 int ldt, IComplexNumber q, int ldq,
                 int ifst, int ilst, int info );
    void strsen( String job, String compq,  int select,
                 int n, INDArray t, int ldt, INDArray q,
                 int ldq, INDArray wr, INDArray wi, int m,
                 INDArray s, INDArray sep, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dtrsen( String job, String compq,  int select,
                 int n, INDArray t, int ldt, INDArray q,
                 int ldq, INDArray wr, INDArray wi, int m,
                 INDArray s, INDArray sep, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void ctrsen( String job, String compq,  int select,
                 int n, IComplexNumber t, int ldt,
                 IComplexNumber q, int ldq,
                 IComplexNumber w, int m, INDArray s,
                 INDArray sep, IComplexNumber work, int lwork,
                 int info );
    void ztrsen( String job, String compq,  int select,
                 int n, IComplexNumber t, int ldt,
                 IComplexNumber q, int ldq,
                 IComplexNumber w, int m, INDArray s,
                 INDArray sep, IComplexNumber work, int lwork,
                 int info );
    void strsyl( String trana, String tranb, int isgn, int m,
                 int n,  INDArray a, int lda,
                 INDArray b, int ldb, INDArray c, int ldc,
                 INDArray scale, int info );
    void dtrsyl( String trana, String tranb, int isgn, int m,
                 int n,  INDArray a, int lda,
                 INDArray b, int ldb, INDArray c,
                 int ldc, INDArray scale, int info );
    void ctrsyl( String trana, String tranb, int isgn, int m,
                 int n,  IComplexNumber a,
                 int lda,  IComplexNumber b,
                 int ldb, IComplexNumber c, int ldc,
                 INDArray scale, int info );
    void ztrsyl( String trana, String tranb, int isgn, int m,
                 int n,  IComplexNumber a,
                 int lda,  IComplexNumber b,
                 int ldb, IComplexNumber c, int ldc,
                 INDArray scale, int info );
    void sgghrd( String compq, String compz, int n, int ilo,
                 int ihi, INDArray a, int lda, INDArray b,
                 int ldb, INDArray q, int ldq, INDArray z,
                 int ldz, int info );
    void dgghrd( String compq, String compz, int n, int ilo,
                 int ihi, INDArray a, int lda, INDArray b,
                 int ldb, INDArray q, int ldq, INDArray z,
                 int ldz, int info );
    void cgghrd( String compq, String compz, int n, int ilo,
                 int ihi, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz,
                 int info );
    void zgghrd( String compq, String compz, int n, int ilo,
                 int ihi, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz,
                 int info );
    void sgghd3( String compq, String compz, int n, int ilo,
                 int ihi, INDArray a, int lda, INDArray b,
                 int ldb, INDArray q, int ldq, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int info );
    void dgghd3( String compq, String compz, int n, int ilo,
                 int ihi, INDArray a, int lda, INDArray b,
                 int ldb, INDArray q, int ldq, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int info );
    void cgghd3( String compq, String compz, int n,
                 int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 int info );
    void zgghd3( String compq, String compz, int n,
                 int ilo, int ihi,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 int info );
    void sggbal( String job, int n, INDArray a, int lda,
                 INDArray b, int ldb, int ilo, int ihi,
                 INDArray lscale, INDArray rscale, INDArray work,
                 int info );
    void dggbal( String job, int n, INDArray a, int lda,
                 INDArray b, int ldb, int ilo,
                 int ihi, INDArray lscale, INDArray rscale,
                 INDArray work, int info );
    void cggbal( String job, int n, IComplexNumber a,
                 int lda, IComplexNumber b, int ldb,
                 int ilo, int ihi, INDArray lscale,
                 INDArray rscale, INDArray work, int info );
    void zggbal( String job, int n, IComplexNumber a,
                 int lda, IComplexNumber b, int ldb,
                 int ilo, int ihi, INDArray lscale,
                 INDArray rscale, INDArray work, int info );
    void sggbak( String job, String side, int n, int ilo,
                 int ihi,  INDArray lscale,  INDArray rscale,
                 int m, INDArray v, int ldv,
                 int info );
    void dggbak( String job, String side, int n, int ilo,
                 int ihi,  INDArray lscale,  INDArray rscale,
                 int m, INDArray v, int ldv,
                 int info );
    void cggbak( String job, String side, int n, int ilo,
                 int ihi,  INDArray lscale,  INDArray rscale,
                 int m, IComplexNumber v, int ldv,
                 int info );
    void zggbak( String job, String side, int n, int ilo,
                 int ihi,  INDArray lscale,  INDArray rscale,
                 int m, IComplexNumber v, int ldv,
                 int info );
    void shgeqz( String job, String compq, String compz, int n,
                 int ilo, int ihi, INDArray h, int ldh,
                 INDArray t, int ldt, INDArray alphar, INDArray alphai,
                 INDArray beta, INDArray q, int ldq, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int info );
    void dhgeqz( String job, String compq, String compz, int n,
                 int ilo, int ihi, INDArray h,
                 int ldh, INDArray t, int ldt, INDArray alphar,
                 INDArray alphai, INDArray beta, INDArray q, int ldq,
                 INDArray z, int ldz, INDArray work, int lwork,
                 int info );
    void chgeqz( String job, String compq, String compz, int n,
                 int ilo, int ihi, IComplexNumber h,
                 int ldh, IComplexNumber t, int ldt,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int info );
    void zhgeqz( String job, String compq, String compz, int n,
                 int ilo, int ihi, IComplexNumber h,
                 int ldh, IComplexNumber t, int ldt,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void stgevc( String side, String howmny,  int select,
                 int n,  INDArray s, int lds,
                 INDArray p, int ldp, INDArray vl,
                 int ldvl, INDArray vr, int ldvr,
                 int mm, int m, INDArray work,
                 int info );
    void dtgevc( String side, String howmny,  int select,
                 int n,  INDArray s, int lds,
                 INDArray p, int ldp, INDArray vl,
                 int ldvl, INDArray vr, int ldvr,
                 int mm, int m, INDArray work,
                 int info );
    void ctgevc( String side, String howmny,  int select,
                 int n,  IComplexNumber s,
                 int lds,  IComplexNumber p,
                 int ldp, IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr, int mm,
                 int m, IComplexNumber work, INDArray rwork,
                 int info );
    void ztgevc( String side, String howmny,  int select,
                 int n,  IComplexNumber s,
                 int lds,  IComplexNumber p,
                 int ldp, IComplexNumber vl,
                 int ldvl, IComplexNumber vr,
                 int ldvr, int mm, int m,
                 IComplexNumber work, INDArray rwork,
                 int info );
    void stgexc( int wantq, int wantz, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray q, int ldq, INDArray z, int ldz,
                 int ifst, int ilst, INDArray work,
                 int lwork, int info );
    void dtgexc( int wantq, int wantz, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray q, int ldq, INDArray z, int ldz,
                 int ifst, int ilst, INDArray work,
                 int lwork, int info );
    void ctgexc( int wantq, int wantz, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz, int ifst,
                 int ilst, int info );
    void ztgexc( int wantq, int wantz, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz, int ifst,
                 int ilst, int info );
    void stgsen( int ijob, int wantq,
                 int wantz,  int select,
                 int n, INDArray a, int lda, INDArray b,
                 int ldb, INDArray alphar, INDArray alphai, INDArray beta,
                 INDArray q, int ldq, INDArray z, int ldz,
                 int m, INDArray pl, INDArray pr, INDArray dif,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void dtgsen( int ijob, int wantq,
                 int wantz,  int select,
                 int n, INDArray a, int lda, INDArray b,
                 int ldb, INDArray alphar, INDArray alphai,
                 INDArray beta, INDArray q, int ldq, INDArray z,
                 int ldz, int m, INDArray pl, INDArray pr,
                 INDArray dif, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void ctgsen( int ijob, int wantq,
                 int wantz,  int select,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz, int m,
                 INDArray pl, INDArray pr, INDArray dif,
                 IComplexNumber work, int lwork,
                 int iwork, int liwork, int info );
    void ztgsen( int ijob, int wantq,
                 int wantz,  int select,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber q, int ldq,
                 IComplexNumber z, int ldz, int m,
                 INDArray pl, INDArray pr, INDArray dif,
                 IComplexNumber work, int lwork,
                 int iwork, int liwork, int info );
    void stgsyl( String trans, int ijob, int m, int n,
                 INDArray a, int lda,  INDArray b,
                 int ldb, INDArray c, int ldc,  INDArray d,
                 int ldd,  INDArray e, int lde, INDArray f,
                 int ldf, INDArray scale, INDArray dif, INDArray work,
                 int lwork, int iwork, int info );
    void dtgsyl( String trans, int ijob, int m, int n,
                 INDArray a, int lda,  INDArray b,
                 int ldb, INDArray c, int ldc,
                 INDArray d, int ldd,  INDArray e,
                 int lde, INDArray f, int ldf, INDArray scale,
                 INDArray dif, INDArray work, int lwork,
                 int iwork, int info );
    void ctgsyl( String trans, int ijob, int m, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber c, int ldc,
                 IComplexNumber d, int ldd,
                 IComplexNumber e, int lde,
                 IComplexNumber f, int ldf, INDArray scale,
                 INDArray dif, IComplexNumber work, int lwork,
                 int iwork, int info );
    void ztgsyl( String trans, int ijob, int m, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber c, int ldc,
                 IComplexNumber d, int ldd,
                 IComplexNumber e, int lde,
                 IComplexNumber f, int ldf, INDArray scale,
                 INDArray dif, IComplexNumber work, int lwork,
                 int iwork, int info );
    void stgsna( String job, String howmny,  int select,
                 int n,  INDArray a, int lda,
                 INDArray b, int ldb,  INDArray vl,
                 int ldvl,  INDArray vr, int ldvr,
                 INDArray s, INDArray dif, int mm, int m,
                 INDArray work, int lwork, int iwork,
                 int info );
    void dtgsna( String job, String howmny,  int select,
                 int n,  INDArray a, int lda,
                 INDArray b, int ldb,  INDArray vl,
                 int ldvl,  INDArray vr, int ldvr,
                 INDArray s, INDArray dif, int mm, int m,
                 INDArray work, int lwork, int iwork,
                 int info );
    void ctgsna( String job, String howmny,  int select,
                 int n,  IComplexNumber a,
                 int lda,  IComplexNumber b,
                 int ldb,  IComplexNumber vl,
                 int ldvl,  IComplexNumber vr,
                 int ldvr, INDArray s, INDArray dif, int mm,
                 int m, IComplexNumber work,
                 int lwork, int iwork, int info );
    void ztgsna( String job, String howmny,  int select,
                 int n,  IComplexNumber a,
                 int lda,  IComplexNumber b,
                 int ldb,  IComplexNumber vl,
                 int ldvl,  IComplexNumber vr,
                 int ldvr, INDArray s, INDArray dif, int mm,
                 int m, IComplexNumber work,
                 int lwork, int iwork, int info );
    void sggsvp( String jobu, String jobv, String jobq, int m,
                 int p, int n, INDArray a, int lda,
                 INDArray b, int ldb, INDArray tola, INDArray tolb,
                 int k, int l, INDArray u, int ldu,
                 INDArray v, int ldv, INDArray q, int ldq,
                 int iwork, INDArray tau, INDArray work,
                 int info );
    void dggsvp( String jobu, String jobv, String jobq, int m,
                 int p, int n, INDArray a, int lda,
                 INDArray b, int ldb, INDArray tola, INDArray tolb,
                 int k, int l, INDArray u, int ldu,
                 INDArray v, int ldv, INDArray q, int ldq,
                 int iwork, INDArray tau, INDArray work,
                 int info );
    void cggsvp( String jobu, String jobv, String jobq, int m,
                 int p, int n, IComplexNumber a,
                 int lda, IComplexNumber b, int ldb,
                 INDArray tola, INDArray tolb, int k, int l,
                 IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv,
                 IComplexNumber q, int ldq, int iwork,
                 INDArray rwork, IComplexNumber tau,
                 IComplexNumber work, int info );
    void zggsvp( String jobu, String jobv, String jobq, int m,
                 int p, int n, IComplexNumber a,
                 int lda, IComplexNumber b, int ldb,
                 INDArray tola, INDArray tolb, int k, int l,
                 IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv,
                 IComplexNumber q, int ldq,
                 int iwork, INDArray rwork,
                 IComplexNumber tau, IComplexNumber work,
                 int info );
    void sggsvp3( String jobu, String jobv, String jobq, int m,
                  int p, int n, INDArray a, int lda,
                  INDArray b, int ldb, INDArray tola, INDArray tolb,
                  int k, int l, INDArray u, int ldu,
                  INDArray v, int ldv, INDArray q, int ldq,
                  int iwork, INDArray tau, INDArray work,
                  int lwork, int info );
    void dggsvp3( String jobu, String jobv, String jobq, int m,
                  int p, int n, INDArray a, int lda,
                  INDArray b, int ldb, INDArray tola, INDArray tolb,
                  int k, int l, INDArray u, int ldu,
                  INDArray v, int ldv, INDArray q, int ldq,
                  int iwork, INDArray tau, INDArray work,
                  int lwork, int info );
    void cggsvp3( String jobu, String jobv, String jobq, int m,
                  int p, int n, IComplexNumber a,
                  int lda, IComplexNumber b, int ldb,
                  INDArray tola, INDArray tolb, int k, int l,
                  IComplexNumber u, int ldu,
                  IComplexNumber v, int ldv,
                  IComplexNumber q, int ldq, int iwork,
                  INDArray rwork, IComplexNumber tau,
                  IComplexNumber work, int lwork,
                  int info );
    void zggsvp3( String jobu, String jobv, String jobq, int m,
                  int p, int n, IComplexNumber a,
                  int lda, IComplexNumber b, int ldb,
                  INDArray tola, INDArray tolb, int k, int l,
                  IComplexNumber u, int ldu,
                  IComplexNumber v, int ldv,
                  IComplexNumber q, int ldq,
                  int iwork, INDArray rwork,
                  IComplexNumber tau, IComplexNumber work,
                  int lwork, int info );
    void stgsja( String jobu, String jobv, String jobq, int m,
                 int p, int n, int k, int l,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray tola, INDArray tolb, INDArray alpha, INDArray beta,
                 INDArray u, int ldu, INDArray v, int ldv,
                 INDArray q, int ldq, INDArray work, int ncycle,
                 int info );
    void dtgsja( String jobu, String jobv, String jobq, int m,
                 int p, int n, int k, int l,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray tola, INDArray tolb, INDArray alpha, INDArray beta,
                 INDArray u, int ldu, INDArray v, int ldv,
                 INDArray q, int ldq, INDArray work,
                 int ncycle, int info );
    void ctgsja( String jobu, String jobv, String jobq, int m,
                 int p, int n, int k, int l,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray tola,
                 INDArray tolb, INDArray alpha, INDArray beta,
                 IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv,
                 IComplexNumber q, int ldq,
                 IComplexNumber work, int ncycle,
                 int info );
    void ztgsja( String jobu, String jobv, String jobq, int m,
                 int p, int n, int k, int l,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray tola,
                 INDArray tolb, INDArray alpha, INDArray beta,
                 IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv,
                 IComplexNumber q, int ldq,
                 IComplexNumber work, int ncycle,
                 int info );
    void sgels( String trans, int m, int n, int nrhs,
                INDArray a, int lda, INDArray b, int ldb,
                INDArray work, int lwork, int info );
    void dgels( String trans, int m, int n, int nrhs,
                INDArray a, int lda, INDArray b, int ldb,
                INDArray work, int lwork, int info );
    void cgels( String trans, int m, int n, int nrhs,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb,
                IComplexNumber work, int lwork,
                int info );
    void zgels( String trans, int m, int n, int nrhs,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb,
                IComplexNumber work, int lwork,
                int info );
    void sgelsy( int m, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb,
                 int jpvt, INDArray rcond, int rank,
                 INDArray work, int lwork, int info );
    void dgelsy( int m, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb,
                 int jpvt, INDArray rcond, int rank,
                 INDArray work, int lwork, int info );
    void cgelsy( int m, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, int jpvt,
                 INDArray rcond, int rank, IComplexNumber work,
                 int lwork, INDArray rwork, int info );
    void zgelsy( int m, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, int jpvt,
                 INDArray rcond, int rank,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void sgelss( int m, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb, INDArray s,
                 INDArray rcond, int rank, INDArray work,
                 int lwork, int info );
    void dgelss( int m, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb, INDArray s,
                 INDArray rcond, int rank, INDArray work,
                 int lwork, int info );
    void cgelss( int m, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray s,
                 INDArray rcond, int rank, IComplexNumber work,
                 int lwork, INDArray rwork, int info );
    void zgelss( int m, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray s,
                 INDArray rcond, int rank,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void sgelsd( int m, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb, INDArray s,
                 INDArray rcond, int rank, INDArray work,
                 int lwork, int iwork, int info );
    void dgelsd( int m, int n, int nrhs, INDArray a,
                 int lda, INDArray b, int ldb, INDArray s,
                 INDArray rcond, int rank, INDArray work,
                 int lwork, int iwork, int info );
    void cgelsd( int m, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray s,
                 INDArray rcond, int rank, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int info );
    void zgelsd( int m, int n, int nrhs,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray s,
                 INDArray rcond, int rank,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int iwork, int info );
    void sgglse( int m, int n, int p, INDArray a,
                 int lda, INDArray b, int ldb, INDArray c,
                 INDArray d, INDArray x, INDArray work, int lwork,
                 int info );
    void dgglse( int m, int n, int p, INDArray a,
                 int lda, INDArray b, int ldb, INDArray c,
                 INDArray d, INDArray x, INDArray work, int lwork,
                 int info );
    void cgglse( int m, int n, int p,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber c, IComplexNumber d,
                 IComplexNumber x, IComplexNumber work,
                 int lwork, int info );
    void zgglse( int m, int n, int p,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber c, IComplexNumber d,
                 IComplexNumber x, IComplexNumber work,
                 int lwork, int info );
    void sggglm( int n, int m, int p, INDArray a,
                 int lda, INDArray b, int ldb, INDArray d,
                 INDArray x, INDArray y, INDArray work, int lwork,
                 int info );
    void dggglm( int n, int m, int p, INDArray a,
                 int lda, INDArray b, int ldb, INDArray d,
                 INDArray x, INDArray y, INDArray work, int lwork,
                 int info );
    void cggglm( int n, int m, int p,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber d, IComplexNumber x,
                 IComplexNumber y, IComplexNumber work,
                 int lwork, int info );
    void zggglm( int n, int m, int p,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber d, IComplexNumber x,
                 IComplexNumber y, IComplexNumber work,
                 int lwork, int info );
    void ssyev( String jobz, String uplo, int n, INDArray a,
                int lda, INDArray w, INDArray work, int lwork,
                int info );
    void dsyev( String jobz, String uplo, int n, INDArray a,
                int lda, INDArray w, INDArray work, int lwork,
                int info );
    void cheev( String jobz, String uplo, int n,
                IComplexNumber a, int lda, INDArray w,
                IComplexNumber work, int lwork, INDArray rwork,
                int info );
    void zheev( String jobz, String uplo, int n,
                IComplexNumber a, int lda, INDArray w,
                IComplexNumber work, int lwork,
                INDArray rwork, int info );
    void ssyevd( String jobz, String uplo, int n, INDArray a,
                 int lda, INDArray w, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dsyevd( String jobz, String uplo, int n, INDArray a,
                 int lda, INDArray w, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void cheevd( String jobz, String uplo, int n,
                 IComplexNumber a, int lda, INDArray w,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zheevd( String jobz, String uplo, int n,
                 IComplexNumber a, int lda, INDArray w,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void ssyevx( String jobz, String range, String uplo, int n,
                 INDArray a, int lda, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, INDArray z, int ldz,
                 INDArray work, int lwork, int iwork,
                 int ifail, int info );
    void dsyevx( String jobz, String range, String uplo, int n,
                 INDArray a, int lda, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, INDArray z, int ldz,
                 INDArray work, int lwork, int iwork,
                 int ifail, int info );
    void cheevx( String jobz, String range, String uplo, int n,
                 IComplexNumber a, int lda, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int ifail, int info );
    void zheevx( String jobz, String range, String uplo, int n,
                 IComplexNumber a, int lda, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int ifail, int info );
    void ssyevr( String jobz, String range, String uplo, int n,
                 INDArray a, int lda, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, INDArray z, int ldz,
                 int isuppz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dsyevr( String jobz, String range, String uplo, int n,
                 INDArray a, int lda, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, INDArray z, int ldz,
                 int isuppz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void cheevr( String jobz, String range, String uplo, int n,
                 IComplexNumber a, int lda, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, int isuppz,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zheevr( String jobz, String range, String uplo, int n,
                 IComplexNumber a, int lda, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, int isuppz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void sspev( String jobz, String uplo, int n, INDArray ap, INDArray w,
                INDArray z, int ldz, INDArray work, int info );
    void dspev( String jobz, String uplo, int n, INDArray ap, INDArray w,
                INDArray z, int ldz, INDArray work, int info );
    void chpev( String jobz, String uplo, int n,
                IComplexNumber ap, INDArray w, IComplexNumber z,
                int ldz, IComplexNumber work, INDArray rwork,
                int info );
    void zhpev( String jobz, String uplo, int n,
                IComplexNumber ap, INDArray w,
                IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork,
                int info );
    void sspevd( String jobz, String uplo, int n, INDArray ap, INDArray w,
                 INDArray z, int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dspevd( String jobz, String uplo, int n, INDArray ap,
                 INDArray w, INDArray z, int ldz, INDArray work,
                 int lwork, int iwork, int liwork,
                 int info );
    void chpevd( String jobz, String uplo, int n,
                 IComplexNumber ap, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, INDArray rwork, int lrwork,
                 int iwork, int liwork, int info );
    void zhpevd( String jobz, String uplo, int n,
                 IComplexNumber ap, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void sspevx( String jobz, String range, String uplo, int n,
                 INDArray ap, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void dspevx( String jobz, String range, String uplo, int n,
                 INDArray ap, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void chpevx( String jobz, String range, String uplo, int n,
                 IComplexNumber ap, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void zhpevx( String jobz, String range, String uplo, int n,
                 IComplexNumber ap, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void ssbev( String jobz, String uplo, int n, int kd,
                INDArray ab, int ldab, INDArray w, INDArray z,
                int ldz, INDArray work, int info );
    void dsbev( String jobz, String uplo, int n, int kd,
                INDArray ab, int ldab, INDArray w, INDArray z,
                int ldz, INDArray work, int info );
    void chbev( String jobz, String uplo, int n, int kd,
                IComplexNumber ab, int ldab, INDArray w,
                IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork, int info );
    void zhbev( String jobz, String uplo, int n, int kd,
                IComplexNumber ab, int ldab, INDArray w,
                IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork,
                int info );
    void ssbevd( String jobz, String uplo, int n, int kd,
                 INDArray ab, int ldab, INDArray w, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dsbevd( String jobz, String uplo, int n, int kd,
                 INDArray ab, int ldab, INDArray w, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void chbevd( String jobz, String uplo, int n, int kd,
                 IComplexNumber ab, int ldab, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zhbevd( String jobz, String uplo, int n, int kd,
                 IComplexNumber ab, int ldab, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void ssbevx( String jobz, String range, String uplo, int n,
                 int kd, INDArray ab, int ldab, INDArray q,
                 int ldq, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void dsbevx( String jobz, String range, String uplo, int n,
                 int kd, INDArray ab, int ldab, INDArray q,
                 int ldq, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void chbevx( String jobz, String range, String uplo, int n,
                 int kd, IComplexNumber ab, int ldab,
                 IComplexNumber q, int ldq, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void zhbevx( String jobz, String range, String uplo, int n,
                 int kd, IComplexNumber ab, int ldab,
                 IComplexNumber q, int ldq, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void sstev( String jobz, int n, INDArray d, INDArray e, INDArray z,
                int ldz, INDArray work, int info );
    void dstev( String jobz, int n, INDArray d, INDArray e, INDArray z,
                int ldz, INDArray work, int info );
    void sstevd( String jobz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void dstevd( String jobz, int n, INDArray d, INDArray e, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void sstevx( String jobz, String range, int n, INDArray d, INDArray e,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w, INDArray z,
                 int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void dstevx( String jobz, String range, int n, INDArray d,
                 INDArray e, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void sstevr( String jobz, String range, int n, INDArray d, INDArray e,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w, INDArray z,
                 int ldz, int isuppz, INDArray work,
                 int lwork, int iwork, int liwork,
                 int info );
    void dstevr( String jobz, String range, int n, INDArray d,
                 INDArray e, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, int isuppz,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void sgees( String jobvs, String sort, int select,
                int n, INDArray a, int lda, int sdim,
                INDArray wr, INDArray wi, INDArray vs, int ldvs,
                INDArray work, int lwork, int bwork,
                int info );
    void dgees( String jobvs, String sort, int select,
                int n, INDArray a, int lda, int sdim,
                INDArray wr, INDArray wi, INDArray vs, int ldvs,
                INDArray work, int lwork, int bwork,
                int info );
    void cgees( String jobvs, String sort, int select,
                int n, IComplexNumber a, int lda,
                int sdim, IComplexNumber w,
                IComplexNumber vs, int ldvs,
                IComplexNumber work, int lwork, INDArray rwork,
                int bwork, int info );
    void zgees( String jobvs, String sort, int select,
                int n, IComplexNumber a, int lda,
                int sdim, IComplexNumber w,
                IComplexNumber vs, int ldvs,
                IComplexNumber work, int lwork,
                INDArray rwork, int bwork, int info );
    void sgeesx( String jobvs, String sort, int select,
                 String sense, int n, INDArray a, int lda,
                 int sdim, INDArray wr, INDArray wi, INDArray vs,
                 int ldvs, INDArray rconde, INDArray rcondv, INDArray work,
                 int lwork, int iwork, int liwork,
                 int bwork, int info );
    void dgeesx( String jobvs, String sort, int select,
                 String sense, int n, INDArray a, int lda,
                 int sdim, INDArray wr, INDArray wi, INDArray vs,
                 int ldvs, INDArray rconde, INDArray rcondv,
                 INDArray work, int lwork, int iwork,
                 int liwork, int bwork,
                 int info );
    void cgeesx( String jobvs, String sort, int select,
                 String sense, int n, IComplexNumber a,
                 int lda, int sdim, IComplexNumber w,
                 IComplexNumber vs, int ldvs, INDArray rconde,
                 INDArray rcondv, IComplexNumber work,
                 int lwork, INDArray rwork, int bwork,
                 int info );
    void zgeesx( String jobvs, String sort, int select,
                 String sense, int n, IComplexNumber a,
                 int lda, int sdim, IComplexNumber w,
                 IComplexNumber vs, int ldvs, INDArray rconde,
                 INDArray rcondv, IComplexNumber work,
                 int lwork, INDArray rwork, int bwork,
                 int info );
    void sgeev( String jobvl, String jobvr, int n, INDArray a,
                int lda, INDArray wr, INDArray wi, INDArray vl,
                int ldvl, INDArray vr, int ldvr, INDArray work,
                int lwork, int info );
    void dgeev( String jobvl, String jobvr, int n, INDArray a,
                int lda, INDArray wr, INDArray wi, INDArray vl,
                int ldvl, INDArray vr, int ldvr, INDArray work,
                int lwork, int info );
    void cgeev( String jobvl, String jobvr, int n,
                IComplexNumber a, int lda,
                IComplexNumber w, IComplexNumber vl,
                int ldvl, IComplexNumber vr, int ldvr,
                IComplexNumber work, int lwork, INDArray rwork,
                int info );
    void zgeev( String jobvl, String jobvr, int n,
                IComplexNumber a, int lda,
                IComplexNumber w, IComplexNumber vl,
                int ldvl, IComplexNumber vr,
                int ldvr, IComplexNumber work,
                int lwork, INDArray rwork, int info );
    void sgeevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, INDArray a, int lda, INDArray wr,
                 INDArray wi, INDArray vl, int ldvl, INDArray vr,
                 int ldvr, int ilo, int ihi,
                 INDArray scale, INDArray abnrm, INDArray rconde, INDArray rcondv,
                 INDArray work, int lwork, int iwork,
                 int info );
    void dgeevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, INDArray a, int lda, INDArray wr,
                 INDArray wi, INDArray vl, int ldvl, INDArray vr,
                 int ldvr, int ilo, int ihi,
                 INDArray scale, INDArray abnrm, INDArray rconde,
                 INDArray rcondv, INDArray work, int lwork,
                 int iwork, int info );
    void cgeevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber w, IComplexNumber vl,
                 int ldvl, IComplexNumber vr,
                 int ldvr, int ilo, int ihi,
                 INDArray scale, INDArray abnrm, INDArray rconde, INDArray rcondv,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int info );
    void zgeevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber w, IComplexNumber vl,
                 int ldvl, IComplexNumber vr,
                 int ldvr, int ilo, int ihi,
                 INDArray scale, INDArray abnrm, INDArray rconde,
                 INDArray rcondv, IComplexNumber work,
                 int lwork, INDArray rwork, int info );
    void sgesvd( String jobu, String jobvt, int m, int n,
                 INDArray a, int lda, INDArray s, INDArray u,
                 int ldu, INDArray vt, int ldvt, INDArray work,
                 int lwork, int info );
    void dgesvd( String jobu, String jobvt, int m, int n,
                 INDArray a, int lda, INDArray s, INDArray u,
                 int ldu, INDArray vt, int ldvt, INDArray work,
                 int lwork, int info );
    void cgesvd( String jobu, String jobvt, int m, int n,
                 IComplexNumber a, int lda, INDArray s,
                 IComplexNumber u, int ldu,
                 IComplexNumber vt, int ldvt,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int info );
    void zgesvd( String jobu, String jobvt, int m, int n,
                 IComplexNumber a, int lda, INDArray s,
                 IComplexNumber u, int ldu,
                 IComplexNumber vt, int ldvt,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void sgesvdx( String jobu, String jobvt, String range, int m, int n,
                  INDArray a, int lda, int vl, int vu,
                  int il, int iu, int ns, INDArray s, INDArray u,
                  int ldu, INDArray vt, int ldvt, INDArray work,
                  int lwork, int iwork, int info );
    void dgesvdx( String jobu, String jobvt, String range, int m, int n,
                  INDArray a, int lda, int vl, int vu,
                  int il, int iu, int ns, INDArray s, INDArray u,
                  int ldu, INDArray vt, int ldvt, INDArray work,
                  int lwork, int iwork, int info );
    void cgesvdx( String jobu, String jobvt, String range, int m, int n,
                  IComplexNumber a, int lda, int vl, int vu,
                  int il, int iu, int ns, INDArray s,
                  IComplexNumber u, int ldu,
                  IComplexNumber vt, int ldvt,
                  IComplexNumber work, int lwork, INDArray rwork,
                  int iwork, int info );
    void zgesvdx( String jobu, String jobvt, String range, int m, int n,
                  IComplexNumber a, int lda, int vl, int vu,
                  int il, int iu, int ns, INDArray s,
                  IComplexNumber u, int ldu,
                  IComplexNumber vt, int ldvt,
                  IComplexNumber work, int lwork,
                  INDArray rwork, int iwork, int info );
    void sgesdd( String jobz, int m, int n, INDArray a,
                 int lda, INDArray s, INDArray u, int ldu,
                 INDArray vt, int ldvt, INDArray work, int lwork,
                 int iwork, int info );
    void dgesdd( String jobz, int m, int n, INDArray a,
                 int lda, INDArray s, INDArray u, int ldu,
                 INDArray vt, int ldvt, INDArray work,
                 int lwork, int iwork, int info );
    void cgesdd( String jobz, int m, int n,
                 IComplexNumber a, int lda, INDArray s,
                 IComplexNumber u, int ldu,
                 IComplexNumber vt, int ldvt,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int iwork, int info );
    void zgesdd( String jobz, int m, int n,
                 IComplexNumber a, int lda, INDArray s,
                 IComplexNumber u, int ldu,
                 IComplexNumber vt, int ldvt,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int iwork, int info );
    void dgejsv( String joba, String jobu, String jobv, String jobr, String jobt,
                 String jobp, int m, int n, INDArray a,
                 int lda, INDArray sva, INDArray u, int ldu,
                 INDArray v, int ldv, INDArray work, int lwork,
                 int iwork, int info );
    void sgejsv( String joba, String jobu, String jobv, String jobr, String jobt,
                 String jobp, int m, int n, INDArray a,
                 int lda, INDArray sva, INDArray u, int ldu,
                 INDArray v, int ldv, INDArray work, int lwork,
                 int iwork, int info );
    void cgejsv( String joba, String jobu, String jobv, String jobr, String jobt,
                 String jobp, int m, int n, IComplexNumber a,
                 int lda, INDArray sva, IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv, IComplexNumber cwork,
                 int lwork, INDArray work, int lrwork,
                 int iwork, int info );
    void zgejsv( String joba, String jobu, String jobv, String jobr, String jobt,
                 String jobp, int m, int n, IComplexNumber a,
                 int lda, INDArray sva, IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv, IComplexNumber cwork,
                 int lwork, INDArray work, int lrwork,
                 int iwork, int info );
    void dgesvj( String joba, String jobu, String jobv, int m,
                 int n, INDArray a, int lda, INDArray sva,
                 int mv, INDArray v, int ldv, INDArray work,
                 int lwork, int info );
    void sgesvj( String joba, String jobu, String jobv, int m,
                 int n, INDArray a, int lda, INDArray sva,
                 int mv, INDArray v, int ldv, INDArray work,
                 int lwork, int info );
    void cgesvj( String joba, String jobu, String jobv, int m,
                 int n, IComplexNumber a, int lda, INDArray sva,
                 int mv, IComplexNumber v, int ldv,
                 IComplexNumber cwork, int lwork, INDArray rwork,
                 int lrwork, int info );
    void zgesvj( String joba, String jobu, String jobv, int m,
                 int n, IComplexNumber a, int lda, INDArray sva,
                 int mv, IComplexNumber v, int ldv,
                 IComplexNumber cwork, int lwork, INDArray rwork,
                 int lrwork, int info );
    void sggsvd( String jobu, String jobv, String jobq, int m,
                 int n, int p, int k, int l,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray alpha, INDArray beta, INDArray u, int ldu,
                 INDArray v, int ldv, INDArray q, int ldq,
                 INDArray work, int iwork, int info );
    void dggsvd( String jobu, String jobv, String jobq, int m,
                 int n, int p, int k, int l,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray alpha, INDArray beta, INDArray u, int ldu,
                 INDArray v, int ldv, INDArray q, int ldq,
                 INDArray work, int iwork, int info );
    void cggsvd( String jobu, String jobv, String jobq, int m,
                 int n, int p, int k, int l,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray alpha,
                 INDArray beta, IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv,
                 IComplexNumber q, int ldq,
                 IComplexNumber work, INDArray rwork, int iwork,
                 int info );
    void zggsvd( String jobu, String jobv, String jobq, int m,
                 int n, int p, int k, int l,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray alpha,
                 INDArray beta, IComplexNumber u, int ldu,
                 IComplexNumber v, int ldv,
                 IComplexNumber q, int ldq,
                 IComplexNumber work, INDArray rwork,
                 int iwork, int info );
    void sggsvd3( String jobu, String jobv, String jobq, int m,
                  int n, int p, int k, int l,
                  INDArray a, int lda, INDArray b, int ldb,
                  INDArray alpha, INDArray beta, INDArray u, int ldu,
                  INDArray v, int ldv, INDArray q, int ldq,
                  INDArray work, int lwork, int iwork,
                  int info );
    void dggsvd3( String jobu, String jobv, String jobq, int m,
                  int n, int p, int k, int l,
                  INDArray a, int lda, INDArray b, int ldb,
                  INDArray alpha, INDArray beta, INDArray u, int ldu,
                  INDArray v, int ldv, INDArray q, int ldq,
                  INDArray work, int lwork, int iwork,
                  int info );
    void cggsvd3( String jobu, String jobv, String jobq, int m,
                  int n, int p, int k, int l,
                  IComplexNumber a, int lda,
                  IComplexNumber b, int ldb, INDArray alpha,
                  INDArray beta, IComplexNumber u, int ldu,
                  IComplexNumber v, int ldv,
                  IComplexNumber q, int ldq,
                  IComplexNumber work, int lwork, INDArray rwork,
                  int iwork, int info );
    void zggsvd3( String jobu, String jobv, String jobq, int m,
                  int n, int p, int k, int l,
                  IComplexNumber a, int lda,
                  IComplexNumber b, int ldb, INDArray alpha,
                  INDArray beta, IComplexNumber u, int ldu,
                  IComplexNumber v, int ldv,
                  IComplexNumber q, int ldq,
                  IComplexNumber work, int lwork,
                  INDArray rwork, int iwork, int info );
    void ssygv( int itype, String jobz, String uplo, int n,
                INDArray a, int lda, INDArray b, int ldb,
                INDArray w, INDArray work, int lwork, int info );
    void dsygv( int itype, String jobz, String uplo, int n,
                INDArray a, int lda, INDArray b, int ldb,
                INDArray w, INDArray work, int lwork,
                int info );
    void chegv( int itype, String jobz, String uplo, int n,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb, INDArray w,
                IComplexNumber work, int lwork, INDArray rwork,
                int info );
    void zhegv( int itype, String jobz, String uplo, int n,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb, INDArray w,
                IComplexNumber work, int lwork,
                INDArray rwork, int info );
    void ssygvd( int itype, String jobz, String uplo, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray w, INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void dsygvd( int itype, String jobz, String uplo, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray w, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void chegvd( int itype, String jobz, String uplo, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray w,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zhegvd( int itype, String jobz, String uplo, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray w,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void ssygvx( int itype, String jobz, String range, String uplo,
                 int n, INDArray a, int lda, INDArray b,
                 int ldb, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int lwork,
                 int iwork, int ifail, int info );
    void dsygvx( int itype, String jobz, String range, String uplo,
                 int n, INDArray a, int lda, INDArray b,
                 int ldb, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int lwork,
                 int iwork, int ifail, int info );
    void chegvx( int itype, String jobz, String range, String uplo,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int ifail, int info );
    void zhegvx( int itype, String jobz, String range, String uplo,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int ifail, int info );
    void sspgv( int itype, String jobz, String uplo, int n,
                INDArray ap, INDArray bp, INDArray w, INDArray z, int ldz,
                INDArray work, int info );
    void dspgv( int itype, String jobz, String uplo, int n,
                INDArray ap, INDArray bp, INDArray w, INDArray z,
                int ldz, INDArray work, int info );
    void chpgv( int itype, String jobz, String uplo, int n,
                IComplexNumber ap, IComplexNumber bp, INDArray w,
                IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork, int info );
    void zhpgv( int itype, String jobz, String uplo, int n,
                IComplexNumber ap, IComplexNumber bp,
                INDArray w, IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork,
                int info );
    void sspgvd( int itype, String jobz, String uplo, int n,
                 INDArray ap, INDArray bp, INDArray w, INDArray z, int ldz,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void dspgvd( int itype, String jobz, String uplo, int n,
                 INDArray ap, INDArray bp, INDArray w, INDArray z,
                 int ldz, INDArray work, int lwork,
                 int iwork, int liwork, int info );
    void chpgvd( int itype, String jobz, String uplo, int n,
                 IComplexNumber ap, IComplexNumber bp,
                 INDArray w, IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zhpgvd( int itype, String jobz, String uplo, int n,
                 IComplexNumber ap, IComplexNumber bp,
                 INDArray w, IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void sspgvx( int itype, String jobz, String range, String uplo,
                 int n, INDArray ap, INDArray bp, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, INDArray z, int ldz,
                 INDArray work, int iwork, int ifail,
                 int info );
    void dspgvx( int itype, String jobz, String range, String uplo,
                 int n, INDArray ap, INDArray bp, INDArray vl,
                 INDArray vu, int il, int iu, INDArray abstol,
                 int m, INDArray w, INDArray z, int ldz,
                 INDArray work, int iwork, int ifail,
                 int info );
    void chpgvx( int itype, String jobz, String range, String uplo,
                 int n, IComplexNumber ap,
                 IComplexNumber bp, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void zhpgvx( int itype, String jobz, String range, String uplo,
                 int n, IComplexNumber ap,
                 IComplexNumber bp, INDArray vl, INDArray vu,
                 int il, int iu, INDArray abstol,
                 int m, INDArray w, IComplexNumber z,
                 int ldz, IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void ssbgv( String jobz, String uplo, int n, int ka,
                int kb, INDArray ab, int ldab, INDArray bb,
                int ldbb, INDArray w, INDArray z, int ldz,
                INDArray work, int info );
    void dsbgv( String jobz, String uplo, int n, int ka,
                int kb, INDArray ab, int ldab, INDArray bb,
                int ldbb, INDArray w, INDArray z, int ldz,
                INDArray work, int info );
    void chbgv( String jobz, String uplo, int n, int ka,
                int kb, IComplexNumber ab, int ldab,
                IComplexNumber bb, int ldbb, INDArray w,
                IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork, int info );
    void zhbgv( String jobz, String uplo, int n, int ka,
                int kb, IComplexNumber ab, int ldab,
                IComplexNumber bb, int ldbb, INDArray w,
                IComplexNumber z, int ldz,
                IComplexNumber work, INDArray rwork,
                int info );
    void ssbgvd( String jobz, String uplo, int n, int ka,
                 int kb, INDArray ab, int ldab, INDArray bb,
                 int ldbb, INDArray w, INDArray z, int ldz,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void dsbgvd( String jobz, String uplo, int n, int ka,
                 int kb, INDArray ab, int ldab, INDArray bb,
                 int ldbb, INDArray w, INDArray z, int ldz,
                 INDArray work, int lwork, int iwork,
                 int liwork, int info );
    void chbgvd( String jobz, String uplo, int n, int ka,
                 int kb, IComplexNumber ab, int ldab,
                 IComplexNumber bb, int ldbb, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int lrwork, int iwork, int liwork,
                 int info );
    void zhbgvd( String jobz, String uplo, int n, int ka,
                 int kb, IComplexNumber ab, int ldab,
                 IComplexNumber bb, int ldbb, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork, int iwork,
                 int liwork, int info );
    void ssbgvx( String jobz, String range, String uplo, int n,
                 int ka, int kb, INDArray ab, int ldab,
                 INDArray bb, int ldbb, INDArray q, int ldq,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w, INDArray z,
                 int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void dsbgvx( String jobz, String range, String uplo, int n,
                 int ka, int kb, INDArray ab,
                 int ldab, INDArray bb, int ldbb, INDArray q,
                 int ldq, INDArray vl, INDArray vu, int il,
                 int iu, INDArray abstol, int m, INDArray w,
                 INDArray z, int ldz, INDArray work, int iwork,
                 int ifail, int info );
    void chbgvx( String jobz, String range, String uplo, int n,
                 int ka, int kb, IComplexNumber ab,
                 int ldab, IComplexNumber bb,
                 int ldbb, IComplexNumber q, int ldq,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, INDArray rwork, int iwork,
                 int ifail, int info );
    void zhbgvx( String jobz, String range, String uplo, int n,
                 int ka, int kb, IComplexNumber ab,
                 int ldab, IComplexNumber bb,
                 int ldbb, IComplexNumber q, int ldq,
                 INDArray vl, INDArray vu, int il, int iu,
                 INDArray abstol, int m, INDArray w,
                 IComplexNumber z, int ldz,
                 IComplexNumber work, INDArray rwork,
                 int iwork, int ifail, int info );
    void sgges( String jobvsl, String jobvsr, String sort,
                int selctg, int n, INDArray a,
                int lda, INDArray b, int ldb, int sdim,
                INDArray alphar, INDArray alphai, INDArray beta, INDArray vsl,
                int ldvsl, INDArray vsr, int ldvsr,
                INDArray work, int lwork, int bwork,
                int info );
    void dgges( String jobvsl, String jobvsr, String sort,
                int selctg, int n, INDArray a,
                int lda, INDArray b, int ldb,
                int sdim, INDArray alphar, INDArray alphai,
                INDArray beta, INDArray vsl, int ldvsl, INDArray vsr,
                int ldvsr, INDArray work, int lwork,
                int bwork, int info );
    void cgges( String jobvsl, String jobvsr, String sort,
                int selctg, int n,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb, int sdim,
                IComplexNumber alpha, IComplexNumber beta,
                IComplexNumber vsl, int ldvsl,
                IComplexNumber vsr, int ldvsr,
                IComplexNumber work, int lwork, INDArray rwork,
                int bwork, int info );
    void zgges( String jobvsl, String jobvsr, String sort,
                int selctg, int n,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb, int sdim,
                IComplexNumber alpha, IComplexNumber beta,
                IComplexNumber vsl, int ldvsl,
                IComplexNumber vsr, int ldvsr,
                IComplexNumber work, int lwork,
                INDArray rwork, int bwork, int info );
    void sgges3( String jobvsl, String jobvsr, String sort,
                 int selctg, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 int sdim, INDArray alphar, INDArray alphai,
                 INDArray beta, INDArray vsl, int ldvsl,
                 INDArray vsr, int ldvsr,
                 INDArray work, int lwork, int bwork,
                 int info );
    void dgges3( String jobvsl, String jobvsr, String sort,
                 int selctg, int n, INDArray a,
                 int lda, INDArray b, int ldb,
                 int sdim, INDArray alphar, INDArray alphai,
                 INDArray beta, INDArray vsl, int ldvsl, INDArray vsr,
                 int ldvsr, INDArray work, int lwork,
                 int bwork, int info );
    void cgges3( String jobvsl, String jobvsr, String sort,
                 int selctg, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 int sdim,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vsl, int ldvsl,
                 IComplexNumber vsr, int ldvsr,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int bwork, int info );
    void zgges3( String jobvsl, String jobvsr, String sort,
                 int selctg, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, int sdim,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vsl, int ldvsl,
                 IComplexNumber vsr, int ldvsr,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int bwork, int info );
    void sggesx( String jobvsl, String jobvsr, String sort,
                 int selctg, String sense, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 int sdim, INDArray alphar, INDArray alphai, INDArray beta,
                 INDArray vsl, int ldvsl, INDArray vsr,
                 int ldvsr, INDArray rconde, INDArray rcondv,
                 INDArray work, int lwork, int iwork,
                 int liwork, int bwork,
                 int info );
    void dggesx( String jobvsl, String jobvsr, String sort,
                 int selctg, String sense, int n,
                 INDArray a, int lda, INDArray b, int ldb,
                 int sdim, INDArray alphar, INDArray alphai,
                 INDArray beta, INDArray vsl, int ldvsl, INDArray vsr,
                 int ldvsr, INDArray rconde, INDArray rcondv,
                 INDArray work, int lwork, int iwork,
                 int liwork, int bwork,
                 int info );
    void cggesx( String jobvsl, String jobvsr, String sort,
                 int selctg, String sense, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, int sdim,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vsl, int ldvsl,
                 IComplexNumber vsr, int ldvsr, INDArray rconde,
                 INDArray rcondv, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int liwork, int bwork,
                 int info );
    void zggesx( String jobvsl, String jobvsr, String sort,
                 int selctg, String sense, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb, int sdim,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vsl, int ldvsl,
                 IComplexNumber vsr, int ldvsr,
                 INDArray rconde, INDArray rcondv, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int liwork, int bwork,
                 int info );
    void sggev( String jobvl, String jobvr, int n, INDArray a,
                int lda, INDArray b, int ldb, INDArray alphar,
                INDArray alphai, INDArray beta, INDArray vl, int ldvl,
                INDArray vr, int ldvr, INDArray work, int lwork,
                int info );
    void dggev( String jobvl, String jobvr, int n, INDArray a,
                int lda, INDArray b, int ldb, INDArray alphar,
                INDArray alphai, INDArray beta, INDArray vl, int ldvl,
                INDArray vr, int ldvr, INDArray work,
                int lwork, int info );
    void cggev( String jobvl, String jobvr, int n,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb,
                IComplexNumber alpha, IComplexNumber beta,
                IComplexNumber vl, int ldvl,
                IComplexNumber vr, int ldvr,
                IComplexNumber work, int lwork, INDArray rwork,
                int info );
    void zggev( String jobvl, String jobvr, int n,
                IComplexNumber a, int lda,
                IComplexNumber b, int ldb,
                IComplexNumber alpha, IComplexNumber beta,
                IComplexNumber vl, int ldvl,
                IComplexNumber vr, int ldvr,
                IComplexNumber work, int lwork,
                INDArray rwork, int info );
    void sggev3( String jobvl, String jobvr, int n, INDArray a,
                 int lda, INDArray b, int ldb, INDArray alphar,
                 INDArray alphai, INDArray beta, INDArray vl, int ldvl,
                 INDArray vr, int ldvr, INDArray work, int lwork,
                 int info );
    void dggev3( String jobvl, String jobvr, int n, INDArray a,
                 int lda, INDArray b, int ldb, INDArray alphar,
                 INDArray alphai, INDArray beta, INDArray vl, int ldvl,
                 INDArray vr, int ldvr, INDArray work,
                 int lwork, int info );
    void cggev3( String jobvl, String jobvr, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int info );
    void zggev3( String jobvl, String jobvr, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int info );
    void sggevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, INDArray a, int lda, INDArray b,
                 int ldb, INDArray alphar, INDArray alphai, INDArray beta,
                 INDArray vl, int ldvl, INDArray vr, int ldvr,
                 int ilo, int ihi, INDArray lscale,
                 INDArray rscale, INDArray abnrm, INDArray bbnrm, INDArray rconde,
                 INDArray rcondv, INDArray work, int lwork,
                 int iwork, int bwork,
                 int info );
    void dggevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, INDArray a, int lda, INDArray b,
                 int ldb, INDArray alphar, INDArray alphai,
                 INDArray beta, INDArray vl, int ldvl, INDArray vr,
                 int ldvr, int ilo, int ihi,
                 INDArray lscale, INDArray rscale, INDArray abnrm,
                 INDArray bbnrm, INDArray rconde, INDArray rcondv, INDArray work,
                 int lwork, int iwork, int bwork,
                 int info );
    void cggevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr, int ilo,
                 int ihi, INDArray lscale, INDArray rscale, INDArray abnrm,
                 INDArray bbnrm, INDArray rconde, INDArray rcondv,
                 IComplexNumber work, int lwork, INDArray rwork,
                 int iwork, int bwork,
                 int info );
    void zggevx( String balanc, String jobvl, String jobvr, String sense,
                 int n, IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber vl, int ldvl,
                 IComplexNumber vr, int ldvr,
                 int ilo, int ihi, INDArray lscale,
                 INDArray rscale, INDArray abnrm, INDArray bbnrm,
                 INDArray rconde, INDArray rcondv, IComplexNumber work,
                 int lwork, INDArray rwork, int iwork,
                 int bwork, int info );
    void dsfrk( String transr, String uplo, String trans, int n,
                int k, INDArray alpha,  INDArray a,
                int lda, INDArray beta, INDArray c );
    void ssfrk( String transr, String uplo, String trans, int n,
                int k, INDArray alpha,  INDArray a, int lda,
                INDArray beta, INDArray c );
    void zhfrk( String transr, String uplo, String trans, int n,
                int k, INDArray alpha,  IComplexNumber a,
                int lda, INDArray beta, IComplexNumber c );
    void chfrk( String transr, String uplo, String trans, int n,
                int k, INDArray alpha,  IComplexNumber a,
                int lda, INDArray beta, IComplexNumber c );
    void dtfsm( String transr, String side, String uplo, String trans,
                String diag, int m, int n, INDArray alpha,
                INDArray a, INDArray b, int ldb );
    void stfsm( String transr, String side, String uplo, String trans,
                String diag, int m, int n, INDArray alpha,
                INDArray a, INDArray b, int ldb );
    void ztfsm( String transr, String side, String uplo, String trans,
                String diag, int m, int n,
                IComplexNumber alpha,  IComplexNumber a,
                IComplexNumber b, int ldb );
    void ctfsm( String transr, String side, String uplo, String trans,
                String diag, int m, int n,
                IComplexNumber alpha,  IComplexNumber a,
                IComplexNumber b, int ldb );
    void dtfttp( String transr, String uplo, int n,  INDArray arf,
                 INDArray ap, int info );
    void stfttp( String transr, String uplo, int n,  INDArray arf,
                 INDArray ap, int info );
    void ztfttp( String transr, String uplo, int n,
                 IComplexNumber arf, IComplexNumber ap,
                 int info );
    void ctfttp( String transr, String uplo, int n,
                 IComplexNumber arf, IComplexNumber ap,
                 int info );
    void dtfttr( String transr, String uplo, int n,  INDArray arf,
                 INDArray a, int lda, int info );
    void stfttr( String transr, String uplo, int n,  INDArray arf,
                 INDArray a, int lda, int info );
    void ztfttr( String transr, String uplo, int n,
                 IComplexNumber arf, IComplexNumber a,
                 int lda, int info );
    void ctfttr( String transr, String uplo, int n,
                 IComplexNumber arf, IComplexNumber a,
                 int lda, int info );
    void dtpttf( String transr, String uplo, int n,  INDArray ap,
                 INDArray arf, int info );
    void stpttf( String transr, String uplo, int n,  INDArray ap,
                 INDArray arf, int info );
    void ztpttf( String transr, String uplo, int n,
                 IComplexNumber ap, IComplexNumber arf,
                 int info );
    void ctpttf( String transr, String uplo, int n,
                 IComplexNumber ap, IComplexNumber arf,
                 int info );
    void dtpttr( String uplo, int n,  INDArray ap, INDArray a,
                 int lda, int info );
    void stpttr( String uplo, int n,  INDArray ap, INDArray a,
                 int lda, int info );
    void ztpttr( String uplo, int n,  IComplexNumber ap,
                 IComplexNumber a, int lda,
                 int info );
    void ctpttr( String uplo, int n,  IComplexNumber ap,
                 IComplexNumber a, int lda,
                 int info );
    void dtrttf( String transr, String uplo, int n,  INDArray a,
                 int lda, INDArray arf, int info );
    void strttf( String transr, String uplo, int n,  INDArray a,
                 int lda, INDArray arf, int info );
    void ztrttf( String transr, String uplo, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber arf, int info );
    void ctrttf( String transr, String uplo, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber arf, int info );
    void dtrttp( String uplo, int n,  INDArray a, int lda,
                 INDArray ap, int info );
    void strttp( String uplo, int n,  INDArray a, int lda,
                 INDArray ap, int info );
    void ztrttp( String uplo, int n,  IComplexNumber a,
                 int lda, IComplexNumber ap,
                 int info );
    void ctrttp( String uplo, int n,  IComplexNumber a,
                 int lda, IComplexNumber ap,
                 int info );
    void sgeqrfp( int m, int n, INDArray a, int lda,
                  INDArray tau, INDArray work, int lwork,
                  int info );
    void dgeqrfp( int m, int n, INDArray a, int lda,
                  INDArray tau, INDArray work, int lwork,
                  int info );
    void cgeqrfp( int m, int n, IComplexNumber a,
                  int lda, IComplexNumber tau,
                  IComplexNumber work, int lwork,
                  int info );
    void zgeqrfp( int m, int n, IComplexNumber a,
                  int lda, IComplexNumber tau,
                  IComplexNumber work, int lwork,
                  int info );
    void clacgv( int n, IComplexNumber x, int incx );
    void zlacgv( int n, IComplexNumber x, int incx );
    void slarnv( int idist, int iseed, int n,
                 INDArray x );
    void dlarnv( int idist, int iseed, int n,
                 INDArray x );
    void clarnv( int idist, int iseed, int n,
                 IComplexNumber x );
    void zlarnv( int idist, int iseed, int n,
                 IComplexNumber x );
    void sgeqr2( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int info );
    void dgeqr2( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int info );
    void cgeqr2( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int info );
    void zgeqr2( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int info );
    void slacn2( int n, INDArray v, INDArray x, int isgn,
                 INDArray est, int kase, int isave );
    void dlacn2( int n, INDArray v, INDArray x, int isgn,
                 INDArray est, int kase, int isave );
    void clacn2( int n, IComplexNumber v,
                 IComplexNumber x, INDArray est,
                 int kase, int isave );
    void zlacn2( int n, IComplexNumber v,
                 IComplexNumber x, INDArray est,
                 int kase, int isave );
    void slacpy( String uplo, int m, int n,  INDArray a,
                 int lda, INDArray b, int ldb );
    void dlacpy( String uplo, int m, int n,  INDArray a,
                 int lda, INDArray b, int ldb );
    void clacpy( String uplo, int m, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb );
    void zlacpy( String uplo, int m, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb );

    void clacp2( String uplo, int m, int n,  INDArray a,
                 int lda, IComplexNumber b, int ldb );
    void zlacp2( String uplo, int m, int n,  INDArray a,
                 int lda, IComplexNumber b,
                 int ldb );

    void sgetf2( int m, int n, INDArray a, int lda,
                 int ipiv, int info );
    void dgetf2( int m, int n, INDArray a, int lda,
                 int ipiv, int info );
    void cgetf2( int m, int n, IComplexNumber a,
                 int lda, int ipiv, int info );
    void zgetf2( int m, int n, IComplexNumber a,
                 int lda, int ipiv, int info );
    void slaswp( int n, INDArray a, int lda, int k1,
                 int k2,  int ipiv, int incx );
    void dlaswp( int n, INDArray a, int lda, int k1,
                 int k2,  int ipiv, int incx );
    void claswp( int n, IComplexNumber a, int lda,
                 int k1, int k2,  int ipiv,
                 int incx );
    void zlaswp( int n, IComplexNumber a, int lda,
                 int k1, int k2,  int ipiv,
                 int incx );
    float slange( String norm, int m, int n,  INDArray a,
                  int lda, INDArray work );
    double dlange( String norm, int m, int n,  INDArray a,
                   int lda, INDArray work );
    float clange( String norm, int m, int n,
                  IComplexNumber a, int lda, INDArray work );
    double zlange( String norm, int m, int n,
                   IComplexNumber a, int lda, INDArray work );
    float clanhe( String norm, String uplo, int n,
                  IComplexNumber a, int lda, INDArray work );
    double zlanhe( String norm, String uplo, int n,
                   IComplexNumber a, int lda, INDArray work );
    float slansy( String norm, String uplo, int n,  INDArray a,
                  int lda, INDArray work );
    double dlansy( String norm, String uplo, int n,  INDArray a,
                   int lda, INDArray work );
    float clansy( String norm, String uplo, int n,
                  IComplexNumber a, int lda, INDArray work );
    double zlansy( String norm, String uplo, int n,
                   IComplexNumber a, int lda, INDArray work );
    float slantr( String norm, String uplo, String diag, int m,
                  int n,  INDArray a, int lda, INDArray work );
    double dlantr( String norm, String uplo, String diag, int m,
                   int n,  INDArray a, int lda, INDArray work );
    float clantr( String norm, String uplo, String diag, int m,
                  int n,  IComplexNumber a, int lda,
                  INDArray work );
    double zlantr( String norm, String uplo, String diag, int m,
                   int n,  IComplexNumber a, int lda,
                   INDArray work );
    float slamch( String cmach );
    double dlamch( String cmach );
    void sgelq2( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int info );
    void dgelq2( int m, int n, INDArray a, int lda,
                 INDArray tau, INDArray work, int info );
    void cgelq2( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int info );
    void zgelq2( int m, int n, IComplexNumber a,
                 int lda, IComplexNumber tau,
                 IComplexNumber work, int info );
    void slarfb( String side, String trans, String direct, String storev,
                 int m, int n, int k,  INDArray v,
                 int ldv,  INDArray t, int ldt, INDArray c,
                 int ldc, INDArray work, int ldwork );
    void dlarfb( String side, String trans, String direct, String storev,
                 int m, int n, int k,
                 INDArray v, int ldv,  INDArray t,
                 int ldt, INDArray c, int ldc, INDArray work,
                 int ldwork );
    void clarfb( String side, String trans, String direct, String storev,
                 int m, int n, int k,
                 IComplexNumber v, int ldv,
                 IComplexNumber t, int ldt,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int ldwork );
    void zlarfb( String side, String trans, String direct, String storev,
                 int m, int n, int k,
                 IComplexNumber v, int ldv,
                 IComplexNumber t, int ldt,
                 IComplexNumber c, int ldc,
                 IComplexNumber work, int ldwork );
    void slarfg( int n, INDArray alpha, INDArray x, int incx,
                 INDArray tau );
    void dlarfg( int n, INDArray alpha, INDArray x, int incx,
                 INDArray tau );
    void clarfg( int n, IComplexNumber alpha,
                 IComplexNumber x, int incx,
                 IComplexNumber tau );
    void zlarfg( int n, IComplexNumber alpha,
                 IComplexNumber x, int incx,
                 IComplexNumber tau );
    void slarft( String direct, String storev, int n, int k,
                 INDArray v, int ldv,  INDArray tau, INDArray t,
                 int ldt );
    void dlarft( String direct, String storev, int n, int k,
                 INDArray v, int ldv,  INDArray tau,
                 INDArray t, int ldt );
    void clarft( String direct, String storev, int n, int k,
                 IComplexNumber v, int ldv,
                 IComplexNumber tau, IComplexNumber t,
                 int ldt );
    void zlarft( String direct, String storev, int n, int k,
                 IComplexNumber v, int ldv,
                 IComplexNumber tau, IComplexNumber t,
                 int ldt );
    void slarfx( String side, int m, int n,  INDArray v,
                 INDArray tau, INDArray c, int ldc, INDArray work );
    void dlarfx( String side, int m, int n,  INDArray v,
                 INDArray tau, INDArray c, int ldc, INDArray work );
    void clarfx( String side, int m, int n,
                 IComplexNumber v, IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work );
    void zlarfx( String side, int m, int n,
                 IComplexNumber v, IComplexNumber tau,
                 IComplexNumber c, int ldc,
                 IComplexNumber work );
    void slatms( int m, int n, String dist, int iseed,
                 String sym, INDArray d, int mode, INDArray cond,
                 INDArray dmax, int kl, int ku, String pack,
                 INDArray a, int lda, INDArray work, int info );
    void dlatms( int m, int n, String dist, int iseed,
                 String sym, INDArray d, int mode, INDArray cond,
                 INDArray dmax, int kl, int ku, String pack,
                 INDArray a, int lda, INDArray work,
                 int info );
    void clatms( int m, int n, String dist, int iseed,
                 String sym, INDArray d, int mode, INDArray cond,
                 INDArray dmax, int kl, int ku, String pack,
                 IComplexNumber a, int lda,
                 IComplexNumber work, int info );
    void zlatms( int m, int n, String dist, int iseed,
                 String sym, INDArray d, int mode, INDArray cond,
                 INDArray dmax, int kl, int ku, String pack,
                 IComplexNumber a, int lda,
                 IComplexNumber work, int info );
    void slag2d( int m, int n,  INDArray sa,
                 int ldsa, INDArray a, int lda,
                 int info );
    void dlag2s( int m, int n,  INDArray a,
                 int lda, INDArray sa, int ldsa,
                 int info );
    void clag2z( int m, int n,
                 IComplexNumber sa, int ldsa,
                 IComplexNumber a, int lda,
                 int info );
    void zlag2c( int m, int n,
                 IComplexNumber a, int lda,
                 IComplexNumber sa, int ldsa,
                 int info );
    void slauum( String uplo, int n, INDArray a, int lda,
                 int info );
    void dlauum( String uplo, int n, INDArray a, int lda,
                 int info );
    void clauum( String uplo, int n, IComplexNumber a,
                 int lda, int info );
    void zlauum( String uplo, int n, IComplexNumber a,
                 int lda, int info );
    void slagge( int m, int n, int kl,
                 int ku,  INDArray d, INDArray a, int lda,
                 int iseed, INDArray work, int info );
    void dlagge( int m, int n, int kl,
                 int ku,  INDArray d, INDArray a, int lda,
                 int iseed, INDArray work, int info );
    void clagge( int m, int n, int kl,
                 int ku,  INDArray d, IComplexNumber a,
                 int lda, int iseed,
                 IComplexNumber work, int info );
    void zlagge( int m, int n, int kl,
                 int ku,  INDArray d, IComplexNumber a,
                 int lda, int iseed,
                 IComplexNumber work, int info );
    void slascl( String type, int kl, int ku, INDArray cfrom,
                 INDArray cto, int m, int n, INDArray a,
                 int lda, int info );
    void dlascl( String type, int kl, int ku, INDArray cfrom,
                 INDArray cto, int m, int n, INDArray a,
                 int lda, int info );
    void clascl( String type, int kl, int ku, INDArray cfrom,
                 INDArray cto, int m, int n, IComplexNumber a,
                 int lda, int info );
    void zlascl( String type, int kl, int ku, INDArray cfrom,
                 INDArray cto, int m, int n, IComplexNumber a,
                 int lda, int info );
    void slaset( String uplo, int m, int n, INDArray alpha,
                 INDArray beta, INDArray a, int lda );
    void dlaset( String uplo, int m, int n, INDArray alpha,
                 INDArray beta, INDArray a, int lda );
    void claset( String uplo, int m, int n,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber a, int lda );
    void zlaset( String uplo, int m, int n,
                 IComplexNumber alpha, IComplexNumber beta,
                 IComplexNumber a, int lda );
    void slasrt( String id, int n, INDArray d, int info );
    void dlasrt( String id, int n, INDArray d, int info );
    void claghe( int n, int k,  INDArray d,
                 IComplexNumber a, int lda, int iseed,
                 IComplexNumber work, int info );
    void zlaghe( int n, int k,  INDArray d,
                 IComplexNumber a, int lda,
                 int iseed, IComplexNumber work,
                 int info );
    void slagsy( int n, int k,  INDArray d, INDArray a,
                 int lda, int iseed, INDArray work,
                 int info );
    void dlagsy( int n, int k,  INDArray d, INDArray a,
                 int lda, int iseed, INDArray work,
                 int info );
    void clagsy( int n, int k,  INDArray d,
                 IComplexNumber a, int lda, int iseed,
                 IComplexNumber work, int info );
    void zlagsy( int n, int k,  INDArray d,
                 IComplexNumber a, int lda,
                 int iseed, IComplexNumber work,
                 int info );
    void slapmr( int forwrd, int m, int n,
                 INDArray x, int ldx, int k );
    void dlapmr( int forwrd, int m, int n,
                 INDArray x, int ldx, int k );
    void clapmr( int forwrd, int m, int n,
                 IComplexNumber x, int ldx, int k );
    void zlapmr( int forwrd, int m, int n,
                 IComplexNumber x, int ldx, int k );
    void slapmt( int forwrd, int m, int n,
                 INDArray x, int ldx, int k );
    void dlapmt( int forwrd, int m, int n,
                 INDArray x, int ldx, int k );
    void clapmt( int forwrd, int m, int n,
                 IComplexNumber x, int ldx, int k );
    void zlapmt( int forwrd, int m, int n,
                 IComplexNumber x, int ldx, int k );
    float slapy2( INDArray x, INDArray y );
    double dlapy2( INDArray x, INDArray y );
    float slapy3( INDArray x, INDArray y, INDArray z );
    double dlapy3( INDArray x, INDArray y, INDArray z );
    void slartgp( INDArray f, INDArray g, INDArray cs, INDArray sn, INDArray r );
    void dlartgp( INDArray f, INDArray g, INDArray cs, INDArray sn, INDArray r );
    void slartgs( INDArray x, INDArray y, INDArray sigma, INDArray cs, INDArray sn );
    void dlartgs( INDArray x, INDArray y, INDArray sigma, INDArray cs,
                  INDArray sn );
    // LAPACK 3.3.0
    void cbbcsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 int m, int p, int q,
                 INDArray theta, INDArray phi,
                 IComplexNumber u1, int ldu1,
                 IComplexNumber u2, int ldu2,
                 IComplexNumber v1t, int ldv1t,
                 IComplexNumber v2t, int ldv2t,
                 INDArray b11d, INDArray b11e, INDArray b12d,
                 INDArray b12e, INDArray b21d, INDArray b21e,
                 INDArray b22d, INDArray b22e, INDArray rwork,
                 int lrwork , int info );
    void cheswapr( String uplo, int n,
                   IComplexNumber a, int i1,
                   int i2 );
    void chetri2( String uplo, int n,
                  IComplexNumber a, int lda,
                  int ipiv,
                  IComplexNumber work, int lwork , int info );
    void chetri2x( String uplo, int n,
                   IComplexNumber a, int lda,
                   int ipiv,
                   IComplexNumber work, int nb , int info );
    void chetrs2( String uplo, int n,
                  int nrhs,  IComplexNumber a,
                  int lda,  int ipiv,
                  IComplexNumber b, int ldb,
                  IComplexNumber work , int info );
    void csyconv( String uplo, String way,
                  int n, IComplexNumber a,
                  int lda,  int ipiv,
                  IComplexNumber work , int info );
    void csyswapr( String uplo, int n,
                   IComplexNumber a, int i1,
                   int i2 );
    void csytri2( String uplo, int n,
                  IComplexNumber a, int lda,
                  int ipiv,
                  IComplexNumber work, int lwork , int info );
    void csytri2x( String uplo, int n,
                   IComplexNumber a, int lda,
                   int ipiv,
                   IComplexNumber work, int nb , int info );
    void csytrs2( String uplo, int n,
                  int nrhs,  IComplexNumber a,
                  int lda,  int ipiv,
                  IComplexNumber b, int ldb,
                  IComplexNumber work , int info );
    void cunbdb( String trans, String signs,
                 int m, int p, int q,
                 IComplexNumber x11, int ldx11,
                 IComplexNumber x12, int ldx12,
                 IComplexNumber x21, int ldx21,
                 IComplexNumber x22, int ldx22,
                 INDArray theta, INDArray phi,
                 IComplexNumber taup1,
                 IComplexNumber taup2,
                 IComplexNumber tauq1,
                 IComplexNumber tauq2,
                 IComplexNumber work, int lwork , int info );
    void cuncsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 String signs, int m, int p,
                 int q, IComplexNumber x11,
                 int ldx11, IComplexNumber x12,
                 int ldx12, IComplexNumber x21,
                 int ldx21, IComplexNumber x22,
                 int ldx22, INDArray theta,
                 IComplexNumber u1, int ldu1,
                 IComplexNumber u2, int ldu2,
                 IComplexNumber v1t, int ldv1t,
                 IComplexNumber v2t, int ldv2t,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork,
                 int iwork , int info );
    void cuncsd2by1( String jobu1, String jobu2,
                     String jobv1t, int m, int p,
                     int q, IComplexNumber x11,
                     int ldx11, IComplexNumber x21,
                     int ldx21, IComplexNumber theta,
                     IComplexNumber u1, int ldu1,
                     IComplexNumber u2, int ldu2,
                     IComplexNumber v1t, int ldv1t,
                     IComplexNumber work, int lwork,
                     INDArray rwork, int lrwork,
                     int iwork , int info );
    void dbbcsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 int m, int p, int q,
                 INDArray theta, INDArray phi, INDArray u1,
                 int ldu1, INDArray u2, int ldu2,
                 INDArray v1t, int ldv1t, INDArray v2t,
                 int ldv2t, INDArray b11d, INDArray b11e,
                 INDArray b12d, INDArray b12e, INDArray b21d,
                 INDArray b21e, INDArray b22d, INDArray b22e,
                 INDArray work, int lwork , int info );
    void dorbdb( String trans, String signs,
                 int m, int p, int q,
                 INDArray x11, int ldx11, INDArray x12,
                 int ldx12, INDArray x21, int ldx21,
                 INDArray x22, int ldx22, INDArray theta,
                 INDArray phi, INDArray taup1, INDArray taup2,
                 INDArray tauq1, INDArray tauq2, INDArray work,
                 int lwork , int info );
    void dorcsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 String signs, int m, int p,
                 int q, INDArray x11, int ldx11,
                 INDArray x12, int ldx12, INDArray x21,
                 int ldx21, INDArray x22, int ldx22,
                 INDArray theta, INDArray u1, int ldu1,
                 INDArray u2, int ldu2, INDArray v1t,
                 int ldv1t, INDArray v2t, int ldv2t,
                 INDArray work, int lwork,
                 int iwork , int info );
    void dorcsd2by1( String jobu1, String jobu2,
                     String jobv1t, int m, int p,
                     int q, INDArray x11, int ldx11,
                     INDArray x21, int ldx21,
                     INDArray theta, INDArray u1, int ldu1,
                     INDArray u2, int ldu2, INDArray v1t,
                     int ldv1t, INDArray work, int lwork,
                     int iwork , int info );
    void dsyconv( String uplo, String way,
                  int n, INDArray a, int lda,
                  int ipiv, INDArray work , int info );
    void dsyswapr( String uplo, int n,
                   INDArray a, int i1, int i2 );
    void dsytri2( String uplo, int n,
                  INDArray a, int lda,
                  int ipiv,
                  IComplexNumber work, int lwork , int info );
    void dsytri2x( String uplo, int n,
                   INDArray a, int lda,
                   int ipiv, INDArray work,
                   int nb , int info );
    void dsytrs2( String uplo, int n,
                  int nrhs,  INDArray a,
                  int lda,  int ipiv,
                  INDArray b, int ldb, INDArray work , int info );
    void sbbcsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 int m, int p, int q,
                 INDArray theta, INDArray phi, INDArray u1,
                 int ldu1, INDArray u2, int ldu2,
                 INDArray v1t, int ldv1t, INDArray v2t,
                 int ldv2t, INDArray b11d, INDArray b11e,
                 INDArray b12d, INDArray b12e, INDArray b21d,
                 INDArray b21e, INDArray b22d, INDArray b22e,
                 INDArray work, int lwork , int info );
    void sorbdb( String trans, String signs,
                 int m, int p, int q,
                 INDArray x11, int ldx11, INDArray x12,
                 int ldx12, INDArray x21, int ldx21,
                 INDArray x22, int ldx22, INDArray theta,
                 INDArray phi, INDArray taup1, INDArray taup2,
                 INDArray tauq1, INDArray tauq2, INDArray work,
                 int lwork , int info );
    void sorcsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 String signs, int m, int p,
                 int q, INDArray x11, int ldx11,
                 INDArray x12, int ldx12, INDArray x21,
                 int ldx21, INDArray x22, int ldx22,
                 INDArray theta, INDArray u1, int ldu1,
                 INDArray u2, int ldu2, INDArray v1t,
                 int ldv1t, INDArray v2t, int ldv2t,
                 INDArray work, int lwork,
                 int iwork , int info );
    void sorcsd2by1( String jobu1, String jobu2,
                     String jobv1t, int m, int p,
                     int q, INDArray x11, int ldx11,
                     INDArray x21, int ldx21,
                     INDArray theta, INDArray u1, int ldu1,
                     INDArray u2, int ldu2, INDArray v1t,
                     int ldv1t, INDArray work, int lwork,
                     int iwork , int info );
    void ssyconv( String uplo, String way,
                  int n, INDArray a, int lda,
                  int ipiv, INDArray work , int info );
    void ssyswapr( String uplo, int n,
                   INDArray a, int i1, int i2 );
    void ssytri2( String uplo, int n,
                  INDArray a, int lda,
                  int ipiv,
                  IComplexNumber work, int lwork , int info );
    void ssytri2x( String uplo, int n,
                   INDArray a, int lda,
                   int ipiv, INDArray work,
                   int nb , int info );
    void ssytrs2( String uplo, int n,
                  int nrhs,  INDArray a,
                  int lda,  int ipiv,
                  INDArray b, int ldb, INDArray work , int info );
    void zbbcsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 int m, int p, int q,
                 INDArray theta, INDArray phi,
                 IComplexNumber u1, int ldu1,
                 IComplexNumber u2, int ldu2,
                 IComplexNumber v1t, int ldv1t,
                 IComplexNumber v2t, int ldv2t,
                 INDArray b11d, INDArray b11e, INDArray b12d,
                 INDArray b12e, INDArray b21d, INDArray b21e,
                 INDArray b22d, INDArray b22e, INDArray rwork,
                 int lrwork , int info );
    void zheswapr( String uplo, int n,
                   IComplexNumber a, int i1,
                   int i2 );
    void zhetri2( String uplo, int n,
                  IComplexNumber a, int lda,
                  int ipiv,
                  IComplexNumber work, int lwork , int info );
    void zhetri2x( String uplo, int n,
                   IComplexNumber a, int lda,
                   int ipiv,
                   IComplexNumber work, int nb , int info );
    void zhetrs2( String uplo, int n,
                  int nrhs,
                  IComplexNumber a, int lda,
                  int ipiv,
                  IComplexNumber b, int ldb,
                  IComplexNumber work , int info );
    void zsyconv( String uplo, String way,
                  int n, IComplexNumber a,
                  int lda,  int ipiv,
                  IComplexNumber work , int info );
    void zsyswapr( String uplo, int n,
                   IComplexNumber a, int i1,
                   int i2 );
    void zsytri2( String uplo, int n,
                  IComplexNumber a, int lda,
                  int ipiv,
                  IComplexNumber work, int lwork , int info );
    void zsytri2x( String uplo, int n,
                   IComplexNumber a, int lda,
                   int ipiv,
                   IComplexNumber work, int nb , int info );
    void zsytrs2( String uplo, int n,
                  int nrhs,
                  IComplexNumber a, int lda,
                  int ipiv,
                  IComplexNumber b, int ldb,
                  IComplexNumber work , int info );
    void zunbdb( String trans, String signs,
                 int m, int p, int q,
                 IComplexNumber x11, int ldx11,
                 IComplexNumber x12, int ldx12,
                 IComplexNumber x21, int ldx21,
                 IComplexNumber x22, int ldx22,
                 INDArray theta, INDArray phi,
                 IComplexNumber taup1,
                 IComplexNumber taup2,
                 IComplexNumber tauq1,
                 IComplexNumber tauq2,
                 IComplexNumber work, int lwork , int info );
    void zuncsd( String jobu1, String jobu2,
                 String jobv1t, String jobv2t, String trans,
                 String signs, int m, int p,
                 int q, IComplexNumber x11,
                 int ldx11, IComplexNumber x12,
                 int ldx12, IComplexNumber x21,
                 int ldx21, IComplexNumber x22,
                 int ldx22, INDArray theta,
                 IComplexNumber u1, int ldu1,
                 IComplexNumber u2, int ldu2,
                 IComplexNumber v1t, int ldv1t,
                 IComplexNumber v2t, int ldv2t,
                 IComplexNumber work, int lwork,
                 INDArray rwork, int lrwork,
                 int iwork , int info );
    void zuncsd2by1( String jobu1, String jobu2,
                     String jobv1t, int m, int p,
                     int q, IComplexNumber x11,
                     int ldx11, IComplexNumber x21,
                     int ldx21, IComplexNumber theta,
                     IComplexNumber u1, int ldu1,
                     IComplexNumber u2, int ldu2,
                     IComplexNumber v1t, int ldv1t,
                     IComplexNumber work, int lwork,
                     INDArray rwork, int lrwork,
                     int iwork , int info );
    // LAPACK 3.4.0
    void sgemqrt( String side, String trans, int m, int n,
                  int k, int nb,  INDArray v,
                  int ldv,  INDArray t, int ldt, INDArray c,
                  int ldc, INDArray work, int info );
    void dgemqrt( String side, String trans, int m, int n,
                  int k, int nb,  INDArray v,
                  int ldv,  INDArray t, int ldt,
                  INDArray c, int ldc, INDArray work,
                  int info );
    void cgemqrt( String side, String trans, int m, int n,
                  int k, int nb,
                  IComplexNumber v, int ldv,
                  IComplexNumber t, int ldt,
                  IComplexNumber c, int ldc,
                  IComplexNumber work, int info );
    void zgemqrt( String side, String trans, int m, int n,
                  int k, int nb,
                  IComplexNumber v, int ldv,
                  IComplexNumber t, int ldt,
                  IComplexNumber c, int ldc,
                  IComplexNumber work, int info );
    void sgeqrt( int m, int n, int nb, INDArray a,
                 int lda, INDArray t, int ldt, INDArray work,
                 int info );
    void dgeqrt( int m, int n, int nb, INDArray a,
                 int lda, INDArray t, int ldt, INDArray work,
                 int info );
    void cgeqrt( int m, int n, int nb,
                 IComplexNumber a, int lda,
                 IComplexNumber t, int ldt,
                 IComplexNumber work, int info );
    void zgeqrt( int m, int n, int nb,
                 IComplexNumber a, int lda,
                 IComplexNumber t, int ldt,
                 IComplexNumber work, int info );
    void sgeqrt2( int m, int n, INDArray a, int lda,
                  INDArray t, int ldt, int info );
    void dgeqrt2( int m, int n, INDArray a, int lda,
                  INDArray t, int ldt, int info );
    void cgeqrt2( int m, int n, IComplexNumber a,
                  int lda, IComplexNumber t, int ldt,
                  int info );
    void zgeqrt2( int m, int n, IComplexNumber a,
                  int lda, IComplexNumber t, int ldt,
                  int info );
    void sgeqrt3( int m, int n, INDArray a, int lda,
                  INDArray t, int ldt, int info );
    void dgeqrt3( int m, int n, INDArray a, int lda,
                  INDArray t, int ldt, int info );
    void cgeqrt3( int m, int n, IComplexNumber a,
                  int lda, IComplexNumber t, int ldt,
                  int info );
    void zgeqrt3( int m, int n, IComplexNumber a,
                  int lda, IComplexNumber t, int ldt,
                  int info );
    void stpmqrt( String side, String trans, int m, int n,
                  int k, int l, int nb,
                  INDArray v, int ldv,  INDArray t,
                  int ldt, INDArray a, int lda, INDArray b,
                  int ldb, INDArray work, int info );
    void dtpmqrt( String side, String trans, int m, int n,
                  int k, int l, int nb,
                  INDArray v, int ldv,  INDArray t,
                  int ldt, INDArray a, int lda, INDArray b,
                  int ldb, INDArray work, int info );
    void ctpmqrt( String side, String trans, int m, int n,
                  int k, int l, int nb,
                  IComplexNumber v, int ldv,
                  IComplexNumber t, int ldt,
                  IComplexNumber a, int lda,
                  IComplexNumber b, int ldb,
                  IComplexNumber work, int info );
    void ztpmqrt( String side, String trans, int m, int n,
                  int k, int l, int nb,
                  IComplexNumber v, int ldv,
                  IComplexNumber t, int ldt,
                  IComplexNumber a, int lda,
                  IComplexNumber b, int ldb,
                  IComplexNumber work, int info );
    void stpqrt( int m, int n, int l, int nb,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray t, int ldt, INDArray work, int info );
    void dtpqrt( int m, int n, int l, int nb,
                 INDArray a, int lda, INDArray b, int ldb,
                 INDArray t, int ldt, INDArray work,
                 int info );
    void ctpqrt( int m, int n, int l, int nb,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber t, int ldt,
                 IComplexNumber work, int info );
    void ztpqrt( int m, int n, int l, int nb,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber t, int ldt,
                 IComplexNumber work, int info );
    void stpqrt2( int m, int n, int l,
                  INDArray a, int lda,
                  INDArray b, int ldb,
                  INDArray t, int ldt,
                  int info );
    void dtpqrt2( int m, int n, int l,
                  INDArray a, int lda,
                  INDArray b, int ldb,
                  INDArray t, int ldt,
                  int info );
    void ctpqrt2( int m, int n, int l,
                  IComplexNumber a, int lda,
                  IComplexNumber b, int ldb,
                  IComplexNumber t, int ldt,
                  int info );
    void ztpqrt2( int m, int n, int l,
                  IComplexNumber a, int lda,
                  IComplexNumber b, int ldb,
                  IComplexNumber t, int ldt,
                  int info );
    void stprfb( String side, String trans, String direct, String storev,
                 int m, int n, int k, int l,
                 INDArray v, int ldv,  INDArray t,
                 int ldt, INDArray a, int lda, INDArray b,
                 int ldb,  INDArray work,
                 int ldwork );
    void dtprfb( String side, String trans, String direct, String storev,
                 int m, int n, int k, int l,
                 INDArray v, int ldv,  INDArray t,
                 int ldt, INDArray a, int lda, INDArray b,
                 int ldb,  INDArray work,
                 int ldwork );
    void ctprfb( String side, String trans, String direct, String storev,
                 int m, int n, int k, int l,
                 IComplexNumber v, int ldv,
                 IComplexNumber t, int ldt,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber work, int ldwork );
    void ztprfb( String side, String trans, String direct, String storev,
                 int m, int n, int k, int l,
                 IComplexNumber v, int ldv,
                 IComplexNumber t, int ldt,
                 IComplexNumber a, int lda,
                 IComplexNumber b, int ldb,
                 IComplexNumber work, int ldwork );
    // LAPACK 3.5.0
    void ssysint( String uplo, int n, int nrhs, INDArray a,
                  int lda, int ipiv, INDArray b,
                  int ldb, INDArray work, int lwork,
                  int info );
    void ssytrint( String uplo, int n, INDArray a, int lda,
                   int ipiv, INDArray work, int lwork,
                   int info );
    void dsysint( String uplo, int n, int nrhs, INDArray a,
                  int lda, int ipiv, INDArray b,
                  int ldb, INDArray work, int lwork,
                  int info );
    void dsytrint( String uplo, int n, INDArray a, int lda,
                   int ipiv, INDArray work, int lwork,
                   int info );
    void csysint( String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  int ipiv, IComplexNumber b,
                  int ldb, IComplexNumber work,
                  int lwork, int info );
    void csytrint( String uplo, int n, IComplexNumber a,
                   int lda, int ipiv,
                   IComplexNumber work, int lwork,
                   int info );
    void zsysint( String uplo, int n, int nrhs,
                  IComplexNumber a, int lda,
                  int ipiv, IComplexNumber b,
                  int ldb, IComplexNumber work,
                  int lwork, int info );
    void zsytrint( String uplo, int n, IComplexNumber a,
                   int lda, int ipiv,
                   IComplexNumber work, int lwork,
                   int info );
    void ssytrint( String uplo, int n, int nrhs,  INDArray a,
                   int lda,  int ipiv, INDArray b,
                   int ldb, int info );
    void dsytrint( String uplo, int n, int nrhs,
                   INDArray a, int lda,  int ipiv,
                   INDArray b, int ldb, int info );
    void csytrint( String uplo, int n, int nrhs,
                   IComplexNumber a, int lda,
                   int ipiv, IComplexNumber b,
                   int ldb, int info );
    void zsytrint( String uplo, int n, int nrhs,
                   IComplexNumber a, int lda,
                   int ipiv, IComplexNumber b,
                   int ldb, int info );
    void chetrint( String uplo, int n, IComplexNumber a,
                   int lda, int ipiv,
                   IComplexNumber work, int lwork,
                   int info );
    void zhetrint( String uplo, int n, IComplexNumber a,
                   int lda, int ipiv,
                   IComplexNumber work, int lwork,
                   int info );
    void chetrint( String uplo, int n, int nrhs,
                   IComplexNumber a, int lda,
                   int ipiv, IComplexNumber b,
                   int ldb, int info );
    void zhetrint( String uplo, int n, int nrhs,
                   IComplexNumber a, int lda,
                   int ipiv, IComplexNumber b,
                   int ldb, int info );

    void csyr( String uplo, int n, IComplexNumber alpha,
               IComplexNumber x, int incx,
               IComplexNumber a, int lda );
    void zsyr( String uplo, int n, IComplexNumber alpha,
               IComplexNumber x, int incx,
               IComplexNumber a, int lda );


}