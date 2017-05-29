package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Lapack interface
 *
 * @author Adam Gibson
 * @author rcorbish
 */
public interface Lapack {

    /**
     * LU decomposiiton of a matrix
     * Factorize a matrix A
     *
     * The matrix A is overridden by the L & U combined.
     * The permutation results are returned directly as a vector. To 
     * create the permutation matrix use getPFactor method
     * To split out the L & U matrix use getLFactor and getUFactor methods
     *
     * getrf = triangular factorization (TRF) of a general matrix (GE)
     *
     * @param A the input matrix, it will be overwritten with the factors
     * @returns Permutation array
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    INDArray getrf(INDArray A);


    /**
    * Float/Double versions of cholesky decomp for positive definite matrices
    *
    *   A = LL*
    *
    * @param uplo which factor to return L or U
    * @param M  the number of rows & cols in the matrix A
    * @param A  the matrix to factorize - data must be in column order ( create with 'f' ordering )
    * @param INFO error details 1 int array, a positive number (i) implies row i cannot be factored, a negative value implies paramtere i is invalid
    */
    void spotrf(byte uplo, int N, INDArray A, INDArray INFO)  ;

    void dpotrf(byte uplo, int N, INDArray A, INDArray INFO) ;

    /**
     * QR decomposiiton of a matrix
     * Factorize a matrix A such that A = QR
     *
     * The matrix A is overwritten by the Q component (i.e. destroyed)
     *
     * geqrf = QR factorization of a general matrix (GE) into an orthogonal
     *         matrix Q and an upper triangular R matrix
     *
     * @param A the input matrix, it will be overwritten with the factors
     * @param The R array if null R is not returned
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    public void geqrf(INDArray A, INDArray R );


    /**
    * Float/Double versions of LU decomp.
    * This is the official LAPACK interface (in case you want to call this directly)
    * See getrf for full details on LU Decomp
    *
    * @param M  the number of rows in the matrix A
    * @param N  the number of cols in the matrix A
    * @param A  the matrix to factorize - data must be in column order ( create with 'f' ordering )
    * @param IPIV an output array for the permutations ( must be int based storage )
    * @param INFO error details 1 int array, a positive number (i) implies row i cannot be factored, a negative value implies paramtere i is invalid
    */
    void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO);

    void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO);

    /**
     * Triangular decomposiiton of a positive definite matrix ( cholesky )
     * Factorize a matrix A such that A = LL* (assuming lower==true) or
     * A = U*U   (a * represents conjugate i.e. if matrix is real U* is a transpose) 
     *
     * The matrix A is overridden by the L (or U).
     *
     * potrf = LU factorization of a positive definite matrix (PO) into a
     *         lower L ( or upper U ) triangular matrix
     *
     * @param A the input matrix, it will be overwritten with the factors
     * @param whether to return the upper (false) or lower factor
     * @returns Permutation array
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    void potrf(INDArray A, boolean lower) ;



    /**
     * SVD decomposiiton of a matrix
     * Factorize a matrix into its singular vectors and eigenvalues
     * The decomposition is such that:
     *
     * 		A = U x S x VT
     *
     * gesvd = singular value decomposition (SVD) of a general matrix (GE)
     *
     * @param A the input matrix
     * @param S the eigenvalues as a vector
     * @param U the left singular vectors as a matrix. Maybe null if no S required
     * @param VT the right singular vectors as a (transposed) matrix. Maybe null if no V required
     * @throws Error - with a message to indicate failure (usu. bad params)
     */
    void gesvd(INDArray A, INDArray S, INDArray U, INDArray VT);


    void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                INDArray INFO);

    void dgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                INDArray INFO);

    /**
    * This method takes one of the ipiv returns from LAPACK and creates
    * the permutation matrix. When factorizing, it is useful to avoid underflows
    * and overflows by reordering rows/and or columns of the input matrix (mostly
    * these methods solve simultaneous equations, so order is not important). 
    * The ipiv method assumes that only row ordering is done ( never seen column 
    * ordering done )
    *
    * @param M - the size of the permutation matrix ( usu. the # rows in factored matrix )
    * @param ipiv - the vector returned from a refactoring
    * @returned the square permutation matrix - size is the M x M
    */
    public INDArray getPFactor(int M, INDArray ipiv);


    /**
    * extracts the L (lower triangular) matrix from the LU factor result
    * L will be the same dimensions as A
    *
    * @param A - the combined L & U matrices returned from factorization
    * @returned the lower triangular with unit diagonal
    */
    INDArray getLFactor(INDArray A);


    /**
    * extracts the U (upper triangular) matrix from the LU factor result
    * U will be n x n matrix where n = num cols in A
    *
    * @param A - the combined L & U matrices returned from factorization
    * @returned the upper triangular matrix
    */
    INDArray getUFactor(INDArray A);


    // generate inverse of a matrix given its LU decomposition

    /**
     * Generate inverse ggiven LU decomp
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO);

    /**
     *
     * @param _jobz
     * @param _uplo
     * @param N
     * @param A
     * @param R
     * @return
     */
    int dsyev(char _jobz, char _uplo, int N, INDArray A, INDArray R);

    /**
     *
     * @param _jobz
     * @param _uplo
     * @param N
     * @param A
     * @param R
     * @return
     */
    int ssyev(char _jobz, char _uplo, int N, INDArray A, INDArray R);
}
