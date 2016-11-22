package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * CPU lapack implementation
 */
public class CpuLapack extends BaseLapack {



    /**
     * LU decomposiiton of a matrix
     *
     * @param M
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param INFO
     */
    @Override
    public void getrf(int M, int N, INDArray A, int lda, INDArray IPIV, INDArray INFO) {

    }

    /**
     * Generate inverse ggiven LU decomp
     *
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {

    }
}
