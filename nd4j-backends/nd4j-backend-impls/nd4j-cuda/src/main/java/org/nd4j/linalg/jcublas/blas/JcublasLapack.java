package org.nd4j.linalg.jcublas.blas;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.bytedeco.javacpp.cusolver.*;

/**
 * JCublas lapack
 *
 * @author Adam Gibson
 */
public class JcublasLapack extends BaseLapack {

    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private Allocator allocator = AtomicAllocator.getInstance();
    private static Logger logger = LoggerFactory.getLogger(JcublasLapack.class);


    @Override
    public void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        INDArray a = A;
        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT getrf called in DOUBLE environment");

        if (A.ordering() == 'c')
            a = A.dup('f');


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgetrf_bufferSize(solverDn, M, N, (FloatPointer) xAPointer.getDevicePointer(), M,
                            (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnSgetrf_bufferSize failed with code: " + stat);
            }

            int worksize = worksizeBuffer.getInt(0);
            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual LU decomp
            stat = cusolverDnSgetrf(solverDn, M, N, (FloatPointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(workspace).asFloatPointer(),
                            new CudaPointer(allocator.getPointer(IPIV, ctx)).asIntPointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

            // we do sync to make sure getrf is finished
            //ctx.syncOldStream();

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnSgetrf failed with code: " + stat);
            }
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, IPIV);

        if (a != A)
            A.assign(a);

        logger.info("A: {}", A);
    }



    @Override
    public void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        INDArray a = A;

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("FLOAT getrf called in FLOAT environment");

        if (A.ordering() == 'c')
            a = A.dup('f');

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnDgetrf_bufferSize(solverDn, M, N, (DoublePointer) xAPointer.getDevicePointer(), M,
                            (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnDgetrf_bufferSize failed with code: " + stat);
            }
            int worksize = worksizeBuffer.getInt(0);

            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual LU decomp
            stat = cusolverDnDgetrf(solverDn, M, N, (DoublePointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(workspace).asDoublePointer(),
                            new CudaPointer(allocator.getPointer(IPIV, ctx)).asIntPointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnSgetrf failed with code: " + stat);
            }
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, IPIV);

        if (a != A)
            A.assign(a);
    }



    /**
     * Generate inverse ggiven LU decomp
     *
     * @param N
     * @param A
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {

    }


    @Override
    public void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {

        INDArray a = A;
        INDArray u = U;
        INDArray vt = VT;

        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT gesvd called in DOUBLE environment");

        // cuda requires column ordering - we'll register a warning in case
        if (A.ordering() == 'c')
            a = A.dup('f');

        if (U != null && U.ordering() == 'c')
            u = U.dup('f');

        if (VT != null && VT.ordering() == 'c')
            vt = VT.dup('f');


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgesvd_bufferSize(solverDn, M, N, (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnSgesvd_bufferSize failed with code: " + stat);
            }
            int worksize = worksizeBuffer.getInt(0);

            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());
            DataBuffer rwork = Nd4j.getDataBufferFactory().createFloat((M < N ? M : N) - 1);

            // Do the actual decomp
            stat = cusolverDnSgesvd(solverDn, jobu, jobvt, M, N, (FloatPointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(allocator.getPointer(S, ctx)).asFloatPointer(),
                            U == null ? null : new CudaPointer(allocator.getPointer(u, ctx)).asFloatPointer(), M,
                            VT == null ? null : new CudaPointer(allocator.getPointer(vt, ctx)).asFloatPointer(), N,
                            new CudaPointer(workspace).asFloatPointer(), worksize,
                            new CudaPointer(allocator.getPointer(rwork, ctx)).asFloatPointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnSgesvd failed with code: " + stat);
            }
        }
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, S);
        allocator.registerAction(ctx, a);

        if (U != null)
            allocator.registerAction(ctx, u);
        if (VT != null)
            allocator.registerAction(ctx, vt);

        if (a != A)
            A.assign(a);

        if (u != U)
            U.assign(u);

        if (vt != VT)
            VT.assign(vt);
    }


    @Override
    public void dgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {

        INDArray a = A;
        INDArray u = U;
        INDArray vt = VT;

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            logger.warn("DOUBLE gesvd called in FLOAT environment");

        // cuda requires column ordering - we'll register a warning in case
        if (A.ordering() == 'c')
            a = A.dup('f');

        if (U != null && U.ordering() == 'c')
            u = U.dup('f');

        if (VT != null && VT.ordering() == 'c')
            vt = VT.dup('f');


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgesvd_bufferSize(solverDn, M, N, (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnSgesvd_bufferSize failed with code: " + stat);
            }
            int worksize = worksizeBuffer.getInt(0);

            // Now allocate memory for the workspace, the non-converging row buffer and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());
            DataBuffer rwork = Nd4j.getDataBufferFactory().createDouble((M < N ? M : N) - 1);

            // Do the actual decomp
            stat = cusolverDnDgesvd(solverDn, jobu, jobvt, M, N, (DoublePointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(allocator.getPointer(S, ctx)).asDoublePointer(),
                            U == null ? null : new CudaPointer(allocator.getPointer(u, ctx)).asDoublePointer(), M,
                            VT == null ? null : new CudaPointer(allocator.getPointer(vt, ctx)).asDoublePointer(), N,
                            new CudaPointer(workspace).asDoublePointer(), worksize,
                            new CudaPointer(allocator.getPointer(rwork, ctx)).asDoublePointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new IllegalStateException("cusolverDnDgesvd failed with code: " + stat);
            }
        }
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, S);
        allocator.registerAction(ctx, a);

        if (U != null)
            allocator.registerAction(ctx, u);

        if (VT != null)
            allocator.registerAction(ctx, vt);


        if (a != A)
            A.assign(a);

        if (u != U)
            U.assign(u);

        if (vt != VT)
            VT.assign(vt);
    }



    static class Workspace extends Pointer {
        public Workspace(long size) {
            super(NativeOpsHolder.getInstance().getDeviceNativeOps().mallocDevice(size, null, 0));
            deallocator(new Deallocator() {
                @Override
                public void deallocate() {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(Workspace.this, null);
                }
            });
        }
    }
}
