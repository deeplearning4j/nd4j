package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;

public abstract class BaseCPUScalarOpAction extends BaseCPUAction {
    protected final ScalarOp op;

    /**
     * Constructor for operating on subset of NDArray
     */
    public BaseCPUScalarOpAction(ScalarOp op, int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ) {
        super(threshold, n, offsetX, 0, offsetZ, incrX, 0, incrZ);
        this.op = op;
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public BaseCPUScalarOpAction(ScalarOp op, int threshold) {
        super(op, threshold);
        this.op = op;
    }

    protected void doOp(final int startIdx, final int endIdx, final int offsetX, final int offsetY, final int offsetZ ){

        DataBuffer x = op.x().data();
        DataBuffer z = op.z().data();

        //Task: Z = Op(X)
        if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {

            //Heap allocation (float[] or double[])
            if (x.dataType() == DataBuffer.Type.FLOAT) {
                float[] xf = (float[]) x.array();
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = startIdx; i < endIdx; i++) {
                            int xIdx = offsetX + i;
                            xf[xIdx] = op.op(xf[xIdx]);
                        }
                    } else {
                        float[] zf = (float[]) z.array();
                        for (int i = startIdx; i < endIdx; i++) {
                            zf[offsetZ + i] = op.op(xf[offsetX + i]);
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = startIdx; i < endIdx; i++) {
                            int xIdx = offsetX + i * incrX;
                            xf[xIdx] = op.op(xf[xIdx]);
                        }
                    } else {
                        float[] zf = (float[]) z.array();
                        for (int i = startIdx; i < endIdx; i++) {
                            zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX]);
                        }
                    }
                }
            } else {
                double[] xd = (double[]) x.array();
                if (incrX == 1 && incrZ == 1) {
                    if (x == z) {
                        for (int i = startIdx; i < endIdx; i++) {
                            int xIdx = offsetX + i;
                            xd[xIdx] = op.op(xd[xIdx]);
                        }
                    } else {
                        double[] zd = (double[]) z.array();
                        for (int i = startIdx; i < endIdx; i++) {
                            zd[offsetZ + i] = op.op(xd[offsetX + i]);
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = startIdx; i < endIdx; i++) {
                            int xIdx = offsetX + i * incrX;
                            xd[xIdx] = op.op(xd[xIdx]);
                        }
                    } else {
                        double[] zd = (double[]) z.array();
                        for (int i = startIdx; i < endIdx; i++) {
                            zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX]);
                        }
                    }
                }
            }
        } else {
            //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
            ByteBuf nbbx = x.asNetty();
            ByteBuf nbbz = z.asNetty();
            if (x.dataType() == DataBuffer.Type.FLOAT) {

                final int startIdxBytes = startIdx * 4;
                final int endIdxBytes = endIdx * 4;

                int byteOffsetX = 4 * offsetX;
                int byteOffsetZ = 4 * offsetZ;
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                            int xbIdx = byteOffsetX + i;
                            nbbx.setFloat(xbIdx, op.op(nbbx.getFloat(xbIdx)));
                        }
                    } else {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                            nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i)));
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                            int xbIdx = byteOffsetX + i * incrX;
                            nbbx.setFloat(xbIdx, op.op(nbbx.getFloat(xbIdx)));
                        }
                    } else {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                            nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX)));
                        }
                    }
                }
            } else {

                final int startIdxBytes = startIdx * 8;
                final int endIdxBytes = endIdx * 8;

                int byteOffsetX = 8 * offsetX;
                int byteOffsetZ = 8 * offsetZ;
                if (incrX == 1 && (x == z || incrZ == 1)) {
                    if (x == z) {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                            int xbIdx = byteOffsetX + i;
                            nbbx.setDouble(xbIdx, op.op(nbbx.getDouble(xbIdx)));
                        }
                    } else {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                            nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i)));
                        }
                    }
                } else {
                    if (x == z) {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                            int xbIdx = byteOffsetX + i * incrX;
                            nbbx.setDouble(xbIdx, op.op(nbbx.getDouble(xbIdx)));
                        }
                    } else {
                        for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                            nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX)));
                        }
                    }
                }
            }
        }
    }
}
