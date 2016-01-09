package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;

public abstract class BaseCPUTransformOpAction extends BaseCPUAction {

    protected final TransformOp op;

    /**
     * Constructor for operating on subset of NDArray
     */
    public BaseCPUTransformOpAction(TransformOp op, int threshold, int n, int offsetX, int offsetY, int offsetZ,
                                    int incrX, int incrY, int incrZ) {
        super(threshold, n, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        this.op = op;
        this.offsetY = (op.y() != null ? op.y().offset() : 0);
        this.incrY = (op.y() != null ? op.y().elementWiseStride() : 0);
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public BaseCPUTransformOpAction(TransformOp op, int threshold) {
        super(op, threshold);
        this.op = op;
        this.offsetY = (op.y() != null ? op.y().offset() : 0);
        this.incrY = (op.y() != null ? op.y().elementWiseStride() : 0);
    }

    protected void doOp(final int startIdx, final int endIdx, final int offsetX, final int offsetY, final int offsetZ){
        DataBuffer x = op.x().data();
        DataBuffer y = (op.y() != null ? op.y().data() : null);
        DataBuffer z = op.z().data();

        if (y != null) {
            //Task: Z = Op(X,Y)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {

                //Heap allocation: float[] or double[]
                if (x.dataType() == DataBuffer.Type.FLOAT) {
                    float[] xf = (float[]) x.array();
                    float[] yf = (float[]) y.array();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = startIdx; i < endIdx; i++) {
                                int xIdx = offsetX + i;
                                xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = startIdx; i < endIdx; i++) {
                                zf[offsetZ + i] = op.op(xf[offsetX + i], yf[offsetY + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = startIdx; i < endIdx; i++) {
                                int xIdx = offsetX + i * incrX;
                                xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i * incrY]);
                            }
                        } else {
                            float[] zf = (float[]) z.array();
                            for (int i = startIdx; i < endIdx; i++) {
                                zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                            }
                        }
                    }
                } else {
                    double[] xd = (double[]) x.array();
                    double[] yd = (double[]) y.array();
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = startIdx; i < endIdx; i++) {
                                int xIdx = offsetX + i;
                                xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = startIdx; i < endIdx; i++) {
                                zd[offsetZ + i] = op.op(xd[offsetX + i], yd[offsetY + i]);
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = startIdx; i < endIdx; i++) {
                                int xIdx = offsetX + i * incrX;
                                xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i * incrY]);
                            }
                        } else {
                            double[] zd = (double[]) z.array();
                            for (int i = startIdx; i < endIdx; i++) {
                                zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX], yd[offsetY + i * incrY]);
                            }
                        }
                    }
                }
            } else {

                //Direct allocation (FloatBuffer / DoubleBuffer backed by a Netty ByteBuf)
                ByteBuf nbbx = x.asNetty();
                ByteBuf nbby = y.asNetty();
                ByteBuf nbbz = z.asNetty();
                if (x.dataType() == DataBuffer.Type.FLOAT) {

                    final int startIdxBytes = startIdx * 4;
                    final int endIdxBytes = endIdx * 4;  //End index, exclusive

                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetY = 4 * offsetY;
                    int byteOffsetZ = 4 * offsetZ;
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i), nbby.getFloat(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX), nbby.getFloat(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                } else {

                    final int startIdxBytes = startIdx * 8;
                    final int endIdxBytes = endIdx * 8;

                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetY = 8 * offsetY;
                    int byteOffsetZ = 8 * offsetZ;
                    if (incrX == 1 && incrY == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i), nbby.getDouble(byteOffsetY + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i * incrZ, op.op(nbbx.getDouble(byteOffsetX + i * incrX), nbby.getDouble(byteOffsetY + i * incrY)));
                            }
                        }
                    }
                }
            }
        } else {
            //Task: Z = Op(X)
            if (x.allocationMode() == DataBuffer.AllocationMode.HEAP) {
                //Heap allocation: float[] or double[]
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
                    if (incrX == 1 && (x == z || incrZ == 1)) {
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
                    final int endIdxBytes = endIdx * 4;  //End index, exclusive

                    int byteOffsetX = 4 * offsetX;
                    int byteOffsetZ = 4 * offsetZ;
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i, op.op(nbbx.getFloat(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setFloat(xbOffset, op.op(nbbx.getFloat(xbOffset)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 4) {
                                nbbz.setFloat(byteOffsetZ + i * incrZ, op.op(nbbx.getFloat(byteOffsetX + i * incrX)));
                            }
                        }
                    }
                } else {
                    final int startIdxBytes = startIdx * 8;
                    final int endIdxBytes = endIdx * 8;  //End index, exclusive

                    //Double
                    int byteOffsetX = 8 * offsetX;
                    int byteOffsetZ = 8 * offsetZ;
                    if (incrX == 1 && (x == z || incrZ == 1)) {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                int xbOffset = byteOffsetX + i;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset)));
                            }
                        } else {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                nbbz.setDouble(byteOffsetZ + i, op.op(nbbx.getDouble(byteOffsetX + i)));
                            }
                        }
                    } else {
                        if (x == z) {
                            for (int i = startIdxBytes; i < endIdxBytes; i += 8) {
                                int xbOffset = byteOffsetX + i * incrX;
                                nbbx.setDouble(xbOffset, op.op(nbbx.getDouble(xbOffset)));
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
}
