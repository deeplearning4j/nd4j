package org.nd4j.linalg.api.ops.random.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

/**
 * @author raver119@gmail.com
 */
public class UniformDistribution extends BaseRandomOp {
    private double from;
    private double to;

    public UniformDistribution(){
        super();
    }

    /**
     * This op fills Z with random values within from...to boundaries
     * @param z
     * @param from
     * @param to
     */
    public UniformDistribution(INDArray z, double from, double to) {
        init(null, null, z, z.length());
        this.from = from;
        this.to = to;
        this.extraArgs = new Object[]{this.from, this.to};
    }

    /**
     * This op fills Z with random values within 0...1
     * @param z
     */
    public UniformDistribution(INDArray z) {
        this(z, 0.0, 1.0);
    }

    /**
     * This op fills Z with random values within 0...to
     * @param z
     */
    public UniformDistribution(INDArray z, double to) {
        this(z, 0.0, to);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "distribution_uniform";
    }
}
