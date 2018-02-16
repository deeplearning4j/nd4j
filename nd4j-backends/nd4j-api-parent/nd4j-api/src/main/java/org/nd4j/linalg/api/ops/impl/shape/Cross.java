package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

public class Cross extends DynamicCustomOp {

    public Cross() {
    }


    public Cross(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args, false);
    }

    @Override
    public String opName() {
        return "cross";
    }


    @Override
    public String tensorflowName() {
        return "Cross";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        /**
         * dL / dx = dL / dCross * dCross / dx
         * dCross(a,b) / da = Cross(1, b)
         * dCross(a,b) / db = Cross(a, 1)
         *
         * return (grad * Cross(1, b), grad * Cross(a, 1)
         */
        SDVariable grad = gradients.get(0);
        SDVariable a = larg();
        SDVariable b = rarg();
        SDVariable ones = sameDiff.onesLike(a);

        SDVariable gradLeft = grad.mul(sameDiff.cross(ones, b));
        SDVariable gradRight = grad.mul(sameDiff.cross(a, ones));

        return Arrays.asList(gradLeft, gradRight);
    }
}
