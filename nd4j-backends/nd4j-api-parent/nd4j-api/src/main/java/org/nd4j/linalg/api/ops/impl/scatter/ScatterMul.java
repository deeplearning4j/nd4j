package org.nd4j.linalg.api.ops.impl.scatter;

import com.sun.tools.sjavac.pubapi.PubApi;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class ScatterMul extends DynamicCustomOp {

    public ScatterMul(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterMul() {}

    @Override
    public String opName() {
        return "scatter_mul";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterMul";
    }

}
