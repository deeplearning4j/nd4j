package org.nd4j.linalg.api.ops.impl.scatter;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;


/**
 * Created by farizrahman4u on 3/23/18.
 */

public class ScatterDiv extends DynamicCustomOp {

    public ScatterDiv(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterDiv() {}

    @Override
    public String opName() {
        return "scatter_div";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterDiv";
    }

}
