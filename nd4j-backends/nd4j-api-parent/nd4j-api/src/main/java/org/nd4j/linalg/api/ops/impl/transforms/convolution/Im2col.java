package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;


/**
 * Im2col operation
 */
public class Im2col extends DynamicCustomOp {

    private int kh, kw, sy, sx, ph, pw, dh, dw;
    boolean isSameMode;

    @Builder(builderMethodName = "sameDiffBuilder")
    public Im2col(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(null,sameDiff, inputs, inPlace);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
        addArgs();
    }

    public Im2col() {}


    @Builder(builderMethodName = "execBuilder")
    public Im2col(INDArray[] inputs,INDArray[] outputs, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(null,inputs,outputs );
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
        addArgs();
    }

    @Override
    public String opName() {
        return "im2col";
    }


    private void addArgs() {
        getIArguments().add(kh);
        getIArguments().add(kw);
        getIArguments().add(sy);
        getIArguments().add(sx);
        getIArguments().add(ph);
        getIArguments().add(pw);
        getIArguments().add(dh);
        getIArguments().add(dw);
        getIArguments().add(fromBoolean(isSameMode));

    }


    private static INDArray getNewOutputArray(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX,
                                              int padHeight, int padWidth, int dilationH, int dilationW, boolean coverAll) {
        //number of images
        int n = img.size(0);
        //number of channels (depth)
        int c = img.size(1);
        //image height
        int h = img.size(2);
        //image width
        int w = img.size(3);
        int outHeight = Convolution.outSize(h, kernelHeight, strideY, padHeight, dilationH, coverAll);
        int outWidth = Convolution.outSize(w, kernelWidth, strideX, padWidth, dilationW, coverAll);

        return Nd4j.createUninitialized(new int[] {n, c, kernelHeight, kernelWidth, outHeight, outWidth}, 'c');
    }


    @Override
    public ArrayField doGetValue() {
        return null;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
