package org.nd4j.weightinit;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;

import java.util.Arrays;

/**
 * Abstract class for {@link WeightInitScheme}
 * This handles boilerplate like delegating to the parameters view.
 *
 *
 *
 * @author Adam Gibson
 */
public abstract class BaseWeightInitScheme implements WeightInitScheme {
    private char order;

    public BaseWeightInitScheme(char order) {
        this.order = order;
    }

    public abstract INDArray doCreate(int[] shape, INDArray paramsView);

    @Override
    public INDArray create(int[] shape, INDArray paramsView) {
        return handleParamsView(doCreate(shape,paramsView),paramsView);
    }

    @Override
    public char order() {
        return order;
    }

    protected INDArray handleParamsView(INDArray outputArray, INDArray paramView) {
        INDArray flat = Nd4j.toFlattened(order(), outputArray);
        if (flat.length() != paramView.length())
            throw new RuntimeException("ParamView length does not match initialized weights length (view length: "
                    + paramView.length() + ", view shape: " + Arrays.toString(paramView.shape())
                    + "; flattened length: " + flat.length());

        paramView.assign(flat);

        return paramView.reshape(order(), outputArray.shape());
    }
}
