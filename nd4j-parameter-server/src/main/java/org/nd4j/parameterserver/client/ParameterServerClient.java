package org.nd4j.parameterserver.client;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Parameter server client for
 * publishing and retrieving ndarrays
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class ParameterServerClient {

    private String ndarraySendUrl;
    private String ndarrayRetrieveUrl;


    public void pushNDArray(INDArray arr) {

    }

    public INDArray getArray() {
        return null;
    }

}
