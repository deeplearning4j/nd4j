package org.nd4j.parameterserver.model;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/16.
 */
@Data
@Builder
public class MasterConnectionInfo implements Serializable {
    private String connectionUrl;
    private String responderUrl;
    private List<String> slaveUrls;
}
