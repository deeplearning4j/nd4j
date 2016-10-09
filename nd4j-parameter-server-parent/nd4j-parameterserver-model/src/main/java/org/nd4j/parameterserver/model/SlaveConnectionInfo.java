package org.nd4j.parameterserver.model;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;
import java.util.List;

/**
 * Slave connection info,
 * including the connection url,
 * and the associated master.
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class SlaveConnectionInfo implements Serializable {
    private String connectionUrl;
    private String masterUrl;
}
