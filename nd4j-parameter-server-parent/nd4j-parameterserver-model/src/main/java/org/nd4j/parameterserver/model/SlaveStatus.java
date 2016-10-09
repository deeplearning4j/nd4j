package org.nd4j.parameterserver.model;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * The slave status of whether the
 * slave node is started or not.
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class SlaveStatus implements Serializable {
    private String slave;

    /**
     * Whether the slavenode is started or not.
     * @return
     */
    public boolean started() {
        return slave.equals(ServerState.STARTED.name().toLowerCase());
    }

}
