package org.nd4j.parameterserver.model;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;

/**
 * Status of a master node, covered
 * both by the master node itself and its responder.
 *
 * @author Adam Gibson
 */
@Data @Builder
public class MasterStatus implements Serializable {
    private String master,responder;


    /**
     * Returns true if bth
     * the master and responder are started.
     * @return
     */
    public boolean started() {
        return master.equals(ServerState.STARTED.name().toLowerCase()) && responder.equals(ServerState.STARTED.name().toLowerCase());
    }

}
