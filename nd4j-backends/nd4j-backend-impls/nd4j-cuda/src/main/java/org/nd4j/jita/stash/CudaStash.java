package org.nd4j.jita.stash;

import org.nd4j.linalg.memory.stash.BasicStash;

/**
 * @author raver119@gmail.com
 */
public class CudaStash<T> extends BasicStash<T> {
    public CudaStash(String id) {
        super(id);
    }

    @Override
    protected void init() {
        // TODO: to be implemented
    }
}
