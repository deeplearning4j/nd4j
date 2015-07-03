package org.nd4j.paramserver.pistachio.pb;
import org.apache.commons.configuration.Configuration;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Arrays;

public class ExamplePartitioner implements Partitioner {
    private Configuration conf = ConfigurationManager.getConfiguration();
    private static Logger logger = LoggerFactory.getLogger(ExamplePartitioner.class);

    public long getPartition(byte[] id, long totalParition) {

        long shard = Arrays.hashCode(id)  % totalParition;
        shard = shard < 0 ? shard + totalParition: shard;
        logger.debug("example partitioner shard id {} into parittion {}", id, shard);
        return shard;
    }
}


