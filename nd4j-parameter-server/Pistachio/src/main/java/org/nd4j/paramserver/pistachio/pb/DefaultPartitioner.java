package org.nd4j.paramserver.pistachio.pb;
import org.apache.commons.configuration.Configuration;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Arrays;

public class DefaultPartitioner implements Partitioner {
    private Configuration conf = ConfigurationManager.getConfiguration();
    private static Logger logger = LoggerFactory.getLogger(DefaultPartitioner.class);
    Partitioner customizedPartitioner = null;
    boolean checkedCustomizedPartitioner = false;

    public long getPartition(byte[] id, long totalParition) {

        String customizedPartitionerClassName = ConfigurationManager.getConfiguration().getString("pistachio.partitioner");
        logger.debug("found customized partitioner {}",customizedPartitionerClassName);
        if (!checkedCustomizedPartitioner && customizedPartitionerClassName != null) {
            checkedCustomizedPartitioner = true;
            logger.info("found customized partitioner {}",customizedPartitionerClassName); 
            try {
                customizedPartitioner = (Partitioner) Class.forName(customizedPartitionerClassName).newInstance();
            } catch (Exception e) {
                logger.debug("error partitioning with customized partitioner, fall back to default", e);
            }
        }

        if (customizedPartitioner != null) {
           return customizedPartitioner.getPartition(id,totalParition);
        }


        long shard = Arrays.hashCode(id) % totalParition;
        shard = shard < 0 ? shard + totalParition: shard;

        if (logger.isDebugEnabled())
            logger.debug("id {} hash code {} partition {}", id, Arrays.hashCode(id), shard);

        return shard;
    }
}

