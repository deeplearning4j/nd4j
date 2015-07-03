/*
 * Copyright 2014 Yahoo! Inc. Licensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
 * agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.nd4j.paramserver.pistachio.pb.store;

import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.configuration.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.nd4j.paramserver.pistachio.pb.kafka.KafkaSimpleConsumer;
import org.nd4j.paramserver.pistachio.pb.kafka.KafkaSimpleConsumer.BytesMessageWithOffset;
import org.nd4j.paramserver.pistachio.pb.helix.BootstrapPartitionHandler;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
//import org.nd4j.paramserver.pistachio.pb.util.ModuleManager;
import java.util.concurrent.atomic.AtomicLong;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import org.nd4j.paramserver.pistachio.pb.kafka.KeyValue;
import com.ibm.icu.util.ByteArrayWrapper;


public class StorePartition implements BootstrapPartitionHandler, StoreChangable, StorePartitionMBean {
    volatile boolean isRunning = false;
    private final int partitionId;
    private Thread t;
    private static Logger logger = LoggerFactory
            .getLogger(StorePartition.class);
    private final ConcurrentHashMap<String, Long> partitionQueueStatus = new ConcurrentHashMap<>();
    private StoreFactory storeFactory;
    public static final long TIME_THRESHHOLD = 1000L;
    public static final long COUNT_THRESHHOLD = 10000L;
    volatile boolean isTransfering = false;
    volatile boolean receiveData = true;
    volatile boolean transferFinish = false;
    private static final int transferTimeOutThreshHold = 60000;
    private long currentOffset = 0;
    private long consumedMaxOffset = -1;
    private Store s ;
    private Configuration conf = ConfigurationManager.getConfiguration();
    private static final int maxTryTime = 3;
    private static final int offsetThreadHold = ConfigurationManager.getConfiguration().getInt(
            "Pistachio.offsetGapThreadHold");
    private KafkaSimpleConsumer consumer ;

    private AtomicLong seqId = new AtomicLong(0);
    private AtomicLong nextSeqId = new AtomicLong(-1);
    ConcurrentHashMap<ByteArrayWrapper, KeyValue> writeCache = new ConcurrentHashMap<>();

    public static Integer[] keyLocks = new Integer[1024];


    static {
        for (int i =0; i<1024;i++) {
            keyLocks[i] = i;
        }
    }

    public Integer getKeyLock(int key) {
        return keyLocks[key];
    }



    public ConcurrentHashMap<ByteArrayWrapper, KeyValue> getWriteCache() { return writeCache; }
    public KeyValue getFromWriteCache(byte[] key) {
        return writeCache.get(new ByteArrayWrapper(key, key.length));
    }
    public void removeIteamFromCacheAccordingToSeqId(byte[] key, long seqId) {
        ByteArrayWrapper byteArrayWrapper = new ByteArrayWrapper(key, key.length);
        KeyValue keyValueInCache;
        if (writeCache.containsKey(byteArrayWrapper) &&
                (keyValueInCache = writeCache.get(byteArrayWrapper)).seqId == seqId) {
            writeCache.remove(byteArrayWrapper, keyValueInCache);
        }
    }
    public void setSeqId(long id) {
        seqId.set(id);
    }
    public void setNextSeqId(long id) {
        nextSeqId.set(id);
    }
    public long getSeqId() {
        return seqId.get();
    }

    public long getNextSeqId() {
        if (nextSeqId.get() == -1)
            return -1;
        return nextSeqId.incrementAndGet();
    }

    public StorePartition(int partitionId) {
        this.partitionId = partitionId;
    }

    @Override
    public long getKafkaLatestOffset() {
        if (consumer != null)
            return consumer.getLatestOffset();

        return -1;
    }

    @Override
    public long getCurrentConsumeOffset() {
        return consumedMaxOffset;
    }

    @Override
    public long getConsumeOffsetGap() {
        if (getKafkaLatestOffset() == -1)
            return -1;
        if (getCurrentConsumeOffset() == -1)
            return -1;

        return getKafkaLatestOffset() - getCurrentConsumeOffset();
    }



    @Override
    public void startServing() {
        logger.info("startServing()");
        s = storeFactory.getInstance();
        s.open(partitionId);
        if (t == null) {
            if (this.storeFactory == null) {
                throw new RuntimeException("should set storeFactory first");
            }
            isRunning = true;
            logger.info("startServing Partition {}", partitionId);

            t = new Thread() {
                @Override
                public void run() {
                    String partitionTopic = ConfigurationManager.getConfiguration().getString("Profile.Kafka.TopicPrefix") + partitionId;
                    logger.info("run Partition Serving {}", partitionTopic);

                    consumer = new KafkaSimpleConsumer(partitionTopic,
                            0, ConfigurationManager.getConfiguration().getString(
                            "Profile.Helix.InstanceId"), false);
                    //KafkaSimpleConsumer previousConsumer = consumer;
                    //consumer = new KafkaSimpleConsumer(previousConsumer); // create a new KafkaSimpleConsumer as it will be used in another thread
                    //previousConsumer.stop();
                    serving(s, false);
                }
            };
            t.setName("StorePartitionServing - " + partitionId);
            t.start();
        } else {
            t.notify();
        }
    }

    public void serving(Store store, boolean offsetChecking) {

        String partitionTopic;
        long readOffset;
        long startTime = System.currentTimeMillis();
        long recMsgTmpCount = 0;

        try  {
            partitionTopic = ConfigurationManager.getConfiguration().getString("Profile.Kafka.TopicPrefix") + partitionId;
            logger.info("run Partition Serving {} need to check offset {}", partitionTopic,offsetChecking);

            readOffset = store.getCurrentOffset();
            logger.info("partiton {} readoffset {}", partitionId, readOffset);
            partitionQueueStatus.put(partitionTopic, 0L);
        } catch (Exception e) {
            logger.info("error ", e);
            return;
        }

        Kryo kryo = new Kryo();
        while (isRunning) {
            try {
                if (readOffset > 0)
                    logger.debug("parition {} read offset " + readOffset,
                            partitionTopic);

                Iterable<BytesMessageWithOffset> msgs = consumer.fetch(
                        readOffset, 1000);
                long receivedMsgCnt = 0;
                if (isTransfering) {
                    s.flush();
                    currentOffset = readOffset;
                    receiveData = false;
                    wait();
                    receiveData = true;
                }
                for (BytesMessageWithOffset msgWithOffset : msgs) {
                    // to do , how to avoid byte array copy
                    // logger.info(Arrays.toString(msg)+"array length:"+msg.length);
                    byte[] msg = msgWithOffset.message();
                    readOffset = msgWithOffset.offset();
                    if (msg.length == 0 || msg == null) {
                        continue;
                    }
                    int saveTime = 0;

                    //deserialize
                    Input input = new Input(msg);

                    KeyValue keyValue = kryo.readObject(input, KeyValue.class);
                    input.close();


                    logger.debug("adding {} offset {} to store", msg, readOffset);
                    while(!store.add(msg, readOffset) ){
                        saveTime++;
                        if(saveTime > maxTryTime){
                            logger.error("partition {} exceed max try times ", partitionId);
                        }
                    }
                    if (keyValue.seqId <= readOffset)
                        setSeqId(keyValue.seqId);

                    //writeCache.remove(keyValue.key, keyValue);


                    saveTime = 0;

                    logger.debug("Reading    offset {}.", readOffset);
                    receivedMsgCnt++;
                    recMsgTmpCount++;

                }
                if (System.currentTimeMillis() - startTime > TIME_THRESHHOLD
                        || recMsgTmpCount > COUNT_THRESHHOLD) {
                    startTime = System.currentTimeMillis();
                    recMsgTmpCount = 0;
                    store.commitOffset(readOffset);
                    logger.debug("start to commit current offset {}", readOffset);

                    if (readOffset > consumedMaxOffset)
                        consumedMaxOffset = readOffset;

                }
                Long originalValue = partitionQueueStatus.get(partitionTopic);
                if (originalValue == null) {
                    originalValue = 0L;
                }
                partitionQueueStatus.put(partitionTopic, originalValue
                        + receivedMsgCnt);
                if( offsetChecking && (receivedMsgCnt == 0 || consumer.getLastOffset() - readOffset<offsetThreadHold)){
                    store.commitOffset(readOffset);
                    return;
                }
            } catch (Exception e) {
                logger.error("failed to update profile partitionId {}",
                        partitionId, e);
                logger.error("failed to update profile " + e.getMessage());
                logger.error("failed to update profile " , e);
            }
        }
        store.commitOffset(readOffset);

        if (readOffset > consumedMaxOffset)
            consumedMaxOffset = readOffset;

        store.close();
        consumer.stop();
    }

    @Override
    public void stopServing() {
        logger.info("stop serving.........." + partitionId);
        isRunning = false;
        t.interrupt();
        try {
            t.join(3000L);
        } catch (InterruptedException e) {
        }
    }

    @Override
    public void setStoreFactory(StoreFactory sf) {
        // TODO Auto-generated method stub
        this.storeFactory = sf;
    }

    @Override
    public void selfBootstraping(){
        //s = storeFactory.getInstance();
        //s.open(partitionId);
        String partitionTopic = ConfigurationManager.getConfiguration().getString("Profile.Kafka.TopicPrefix") + partitionId;
        logger.info("run Partition Serving {}", partitionTopic);

        KafkaSimpleConsumer consumer1 = new KafkaSimpleConsumer(partitionTopic,
                0, ConfigurationManager.getConfiguration().getString(
                "Profile.Helix.InstanceId"), false);
        long currentOffset = s.getCurrentOffset();
        long kafkaOffset = -1;
        try{
            kafkaOffset = consumer1.getLastOffset();
        }catch(InterruptedException e){
            logger.error("interrupted when getting last offset");
        }

        if (currentOffset > kafkaOffset) {
            currentOffset = kafkaOffset;
            s.commitOffset(kafkaOffset);
        }
        long earlistOffset = consumer1.getEarlistOffset();
        if (currentOffset < earlistOffset) {
            currentOffset = earlistOffset;
            s.commitOffset(earlistOffset);
        }

        while (kafkaOffset >  currentOffset) {
            currentOffset = s.getCurrentOffset();
            logger.info("partition catching up:"+partitionId+"  kafkaoffset:"+kafkaOffset+" currentoffset:"+currentOffset);
            try {
                Thread.sleep(1000);
            }catch (Exception e)
            {
            }
        }

        nextSeqId.set(kafkaOffset);
        setSeqId(kafkaOffset - 1);
        logger.info("partition:"+partitionId+"  caught up, start to serving as master , setting next seq to: {}", kafkaOffset);
        /*
        */
    }




    @Override
    public void bootstrapingOthers() {
    }
}
