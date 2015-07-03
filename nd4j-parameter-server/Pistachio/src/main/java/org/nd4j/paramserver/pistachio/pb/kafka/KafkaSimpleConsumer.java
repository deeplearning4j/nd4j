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

package org.nd4j.paramserver.pistachio.pb.kafka;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.configuration.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.nd4j.paramserver.pistachio.pb.platform.kafka.KafkaConnection.CrossColoDedoder;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;

import kafka.api.FetchRequest;
import kafka.message.Message;
import kafka.api.FetchRequestBuilder;
import kafka.api.PartitionOffsetRequestInfo;
import kafka.javaapi.FetchResponse;
import kafka.cluster.Broker;
import kafka.common.ErrorMapping;
import kafka.common.TopicAndPartition;
import kafka.javaapi.OffsetRequest;
import kafka.javaapi.OffsetResponse;
import kafka.javaapi.PartitionMetadata;
import java.nio.ByteBuffer;

import kafka.javaapi.TopicMetadata;
import kafka.javaapi.TopicMetadataRequest;
import kafka.javaapi.TopicMetadataResponse;
import kafka.javaapi.consumer.SimpleConsumer;
import kafka.message.MessageAndOffset;
import kafka.serializer.Decoder;
import kafka.serializer.Encoder;
import kafka.utils.VerifiableProperties;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;


/**
 * refer @{link https://cwiki.apache.org/confluence/display/KAFKA/0.8.0+SimpleConsumer+Example}
 *
 * This class is not thread safe, the caller must ensure all the methods be called from single thread
 */
public class KafkaSimpleConsumer {
    public static class CrossColoEncoder implements Encoder<byte[]> {
        public CrossColoEncoder(VerifiableProperties v) {

        }

        @Override
        public byte[] toBytes(byte[] msg) {
            return msg;
            /*
            try {
                //YCR ycr = YCR.createYCR(YCR.NATIVE_YCR);
                //String signature = ycr.getKey(conf.getString("Kafka.Encrpt.Sign.Key"));

                //SymmetricContext sc = ycr.createSymmetricContext(conf
                //        .getString("Kafka.Encrpt.Queue.Key"));

                //byte[] encryptMsg = sc.encryptSign64Bytes(msg, signature.getBytes());
                logger.debug("Encode msg---" + new String(msg));
                logger.debug("Encode result ---" + encryptMsg);
                return encryptMsg;
            } catch (YCRException | KeyDBException e) {

                logger.error("Can't encode msg, msg---" + new String(msg));
                logger.error("Can't encode msg, reason---" + e.getMessage());
                logger.error("Can't encode msg, st---" + e);
                return "".getBytes();
            }
            */
        }

    }

    public static class CrossColoDedoder implements Decoder<byte[]> {
        public CrossColoDedoder() {
        }

        @Override
        public byte[] fromBytes(byte[] msg) {
            try {
                Kryo kryo = new Kryo();
                Input input = new Input(msg);

                KeyValue keyValue = kryo.readObject(input, KeyValue.class);
                input.close();
            } catch (Exception e)
            {}

            return msg;
            /*
            logger.debug("decode message");
            YCR ycr;
            try {
                ycr = YCR.createYCR(YCR.NATIVE_YCR);
                SymmetricContext sc = ycr.createSymmetricContext(conf
                        .getString("Kafka.Encrpt.Queue.Key"));
                String signature = ycr.getKey(conf.getString("Kafka.Encrpt.Sign.Key"));
                logger.debug("Decode msg---" + new String(msg));
                //                ByteBuffer bb = msg.payload();
                //                byte[] result = new byte[bb.remaining()];
                //                bb.get(msg);

                logger.debug("Decode result ---"
                        + sc.decryptSign64Bytes(msg, signature.getBytes()));
                return sc.decryptSign64Bytes(msg, signature.getBytes());
            } catch (YCRException | KeyDBException e) {
                logger.error("Can't decode msg, msg---" + msg);
                logger.error("Can't decode msg, reason---" + e.getMessage());
                logger.error("Can't decode msg, st---" + e);
                return "".getBytes();
            }
            */
        }

        public byte[] fromMessage(Message msg) {
            ByteBuffer bb = msg.payload();
            byte[] result = new byte[bb.remaining()];
            bb.get(result);
            return fromBytes(result);
        }
    }

    public static final List<BytesMessageWithOffset> EMPTY_MSGS = new ArrayList<>();

    private static final Logger logger = LoggerFactory.getLogger(KafkaSimpleConsumer.class);
    private static final Configuration conf = ConfigurationManager.getConfiguration();

    private static final CrossColoDedoder decoder = new CrossColoDedoder();

    private final List<KafkaBroker> allBrokers;
    private final String topic;
    private final int partitionId;
    private final String clientId;

    private volatile Broker leaderBroker;
    private List<KafkaBroker> replicaBrokers;
    private SimpleConsumer consumer = null;


    public KafkaSimpleConsumer(String topic, int partitionId, String clientId) {
        this(topic, partitionId, clientId, true);
    }

    @SuppressWarnings("unchecked")
    public KafkaSimpleConsumer(String topic, int partitionId, String clientId, boolean crossColo) {
        List<String> brokers = null;
        brokers = conf.getList("Kafka.Broker");
        List<KafkaBroker> brokerList = new ArrayList<KafkaBroker>();
        for (String broker: brokers) {
            String[] tokens = broker.split(":");
            if (tokens.length != 2) {
                logger.warn("wrong broker name {}, its format should be host:port", broker);
                continue;
            }
            String host = tokens[0];
            int port = -1;
            try {
                port = Integer.parseInt(tokens[1]);
            } catch (NumberFormatException e) {
                logger.warn("wrong broker name {}, its format should be host:port", broker);
                continue;
            }
            brokerList.add(new KafkaBroker(host, port));
        }
        this.allBrokers = Collections.unmodifiableList(brokerList);
        this.topic = topic;
        this.partitionId = partitionId;
        this.clientId = String.format("%s_%d_%s", topic, partitionId, clientId);

        this.replicaBrokers = new ArrayList<>();
        this.replicaBrokers.addAll(this.allBrokers);
    }

    public long getEarlistOffset() {
        if (consumer == null)
            return -1;

        try {
            return getOffset(true);
        } catch (Exception e) {
            return -1;
        }
    }
    public long getLatestOffset() {
        if (consumer == null)
            return -1;

        try {
            return getOffset(false);
        } catch (Exception e) {
            return -1;
        }
    }

    /**
     * Copy around the internal state of other KafkaSimpleConsumer to a newly constructed one. Mainly for leaderBroker
     * Can be used to share KafkaSimpleConsumer across different threads
     *
     * @param other
     */
    public KafkaSimpleConsumer(KafkaSimpleConsumer other) {
        this.allBrokers = other.allBrokers;
        this.topic = other.topic;
        this.partitionId = other.partitionId;
        this.clientId = other.clientId;
        this.leaderBroker = other.leaderBroker;
        this.replicaBrokers = new ArrayList<>();
        this.replicaBrokers.addAll(this.allBrokers);
    }

    private SimpleConsumer ensureConsumer(Broker leader) throws InterruptedException {
        if (consumer == null) {
            while (leaderBroker == null) {
                leaderBroker = findNewLeader(leader);
            }

            logger.info("create SimpleConsumer for {} - {}, leader broker {}:{}", topic, partitionId, leaderBroker.host(), leaderBroker.port());

            consumer = new SimpleConsumer(leaderBroker.host(),
                    leaderBroker.port(), conf.getInt("kafka.soTimeout"),
                    conf.getInt("kafka.bufferSize"), clientId);
        }
        return consumer;
    }

    public static class BytesMessageWithOffset {
        final byte[] msg;
        final long offset;

        public BytesMessageWithOffset(byte[] msg, long offset) {
            this.msg = msg;
            this.offset = offset;
        }

        public byte[] message() {
            return msg;
        }

        public long offset() {
            return offset;
        }
    }

    static class KafkaBroker {
        final String host;
        final int port;

        KafkaBroker(String host, int port) {
            this.host = host;
            this.port = port;
        }

        @Override
        public String toString() {
            return String.format("%s:%d", host, port);
        }
    }

    private Iterable<BytesMessageWithOffset> filterAndDecode(Iterable<MessageAndOffset> kafkaMessages, long offset) {
        List<BytesMessageWithOffset> ret = new LinkedList<>();
        for (MessageAndOffset msgAndOffset: kafkaMessages) {
            if (msgAndOffset.offset() >= offset) {
                byte[] payload = decoder.fromMessage(msgAndOffset.message());
                // add nextOffset here, thus next fetch will use nextOffset instead of current offset
                ret.add(new BytesMessageWithOffset(payload, msgAndOffset.nextOffset()));
            }
        }
        return ret;
    }

    private long getOffset(boolean earliest) throws InterruptedException {
        TopicAndPartition topicAndPartition = new TopicAndPartition(topic, partitionId);
        Map<TopicAndPartition, PartitionOffsetRequestInfo> requestInfo = new HashMap<TopicAndPartition, PartitionOffsetRequestInfo>();
        requestInfo.put(topicAndPartition, new PartitionOffsetRequestInfo(
            earliest ? kafka.api.OffsetRequest.EarliestTime() : kafka.api.OffsetRequest.LatestTime(), 1));
        OffsetRequest request = new OffsetRequest(
                requestInfo, kafka.api.OffsetRequest.CurrentVersion(), clientId);
        OffsetResponse response = null;
        try {
            response = consumer.getOffsetsBefore(request);
        } catch (Exception e) {
            // e could be an instance of ClosedByInterruptException as SimpleConsumer.getOffsetsBefore uses nio
            if (Thread.interrupted()) {
                logger.info("catch exception of {} with interrupted in getOffset({}) for {} - {}",
                        e.getClass().getName(), earliest, topic, partitionId);

                throw new InterruptedException();
            }

            logger.error("caught exception in getOffsetsBefore {} - {}", topic, partitionId, e);
            return -1;
        }
        if (response.hasError()) {
            logger.error("error fetching data Offset from the Broker {}. reason: {}", leaderBroker.host(), response.errorCode(topic, partitionId));
            return -1;
        }
        long[] offsets = response.offsets(topic, partitionId);
        return earliest ? offsets[0] : offsets[offsets.length - 1];
    }

    //
    public Iterable<BytesMessageWithOffset> fetch(long offset, int timeoutMs) throws InterruptedException {
        List<BytesMessageWithOffset> newOffsetMsg = new ArrayList<BytesMessageWithOffset>();
        FetchResponse response = null;
        Broker previousLeader = leaderBroker;
        while (true) {
            ensureConsumer(previousLeader);

            if (offset == Long.MAX_VALUE) {
                offset = getOffset(false);
                logger.info("offset max long, fetch from latest in kafka {}", offset);
            }

            FetchRequest request = new FetchRequestBuilder()
                    .clientId(clientId)
                    .addFetch(topic, partitionId, offset, 100000000)
                    .maxWait(timeoutMs)
                    .minBytes(1)
                    .build();

            //logger.debug("fetch offset {}", offset);

            try {
                response = consumer.fetch(request);
            } catch (Exception e) {
                // e could be an instance of ClosedByInterruptException as SimpleConsumer.fetch uses nio
                if (Thread.interrupted()) {
                    logger.info("catch exception of {} with interrupted in fetch for {} - {} with offset {}",
                            e.getClass().getName(), topic, partitionId, offset);

                    throw new InterruptedException();
                }
                logger.warn("caughte exception in fetch {} - {}", topic, partitionId, e);
                response = null;
            }

            if (response == null || response.hasError()) {
                short errorCode = response != null ? response.errorCode(topic, partitionId) : ErrorMapping.UnknownCode();
                logger.warn("fetch {} - {} with offset {} encounters error: {}", topic, partitionId, offset, errorCode);

                boolean needNewLeader = false;
                if (errorCode == ErrorMapping.RequestTimedOutCode()) {
                    //TODO: leave it here
                } else if (errorCode == ErrorMapping.OffsetOutOfRangeCode()) {
                    //TODO: fetch the earliest offset or latest offset ?
                    // seems no obvious correct way to handle it
                    long earliestOffset = getOffset(true);
                    logger.debug("get earilset offset {} for {} - {}", earliestOffset, topic, partitionId);
                    if (earliestOffset < 0) {
                        needNewLeader = true;
                    } else {
                        newOffsetMsg.add(new BytesMessageWithOffset(null, earliestOffset));
                        offset = earliestOffset;
                        continue;
                    }
                } else {
                    needNewLeader = true;
                }

                if (needNewLeader) {
                    stopConsumer();
                    previousLeader = leaderBroker;
                    leaderBroker = null;
                    continue;
                }
            } else {
                break;
            }
        }

        return response != null ? filterAndDecode(response.messageSet(topic, partitionId), offset) :
            (newOffsetMsg.size() > 0 ? newOffsetMsg : EMPTY_MSGS);
    }

    private void stopConsumer() {
        if (consumer != null) {
            try {
                consumer.close();
                logger.info("stop consumer for {} - {}, leader broker {}", topic, partitionId, leaderBroker);
            } catch (Exception e) {
                logger.warn("stop consumer for {} - {} failed", topic, partitionId, e);
            } finally {
                consumer = null;
            }
        }
    }

    // stop the consumer
    public void stop() {
        stopConsumer();
        logger.info("KafkaSimpleConsumer stopped for {} - {}", topic, partitionId);
    }

    private PartitionMetadata findLeader() throws InterruptedException {
        List<String> topics = new ArrayList<String>();
        topics.add(topic);

        for (KafkaBroker broker: replicaBrokers) {
            SimpleConsumer consumer = null;
            try {
                logger.debug("findLeader, try broker {}:{}", broker.host, broker.port);
                consumer = new SimpleConsumer(broker.host, broker.port, conf.getInt("kafka.soTimeout"),
                        conf.getInt("kafka.bufferSize"), clientId + "leaderLookup");
                TopicMetadataResponse resp = consumer.send(new TopicMetadataRequest(topics));

                // just one topic inside the topics
                List<TopicMetadata> metaData = resp.topicsMetadata();
                for (TopicMetadata item: metaData) {
                    for (PartitionMetadata part: item.partitionsMetadata()) {
                        if (part.partitionId() == partitionId) {
                            replicaBrokers.clear();
                            for (Broker replica: part.replicas()) {
                                replicaBrokers.add(new KafkaBroker(replica.host(), replica.port()));
                            }
                            return part;
                        }
                    }
                }
            } catch (Exception e) {
                // e could be an instance of ClosedByInterruptException as SimpleConsumer.send uses nio
                if (Thread.interrupted()) {
                    logger.info("catch exception of {} with interrupted in find leader for {} - {}",
                            e.getClass().getName(), topic, partitionId);

                    throw new InterruptedException();
                }
                logger.warn("error communicating with Broker {} to find leader for {} - {}", broker, topic, partitionId, e);
            } finally {
                if (consumer != null) {
                    try {
                        consumer.close();
                    } catch (Exception e) {}
                }
            }
        }

        return null;
    }

    private Broker findNewLeader(Broker oldLeader) throws InterruptedException {
        long retryCnt = 0;
        while (true) {
            PartitionMetadata metadata = findLeader();
            logger.debug("findNewLeader - meta leader {}, previous leader {}", metadata, oldLeader);
            if (metadata != null && metadata.leader() != null && (oldLeader == null ||
                    (!(oldLeader.host().equalsIgnoreCase(metadata.leader().host()) &&
                      (oldLeader.port() == metadata.leader().port())) || retryCnt != 0))) {
                // first time through if the leader hasn't changed give ZooKeeper a second to recover
                // second time, assume the broker did recover before failover, or it was a non-Broker issue
                logger.info("findNewLeader - using new leader {} from meta data, previous leader {}", metadata.leader(), oldLeader);
                return metadata.leader();
            }
            //TODO: backoff retry
            Thread.sleep(1000L);
            retryCnt ++;
            // if could not find the leader for current replicaBrokers, let's try to find one via allBrokers
            if (retryCnt >= 3 && (retryCnt - 3) % 5 == 0) {
                logger.warn("can nof find leader for {} - {} after {} retries", topic, partitionId, retryCnt);
                replicaBrokers.clear();
                replicaBrokers.addAll(allBrokers);
            }
        }
    }

    public long getLastOffset() throws InterruptedException {
        OffsetResponse response = null;
        Broker previousLeader = leaderBroker;
        while (true) {
            TopicAndPartition topicAndPartition = new TopicAndPartition(topic, partitionId);
            Map<TopicAndPartition, PartitionOffsetRequestInfo> requestInfo = new HashMap<TopicAndPartition, PartitionOffsetRequestInfo>();
            requestInfo.put(topicAndPartition, new PartitionOffsetRequestInfo(kafka.api.OffsetRequest.LatestTime(), 1));
            kafka.javaapi.OffsetRequest request = new kafka.javaapi.OffsetRequest(
                    requestInfo, kafka.api.OffsetRequest.CurrentVersion(), clientId);

            ensureConsumer(previousLeader);
            try {
                response = consumer.getOffsetsBefore(request);
            } catch (Exception e) {
                // e could be an instance of ClosedByInterruptException as SimpleConsumer.fetch uses nio
                if (Thread.interrupted()) {
                    logger.info("catch exception of {} with interrupted in getLastOffset for {} - {}",
                            e.getClass().getName(), topic, partitionId);

                    throw new InterruptedException();
                }
                logger.warn("caughte exception in getLastOffset {} - {}", topic, partitionId, e);
                response = null;
            }
            if (response == null || response.hasError()) {
                short errorCode = response != null ? response.errorCode(topic, partitionId) : ErrorMapping.UnknownCode();

                logger.warn("Error fetching data Offset for {} - {}, the Broker. Reason: {}",
                        topic, partitionId, errorCode);

                stopConsumer();
                previousLeader = leaderBroker;
                leaderBroker = null;
                continue;
            }
            break;
        }
        long[] offsets = response.offsets(topic, partitionId);
        return offsets[offsets.length - 1];
    }

    public static void main(String[] args) {
        /**
         */
        if (args.length < 3) {
            System.out.println("Usage: class $topic $clientId $offset");
            return;
        }

        KafkaSimpleConsumer consumer = new KafkaSimpleConsumer(args[0], 0, args[1]);

        long offset = Long.parseLong(args[2]);

        try {
            while (true) {
                Iterable<BytesMessageWithOffset> msgs = consumer.fetch(offset, 1000);
                for (BytesMessageWithOffset msg: msgs) {
                    System.out.println("receive msg >>>" + msg.msg);
                    offset = msg.offset();
                }
                Thread.sleep(1000L);
            }
        } catch (InterruptedException e) {

        }

        consumer.stop();
    }
}
