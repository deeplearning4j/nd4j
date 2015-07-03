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

import org.nd4j.paramserver.pistachio.pb.util.Convert;
import com.ibm.icu.util.ByteArrayWrapper;

import java.util.Iterator;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import kyotocabinet.Cursor;
import kyotocabinet.DB;
import kyotocabinet.Visitor;

import org.apache.commons.lang3.ArrayUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import org.nd4j.paramserver.pistachio.pb.DefaultDataInterpreter;

import org.rocksdb.*;

import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import java.util.Arrays;
//import org.nd4j.paramserver.pistachio.pb.platform.profile.ProfileUtil;

public class LocalStorageEngine {
    private static Logger logger = LoggerFactory.getLogger(LocalStorageEngine.class);
    public static final String storeEngineConf = ConfigurationManager.getConfiguration().getString("LocalStoreEngine", "InMem");
    public static final Boolean ignoreHistoryWhenNoOffsetFoundInStore = ConfigurationManager.getConfiguration().getBoolean("IgnoreHistoryWhenNoOffsetFoundInStore", false);

    private static final int writeBufferSize = 5000;

    private final String baseDir;
    private final int numStores;   //0 means use helix to control partition num
    private final int recordSizeAlignment;
    private final int totalRecords;
    private final long mappedMemorySize;
    private final boolean isReadOnly;
    private StoreEngine[] stores;

    public LocalStorageEngine(
            String baseDir,
            int numStores,
            int recordSizeAlignment,
            int totalRecords,
            long mappedMemorySize) {

        this(baseDir, numStores, recordSizeAlignment, totalRecords, mappedMemorySize, false);
    }

    public LocalStorageEngine(
            String baseDir,
            int numStores,
            int recordSizeAlignment,
            int totalRecords,
            long mappedMemorySize,
            boolean isReadOnly) {

        this.baseDir = baseDir;
        this.recordSizeAlignment = recordSizeAlignment;
        this.totalRecords = totalRecords;
        this.mappedMemorySize = mappedMemorySize;
        this.isReadOnly = isReadOnly;
        this.numStores = numStores;
        stores = new StoreEngine[numStores];
    }
    /**
     * Open the data base
     */
    public void open() throws Exception {
        for (StoreEngine store : stores) {
            store.open();
        }
    }

    /**
     * Open the data base
     */
    public void open(int partitionId) throws Exception {
        if (storeEngineConf.equals("RocksDB")) {
            stores[partitionId] = new RocksDBStoreEngine();
        } else if (storeEngineConf.equals("InMem")) {
            stores[partitionId] = new InMemStoreEngine();
        } else {
            stores[partitionId] = new KCStoreEngine();
        }
        stores[partitionId].init(partitionId);
        stores[partitionId].open();
    }

    /**
     * Close all database.
     */
    public void close() {
        for (StoreEngine store : stores) {
            if(store != null)
                store.close();
        }
    }
    /**
     * Close all database.
     */
    public void close(int partitionId) {
        if(stores[partitionId] != null){
            stores[partitionId].close();
            stores[partitionId] = null;
        }

    }

    /**
     * Get database index based on input key.
     *
     * @param key
     * @return
     */
    protected int getDbIndex(byte[] key) {
        return (int) (Arrays.hashCode(key) % numStores);
    }

    /**
     * Store a record.
     *
     * @param key
     * @param value
     */
    public void forceFlush(int partitionId) throws Exception {
        if(stores[partitionId] != null)
            stores[partitionId].forceFlush(true);
        else
            throw new Exception("partition "+partitionId+" not exist, can't flush");
    }

    /**
     * Store a record.
     *
     * @param key
     * @param value
     */
    public void store(byte[] key, long partition, byte[] value) throws Exception {
        int dbIndex = (int)partition;
        /*
        int dbIndex = getDbIndex(key);
        */
        if(stores[dbIndex] != null)
            stores[dbIndex].store(key, value);
        else
            throw new Exception("partition "+dbIndex+" not exist, can't store");
    }

    public void storeOffset( long value, int partitionId) throws Exception {
        if(stores[partitionId] != null)
            stores[partitionId].storeOffset( value);
        else
            throw new Exception("partition "+partitionId+" not exist, can't store offset");
    }

    public long getOffset(int partitionId) throws Exception{
        if(stores[partitionId] != null)
            return stores[partitionId].getOffset();
        else
            return ignoreHistoryWhenNoOffsetFoundInStore ? Long.MAX_VALUE : 0;
    }

    public boolean isPartitionActive(int partitionId){
        if(stores[partitionId] != null){
            return true;
        }else{
            return false;
        }
    }
    /**
     * Get a record.
     *
     * @param key
     * @return
     */
    public byte[] get(byte[] key, int partitionId) {
        int dbIndex = partitionId;//getDbIndex(key);
        if(stores[dbIndex] != null)
            return stores[dbIndex].get(key);
        else
            return null;
    }

    /**
     * Remove a record.
     *
     * @param key
     * @return
     */
    public boolean delete(byte[] key, long partition) {
        int dbIndex = (int)partition;
        if(stores[dbIndex] != null)
            return stores[dbIndex].delete(key);
        else
            return false;
    }

    /**
     * Get total number of records.
     *
     * @return
     */
    public long count() {
        long count = 0;

        for (StoreEngine store : stores) {
            if (store != null) {
                count += store.count();
            }
        }

        return count;
    }

    /**
     * Go through all the records with the visitor.
     *
     * @param visitor
     */
    public Iterator iterator(int partitionId, long id) {
        return stores[partitionId].iterate(id);
    }

    public void jump(int partitionId, long id, byte[] key) {
        stores[partitionId].jump(key, id);
    }

    /**
     * Go through all the records with the visitor.
     *
     * @param visitor
     */
    public void iterateWithReadLock(Visitor visitor) {
        if (visitor == null) {
            return;
        }

        for (StoreEngine store : stores) {
            if (store != null) {
                store.iterateWithReadLock(visitor);
            }
        }
    }

    /**
     * Go through all the records in the single store with visistor
     *
     * @param visitor
     * @param index
     *
     */
    public void iteratorSingleStore(Visitor visitor, int index) {


        StoreEngine store = stores[index];
        if (store != null) {
            store.iterateWithReadLock(visitor);
        }
    }



    private interface StoreEngine {
        void init(int dbIndex);
        void forceFlush(boolean forceFlush);
        void open() throws Exception;
        void close();
        void store(byte[] key, byte[] value) throws Exception;
        void storeOffset(long value) throws Exception;
        byte[] get(byte[] key);
        long getOffset();
        boolean delete(byte[] key);
        long count();
        Iterator iterate(long id);
        void iterateWithReadLock(Visitor visitor);
        void jump(byte[] key,long id);
    };

    private class InMemStoreEngine implements StoreEngine {
        private Logger logger = LoggerFactory.getLogger(InMemStoreEngine.class);
        private int dbIndex;
        private ConcurrentHashMap<ByteArrayWrapper, byte[]> db = new ConcurrentHashMap<>();
        private AtomicInteger flushCounter;
        private final static int PER_OFFSET_FLUSH = 100;
        private static final String offsetKey  = "offset_storage_tk";
        private long offset  = -1;

        public InMemStoreEngine() {
            logger.info("creating InMemStoreEngine");
        }
        public void init(int dbIndex) {
            this.dbIndex = dbIndex;
            flushCounter = new AtomicInteger(0);

        }
        public void forceFlush(boolean forceFlush) {
            return;
        }
        public void open() throws Exception {
            logger.info("open in mem db");
        }
        public void close() {
            logger.info("close in mem db");
        }
        private void flushOffset() throws RocksDBException{
            if(offset > 0){
                db.put(new ByteArrayWrapper(offsetKey.getBytes(),offsetKey.getBytes().length), Convert.longToBytes(offset));
            }
        }


        public void store(byte[] key, byte[] value) throws Exception {
            try {
                ByteArrayWrapper baw = new ByteArrayWrapper(key, key.length);
                db.put(baw, value);
                logger.debug("{} stored value {} for key {}.{}", db.hashCode(), value, baw,baw.size);

                if(flushCounter.addAndGet(1) % PER_OFFSET_FLUSH == 0) {
                    logger.debug("flushing offset at {}", flushCounter);
                    flushOffset();
                }
            } catch (Exception e) {
                logger.debug("error store {}/{}", key, value, e);
                throw new Exception("error store " + key + " " + value, e);
            }
        }
        public void storeOffset(long value) throws Exception {
            if(value >= 0){
                offset = value;
            }
            else{
                throw new Exception("offset can't be negative value");
            }
        }
        public byte[] get(byte[] key) {
            byte[] result = null;
            byte[] byteKey = (key);
            try{
                result =  db.get(new ByteArrayWrapper(key, key.length));
                logger.debug("{} got value {} for key {}", db.hashCode(), result, key);
            } catch (Exception e) {
                logger.debug("error getting key {}", key, e);
            }
            return result;
        }
        public long getOffset() {
            byte[] result =  null;
            if (offset == -1) {
                try {
                    result = db.get(new ByteArrayWrapper(offsetKey.getBytes(), offsetKey.getBytes().length));
                    if (result != null)
                        offset = Convert.bytesToLong(result);
                    else
                        offset = ignoreHistoryWhenNoOffsetFoundInStore ? Long.MAX_VALUE: 0;
                    return offset;
                } catch (Exception e) {
                    offset = ignoreHistoryWhenNoOffsetFoundInStore ? Long.MAX_VALUE: 0;
                    return offset;
                }
            }

            return offset;
        }
        public boolean delete(byte[] key) {
            try{
                db.remove(new ByteArrayWrapper(key, key.length));
            } catch (Exception e) {
                return false;
            }
            return true;

        }
        public long count() {
            return 0;
        }
        public Iterator iterate(long id) {
            logger.info("dont support iterate");
            return null;
        }
        public void iterateWithReadLock(Visitor visitor) {
            logger.info("dont support iterate");
        }
        @Override
        public void jump(byte[] key, long id) {
            // TODO Auto-generated method stub

        }
    }

    private class RocksDBStoreEngine implements StoreEngine {
        private Logger logger = LoggerFactory.getLogger(RocksDBStoreEngine.class);
        private int dbIndex;
        private RocksDB db;
        private AtomicInteger flushCounter;
        private final static int PER_OFFSET_FLUSH = 100;
        private static final String offsetKey  = "offset_storage_tk";
        private long offset  = -1;


        public RocksDBStoreEngine() {
            logger.info("creating RocksDBStoreEngine");
        }
        public void init(int dbIndex) {
            this.dbIndex = dbIndex;
            flushCounter = new AtomicInteger(0);
        }
        public void forceFlush(boolean forceFlush) {
            return;
        }
        public void open() throws Exception {
            RocksDB.loadLibrary();
            Options options = new Options().setCreateIfMissing(true);
            StringBuilder fileNameBuilder = new StringBuilder();
            fileNameBuilder.append(baseDir).append(dbIndex);
            fileNameBuilder.append("/db_").append(dbIndex).append(".rocks");
            try {
                logger.info("opening rocks db file {}", fileNameBuilder.toString());
                db = RocksDB.open(options, fileNameBuilder.toString());
                //assert(false);
            } catch (RocksDBException e) {
                logger.error("error open rocks db {}", fileNameBuilder.toString(),  e);
                throw new Exception("error open rocks db", e);
            }
        }
        public void close() {
            logger.info("closing db");
            if (db != null)
                db.close();

        }
        private void flushOffset() throws RocksDBException{
            if(offset > 0){
                db.put(offsetKey.getBytes(), Convert.longToBytes(offset));
            }
        }
        public void store(byte[] key, byte[] value) throws Exception {
            try {
                db.put(key, value);


                if(flushCounter.addAndGet(1) % PER_OFFSET_FLUSH == 0) {
                    logger.debug("flushing offset at {}", flushCounter);
                    flushOffset();
                }
            } catch (Exception e) {
                logger.debug("error store {}/{}", key, value, e);
                throw new Exception("error store " + key + " " + value, e);
            }
        }
        public void storeOffset(long value) throws Exception {
            if(value >= 0){
                offset = value;
            }
            else{
                throw new Exception("offset can't be negative value");
            }
        }
        public byte[] get(byte[] key) {
            byte[] result = null;
            byte[] byteKey = (key);
            try{
                result =  db.get(byteKey);
            } catch (RocksDBException e) {
                logger.debug("error getting key {}", key, e);
            } catch (Exception e) {
                logger.debug("error getting key {}", key, e);
                return null;
            }
            return result;
        }
        public long getOffset() {
            byte[] result =  null;
            if (offset == -1) {
                try {
                    result = db.get(offsetKey.getBytes());
                    offset = Convert.bytesToLong(result);
                    return offset;
                } catch (Exception e) {
                    offset = ignoreHistoryWhenNoOffsetFoundInStore ? Long.MAX_VALUE: 0;
                    return offset;
                }
            }

            return offset;
        }
        public boolean delete(byte[] key) {
            try{
                db.remove((key));
            } catch (RocksDBException e) {
                return false;
            }
            return true;

        }
        public long count() {
            return 0;
        }
        public Iterator iterate(long id) {
            logger.info("dont support iterate");
            return null;
        }
        public void iterateWithReadLock(Visitor visitor) {
            logger.info("dont support iterate");
        }
        @Override
        public void jump(byte[] key, long id) {
            // TODO Auto-generated method stub

        }
    }

    private class KCStoreEngine implements StoreEngine{
        private int dbIndex;
        private DB hDb;

        private volatile ConcurrentHashMap<ByteArrayWrapper, byte[]> prevMap;
        private volatile ConcurrentHashMap<ByteArrayWrapper, byte[]> currentMap;
        private volatile ConcurrentHashMap<ByteArrayWrapper, byte[]> offsetMap;
        private volatile boolean isClosing;
        private volatile boolean forceFlush;
        private Thread profileWriter;
        private AtomicInteger flushCounter;
        private final static int PER_OFFSET_FLUSH = 100;
        private long offset  = -1;
        private static final String offsetKey  = "offset_storage_tk";
        private java.util.Random rand = new java.util.Random();
        private static final int maxQueuingCount = 200000;
        private volatile int queuingSize = 0;

        public KCStoreEngine() {
        }

        public void init(int dbIndex) {
            this.dbIndex = dbIndex;

            // use TLINEAR
            hDb = new DB(2);
            currentMap = new ConcurrentHashMap<>(2 * writeBufferSize);
            offsetMap = new ConcurrentHashMap<>();
            flushCounter = new AtomicInteger(0);
            forceFlush = false;
        }

        public void forceFlush(boolean forceFlush) {
            this.forceFlush = forceFlush;
            while(forceFlush){
                try {
                    Thread.sleep(1000L);
                } catch (InterruptedException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }

        /**
         * Open the data base
         */
        public void open() throws Exception {
            int mode;
            if (isReadOnly) {
                mode = DB.OREADER;
            }
            else {
                mode = DB.OWRITER | DB.OCREATE;
            }

            StringBuilder fileNameBuilder = new StringBuilder();
            fileNameBuilder.append(baseDir).append(dbIndex);

            try{
                File baseDirFile = new File(fileNameBuilder.toString());
                if (!baseDirFile.exists()) {
                    logger.info("file not found, create it:{}", fileNameBuilder.toString());
                    baseDirFile.mkdir();
                } else {
                    logger.info("file exists :{}", fileNameBuilder.toString());
                }
            }catch(Exception e) {
                logger.info("error mkdir {}", fileNameBuilder.toString(), e);
            }

            fileNameBuilder.append("/db_").append(dbIndex).append(".kch");
            // specifies the power of the alignment of record size
            fileNameBuilder.append("#apow=").append(recordSizeAlignment);
            // specifies the number of buckets of the hash table
            if(numStores==0){
                fileNameBuilder.append("#bnum=").append(totalRecords * 2 / 8);
            }else{
                fileNameBuilder.append("#bnum=").append(totalRecords * 2 / numStores);
            }
            // specifies the mapped memory size
            if(numStores==0){
                fileNameBuilder.append("#msiz=").append(mappedMemorySize / 8);
            }else{
                fileNameBuilder.append("#msiz=").append(mappedMemorySize / numStores);
            }
            // specifies the unit step number of auto defragmentation
            fileNameBuilder.append("#dfunit=").append(8);

            if (!hDb.open(fileNameBuilder.toString(), mode)){
                throw new Exception(String.format("KC HDB %s open error: " + hDb.error(),
                        fileNameBuilder.toString()));
            }

            profileWriter = new Thread(new Runnable() {

                @Override
                public void run() {
                    while (!isClosing) {
                        flush();

                        try {
                            Thread.sleep(100L);
                        }
                        catch (InterruptedException e) { }
                    }

                    flush();
                }
            });
            profileWriter.setName("ProfileWriter_" + dbIndex);
            profileWriter.setDaemon(true);
            profileWriter.start();
        }

        /**
         * Close all database.
         */
        public void close() {
            isClosing = true;
            try {
                profileWriter.join();
            }
            catch (InterruptedException e) { }

            if (hDb != null) {
                hDb.close();
            }
        }

        /**
         * Store a record.
         *
         * @param key
         * @param value
         */
        public void store(byte[] key, byte[] value) throws Exception {
            if (ArrayUtils.isEmpty(value)) {
                return;
            }

            if (queuingSize > maxQueuingCount || rand.nextInt(100) == 0) {

                while ((queuingSize = currentMap.size()) > maxQueuingCount) {
                    logger.info("back pushing triggered for db indx:{}, size: {}", dbIndex, queuingSize);
                    try {
                        Thread.sleep(1);
                    } catch(Exception e) {
                    }
                }
            }

            currentMap.put(new ByteArrayWrapper(key, key.length), value);
        }

        /**
         * Store a record.
         *
         * @param key
         * @param value
         */
        public void storeOffset(long value) throws Exception {
            if(value >= 0){
                offset = value;
            }
            else{
                throw new Exception("offset can't be negative value");
            }
        }

        /**
         * Get a record.
         *
         * @param key
         * @return
         */
        public byte[] get(byte[] key) {
            byte[] result = currentMap.get(new ByteArrayWrapper(key, key.length));
            if (result == null && prevMap != null) {
                result = prevMap.get(key);
            }

            if (result == null) {
                result = hDb.get(key);
            }

            return result;
        }

        public long getOffset() {
            if (offset == -1) {
                if(hDb == null)
                    throw new IllegalStateException("Unable to get offset hDb is null");
                byte[] result = hDb.get(offsetKey.getBytes());
                if(result == null){
                    offset = ignoreHistoryWhenNoOffsetFoundInStore ? Long.MAX_VALUE: 0;
                    return offset;
                }
                offset = Convert.bytesToLong(result);
            }
            return offset;
        }

        /**
         * Remove a record.
         *
         * @param key
         * @return
         */
        public boolean delete(byte[] key) {
            currentMap.remove(new ByteArrayWrapper(key, key.length));
            if (prevMap != null) {
                prevMap.remove(key);
            }

            return hDb.remove(key);
        }

        /**
         * Get total number of records.
         *
         * @return
         */
        public long count() {
            if (hDb != null) {
                return hDb.count();
            }

            return 0;
        }

        @Override
        public Iterator iterate(long itId) {
            PistachiosTkIterator pit = PistachiosTkIterator.getPistachiosTkIterator(itId);
            if(!pit.isCursorSet()){
                synchronized(pit){
                    if(!pit.isCursorSet()){
                        pit.setCursor(hDb.cursor());
                    }
                }
            }
            return pit;
        }

        public void jump(byte[] key, long itId){
            PistachiosTkIterator pit = PistachiosTkIterator.getPistachiosTkIterator(itId);
            if(!pit.isCursorSet()){
                synchronized(pit){
                    if(!pit.isCursorSet()){
                        pit.setCursor(hDb.cursor());
                    }
                }
            }
            pit.jump(key);
            pit.next();
        }

        /**
         * Go through all the records with the visitor.
         *
         * @param visitor
         */
        public void iterateWithReadLock(Visitor visitor) {
            if (visitor == null) {
                return;
            }

            if (hDb != null) {
                Cursor cursor = hDb.cursor();
                cursor.jump();
                try {
                    byte[][] value = cursor.get(true);
                    while (value!=null && !value[0].equals(offsetKey.getBytes())) {
                        visitor.visit_full(value[0], value[1]);
                        value = cursor.get(true);
                    }
                }
                finally {
                    cursor.disable();
                }
            }
        }

        private void flushOffset(){
            if(offset > 0){
                if (!hDb.set(offsetKey.getBytes(), Convert.longToBytes(offset))) {
                    logger.error("Failed to store offset for {}, Error: {}", offset, hDb.error());
                }else{
                    logger.debug("successfully store offset value {}", offset);
                }
            }
        }
        private void flush() {
            if (currentMap.size() > 0 || isClosing || forceFlush) {

                prevMap = currentMap;
                currentMap = new ConcurrentHashMap<>(2 * writeBufferSize);

                long st = System.currentTimeMillis();
                hDb.begin_transaction(false);
                for (Entry<ByteArrayWrapper, byte[]> item : prevMap.entrySet()) {
                    ByteArrayWrapper key = item.getKey();
                    byte[] value = item.getValue();
                    if (!hDb.set(key.bytes, value)) {
                        logger.error("Failed to store for {}, Error: {}", DefaultDataInterpreter.getDataInterpreter().interpretId(key.bytes), hDb.error());
                    }
                }
                if(numStores == 0 &&(flushCounter.addAndGet(1) % PER_OFFSET_FLUSH == 0 || isClosing || forceFlush))
                    flushOffset();
                hDb.end_transaction(true);
                logger.info("Flushed {} profiles in {}ms", prevMap.size(), System.currentTimeMillis() - st);

                prevMap = null;
                forceFlush = false;
            }
        }

    }
}
