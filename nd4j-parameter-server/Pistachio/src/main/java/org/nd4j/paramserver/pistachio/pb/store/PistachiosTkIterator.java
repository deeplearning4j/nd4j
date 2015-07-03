package org.nd4j.paramserver.pistachio.pb.store;

import java.util.Iterator;
import java.util.Map;
import java.util.WeakHashMap;

import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.ByteBufferOutput;
import org.nd4j.paramserver.pistachio.pb.kafka.KeyValue;

import kyotocabinet.Cursor;

public class PistachiosTkIterator implements Iterator{
    private static org.slf4j.Logger logger = LoggerFactory.getLogger(PistachiosTkIterator.class);
    public volatile static Map<Long,PistachiosTkIterator> iteratorMap = new WeakHashMap<Long,PistachiosTkIterator>();
    private Cursor cursor;
    private Kryo kryo = new Kryo();
    private static final ThreadLocal<ByteBufferOutput> threadByteBuffer =
      new ThreadLocal<ByteBufferOutput>() {
          @Override protected ByteBufferOutput initialValue() {
              return new ByteBufferOutput(10240);
          }
      };
    public static PistachiosTkIterator getPistachiosTkIterator(long id){
        if(iteratorMap.get(id) == null){
            synchronized(iteratorMap){
                if(iteratorMap.get(id) == null){
                    iteratorMap.put(id, new PistachiosTkIterator());
                }
            }
        }
        return iteratorMap.get(id);
    }

    public boolean isCursorSet() {
        return cursor == null ? false : true;
    }
    public void setCursor(Cursor cursor) {
        this.cursor = cursor;
        cursor.jump();
    }

    @Override
    public boolean hasNext() {
            return true;
    }

    @Override
    public Object next() {
            byte[] key =  cursor.get_key(false);
            if(key == null){
                return null;
            }

            byte[] value = cursor.get_value(true);
            KeyValue keyValue = new KeyValue();
            keyValue.key = key;
            keyValue.value = value;
            threadByteBuffer.get().clear();
            kryo.writeObject(threadByteBuffer.get(), keyValue);
            return threadByteBuffer.get().toBytes();
    }

    @Override
    public void remove() {
        cursor.remove();
    }

    public void jump(byte[] key){
        cursor.jump(key);
    }
}
