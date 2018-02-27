// automatically generated by the FlatBuffers compiler, do not modify

package org.nd4j.graph;

import java.nio.*;
import java.lang.*;
import java.nio.ByteOrder;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class FlatConfiguration extends Table {
  public static FlatConfiguration getRootAsFlatConfiguration(ByteBuffer _bb) { return getRootAsFlatConfiguration(_bb, new FlatConfiguration()); }
  public static FlatConfiguration getRootAsFlatConfiguration(ByteBuffer _bb, FlatConfiguration obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; }
  public FlatConfiguration __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public long id() { int o = __offset(4); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public long footprintForward() { int o = __offset(6); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public long footprintBackward() { int o = __offset(8); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public byte executionMode() { int o = __offset(10); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public byte profilingMode() { int o = __offset(12); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public byte outputMode() { int o = __offset(14); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public boolean timestats() { int o = __offset(16); return o != 0 ? 0!=bb.get(o + bb_pos) : false; }

  public static int createFlatConfiguration(FlatBufferBuilder builder,
      long id,
      long footprintForward,
      long footprintBackward,
      byte executionMode,
      byte profilingMode,
      byte outputMode,
      boolean timestats) {
    builder.startObject(7);
    FlatConfiguration.addFootprintBackward(builder, footprintBackward);
    FlatConfiguration.addFootprintForward(builder, footprintForward);
    FlatConfiguration.addId(builder, id);
    FlatConfiguration.addTimestats(builder, timestats);
    FlatConfiguration.addOutputMode(builder, outputMode);
    FlatConfiguration.addProfilingMode(builder, profilingMode);
    FlatConfiguration.addExecutionMode(builder, executionMode);
    return FlatConfiguration.endFlatConfiguration(builder);
  }

  public static void startFlatConfiguration(FlatBufferBuilder builder) { builder.startObject(7); }
  public static void addId(FlatBufferBuilder builder, long id) { builder.addLong(0, id, 0L); }
  public static void addFootprintForward(FlatBufferBuilder builder, long footprintForward) { builder.addLong(1, footprintForward, 0L); }
  public static void addFootprintBackward(FlatBufferBuilder builder, long footprintBackward) { builder.addLong(2, footprintBackward, 0L); }
  public static void addExecutionMode(FlatBufferBuilder builder, byte executionMode) { builder.addByte(3, executionMode, 0); }
  public static void addProfilingMode(FlatBufferBuilder builder, byte profilingMode) { builder.addByte(4, profilingMode, 0); }
  public static void addOutputMode(FlatBufferBuilder builder, byte outputMode) { builder.addByte(5, outputMode, 0); }
  public static void addTimestats(FlatBufferBuilder builder, boolean timestats) { builder.addBoolean(6, timestats, false); }
  public static int endFlatConfiguration(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
  public static void finishFlatConfigurationBuffer(FlatBufferBuilder builder, int offset) { builder.finish(offset); }
}

