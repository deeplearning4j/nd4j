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
package org.nd4j.paramserver.pistachio.pb.util;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;


/**
 * Utility to convert data types.
 *
 *
 */
public class Convert {
    /**
     * Performs conversion of a long value to a byte array representation.
     *
     * @see #bytesToLong(byte[])
     */
    public static byte[] longToBytes(long value) {

        // A long value is 8 bytes in length.
        byte[] bytes = new byte[8];

        // Convert and copy value to byte array:
        // -- Cast long to a byte to retrieve least significant byte;
        // -- Left shift long value by 8 bits to isolate next byte to be converted;
        // -- Repeat until all 8 bytes are converted (long = 64 bits).
        // Note: In the byte array, the least significant byte of the long is held in
        // the highest indexed array bucket.

        for (int i = 0; i < bytes.length; i++) {
            bytes[(bytes.length - 1) - i] = (byte) value;
            value >>>= 8;
        }

        return bytes;
    }

    /**
     * Performs conversion of a byte array to a long representation.
     *
     * @see #longToBytes(long)
     */
    public static long bytesToLong(byte[] value) {

        long longValue = 0L;

        // See method convertLongToBytes(long) for algorithm details.
        if (ArrayUtils.isNotEmpty(value)) {
            for (int i = 0; i < value.length; i++) {
                // Left shift has no effect thru first iteration of loop.
                longValue <<= 8;
                longValue ^= value[i] & 0xFF;
            }
        }

        return longValue;
    }

    /**
     * Convert an integer to byte array.
     *
     * @param value
     * @return
     */
    public static byte[] intToBytes(int value) {
        return new byte[] {
                (byte)(value >>> 24),
                (byte)(value >>> 16),
                (byte)(value >>> 8),
                (byte)value};
    }

    /**
     * Convert a byte array to integer.
     *
     * @param value
     * @return
     */
    public static int bytesToInt(byte[] value) {
        int intValue = 0;

        for (int i = 0; i < value.length; i++) {
            // Left shift has no effect thru first iteration of loop.
            intValue <<= 8;
            intValue ^= value[i] & 0xFF;
        }

        return intValue;
    }

    /**
     * Convert string ip address to long.
     *
     * @param address
     * @return
     */
    public static long convertToLongFromStrIP(String address) {
        if (address==null) return 0;

        // convert string ip to a byte array
        String[] parts = address.split("\\.");
        if (parts == null || parts.length != 4) {
            return 0;
        }

        byte[] addr = new byte[4];
        for (int i = 0; i < 4; i++) {
            addr[i] = Integer.valueOf(parts[i]).byteValue();
        }

        // convert byte array to a long
        return convertToLongFromIPBytes(addr);
    }

    /**
     * Convert IP address byte array to long.
     *
     * @param ipBytes
     * @return
     */
    public static long convertToLongFromIPBytes(byte[] ipBytes) {
        // convert byte array from ip address to a long
        long ipnum = 0L;
        for (int i = 0; i < 4; i++) {
            long y = ipBytes[i];
            if (y < 0L)
                y += 256L;
            ipnum += y << (3 - i) * 8;
        }
        return ipnum;
    }

    /**
     * Convert IP address byte array to long.
     *
     * @param ipBytes
     * @return
     */
    public static long convertToLongFromIPNBytes(byte[] ipBytes) {
        // convert byte array from ip address to a long
        int nBytes = ipBytes.length < 4 ? ipBytes.length : 4;
        long ipnum = 0L;
        for (int i = 0; i < nBytes; i++) {
            long y = ipBytes[i];
            if (y < 0L)
                y += 256L;
            ipnum += y << (3 - i) * 8;
        }
        return ipnum;
    }


    /**
     * Convert IP address byte array to string "x.x.x.x"
     *
     * @param ipBytes
     * @return
     */
    public static String convertToStringFromIPNBytes(byte[] ipBytes) {
        // convert byte array to ip address string
        Integer [] ip = new Integer[ipBytes.length];
        for (int i = 0; i < ipBytes.length; i++) {
            int y = ipBytes[i];
            if (y < 0) y += 256;
            ip[i] = y;
        }
        return StringUtils.join(ip, ".");
    }


    /**
     * Convert String ip address to integer.
     * @param address
     * @return
     */
    public static int convertToIntFromStrIP(String address)    {
        return (int) convertToLongFromStrIP(address);
    }

    /**
     * Convert long ip address to integer.
     *
     * @param ip
     * @return
     */
    public static int convertIPLongToIPInt(long ip)    {
        return (int) ip;
    }

    /**
     * Convert integer ip address to long.
     *
     * @param ip
     * @return
     */
    @SuppressWarnings("cast")
    public static long convertIPIntToIPLong(int ip) {
        if (ip>0) return ip;
        // handle overflow case
        // the casts are very important
        return 2 + (long) Integer.MAX_VALUE + (long) Integer.MAX_VALUE + (long) ip;
    }

    /**
     * Convert long ip address to string.
     *
     * @param ip
     * @return
     */
    public static String convertToStrIPFromLong(long ip) {
        short subD = (short) (ip & 0xFF);
        short subC = (short) ((ip & 0xFF00) >> 8);
        short subB = (short) ((ip & 0xFF0000) >> 16);
        short subA = (short) ((ip & 0xFF000000) >> 24);
        String IP = subA + "." + subB + "." + subC + "." + subD;
        return IP;
    }

    /**
     * Convert integer ip address to string.
     *
     * @param ip
     * @return
     */
    public static String convertToStrIPFromInt(int ip) {
        return convertToStrIPFromLong(convertIPIntToIPLong(ip));
    }

    /**
     * Convert integer list to array.
     *
     * @param integerList
     * @isSorted True to sort the converted array
     * @return
     */
    public static int[] toPrimitive(List<Integer> integerList, boolean isSorted) {
        if (integerList == null || integerList.isEmpty()) {
            return null;
        }

        int[] result = new int[integerList.size()];
        int i = 0;
        for (Integer value : integerList) {
            result[i++] = value.intValue();
        }

        if (isSorted) {
            Arrays.sort(result);
        }

        return result;
    }

}
