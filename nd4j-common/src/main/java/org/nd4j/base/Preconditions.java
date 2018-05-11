package org.nd4j.base;

import java.util.Arrays;

/**
 * Utility method for method checking arguments.
 *
 * @author Alex Black
 */
public class Preconditions {

    private Preconditions(){ }

    /**
     * Check the specified boolean argument. Throws an IllegalArgumentException if {@code b} is false
     *
     * @param b Argument to check
     */
    public static void checkArgument(boolean b) {
        if (!b) {
            throw new IllegalArgumentException();
        }
    }

    /**
     * Check the specified boolean argument. Throws an IllegalArgumentException with the specified message if {@code b} is false
     *
     * @param b       Argument to check
     * @param message Message for exception. May be null
     */
    public static void checkArgument(boolean b, String message) {
        if (!b) {
            throwEx(message);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, int arg1) {
        if (!b) {
            throwEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, long arg1) {
        if (!b) {
            throwEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, double arg1) {
        if (!b) {
            throwEx(msg, arg1);
        }
    }


    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, Object arg1) {
        if (!b) {
            throwEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, int arg1, int arg2) {
        if (!b) {
            throwEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, long arg1, long arg2) {
        if (!b) {
            throwEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, double arg1, double arg2) {
        if (!b) {
            throwEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, Object arg1, Object arg2) {
        if (!b) {
            throwEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, int arg1, int arg2, int arg3) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, long arg1, long arg2, long arg3) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, double arg1, double arg2, double arg3) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, Object arg1, Object arg2, Object arg3) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, int arg1, int arg2, int arg3, int arg4) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, long arg1, long arg2, long arg3, long arg4) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3, arg4);
        }
    }


    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, double arg1, double arg2, double arg3, double arg4) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, Object arg1, Object arg2, Object arg3, Object arg4) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, Object arg1, Object arg2, Object arg3, Object arg4, Object arg5) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3, arg4, arg5);
        }
    }

    /**
     * See {@link #checkArgument(boolean, String, Object...)}
     */
    public static void checkArgument(boolean b, String msg, Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6) {
        if (!b) {
            throwEx(msg, arg1, arg2, arg3, arg4, arg5, arg6);
        }
    }

    /**
     * Check the specified boolean argument. Throws an IllegalArgumentException with the specified message if {@code b} is false.
     * Note that the message may specify argument locations using "%s" - for example,
     * {@code checkArgument(false, "Got %s values, expected %s", 3, "more"} would throw an IllegalArgumentException
     * with the message "Got 3 values, expected more"
     *
     * @param b       Argument to check
     * @param message Message for exception. May be null.
     * @param args    Arguments to place in message
     */
    public static void checkArgument(boolean b, String message, Object... args) {
        if (!b) {
            throwEx(message, args);
        }
    }


    /**
     * Check the specified boolean argument. Throws an IllegalStateException if {@code b} is false
     *
     * @param b State to check
     */
    public static void checkState(boolean b) {
        if (!b) {
            throw new IllegalStateException();
        }
    }

    /**
     * Check the specified boolean argument. Throws an IllegalStateException with the specified message if {@code b} is false
     *
     * @param b       State to check
     * @param message Message for exception. May be null
     */
    public static void checkState(boolean b, String message) {
        if (!b) {
            throwStateEx(message);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, int arg1) {
        if (!b) {
            throwStateEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, long arg1) {
        if (!b) {
            throwStateEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, double arg1) {
        if (!b) {
            throwStateEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, Object arg1) {
        if (!b) {
            throwStateEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, int arg1, int arg2) {
        if (!b) {
            throwStateEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, long arg1, long arg2) {
        if (!b) {
            throwStateEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, double arg1, double arg2) {
        if (!b) {
            throwStateEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, Object arg1, Object arg2) {
        if (!b) {
            throwStateEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, int arg1, int arg2, int arg3) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, long arg1, long arg2, long arg3) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, double arg1, double arg2, double arg3) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, Object arg1, Object arg2, Object arg3) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, int arg1, int arg2, int arg3, int arg4) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, long arg1, long arg2, long arg3, long arg4) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, double arg1, double arg2, double arg3, double arg4) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, Object arg1, Object arg2, Object arg3, Object arg4) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, Object arg1, Object arg2, Object arg3, Object arg4, Object arg5) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3, arg4, arg5);
        }
    }

    /**
     * See {@link #checkState(boolean, String, Object...)}
     */
    public static void checkState(boolean b, String msg, Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6) {
        if (!b) {
            throwStateEx(msg, arg1, arg2, arg3, arg4, arg5, arg6);
        }
    }

    /**
     * Check the specified boolean argument. Throws an IllegalStateException with the specified message if {@code b} is false.
     * Note that the message may specify argument locations using "%s" - for example,
     * {@code checkArgument(false, "Got %s values, expected %s", 3, "more"} would throw an IllegalStateException
     * with the message "Got 3 values, expected more"
     *
     * @param b       Argument to check
     * @param message Message for exception. May be null.
     * @param args    Arguments to place in message
     */
    public static void checkState(boolean b, String message, Object... args) {
        if (!b) {
            throwStateEx(message, args);
        }
    }


    /**
     * Check the specified boolean argument. Throws an NullPointerException if {@code o} is false
     *
     * @param o Object to check
     */
    public static void checkNotNull(Object o) {
        if (o == null) {
            throw new NullPointerException();
        }
    }

    /**
     * Check the specified boolean argument. Throws an NullPointerException with the specified message if {@code o} is false
     *
     * @param o       Object to check
     * @param message Message for exception. May be null
     */
    public static void checkNotNull(Object o, String message) {
        if (o == null) {
            throwNullPointerEx(message);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, int arg1) {
        if (o == null) {
            throwNullPointerEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, long arg1) {
        if (o == null) {
            throwNullPointerEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, double arg1) {
        if (o == null) {
            throwNullPointerEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, Object arg1) {
        if (o == null) {
            throwNullPointerEx(msg, arg1);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, int arg1, int arg2) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, long arg1, long arg2) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, double arg1, double arg2) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, Object arg1, Object arg2) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, int arg1, int arg2, int arg3) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, long arg1, long arg2, long arg3) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, double arg1, double arg2, double arg3) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, Object arg1, Object arg2, Object arg3) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, int arg1, int arg2, int arg3, int arg4) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, long arg1, long arg2, long arg3, long arg4) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, double arg1, double arg2, double arg3, double arg4) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * See {@link #checkNotNull(Object, String, Object...)}
     */
    public static void checkNotNull(Object o, String msg, Object arg1, Object arg2, Object arg3, Object arg4) {
        if (o == null) {
            throwNullPointerEx(msg, arg1, arg2, arg3, arg4);
        }
    }

    /**
     * Check the specified boolean argument. Throws an IllegalStateException with the specified message if {@code o} is false.
     * Note that the message may specify argument locations using "%s" - for example,
     * {@code checkArgument(false, "Got %s values, expected %s", 3, "more"} would throw an IllegalStateException
     * with the message "Got 3 values, expected more"
     *
     * @param o       Object to check
     * @param message Message for exception. May be null.
     * @param args    Arguments to place in message
     */
    public static void checkNotNull(Object o, String message, Object... args) {
        if (o == null) {
            throwStateEx(message, args);
        }
    }

    private static void throwEx(String message, Object... args) {
        String f = format(message, args);
        throw new IllegalArgumentException(f);
    }

    private static void throwStateEx(String message, Object... args) {
        String f = format(message, args);
        throw new IllegalStateException(f);
    }

    private static void throwNullPointerEx(String message, Object... args) {
        String f = format(message, args);
        throw new NullPointerException(f);
    }

    private static String format(String message, Object... args) {
        if (message == null) {
            message = "";
        }
        if (args == null) {
            args = new Object[]{"null"};
        }

        StringBuilder sb = new StringBuilder();

        int indexOfStart = 0;
        boolean consumedMessageFully = false;
        for (int i = 0; i < args.length; i++) {
            int nextIdx = message.indexOf("%s", indexOfStart);
            if (nextIdx < 0) {
                //Malformed message: No more "%s" to replace, but more message args
                if (!consumedMessageFully) {
                    sb.append(message.substring(indexOfStart));
                    consumedMessageFully = true;
                    sb.append(" [");
                    while (i < args.length) {
                        sb.append(formatArg(args[i]));
                        if (i < args.length - 1) {
                            sb.append(",");
                        }
                        i++;
                    }
                    sb.append("]");
                }
            } else {
                sb.append(message.substring(indexOfStart, nextIdx))
                        .append(formatArg(args[i]));
                indexOfStart = nextIdx + 2;
            }
        }
        if (!consumedMessageFully) {
            sb.append(message.substring(indexOfStart));
        }

        return sb.toString();
    }

    private static String formatArg(Object o){
        if(o == null){
            return "null";
        }
        if(o.getClass().isArray()){
            return formatArray(o);
        }
        return o.toString();
    }

    private static String formatArray(Object o){
        if(o == null)
            return "null";

        if(o.getClass().getComponentType().isPrimitive()){
            if(o instanceof byte[]) {
                return Arrays.toString((byte[])o);
            } else if(o instanceof int[]){
                return Arrays.toString((int[])o);
            } else if(o instanceof long[]){
                return Arrays.toString((long[])o);
            } else if(o instanceof float[]){
                return Arrays.toString((float[])o);
            } else if(o instanceof double[]){
                return Arrays.toString((double[])o);
            } else if(o instanceof char[]){
                return Arrays.toString((char[])o);
            } else if(o instanceof boolean[]) {
                return Arrays.toString((boolean[])o);
            } else if(o instanceof short[]){
                return Arrays.toString((short[])o);
            } else {
                //Should never happen
                return o.toString();
            }
        } else {
            Object[] arr = (Object[])o;
            return Arrays.toString(arr);
        }
    }

}
