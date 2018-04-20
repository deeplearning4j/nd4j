package org.nd4j.base;

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

    private static void throwEx(String message, Object... args) {
        String f = format(message, args);
        throw new IllegalArgumentException(f);
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
                        sb.append(args[i]);
                        if (i < args.length - 1) {
                            sb.append(",");
                        }
                        i++;
                    }
                    sb.append("]");
                }
            } else {
                sb.append(message.substring(indexOfStart, nextIdx))
                        .append(args[i]);
                indexOfStart = nextIdx + 2;
            }
        }
        if (!consumedMessageFully) {
            sb.append(message.substring(indexOfStart));
        }

        return sb.toString();
    }

}
