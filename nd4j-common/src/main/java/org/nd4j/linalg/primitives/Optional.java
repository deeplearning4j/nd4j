package org.nd4j.linalg.primitives;

import lombok.NonNull;

import java.util.NoSuchElementException;

/**
 * Simple Optional class, based loosely on Java 8's optional class
 *
 * @param <T> Type for optional
 * @author Alex Black
 */
public class Optional<T> {
    private static final Optional EMPTY = new Optional();

    private final T value;

    private Optional(){
        this(null);
    }

    private Optional(T value){
        this.value = value;
    }

    public static <T> Optional<T> empty(){
        return (Optional<T>)EMPTY;
    }

    public static <T> Optional<T> of(@NonNull T value){
        return new Optional<>(value);
    }

    public static <T> Optional<T> ofNullable(T value){
        if(value == null){
            return empty();
        }
        return new Optional<>(value);
    }

    public T get(){
        if (!isPresent()) {
            throw new NoSuchElementException("Optional is empty");
        }
        return value;
    }

    public boolean isPresent(){
        return value != null;
    }

    public T orElse(T value){
        if(isPresent()){
            return get();
        }
        return value;
    }

    public String toString(){
        if(isPresent()){
            return "Optional(" + value.toString() + ")";
        }
        return "Optional()";
    }
}
