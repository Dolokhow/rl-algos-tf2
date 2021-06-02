import numpy as np
import tensorflow as tf

# Shape modifiers


def shape_expand_axis(shape, axis, size):
    if isinstance(shape, tuple):
        shape_lst = list(shape)
    else:
        shape_lst = shape

    shape_lst[axis] += size
    return tuple(shape_lst)


def shape_expand_dim(shape, axis):
    if isinstance(shape, int):
        shape_lst = [shape]
    # tuple or list
    else:
        shape_lst = list(shape)
    shape_lst.insert(axis, 1)

    if isinstance(shape, tuple):
        return tuple(shape_lst)
    else:
        return shape_lst


# Vector operations: Only meant for vectors up to two dimensions!


# Broadcasts 1d Tensorflow tensor or numpy array with shapes either (x, ) or (1, x)
def broadcast_1d_row_vector(t, axis, new_dim):

    if isinstance(new_dim, int):
        new_dim = (new_dim,)

    if len(t.shape) > 1:
        col_dim = (t.shape[1], )
    else:
        col_dim = t.shape

    if tf.is_tensor(t):
        broadcast_func = tf.broadcast_to
        exp_dim_func = tf.expand_dims
        transpose_func = tf.transpose
    else:
        broadcast_func = np.broadcast_to
        exp_dim_func = np.expand_dims
        transpose_func = np.transpose

    if axis == 0:
        broadcast_shape = new_dim + col_dim
    else:
        broadcast_shape = col_dim + new_dim
        if len(t.shape) == 1:
            t = exp_dim_func(t, axis=axis)
        else:
            t = transpose_func(t)

    t = broadcast_func(t, shape=broadcast_shape)
    return t


# Expensive operation, avoid if possible. To be used when passing a python list of different vector types
# (Python lists vs. TensorFlow Tensors vs. NumPy arrays) to functions that do some operations with these vectors.
# pref_dtype specifies the desired output vector type if there is a mismatch between the passed types. If all passed
# vectors are of the same type, pref_vtype will be ignored.
def homogenize_vector_types(ts, pref_vtype='np'):

    if pref_vtype is None:
        return ts, None

    total = len(ts)
    tensors = 0
    arrays = 0

    tensor_dtype = None
    np_array_dtype = None

    for t in ts:
        if pref_vtype == 'np':
            if not isinstance(t, list) and not tf.is_tensor(t):
                np_array_dtype = t.dtype
                break
        elif pref_vtype == 'tf':
            if tf.is_tensor(t):
                tensor_dtype = t.dtype
                break

    for index in range(total):
        t = ts[index]

        if tf.is_tensor(t):
            tensors += 1
        elif isinstance(t, list):
            if pref_vtype == 'np':
                t = np.array(t, dtype=np_array_dtype)
                arrays += 1
            else:
                t = tf.convert_to_tensor(t, dtype=tensor_dtype)
                tensors += 1
            ts[index] = t
        else:
            arrays += 1

    if total == tensors:
        new_ts = ts
        dtype = 'tf'
    elif total == arrays:
        new_ts = ts
        dtype = 'np'
    else:
        if pref_vtype == 'np':
            new_ts = [np.array(t, dtype=np_array_dtype) for t in ts]
            dtype = 'np'
        else:
            new_ts = [tf.convert_to_tensor(t, dtype=tensor_dtype) for t in ts]
            dtype = 'tf'
    return new_ts, dtype


# Merges two vectors along the specified axis. If two vectors are of different types, pref_vtype will specify the type
# of the output. It will be ignored otherwise. If set to None, t2 vector type (np.array or tf.tensor) will be converted
# into t1 type and output vector will follow t1 type. Setting pref_vtype to None avoids the homogenize_vector_types call
def force_merge_vectors(t1, t2, axis, pref_vtype=None):
    if pref_vtype is not None:
        ts, dtype = homogenize_vector_types(ts=[t1, t2], pref_vtype=pref_vtype)
        t1 = ts[0]
        t2 = ts[1]
    else:
        if tf.is_tensor(t1):
            t2 = tf.convert_to_tensor(t2, dtype=t1.dtype)
            dtype = 'tf'
        elif isinstance(t1, list):
            if tf.is_tensor(t2):
                t1 = tf.convert_to_tensor(t1, dtype=t2.dtype)
                dtype = 'tf'
            elif isinstance(t2, list):
                raise TypeError('Both t1 and t2 are Python lists. Either specify pref_dtype to either TensorFlow '
                                'tensor, or NumPy array, or pass at least one non Python list argument.')
            else:
                t1 = np.array(t1, dtype=t2.dtype)
                dtype = 'np'
        else:
            t2 = np.array(t2, dtype=t1.dtype)
            dtype = 'np'

    # In case both vectors are (x, )
    if len(t1.shape) == len(t2.shape) == 1:
        axis = 0
    else:
        axis = axis

    if dtype == 'tf':
        t = tf.concat([t1, t2], axis=axis)
    else:
        t = np.concatenate((t1, t2), axis=axis)
    return t


def merge_vectors(t1, t2, axis):
    # In case both vectors are (x, )
    if len(t1.shape) == len(t2.shape) == 1:
        axis = 0
    else:
        axis = axis

    if tf.is_tensor(t1):
        if isinstance(t2, list):
            t2 = tf.convert_to_tensor(t2, dtype=t1.dtype)
        elif tf.is_tensor(t2):
            pass
        else:
            raise TypeError('Vector conversions from NumPy ndarrays to TensorFlow tensors may produce unstable '
                            'results. If you do not care, use force_merge_vectors with the same arguments instead.')
        t = tf.concat([t1, t2], axis=axis)
    elif isinstance(t1, list):
        if isinstance(t2, list):
            raise TypeError('Both t1 and t2 are Python lists. At least one argument needs to be either TensorFlow '
                            'tensor or NumPy ndarray.')
        elif tf.is_tensor(t2):
            t1 = tf.convert_to_tensor(t1, dtype=t2.dtype)
            t = tf.concat([t1, t2], axis=axis)
        else:
            t1 = np.array(t1, dtype=t2.dtype)
            t = np.concatenate((t1, t2), axis=axis)
    else:
        if isinstance(t2, list):
            t2 = np.array(t2, dtype=t1.dtype)
        elif tf.is_tensor(t2):
            raise TypeError('Vector conversions from TensorFlow tensors to NumPy ndarrays may produce unstable '
                            'results. If you do not care, use force_merge_vectors with the same arguments instead.')
        else:
            pass
        t = np.concatenate((t1, t2), axis=axis)
    return t


# Splits TensorFlow tensor or NumPy array into two parts based on index, along the specified axis.
def split_vector(t, index, axis):
    if axis == 1:
        t1 = t[:, :index]
        t2 = t[:, index:]
    else:
        t1 = t[:index, :]
        t2 = t[index:, :]

    return t1, t2


def main():
    # np_batch_obs = np.ones((2, 4))
    # np_opts = np.zeros((1, 3))
    # tf_batch_obs = tf.ones((2, 4))
    # tf_opts = tf.zeros((1, 3))
    #
    # py_batch_obs = [[1., 1., 1., 1.], [1., 1., 1., 1.]]
    # py_batch_opts = [[0., 0.], [0., 0.]]
    #
    # # Test broadcast_1d_row_vector use cases
    # # axis=0
    # np_opts_broadcast = broadcast_1d_row_vector(t=np_opts, axis=0, new_dim=np_batch_obs.shape[0])
    # print(np_opts_broadcast)
    # tf_opts_broadcast = broadcast_1d_row_vector(t=tf_opts, axis=0, new_dim=np_batch_obs.shape[0])
    # print(tf_opts_broadcast)
    # print()
    # # axis=1
    # np_opts_broadcast = broadcast_1d_row_vector(t=np_opts, axis=0, new_dim=np_batch_obs.shape[0])
    # print(np_opts_broadcast)
    # tf_opts_broadcast = broadcast_1d_row_vector(t=tf_opts, axis=0, new_dim=np_batch_obs.shape[0])
    # print(tf_opts_broadcast)
    # print()
    # print()
    #
    # # Test merge_vectors use cases
    # # Vectors of the same type
    # # np vectors
    # np_modified_obs = force_merge_vectors(t1=np_batch_obs, t2=np_opts_broadcast, axis=1, pref_vtype=None)
    # print(np_modified_obs)
    # # tf vectors
    # tf_modified_obs = force_merge_vectors(t1=tf_batch_obs, t2=tf_opts_broadcast, axis=1, pref_vtype=None)
    # print(tf_modified_obs)
    # print()
    # # Vectors of different types
    # # No type preference
    # # t1 is np.array
    # np_forced_obs = force_merge_vectors(t1=np_batch_obs, t2=tf_opts_broadcast, axis=1, pref_vtype=None)
    # print(np_forced_obs)
    # # t1 is tf.tensor
    # tf_forced_obs = force_merge_vectors(t1=tf_batch_obs, t2=np_opts_broadcast, axis=1, pref_vtype=None)
    # print(tf_forced_obs)
    # print()
    # # Type is preferred
    # # np is preferred
    # np_pref_obs = force_merge_vectors(t1=tf_batch_obs, t2=np_opts_broadcast, axis=1, pref_vtype='np')
    # print(np_pref_obs)
    # # tf is preferred
    # tf_pref_obs = force_merge_vectors(t1=np_batch_obs, t2=tf_opts_broadcast, axis=1, pref_vtype='tf')
    # print(tf_pref_obs)
    # print()
    # print()
    #
    # # Test splitting op
    # # Positive index
    # # tf
    # tf_split_obs, tf_spit_opts = split_vector(t=tf_pref_obs, index=4, axis=1)
    # print(tf_split_obs)
    # print(tf_spit_opts)
    # print()
    # # np
    # np_split_obs, np_spit_opts = split_vector(t=np_pref_obs, index=4, axis=1)
    # print(np_split_obs)
    # print(np_spit_opts)
    # print()
    # # Negative index
    # # tf
    # tf_split_obs, tf_spit_opts = split_vector(t=tf_pref_obs, index=-3, axis=1)
    # print(tf_split_obs)
    # print(tf_spit_opts)
    # # np
    # np_split_obs, np_spit_opts = split_vector(t=np_pref_obs, index=-3, axis=1)
    # print(np_split_obs)
    # print(np_spit_opts)

    arr = np.array([1., 2, 3, 4, 1., 2, 3])
    true_cond = list(np.where(arr == 1))
    print(true_cond)


if __name__ == '__main__':
    main()
