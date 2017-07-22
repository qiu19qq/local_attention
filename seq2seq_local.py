from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
# import math

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell_impl._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def local_attention_decodere_nowindow(decoder_inputs,
                            initial_state,
                            attention_states,
                            cell,
                            attn_local_D=3,
                            output_size=None,
                            num_heads=1,
                            loop_function=None,
                            dtype=None,
                            scope=None,
                            initial_state_attention=False):
    """RNN decoder with local attention for sequence-to-sequence model"""
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value
        attention_vec_size = attn_size

        state = initial_state

        # print (len(decoder_inputs))

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a, dtype=dtype):
                    attention_vec_size = attn_size  # Size of query vectors for attention.
                    # to calucate wp * ht
                    v_p = variable_scope.get_variable("AttnV_p%d" % a, [attention_vec_size])
                    qiu = linear(query, attention_vec_size, True)
                    qiu = array_ops.reshape(qiu, [-1, 1, 1, attention_vec_size])
                    tan_v = math_ops.reduce_sum(v_p * math_ops.tanh(qiu),
                                                [2, 3])
                    # print(tan_v.get_shape())
                    pt_sig = math_ops.sigmoid(tan_v)
                    # print(pt_sig.get_shape())
                    p = attn_length * pt_sig
                    # print(p.get_shape())
                    # p_t = (array_ops.reshape(p, [-1, attn_length]))
                    p_t = math_ops.cast(p, dtype=dtypes.int32)
                    p_t = math_ops.cast(p_t, dtype=dtypes.float32)
                    # print(p_t.get_shape())
                    # print(4)

                    # To calculate W1 * hi we use a 1-by-1 convolution, need to reshape before.
                    hidden = array_ops.reshape(attention_states,
                                               [-1, attn_length, 1, attn_size])
                    k = variable_scope.get_variable("AttnW_%d" % a,
                                                    [1, 1, attn_size, attention_vec_size])
                    hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
                    v = variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size])

                with variable_scope.variable_scope("Attention_l_%d" % a, dtype=dtype):
                    # w2 * ht
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y),
                                            [2, 3])
                    ai = nn_ops.softmax(s)
                    ai = tf.reshape(ai, [-1, attn_length, 1])
                    # print(5,ai.get_shape())

                    # do the p_t part
                    extent = tf.ones([1, attn_length], dtype=dtypes.float32)
                    p_t = p_t * extent
                    p_t = tf.reshape(p_t, [-1, attn_length, 1])
                    # print (p_t.get_shape())

                    pos = [i for i in xrange(attn_length)]
                    pos = tf.reshape(pos, [attn_length, 1])
                    pos = math_ops.cast(pos, dtype=dtypes.float32)
                    # print((p_t-pos).get_shape(),"jing")

                    value = math_ops.square(p_t - pos) * 2 / (attn_local_D * attn_local_D)
                    pre = math_ops.exp(math_ops.negative(value))
                    # print(pre.get_shape(),"qiu")
                    ai = ai * pre

                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(ai, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            # print (i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    # print(i)
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state


def local_attention_decoder(decoder_inputs,
                                  initial_state,
                                  attention_states,
                                  cell,
                                  batch_size,
                                  attn_local_D=3,
                                  output_size=None,
                                  num_heads=1,
                                  loop_function=None,
                                  dtype=None,
                                  scope=None,
                                  initial_state_attention=False):
    """RNN decoder with local attention for sequence-to-sequence model"""
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype
        batch = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value
        attention_vec_size = attn_size

        # attn fixed length
        attn_fixed_length = 2 * attn_local_D + 1

        state = initial_state

        # print (len(decoder_inputs))

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a, dtype=dtype):
                    attention_vec_size = attn_size  # Size of query vectors for attention.
                    # to calucate wp * ht
                    v_p = variable_scope.get_variable("AttnV_p%d" % a, [attention_vec_size])
                    qiu = linear(query, attention_vec_size, True)
                    qiu = array_ops.reshape(qiu, [batch_size, 1, 1, attention_vec_size])
                    tan_v = math_ops.reduce_sum(v_p * math_ops.tanh(qiu),
                                                [2, 3])
                    # print(tan_v.get_shape())
                    pt_sig = math_ops.sigmoid(tan_v)
                    # print(pt_sig.get_shape())
                    p = attn_length * pt_sig
                    # print(p.get_shape())
                    # p_t = (array_ops.reshape(p, [-1, attn_length]))
                    p_t = math_ops.cast(p, dtype=dtypes.int32)
                    p_t = math_ops.cast(p_t, dtype=dtypes.float32)
                    # print(p_t.get_shape())
                    # print(4)
                    # p_t=tf.convert_to_tensor(p_t)

                    #print(p_t.shape, attention_states.shape)

                    # set a window
                    p_t=array_ops.reshape(p_t,[batch_size,])
                    attention_states_windows = []
                    D = attn_local_D
                    for i in range(attention_states.shape[0]):
                        x = tf.constant(D, dtype=dtypes.float32)
                        y = math_ops.cast(p_t[i], dtype=dtypes.float32)
                        z = tf.constant(attn_length, dtype=dtypes.float32)

                        def f1(): return tf.constant(0,dtype=dtypes.int32), math_ops.cast(D-p_t[i],dtype=dtypes.int32)
                        def f2():
                            return math_ops.cast(p_t[i] - D,dtype=dtypes.int32),tf.constant(0, dtype=dtypes.int32)
                        def f3(): return tf.constant(attn_length, dtype=dtypes.int32), math_ops.cast(p_t[i] + D + 1-attn_length,dtype=dtypes.int32)
                        def f4(): return math_ops.cast(p_t[i] + D + 1,dtype=dtypes.int32),tf.constant(0, dtype=dtypes.int32)

                        begin, pre_num = tf.cond(tf.less(x, y), f2, f1)
                        end , last_num = tf.cond(tf.less(y + D + 1, z), f4, f3)

                        d = tf.constant(attn_fixed_length, dtype=dtypes.int32)
                        #num = tf.cond(tf.less(end - begin, d), f5, f6)
                        pre_tmp=tf.zeros([pre_num, attention_vec_size], dtype=dtypes.float32)
                        last_tmp = tf.zeros([last_num, attention_vec_size], dtype=dtypes.float32)
                        #tmp = tf.zeros([num, attention_vec_size], dtype=dtypes.float32)
                        attention_states_window = math_ops.cast(attention_states[i][begin:end], dtype=dtypes.float32)
                        attention_states_window = tf.concat([pre_tmp,attention_states_window], 0)
                        attention_states_window = tf.concat([attention_states_window, last_tmp], 0)
                        attention_states_window = tf.expand_dims(attention_states_window, 0)
                        attention_states_windows.append(attention_states_window)

                    attention_states_windows = tf.concat(attention_states_windows, 0)
                    attention_states_windows = array_ops.reshape(attention_states_windows,
                                                                 [batch_size, attn_fixed_length, attention_vec_size])
                    # print(attention_states_windows.shape)

                    # To calculate W1 * hi we use a 1-by-1 convolution, need to reshape before.
                    hidden = array_ops.reshape(attention_states_windows,
                                               [batch_size, attn_fixed_length, 1, attn_size])
                    k = variable_scope.get_variable("AttnW_%d" % a,
                                                    [1, 1, attn_size, attention_vec_size])
                    hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
                    v = variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size])

                with variable_scope.variable_scope("Attention_l_%d" % a, dtype=dtype):
                    # w2 * ht
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [batch_size, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y),
                                            [2, 3])
                    ai = nn_ops.softmax(s)
                    ai = tf.reshape(ai, [batch_size, attn_fixed_length, 1])
                    # print(5,ai.get_shape())

                    # do the p_t part
                    p_t=array_ops.reshape(p_t,[batch_size,1])
                    extent = tf.ones([1, attn_fixed_length], dtype=dtypes.float32)
                    p_t = p_t * extent
                    p_t = tf.reshape(p_t, [batch_size, attn_fixed_length, 1])
                    # print (p_t.get_shape())

                    pos = [i for i in xrange(attn_fixed_length)]
                    pos = tf.reshape(pos, [attn_fixed_length, 1])
                    pos = math_ops.cast(pos, dtype=dtypes.float32)
                    # print((p_t-pos).get_shape(),"jing")

                    value = math_ops.square(p_t - pos) * 2 / (D * D)
                    pre = math_ops.exp(math_ops.negative(value))
                    # print(pre.get_shape(),"qiu")
                    ai = ai * pre

                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(ai, [batch_size, attn_fixed_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [batch_size, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            # print (i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    # print(i)
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        # print (hidden.get_shape())

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    # w2 * ht
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                batch_size,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(
                    scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = variable_scope.get_variable("embedding",
                                                [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
        ]
        return local_attention_decoder(
            emb_inp,
            initial_state,
            attention_states,
            cell,
            batch_size,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                enc_cell,
                                dec_cell,
                                batch_size,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
    with variable_scope.variable_scope(
                    scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.

        encoder_cell = enc_cell

        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            encoder_cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.static_rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [
            array_ops.reshape(e, [batch_size, 1, encoder_cell.output_size]) for e in encoder_outputs
        ]
        attention_states = array_ops.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            dec_cell = core_rnn_cell.OutputProjectionWrapper(dec_cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                dec_cell,
                batch_size,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=reuse):
                outputs, state = embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    dec_cell,
                    batch_size,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=initial_state_attention)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(
                structure=encoder_state, flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state
