# -*- coding: utf-8 -*-


import numpy as np
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

from keras.layers.recurrent import _generate_dropout_mask
from keras.utils import conv_utils

from keras.layers import ConvRNN2D
from keras.engine import Layer

from solarpv.layers.chrono_initializer import ChronoInitializer

class ConvJANetCell(Layer):
    """
    Cell class based on the paper ["The unreasonable effectiveness of the forget gate"]
    for the ConvJANet layer.
    
    
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
    """
    
    def __init__(self, filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 beta=1.,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_chrono_initialization=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kargs):
        super(ConvJANetCell, self).__init__(**kargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.beta = beta
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_chrono_initialization = use_chrono_initialization

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
    def build(self, input_shape):
        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
            
        input_dim = input_shape[channel_axis]
        
        if hasattr(self, 'timesteps') and self.timesteps is not None:
            self.max_timesteps = self.timesteps
        else:
            warnings.warn('Maximum number of timesteps must be defined for this model for proper initialization.\n'
                          'Since the maximum timesteps was not defined, defaulting to a very large number of timesteps '
                          '(1000). Note : This may cause irregular training behaviour.')
            self.max_timesteps = 1000
        
        
        kernel_shape = self.kernel_size + (input_dim, self.filters * 2)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 2)
        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        
        
        if self.use_bias:
            if self.use_chrono_initialization:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        ChronoInitializer(self.max_timesteps)((self.filters,), *args, **kwargs),  
                        self.bias_initializer((self.filters,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.filters * 2,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.kernel_f = self.kernel[:, :, :, :self.filters]
        self.kernel_c = self.kernel[:, :, :, self.filters: self.filters * 2]
        
        self.recurrent_kernel_f = (
                self.recurrent_kernel[:, :, :, :self.filters])
        self.recurrent_kernel_c = (
                self.recurrent_kernel[:, :, :, self.filters: self.filters * 2])
        
        if self.use_bias:
            self.bias_f = self.bias[:self.filters]
            self.bias_c = self.bias[self.filters:self.filters * 2]
        else:
            self.bias_f = None
            self.bias_c = None
        self.built = True
        
    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=2)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training=training,
                count=2)
            
        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        
        if 0 < self.dropout < 1.:
            inputs_f = inputs * dp_mask[0]
            inputs_c = inputs * dp_mask[1]
        else:
            inputs_f = inputs
            inputs_c = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_f = h_tm1 * rec_dp_mask[0]
            h_tm1_c = h_tm1 * rec_dp_mask[1]
        else:
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f,
                              padding=self.padding)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c,
                              padding=self.padding)
        h_f = self.recurrent_conv(h_tm1_f,
                                  self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c,
                                  self.recurrent_kernel_c)
        
        f = self.recurrent_activation(x_f + h_f)
        cf = self.recurrent_activation(x_f + h_f - self.beta)
        ci = self.activation(x_c + h_c)
        
        c = f * c_tm1 + (1. - cf) * ci
        
        h = c
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h, c]
    
    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out
    
    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out
    
    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'beta': self.beta,
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  '.use_chrono_initialization': self.use_chrono_initialization,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'max_timesteps': self.max_timesteps}
        base_config = super(ConvJANetCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ConvJANet(ConvRNN2D):
    """Convolutional JANet Layer.

    It is similar to an JANet layer, but the input transformations
    and recurrent transformations are both convolutional.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, time, ..., channels)`
            while `"channels_first"` corresponds to
            inputs with shape `(batch, time, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        beta: float, used on the forget gate. This constant is dataset
            dependent, and we choose beta=1 typically. This beta allows
            slightly more information accumulated than forgotten. 
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        use_chrono_initialization: Boolean.
            If True, overrides the provided kernel initializer, recurrent
            initializer and forget gate initializer of the model to use
            'ChronoInitializer' scheme.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last output
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # Input shape
        - if data_format='channels_first'
            5D tensor with shape:
            `(samples, time, channels, rows, cols)`
        - if data_format='channels_last'
            5D tensor with shape:
            `(samples, time, rows, cols, channels)`

    # Output shape
        - if `return_sequences`
             - if data_format='channels_first'
                5D tensor with shape:
                `(samples, time, filters, output_row, output_col)`
             - if data_format='channels_last'
                5D tensor with shape:
                `(samples, time, output_row, output_col, filters)`
        - else
            - if data_format='channels_first'
                4D tensor with shape:
                `(samples, filters, output_row, output_col)`
            - if data_format='channels_last'
                4D tensor with shape:
                `(samples, output_row, output_col, filters)`
            where o_row and o_col depend on the shape of the filter and
            the padding

    # Raises
        ValueError: in case of invalid constructor arguments.

    # References
        - [The unreasonable effectiveness of the forget gate](https://arxiv.org/abs/1804.04849)
    """
    
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 beta=1.,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_chrono_initialization=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = ConvJANetCell(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              beta=beta,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              use_chrono_initialization=use_chrono_initialization,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout)
        super(ConvJANet, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        def build(self, input_shape):
            self.cell.timesteps = input_shape[1]
            super(ConvJANet, self).build(input_shape)
            
        def call(self, inputs, mask=None, training=None, initial_state=None):
            self.cell._dropout_mask = None
            self.cell._recurrent_dropout_mask = None
            return super(ConvJANet, self).call(inputs,
                                               mask=mask,
                                               training=training,
                                               initial_state=initial_state)
        
    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation
    
    @property
    def beta(self):
        return self.cell.beta

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def use_chrono_initialization(self):
        return self.cell.use_chrono_initialization

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout
    
    @property
    def max_timesteps(self):
        return self.cell.max_timesteps

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'beta': self.beta,
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'use_chrono_initialization': self.use_chrono_initialization,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'max_timesteps': self.max_timesteps}
        base_config = super(ConvJANet, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
        
        
        
            
            
            
            
        
        