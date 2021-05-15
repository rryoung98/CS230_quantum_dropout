import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit



# pseudo-code.
""" create a class that will implement quantum dropout.
we have several methodologies that we are investigating.
schuld and verdon are the two main papers we will focus on but our development will be entirely unique. """


@keras_export('keras.layers.Dropout')
class QuantumDropout(Layer):
    """  
    1. First we need to set the arguments rate of drop, seed, 
    and noise_shape(this might not be necessary if we follow the verdon method since their model 
    inherently applies gaussian noise?)
    The class will then call something similar to the keras class
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        if isinstance(rate, (int, float)) and not rate:
        keras_temporary_dropout_rate.get_cell().set(True)
        else:
        keras_temporary_dropout_rate.get_cell().set(False)
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
    """
    Quantum dropout: The essence of theapproach is to
    randomly select and measure one of thequbits, and set it aside for a certain
    numberNdropoutof parameter update epochs. After that, the qubit isre-added to
    the circuit and another qubit (or, perhaps,no qubit) is randomly dropped. This
    strategy works by“smoothing” the model fit and it generally inflates thetraining
    error, but often deflates the generalization error.
    """

    """ 
    We will make sure that this class follows the tensorflow implementation closely and is
    able to apply our dropout method.
    """
 

@tf_export("nn.dropout", v1=[])
@dispatch.add_dispatch_support
def dropout_v2(circuit, rate, noise_shape=None, seed=None, name=None):
    """ 
    Our method for dropout will go here where we will measure a certain channel.
    Edge cases we have to consider are when the circuit is not
    a cirq circuit object, incorrect parameter inputs in general. 

    - The position at which we apply the dropout and it's effects on a quantum ML layer.
    - We are doing the shuld approach here, but how can we learn from verdon?
    - 

    Our first parameter circuit will be the cirq circuit that we need to manipulate and add the
    measurement (our some other method). 
    
    Then, according to the second rate argument, seed and noise_shape we will calculate 
    which qubits need to be dropped.

    >>> if is_rate_number:
    >>>    keep_prob = 1 - rate
    >>>    scale = 1 / keep_prob
    >>>    scale = ops.convert_to_tensor(scale, dtype=x_dtype)
    >>>    ret = gen_math_ops.mul(x, scale)
    >>> else:
    >>>    raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)

    Finally by getting the appropriate qubits to measure we will append

    >>> cirquit.append(cirq.measure(*target, )
    where *target is the qubit we want to measure. 
    - The above circuit and appending it requires to be changed at every epoch. (we need
    to measure different qubits to successfully apply regulariztion during the training 
    session)
    """
