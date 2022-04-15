# utils function
import math
import numpy as np
import tensorflow as tf


def count_total_parameters(variables):
    """Count the total parameters for tensorflow graph

    Arguments:
        variables {tf.tensor} -- tensorflow variables

    Returns:
        int -- total parameter number
    """
    total_parameters = 0
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


class ExponentialSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1):
        """Exponential interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = tf.minimum(tf.cast(t, tf.float32) /
                              self.schedule_timesteps, 1.0)
        return self.initial_p ** (1 - fraction) * self.final_p ** fraction
