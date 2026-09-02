"""
Author: Javier Sáez (fjaviersaezm@ugr.es), Juan Maroñas (juan.maronnas@uam.es) and Daniel Hernandez-Lobato (daniel.hernandez@uam.es)
Efficient Multiclass Learning with Transformed Gaussian Processes

dsvi_dgp.py. Implements Deep Gaussian Processes with the double stochastic variational inference algorithm, Salimbeni and Deisenroth 2017

"""

## Standard
import numpy as np

## Tensorflow
import tensorflow as tf

## GPflow
import gpflow

## Custom


class DSVI_DGP(gpflow.Module):

    def __init__(
        self,
        GP_list,  # A list containing gpflow.models.SVGPs representing the GPs per layer
        likelihood,  # The likelihood linking latent functions to observations
        num_data,  # Number of total training points for ell rescaling
    ):
        super(DSVI_DGP, self).__init__()

        self.GP_list = GP_list
        self.likelihood = likelihood

        self.num_data = num_data

        assert (
            len(GP_list) > 1
        ), "The length of GP list must be greater than 1. This list contains hidden layers plus output layer"

    def prior_kl(self):
        """Kullback Lieber Divergence from all the layers, i.e returns a scalar value"""
        kld = 0.0
        # loop over each GP Layer
        for gp in self.GP_list:
            kld += gp.prior_kl()
        return kld

    def sample_from_posterior_hierarchy(self, Xnew, S_DGP):
        """
        Draws S_DGP samples from the hierarchy. Returns the sample from the last layer plus the mean and covariance used to draw that sample

        Returns shape ( S_DGP, MB, GPs_per_layer )
        """
        f = Xnew
        _S = S_DGP
        MB, Dx = Xnew.shape
        for gp in self.GP_list:
            f, mean, cov = gp.predict_f_and_draw_samples(
                Xnew=f, num_samples=_S, full_cov=False, full_output_cov=False
            )

            _S = 1  # set to 1. On the first call we draw S_DGP samples for each X. Thus we can keep sampling 1 at each subsequent layer and we will be sampling S_DGP at the end
            f = tf.reshape(f, tf.constant([np.prod(f.shape[:-1]), f.shape[-1]]))

        num_GP = mean.shape[-1]
        f = tf.reshape(f, tf.constant([S_DGP, MB, num_GP]))
        mean = tf.reshape(mean, tf.constant([S_DGP, MB, num_GP]))
        cov = tf.reshape(cov, tf.constant([S_DGP, MB, num_GP]))

        return f, mean, cov

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        """
        Predicts the mean and variance of the last layer of the DGP. This is done by sampling S_DGP samples from the hierarchy and returning the mean and variance of the last layer.
        Note that this is not a posterior predictive distribution, but rather a sample from the posterior hierarchy.
        """
        S_DGP = 1
        f, mean, cov = self.sample_from_posterior_hierarchy(Xnew, S_DGP)
        return mean, cov
