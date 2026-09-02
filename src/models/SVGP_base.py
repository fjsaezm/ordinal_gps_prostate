"""
Author: Juan Maroñas (juan.maronnas@uam.es) and Daniel Hernandez-Lobato (daniel.hernandez@uam.es)
Efficient Multiclass Learning with Transformed Gaussian Processes

svgp_base.py : This file adds functionality to GPflows SVGP class. In particular it is designed to merge predict_f_samples and predict_f methods so that we have a method
               that samples and return mean and cov. In the current GPflow implementation if we wanna do this, we will be computing mean and cov twice which implies double
               N^3 operations.
"""

## Gpflow
import gpflow
from gpflow.conditionals.util import sample_mvn


class SVGP_base(gpflow.models.SVGP):

    def predict_f_and_draw_samples(
        self, Xnew, num_samples, full_cov=True, full_output_cov=False
    ):
        """
        Basically copy paste code from GPModel and return mean and cov
        """
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        # check below for shape info
        mean, cov = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(
                mean_for_sample, cov, full_cov, num_samples=num_samples
            )  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            samples = sample_mvn(
                mean, cov, full_output_cov, num_samples=num_samples
            )  # [..., (S), N, P]
        return samples, mean, cov  # [..., (S), N, P]
