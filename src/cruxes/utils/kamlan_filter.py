import numpy as np


# SimpleKalmanFilter implements a basic Kalman filter for smoothing noisy measurements.
class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-2, measurement_variance=1e-1):
        # process_variance: expected variance in the process (model) dynamics
        # measurement_variance: expected variance in the measurement noise
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        # posteri_estimate: the current best estimate after measurement update
        self.posteri_estimate = None
        # posteri_error_estimate: the current estimate error covariance
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # If this is the first measurement, initialize the filter state
        if self.posteri_estimate is None:
            self.posteri_estimate = np.array(measurement, dtype=np.float32)
            # Return the initial estimate as an integer tuple
            return tuple(self.posteri_estimate.astype(int))

        # Predict step: use previous estimate as the prior
        priori_estimate = self.posteri_estimate
        # Increase error estimate by process variance
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Compute Kalman gain: how much to trust the measurement vs. the prediction
        kalman_gain = priori_error_estimate / (
            priori_error_estimate + self.measurement_variance
        )
        # Update step: combine prior estimate and new measurement
        self.posteri_estimate = priori_estimate + kalman_gain * (
            np.array(measurement, dtype=np.float32) - priori_estimate
        )
        # Update the error estimate for the next iteration
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate

        # Return the filtered estimate as an integer tuple
        return tuple(self.posteri_estimate.astype(int))
