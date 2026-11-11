"""Two estimators that turn late, noisy position measurements of the moving target
into a prediction of where the target is NOW.

  CVKalman     constant-velocity Kalman filter; predicts  p + v * dt.
               (the same idea as predicting a rolling ball: straight-line extrapolation)
  HarmonicEKF  extended Kalman filter with a periodic model (offset + 2 harmonics) that
               also estimates the breathing frequency online.

Both expose the same interface:
    update(z, t_meas)   feed one measurement z that was ACQUIRED at time t_meas
    predict(t)          -> (position, velocity) predicted for time t >= t_meas
"""
import numpy as np


class CVKalman:
    def __init__(self, q: float = 1e-3, r: float = 0.0005 ** 2):
        self.q = q                    # white-acceleration spectral density [(m/s^2)^2 / Hz]
        self.r = r                    # measurement noise variance [m^2]
        self.x = None                 # state [p, v]
        self.P = np.diag([1e-4, 1e-3])
        self.t = None                 # time of the last measurement

    def update(self, z, t):
        if self.x is None:
            self.x, self.t = np.array([z, 0.0]), t
            return
        dt = t - self.t
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        S = P[0, 0] + self.r                       # H = [1, 0]
        K = P[:, 0] / S
        self.x = x + K * (z - x[0])
        self.P = P - np.outer(K, P[0, :])
        self.t = t

    def predict(self, t):
        if self.x is None:
            return None
        return self.x[0] + self.x[1] * (t - self.t), self.x[1]


class HarmonicEKF:
    """State x = [c, a1, b1, a2, b2, theta, omega]
       measurement  z = c + a1 sin(theta) + b1 cos(theta) + a2 sin(2 theta) + b2 cos(2 theta)
       dynamics     theta' = omega ; everything else is a slow random walk.
    """

    def __init__(self, q_amp: float = 1e-7, q_omega: float = 1e-3, r: float = 0.0005 ** 2,
                 rate0_cpm: float = 15.0):
        self.q_amp, self.q_omega, self.r = q_amp, q_omega, r
        self.w0 = 2 * np.pi * rate0_cpm / 60.0     # prior: a typical adult breathing rate
        self.x = None
        self.t = None

    @staticmethod
    def _h(x, lead=0.0):
        """Measurement function, its Jacobian, and the time derivative of the signal."""
        c, a1, b1, a2, b2, th, w = x
        th = th + w * lead
        s1, c1, s2, c2 = np.sin(th), np.cos(th), np.sin(2 * th), np.cos(2 * th)
        z = c + a1 * s1 + b1 * c1 + a2 * s2 + b2 * c2
        dz_dth = a1 * c1 - b1 * s1 + 2 * a2 * c2 - 2 * b2 * s2
        H = np.array([1.0, s1, c1, s2, c2, dz_dth, dz_dth * lead])
        return z, H, dz_dth * w

    def update(self, z, t):
        if self.x is None:
            self.x = np.array([z, 1e-3, 1e-3, 0.0, 0.0, 0.0, self.w0])
            self.P = np.diag([0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.005 ** 2, 0.005 ** 2,
                              np.pi ** 2, (0.3 * self.w0) ** 2])
            self.t = t
            return
        dt = t - self.t
        # predict
        F = np.eye(7)
        F[5, 6] = dt
        x = self.x.copy()
        x[5] += x[6] * dt
        Q = np.diag([self.q_amp] * 5 + [0.0, self.q_omega]) * dt
        P = F @ self.P @ F.T + Q
        # correct
        zhat, H, _ = self._h(x)
        S = H @ P @ H + self.r
        K = P @ H / S
        x = x + K * (z - zhat)
        P = P - np.outer(K, H @ P)
        x[6] = np.clip(x[6], 2 * np.pi * 6 / 60, 2 * np.pi * 40 / 60)   # 6..40 breaths/min
        x[5] = np.mod(x[5], 2 * np.pi)
        self.x, self.P, self.t = x, 0.5 * (P + P.T), t

    def predict(self, t):
        if self.x is None:
            return None
        z, _, v = self._h(self.x, lead=t - self.t)
        return z, v

    @property
    def rate_cpm(self):
        return None if self.x is None else self.x[6] * 60 / (2 * np.pi)


class MeasurementStream:
    """A sensor that samples a signal at `rate` Hz, adds noise, and delivers each sample
    `latency` seconds late (image acquisition + processing time)."""

    def __init__(self, signal, rate=25.0, latency=0.1, noise_std=0.0005, seed=0):
        self.signal, self.dt, self.latency, self.noise_std = signal, 1.0 / rate, latency, noise_std
        self.rng = np.random.default_rng(50_000 + seed)
        self.k = 0                                   # index of the next sample to deliver

    def poll(self, t):
        """Return the list of (z, t_acquired) that have become available by time t."""
        out = []
        while self.k * self.dt + self.latency <= t + 1e-9:
            t_acq = self.k * self.dt
            out.append((self.signal(t_acq) + self.noise_std * self.rng.standard_normal(), t_acq))
            self.k += 1
        return out
