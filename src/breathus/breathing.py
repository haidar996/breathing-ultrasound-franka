"""Synthetic breathing: one scalar signal s(t) that drives BOTH
  - the abdominal surface (moves up, world +z, when the patient inhales), and
  - an internal target such as a liver lesion (moves towards the feet, world -y).

Shape: the Lujan model  s = cos^(2n)(phase)  (Lujan et al., Med. Phys. 1999).
It is asymmetric: the signal rests near 0 (exhale) and peaks briefly at 1 (inhale).

To avoid a "too perfect" signal, every breath has its own duration and depth
(cycle-to-cycle variability) and a slow baseline drift is added.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SubjectParams:
    """One virtual patient."""
    rate_cpm: float = 14.6        # breaths per minute (quiet breathing, supine)
    surface_pp: float = 0.0099    # abdominal wall excursion, peak-to-peak [m]
    target_pp: float = 0.015      # internal target head-feet excursion, peak-to-peak [m]
    n: int = 2                    # Lujan exponent (shape asymmetry)
    period_cv: float = 0.05       # breath-to-breath variation of the period (std / mean)
    amp_cv: float = 0.10          # breath-to-breath variation of the depth (std / mean)
    drift_frac: float = 0.10      # slow baseline drift, fraction of the excursion
    tissue_k: float = 1000.0      # soft-tissue stiffness under the probe [N/m]
    seed: int = 0


class Breathing:
    """Pre-generates the breaths of one subject, then answers s(t) and ds/dt."""

    def __init__(self, p: SubjectParams, duration: float = 200.0):
        self.p = p
        rng = np.random.default_rng(p.seed)
        tau = 60.0 / p.rate_cpm                                  # mean breath duration [s]
        n_cycles = int(duration / tau * 1.5) + 5
        # each breath: its own duration and its own depth (clipped to stay physical)
        self.tau = tau * np.clip(1 + p.period_cv * rng.standard_normal(n_cycles), 0.7, 1.3)
        self.gain = np.clip(1 + p.amp_cv * rng.standard_normal(n_cycles), 0.6, 1.4)
        # breath k occupies [t_start[k], t_start[k+1]); random start inside the first breath
        self.t_start = np.concatenate([[0.0], np.cumsum(self.tau)]) - rng.uniform(0, tau)
        # slow drift (period 40-80 s, random phase)
        self.drift_w = 2 * np.pi / rng.uniform(40, 80)
        self.drift_phase = rng.uniform(0, 2 * np.pi)

    def s(self, t: float):
        """Return (s, ds/dt). s is about 0 at exhale and about 1 at full inhale."""
        p = self.p
        k = int(np.searchsorted(self.t_start, t, side="right") - 1)
        u = (t - self.t_start[k]) / self.tau[k]                  # 0..1 inside the breath
        phi = np.pi * (u - 0.5)                                  # -pi/2..pi/2, cos = 0 at both ends
        c, sn = np.cos(phi), np.sin(phi)
        g = self.gain[k]
        val = g * c ** (2 * p.n)
        dval = g * (2 * p.n) * c ** (2 * p.n - 1) * (-sn) * np.pi / self.tau[k]
        # baseline drift
        a = self.drift_w * t + self.drift_phase
        val += p.drift_frac * np.sin(a)
        dval += p.drift_frac * self.drift_w * np.cos(a)
        return val, dval

    # --- the two physical quantities driven by s(t) -------------------------------------
    def surface(self, t):
        """Abdominal surface height above its exhale position [m] and its velocity."""
        s, ds = self.s(t)
        return self.p.surface_pp * s, self.p.surface_pp * ds

    def target(self, t):
        """Internal target displacement along world y [m] (negative = towards feet) and velocity."""
        s, ds = self.s(t)
        return -self.p.target_pp * s, -self.p.target_pp * ds


def nominal_subject() -> SubjectParams:
    """The 'textbook' subject used for the demo figures and the video."""
    return SubjectParams(seed=1)


def random_subject(seed: int) -> SubjectParams:
    """A random virtual patient. Ranges are stated and justified in docs/research_notes.md."""
    rng = np.random.default_rng(10_000 + seed)
    return SubjectParams(
        rate_cpm=rng.uniform(12, 20),
        surface_pp=rng.uniform(0.005, 0.015),
        target_pp=rng.uniform(0.005, 0.025),
        n=int(rng.integers(1, 4)),
        tissue_k=rng.uniform(500, 2000),
        seed=seed,
    )
