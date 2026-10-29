"""Ride-hailing dynamic pricing environment, calibrated on a TLC demand profile.

One step = 15 minutes, one episode = one day (96 steps). The agent sets a surge multiplier
for every zone each step.

Dynamics per step
    1. drivers finishing trips / relocating arrive back in idle pools
    2. requests ~ Poisson(profile rate x day-level noise x shock)
    3. each rider accepts with prob m^-elasticity
    4. accepted requests are matched to idle drivers in the zone; the rest are lost
    5. matched drivers travel to a destination drawn from the TLC OD matrix, for a trip duration
       drawn around the TLC mean, then become idle there
    6. idle drivers reposition toward zones with high (expected demand x multiplier) / idle,
       arriving one step later
    7. the online fleet drifts toward a target that follows demand only partially and grows with
       the average multiplier (supply response)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

SLOTS = 96
BUFFER = 16  # ring buffer length for drivers in transit (max trip duration < BUFFER)

SCENARIOS = {
    "weekday": dict(dow_choices=[1, 2, 3]),
    "weekend": dict(dow_choices=[5, 6]),
    "friday_night": dict(dow_choices=[4]),
    "event_spike": dict(dow_choices=[1, 2, 3], shock=dict(zone=0, start=72, length=8, factor=2.5)),
    "driver_shortage": dict(dow_choices=[1, 2, 3], fleet_scale=0.6),
    "price_sensitive": dict(dow_choices=[1, 2, 3], elasticity=2.0),
}

"""Ride-hailing dynamic pricing environment, calibrated on a TLC demand profile.

One step = 15 minutes, one episode = one day (96 steps). The agent sets a surge multiplier
for every zone each step.

Dynamics per step
    1. drivers finishing trips / relocating arrive back in idle pools
    2. requests ~ Poisson(profile rate x day-level noise x shock)
    3. each rider accepts with prob m^-elasticity
    4. accepted requests are matched to idle drivers in the zone; the rest are lost
    5. matched drivers travel to a destination drawn from the TLC OD matrix, for a trip duration
       drawn around the TLC mean, then become idle there
    6. idle drivers reposition toward zones with high (expected demand x multiplier) / idle,
       arriving one step later
    7. the online fleet drifts toward a target that follows demand only partially and grows with
       the average multiplier (supply response)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

SLOTS = 96
BUFFER = 16  # ring buffer length for drivers in transit (max trip duration < BUFFER)

SCENARIOS = {
    "weekday": dict(dow_choices=[1, 2, 3]),
    "weekend": dict(dow_choices=[5, 6]),
    "friday_night": dict(dow_choices=[4]),
    "event_spike": dict(dow_choices=[1, 2, 3], shock=dict(zone=0, start=72, length=8, factor=2.5)),
    "driver_shortage": dict(dow_choices=[1, 2, 3], fleet_scale=0.6),
    "price_sensitive": dict(dow_choices=[1, 2, 3], elasticity=2.0),
    # held out: training only ever shows ONE stress at a time, this stacks all three
    "combined_stress": dict(dow_choices=[1, 2, 3], fleet_scale=0.6, elasticity=2.0,
                            shock=dict(zone=0, start=72, length=8, factor=2.5)),
}


class RideHailEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, profile_path="data/profile.npz", max_mult=3.0, elasticity=1.2,
                 elasticity_jitter=0.0, fleet_scale=1.0, target_util=0.8, follow=0.3,
                 supply_eta=0.5, reposition_rate=0.15, reposition_beta=1.0, dow_choices=None,
                 shock=None, reward_w=None):
        p = np.load(profile_path)
        self.rate, self.P = p["rate"], p["P"]
        self.fare, self.trip_steps = p["fare"], p["trip_steps"]
        self.take, self.day_sigma = float(p["take"]), float(p["day_sigma"])
        self.zone_names = [str(n) for n in p["zone_names"]]
        self.K = self.rate.shape[2]

        self.max_mult, self.base_eps, self.eps_jitter = max_mult, elasticity, elasticity_jitter
        self.fleet_scale, self.target_util, self.follow = fleet_scale, target_util, follow
        self.supply_eta, self.rho, self.beta = supply_eta, reposition_rate, reposition_beta
        self.dow_choices = list(dow_choices) if dow_choices is not None else list(range(7))
        self.shock = shock
        self.w = dict(unmet=0.5, volatility=0.1)
        self.w.update(reward_w or {})

        mean_rate = self.rate.mean(axis=(0, 1))
        self.zone_scale = np.maximum(mean_rate, 1.0)
        self.rev_scale = self.take * float((mean_rate * self.fare).sum())
        self.arr_scale = float(mean_rate.sum())
        self.busy_mean = float((mean_rate * self.trip_steps).sum())

        self.action_space = spaces.Box(-1.0, 1.0, (self.K,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6 * self.K + 4,), np.float32)
        self.rng = np.random.default_rng()

    # ---- helpers -------------------------------------------------------------------------
    def _to_mult(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        return 1.0 + (self.max_mult - 1.0) * (a + 1.0) / 2.0

    def mult_to_action(self, m):
        m = np.clip(m, 1.0, self.max_mult)
        return ((m - 1.0) / (self.max_mult - 1.0) * 2.0 - 1.0).astype(np.float32)

    def expected_demand(self, t=None):
        """Forecast of requests per zone at step t (the profile; true noise is not known)."""
        t = self.t if t is None else t
        lam = self.rate[self.dow, t % SLOTS] * self.day_scale
        if self.shock and self.shock["start"] <= t < self.shock["start"] + self.shock["length"]:
            lam = lam.copy()
            lam[self.shock["zone"]] *= self.shock["factor"]
        return lam

    def _target_fleet(self, t, m):
        busy_t = float((self.expected_demand(t) * self.trip_steps).sum())
        busy = self.follow * busy_t + (1 - self.follow) * self.busy_mean * self.day_scale
        w = self.expected_demand(t)
        m_bar = float((m * w).sum() / max(w.sum(), 1e-9))
        return self.fleet_scale * busy / self.target_util * m_bar ** self.supply_eta

    def _obs(self):
        lam, lam_next = self.expected_demand(self.t), self.expected_demand(self.t + 1)
        ang = 2 * np.pi * self.t / SLOTS
        dang = 2 * np.pi * self.dow / 7
        return np.concatenate([
            self.idle / self.zone_scale, lam / self.zone_scale, lam_next / self.zone_scale,
            self.last_m - 1.0, self.last_acc, self.last_match,
            [np.sin(ang), np.cos(ang), np.sin(dang), np.cos(dang)],
        ]).astype(np.float32)

    # ---- gym API -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.dow = int(self.rng.choice(self.dow_choices))
        self.day_scale = float(np.exp(self.day_sigma * self.rng.standard_normal() - self.day_sigma ** 2 / 2))
        self.eps = self.base_eps * (1 + self.eps_jitter * self.rng.uniform(-1, 1))
        self.t = 0
        self.last_m = np.ones(self.K)
        self.last_acc = np.ones(self.K)      # share of riders who accepted the last price
        self.last_match = np.ones(self.K)    # share of accepted riders who got a driver
        self.transit = np.zeros((BUFFER, self.K), dtype=np.int64)
        n0 = int(self._target_fleet(0, self.last_m))
        self.idle = self.rng.multinomial(n0, self._share(self.expected_demand(0))).astype(np.int64)
        self.stats = dict(revenue=0.0, arrivals=np.zeros(self.K), matched=np.zeros(self.K),
                          accepted=0.0, dispatch=0.0, vol=0.0, mult_w=0.0, max_mult=1.0)
        return self._obs(), {}

    @staticmethod
    def _share(w):
        w = np.asarray(w, dtype=float) + 1e-9
        return w / w.sum()

    def step(self, action):
        m = self._to_mult(action)
        t = self.t
        self.idle += self.transit[t % BUFFER]
        self.transit[t % BUFFER] = 0

        lam = self.expected_demand(t)
        arrivals = self.rng.poisson(lam)
        accepted = self.rng.binomial(arrivals, np.minimum(1.0, m ** -self.eps))
        matched = np.minimum(accepted, self.idle)
        idle_before = self.idle.sum()
        self.idle = self.idle - matched
        revenue = float((matched * self.fare * m).sum() * self.take)

        # matched drivers drive to a destination and become idle there after the trip
        for z in np.flatnonzero(matched):
            n = matched[z]
            dest = self.rng.choice(self.K, size=n, p=self.P[z])
            dur = np.minimum(1 + self.rng.poisson(max(self.trip_steps[z] - 1, 0), n), BUFFER - 1)
            np.add.at(self.transit, ((t + dur) % BUFFER, dest), 1)

        # idle drivers reposition toward zones where demand x multiplier per driver is high
        attract = self.expected_demand(t + 1) * m / (self.idle + 1.0)
        score = self.beta * np.log(attract + 1e-6)
        S = np.tile(score, (self.K, 1)) - 1.0 * (1 - np.eye(self.K))
        probs = np.exp(S - S.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        for z in range(self.K):
            movers = self.rng.binomial(self.idle[z], self.rho)
            if movers:
                moved = self.rng.multinomial(movers, probs[z])
                moved[z] = 0
                self.idle[z] -= moved.sum()
                self.transit[(t + 1) % BUFFER] += moved

        # fleet entry / exit toward the target
        diff = self._target_fleet(t + 1, m) - (self.idle.sum() + self.transit.sum())
        if diff > 0:
            self.idle += self.rng.multinomial(self.rng.poisson(0.3 * diff),
                                              self._share(self.expected_demand(t + 1)))
        elif self.idle.sum() > 0:
            n = min(int(0.3 * -diff), int(self.idle.sum()))
            self.idle -= self.rng.multinomial(n, self._share(self.idle))
            self.idle = np.maximum(self.idle, 0)

        vol = float(np.abs(m - self.last_m).mean())
        self.last_m = m
        self.last_acc = np.where(arrivals > 0, accepted / np.maximum(arrivals, 1), 1.0)
        self.last_match = np.where(accepted > 0, matched / np.maximum(accepted, 1), 1.0)
        reward = (revenue / self.rev_scale
                  - self.w["unmet"] * float((arrivals - matched).sum()) / self.arr_scale
                  - self.w["volatility"] * vol)

        s = self.stats
        s["revenue"] += revenue
        s["arrivals"] += arrivals
        s["matched"] += matched
        s["accepted"] += accepted.sum()
        s["dispatch"] += matched.sum() / max(idle_before, 1)
        s["vol"] += vol
        s["mult_w"] += float((m * arrivals).sum())
        s["max_mult"] = max(s["max_mult"], float(m.max()))

        self.t += 1
        truncated = self.t >= SLOTS
        info = {"revenue": revenue, "arrivals": arrivals, "matched": matched, "mult": m}
        if truncated:
            info["episode"] = self.episode_metrics()
        return self._obs(), float(reward), False, truncated, info

    def episode_metrics(self):
        s = self.stats
        zone_service = s["matched"] / np.maximum(s["arrivals"], 1)
        return dict(
            revenue=s["revenue"] / self.rev_scale,  # normalized: 1.0 ~ one average step at m=1
            service_rate=s["matched"].sum() / max(s["arrivals"].sum(), 1),
            fulfil_given_accept=s["matched"].sum() / max(s["accepted"], 1),
            mean_mult=s["mult_w"] / max(s["arrivals"].sum(), 1),
            max_mult=s["max_mult"],
            volatility=s["vol"] / SLOTS,
            dispatch_rate=s["dispatch"] / SLOTS,
            zone_service_gap=float(zone_service.max() - zone_service.min()),
        )


class MixedScenarioEnv(RideHailEnv):
    """Training env: each episode draws at most ONE stress (shortage, price sensitivity or demand
    spike) with random strength, so the agent sees varied conditions. The benchmark's
    combined_stress scenario stacks all three and is never seen in training."""

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.dow_choices = list(range(7))
        self.fleet_scale, self.base_eps, self.shock = 1.0, 1.2, None
        stress = str(r.choice(["none", "shortage", "sensitive", "spike"]))
        if stress == "shortage":
            self.fleet_scale = float(r.uniform(0.5, 0.9))
        elif stress == "sensitive":
            self.base_eps = float(r.uniform(1.6, 2.4))
        elif stress == "spike":
            self.shock = dict(zone=int(r.integers(self.K)), start=int(r.integers(8, 80)),
                              length=int(r.integers(4, 12)), factor=float(r.uniform(1.5, 3.0)))
        return super().reset(seed=seed, options=options)


def make_env(scenario="weekday", profile_path="data/profile.npz", **overrides):
    return RideHailEnv(profile_path, **{**SCENARIOS[scenario], **overrides})