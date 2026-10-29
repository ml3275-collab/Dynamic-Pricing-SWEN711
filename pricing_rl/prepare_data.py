"""Build a demand profile for the simulator from NYC TLC High Volume For-Hire Vehicle data.

HVFHS trip records cover Uber (HV0003) and Lyft, so they are real ride-hail requests.
The profile holds everything the simulator needs, calibrated on the data:

    rate[dow, slot, zone]  mean requests per 15-min slot
    P[zone, zone]          origin -> destination probabilities (drives driver repositioning)
    fare[zone]            mean base passenger fare by pickup zone ($)
    trip_steps[zone]      mean trip duration in 15-min steps
    take                  platform take rate, 1 - driver_pay / base_fare
    day_sigma             std of the log day-to-day demand multiplier

Usage:
    python -m pricing_rl.prepare_data --month 2024-01 --zones 8
    python -m pricing_rl.prepare_data --synthetic        # offline fallback, for smoke tests only
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
SLOT_MIN = 15
SLOTS = 24 * 60 // SLOT_MIN
UBER = "HV0003"
COLUMNS = ["pickup_datetime", "PULocationID", "DOLocationID", "trip_time",
           "base_passenger_fare", "driver_pay"]


def download(url, dest):
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def build_profile(trips, zone_ids, zone_names):
    k = len(zone_ids)
    zidx = {z: i for i, z in enumerate(zone_ids)}
    src = trips[trips["PULocationID"].isin(zone_ids)].copy()

    ts = pd.to_datetime(src["pickup_datetime"])
    src["date"] = ts.dt.normalize()
    src["slot"] = (ts.dt.hour * 60 + ts.dt.minute) // SLOT_MIN
    src["z"] = src["PULocationID"].map(zidx)

    # demand: counts per (date, slot, zone), zero-filled, then averaged per day of week
    counts = src.groupby(["date", "slot", "z"]).size()
    dates = pd.date_range(ts.dt.normalize().min(), ts.dt.normalize().max(), freq="D")
    full = pd.MultiIndex.from_product([dates, range(SLOTS), range(k)])
    counts = counts.reindex(full, fill_value=0).to_numpy().reshape(len(dates), SLOTS, k)
    dow = dates.dayofweek.to_numpy()
    rate = np.stack([counts[dow == d].mean(axis=0) if (dow == d).any() else counts.mean(axis=0)
                     for d in range(7)])

    # day-to-day variability around the day-of-week mean
    daily = counts.sum(axis=(1, 2)).astype(float)
    ratio = daily / np.array([daily[dow == d].mean() for d in dow])
    day_sigma = float(np.std(np.log(ratio)))

    # origin -> destination, restricted to the modeled zones and renormalized
    inside = src[src["DOLocationID"].isin(zone_ids)]
    od = np.zeros((k, k))
    np.add.at(od, (inside["z"].to_numpy(), inside["DOLocationID"].map(zidx).to_numpy()), 1)
    od += 1e-3
    P = od / od.sum(axis=1, keepdims=True)

    g = src.groupby("z")
    fare = g["base_passenger_fare"].mean().reindex(range(k)).to_numpy()
    trip_steps = np.maximum(g["trip_time"].mean().reindex(range(k)).to_numpy() / 60 / SLOT_MIN, 1.0)
    take = float(np.clip(1 - src["driver_pay"].sum() / src["base_passenger_fare"].sum(), 0.1, 0.5))

    return dict(rate=rate, P=P, fare=fare, trip_steps=trip_steps, take=take,
                day_sigma=day_sigma, zone_ids=np.array(zone_ids), zone_names=np.array(zone_names))


def synthetic_profile(k=8, seed=0):
    """Plausible commuter-city shape. NOT real data: only for testing the pipeline offline."""
    rng = np.random.default_rng(seed)
    h = np.arange(SLOTS) / 4.0
    bump = lambda c, w: np.exp(-0.5 * ((h - c) / w) ** 2)
    weekday = 0.15 + bump(8.5, 1.5) + 0.8 * bump(18, 2.0) + 0.3 * bump(13, 3)
    weekend = 0.15 + 0.5 * bump(14, 4) + 0.9 * bump(22.5, 2.5)
    size = rng.uniform(15, 60, k)
    rate = np.stack([(weekend if d >= 5 else weekday)[:, None] * size[None] for d in range(7)])
    P = rng.dirichlet(np.ones(k) * 2, size=k)
    return dict(rate=rate, P=P, fare=rng.uniform(14, 28, k), trip_steps=rng.uniform(1.0, 2.0, k),
                take=0.27, day_sigma=0.1, zone_ids=np.arange(k),
                zone_names=np.array([f"synthetic-{i}" for i in range(k)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2024-01")
    ap.add_argument("--zones", type=int, default=8, help="number of busiest Manhattan zones")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--out", default="data/profile.npz")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        profile = synthetic_profile(args.zones)
    else:
        lookup = pd.read_csv(download(f"{BASE_URL}/misc/taxi+_zone_lookup.csv",
                                      f"{args.cache}/taxi_zone_lookup.csv"))
        path = download(f"{BASE_URL}/trip-data/fhvhv_tripdata_{args.month}.parquet",
                        f"{args.cache}/fhvhv_tripdata_{args.month}.parquet")
        trips = pd.read_parquet(path, columns=COLUMNS + ["hvfhs_license_num"],
                                filters=[("hvfhs_license_num", "==", UBER)])
        manhattan = set(lookup.loc[lookup["Borough"] == "Manhattan", "LocationID"])
        top = (trips[trips["PULocationID"].isin(manhattan)]["PULocationID"]
               .value_counts().head(args.zones).index.tolist())
        names = lookup.set_index("LocationID").loc[top, "Zone"].tolist()
        profile = build_profile(trips, top, names)
        print("zones:", names)

    np.savez(args.out, **profile)
    print(f"saved {args.out}: peak {profile['rate'].max():.1f} req/slot, "
          f"take={profile['take']:.2f}, day_sigma={profile['day_sigma']:.2f}")


if __name__ == "__main__":
    main()
