# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aixinnongnong/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aixinnongnong/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch viewers, hour by hour. Given hour 1's count, it forecasts hours 2–24 as five quantiles each. It learns the shape of a launch day from the case's own comparables — past big launches' hourly Twitch counts — instead of assuming every hour stays at hour 1's level.

## How it works

A console game that unlocks at midnight local time, country by country, does not stay flat after hour 1. Hour 1 only holds the time zones already live (New Zealand, Australia, East Asia, the UK). The count dips in the small hours, jumps when the US East Coast unlocks, dips again mid-morning, then climbs through the US day to a peak in the US afternoon/evening and falls overnight.

For each hour h = 2..24 the program fits, across the comparables, a log-linear regression of that hour's viewers on the launch's own hour-1 print:

```
ln V_h = a_h + b_h * ln p1
```

The slope is near 1 in the small hours and falls to about 0.65 at the peak: the peak grows with the launch, but less than proportionally. The spread is the regression's prediction interval, so it widens when hour 1's print sits far from the comparables' (extrapolation), with a floor of 0.20 log units.

Only console launches are used: GTA VI is console-only, and a PC launch unlocks simultaneously worldwide, a different shape (its hour 1 is already at peak). A comparable is console-only when its `platforms` names no PC. Comparables the case itself flags as early-access-contaminated are left out too — their "release day" is not a launch day. If too few console comparables carry an hour, that hour falls back to all comparables, then to a flat persistence forecast, so every case still gets a valid forecast.

On the four practice launches (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2) this scores a mean pinball loss of **0.195**, against the house baseline's **0.387** — about half the loss, with a typical miss of about 30% against the baseline's 83%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, deterministic: two runs on one case write the same bytes. It reads every `*.in` file in the input directory and writes one `*.out` per case, each `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24.

---

<sub>Entered on [make.fun](https://make.fun/?ref=xhh2wh2q&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
