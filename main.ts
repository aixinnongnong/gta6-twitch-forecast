// gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers, hour by hour.
//
// Idea (the shape of a launch day, learned from past launches, not from the starter):
// A console game that unlocks at midnight local time, country by country, does not stay
// flat after hour 1. Hour 1 (00:00-01:00 UTC) only holds the time zones already live
// (New Zealand, Australia, East Asia, the UK). The count dips in the small hours, jumps
// when the US East Coast unlocks, dips again mid-morning, then climbs through the US day
// to a peak in the US afternoon/evening and falls overnight. The starter's flat
// persistence forecast misses all of that.
//
// This program learns that shape from the case's own comparables: past big launches'
// hourly Twitch counts, 14 days before to 3 days after their release day. For each hour
// h = 2..24 it fits, across the comparables, a log-linear regression of that hour's
// viewers on the launch's own hour-1 print:
//
//   ln V_h = a_h + b_h * ln p1
//
// The slope b_h is near 1 in the small hours (an hour close to hour 1 scales with it)
// and falls to ~0.65 at the peak (the peak grows with the launch, but less than
// proportionally: a launch already huge at hour 1 has a smaller peak multiple). The
// spread is the regression's prediction interval, so it widens when hour 1's print sits
// far from the comparables' (extrapolation), with a floor of 0.20 log units against
// overconfidence from a small fit set.
//
// Which comparables: only console launches, because GTA VI is console-only and a PC
// launch unlocks simultaneously worldwide (a different shape: its hour 1 is already at
// peak). A comparable is console-only when its `platforms` names no PC. Comparables the
// case itself flags as early-access-contaminated are left out too: their "release day"
// is not a launch day (the game was playable days earlier). If too few console
// comparables carry an hour, that hour falls back to all comparables; if even that is
// too few, it falls back to the starter's persistence shape. Every case still gets a
// valid forecast: a case that cannot be read at all is answered with persistence.
//
// Deterministic: ordinary least squares over the case file, no randomness, no network,
// standard library only. Two runs on one case write the same bytes.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The persistence spread, in log units per square-root hour (the starter's number). */
const SIGMA = 0.15;
/** The smallest spread any hour is given, in log units. */
const SPREAD_FLOOR = 0.2;
/** A fit needs at least this many comparables, else it falls back. */
const MIN_FIT = 6;
/** A regression needs at least this many points, else the hour falls back. */
const MIN_POINTS = 3;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  platforms?: string;
  unlock?: string;
  hourly: HourlyPoint[];
}

interface Case {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The starter's persistence quantiles for one hour h around hour 1's print. */
const persistenceQuantiles = (p1: number, h: number): number[] =>
  Z.map((z) => round2(Math.min(1e9, Math.max(1, p1 * Math.exp(z * SIGMA * Math.sqrt(h - 1))))));

/** The hour offset (0..23) of a point's `from` from a comparable's day_start, or -1. */
const hourOffset = (from: string, dayStart: string): number => {
  const t = Date.parse(from);
  const d = Date.parse(dayStart);
  if (Number.isNaN(t) || Number.isNaN(d)) return -1;
  const off = Math.round((t - d) / 3600000);
  return off >= 0 && off < 24 ? off : -1;
};

/** A comparable's release-day viewers by hour offset, from its hourly points. */
const releaseDay = (c: Comparable): Map<number, number> => {
  const day = new Map<number, number>();
  for (const pt of c.hourly ?? []) {
    const off = hourOffset(pt.from, c.day_start);
    if (off >= 0 && typeof pt.viewers === 'number' && pt.viewers > 0) day.set(off, pt.viewers);
  }
  return day;
};

/** True when the comparable is a console-only launch (its platforms name no PC). */
const isConsoleOnly = (c: Comparable): boolean =>
  !/\bPC\b/i.test(c.platforms ?? '');

/** True when the case itself flags the comparable as early-access-contaminated. */
const isEarlyAccessContaminated = (c: Comparable): boolean =>
  /early-access-contaminated/i.test(c.unlock ?? '');

/** Ordinary least squares of ys on xs: intercept, slope, mean of xs, sum of squares of xs. */
const ols = (xs: number[], ys: number[]): { a: number; b: number; mx: number; sxx: number; s2: number } => {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  const sxx = xs.reduce((s, x) => s + (x - mx) * (x - mx), 0);
  const sxy = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  const s2 = ys.reduce((s, y, i) => s + (y - (a + b * xs[i])) ** 2, 0) / Math.max(1, n - 2);
  return { a, b, mx, sxx, s2 };
};

/** The five quantiles of one hour h, from the comparables and hour 1's print p1. */
const hourQuantiles = (
  p1: number,
  h: number,
  days: Array<{ p1: number; day: Map<number, number> }>,
  allDays: Array<{ p1: number; day: Map<number, number> }>,
): number[] => {
  const lp = Math.log(Math.max(1, p1));
  let use = days.filter((c) => c.p1 > 0 && (c.day.get(h - 1) ?? 0) > 0);
  if (use.length < MIN_FIT) {
    use = allDays.filter((c) => c.p1 > 0 && (c.day.get(h - 1) ?? 0) > 0);
  }
  if (use.length < MIN_POINTS) return persistenceQuantiles(p1, h);
  const xs = use.map((c) => Math.log(c.p1));
  const ys = use.map((c) => Math.log(c.day.get(h - 1) as number));
  const fit = ols(xs, ys);
  // A slope outside [0, 1.5] is not a launch shape but a fit artefact (a tiny or
  // degenerate set): clamp it, keeping the line through the data's centre.
  const b = Math.min(1.5, Math.max(0, fit.b));
  const a = ys.reduce((s, y) => s + y, 0) / ys.length - b * fit.mx;
  const s2 = ys.reduce((s, y, i) => s + (y - (a + b * xs[i])) ** 2, 0) / Math.max(1, ys.length - 2);
  const { mx, sxx } = fit;
  const mu = a + b * lp;
  // Prediction interval: the residual spread, widened when p1 sits far from the
  // comparables' (extrapolation), with a floor against overconfidence.
  const n = use.length;
  const leverage = sxx > 0 ? (lp - mx) * (lp - mx) / sxx : 0;
  const sd = Math.max(Math.sqrt(s2 * (1 + 1 / n + leverage)), SPREAD_FLOOR);
  const qs = Z.map((z) => round2(Math.min(1e9, Math.max(1, Math.exp(mu + z * sd)))));
  for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  return qs;
};

/** The forecast of one case: 23 rows, hours 2..24. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const parsed = (c.comparables ?? []).map((comp) => {
    const day = releaseDay(comp);
    return { p1: day.get(0) ?? 0, day, console: isConsoleOnly(comp) && !isEarlyAccessContaminated(comp) };
  });
  const allDays = parsed.map(({ p1, day }) => ({ p1, day }));
  const consoleDays = parsed.filter((d) => d.console).map(({ p1, day }) => ({ p1, day }));
  const days = consoleDays.filter((d) => d.p1 > 0).length >= MIN_FIT ? consoleDays : allDays;
  return c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, hour, days, allDays) }));
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let hours: Array<{ hour: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    hours = forecastCase(c);
  } catch (e) {
    // Never leave a case without a valid forecast: fall back to persistence.
    // Hour 1's print is read again for the fallback, or 1 when it cannot be read.
    let p1 = 1;
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
      const first = c.known.find((k) => k.hour === 1);
      if (first !== undefined) p1 = Math.max(1, first.viewers);
    } catch {
      // keep p1 = 1
    }
    hours = Array.from({ length: 23 }, (_, i) => ({ hour: i + 2, q: persistenceQuantiles(p1, i + 2) }));
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
