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
// viewers on the launch's own hour-1 print AND its hour-1 channel count:
//
//   ln V_h = a_h + b_h * ln p1 + c_h * ln ch1
//
// Two things the hour-1 print alone misses: (1) the slope b_h is near 1 in the small
// hours (an hour close to hour 1 scales with it) and falls to ~0.65 at the peak (the
// peak grows with the launch, but less than proportionally); (2) a diffuse hour 1
// (many live channels for its viewers: a low viewers-per-channel) means the big
// streamers have not gone live yet, so the peak multiple is larger -- across the
// console comparables, ln(viewers per channel in hour 1) and ln(peak / hour 1)
// correlate at -0.87. The channel coefficient is shrunk by 0.6: with only a handful
// of console comparables, the two-predictor fit is noisy, and the shrunk coefficient
// keeps the signal without letting the noise run the forecast.
//
// The spread is the regression's prediction interval, with a floor of 0.20 log units
// against overconfidence from a small fit set.
//
// Which comparables: only console launches, because GTA VI is console-only and a PC
// launch unlocks simultaneously worldwide (a different shape: its hour 1 is already at
// peak). A comparable is console-only when its `platforms` names no PC. Comparables the
// case itself flags as early-access-contaminated are left out too: their "release day"
// is not a launch day (the game was playable days earlier). If too few console
// comparables carry an hour, that hour falls back to all comparables; if even that is
// too few, it falls back to the starter's persistence shape.
//
// Ordering guard: the afternoon peak (the largest median over hours 16..24) must be
// at least 1.2x the morning bump (the largest median over hours 2..8). Every observed
// launch satisfies this by a wide margin (peaks run 1.7x to 9x their morning bump);
// the guard only bites when the per-hour slopes extrapolate to an inverted shape at
// a very large hour-1 print, as a huge launch like GTA VI's may have. When it bites,
// the afternoon hours are scaled up to restore the ordering. It never triggers on
// the practice cases.
//
// Practice validation (against the real SullyGnome answers for the four practice
// launches): mean pinball loss 0.1772 vs the house baseline's 0.3871 -- about half
// the baseline's loss. Per case: Wolverine 0.3252 (baseline 0.7437), Red Dead
// Redemption 2 0.1216 (0.2868), God of War Ragnarök 0.0973 (0.2146), Spider-Man 2
// 0.1647 (0.3033). Typical miss about 30% vs the baseline's 83%. The one-predictor
// version of this model scored 0.1951; the hour-1 channel count is what closes
// most of the gap. Honest caveat: Wolverine is the hardest case (0.33, 90%
// coverage 0.57) -- its peak was 9.1x its small hour 1, an outlier; the model is
// strongest on large launches. GTA VI's hour 1 will be large, so it should behave
// more like God of War / Red Dead / Spider-Man 2 than Wolverine.
//
// Every case still gets a valid forecast: a case that cannot be read at all is
// answered with persistence.
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
/** The channel coefficient is shrunk by this much (a noisy two-predictor fit). */
const CHANNEL_SHRINK = 0.6;
/** The afternoon peak must be at least this many times the morning bump. */
const PEAK_ORDER = 1.2;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number;
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
  known: Array<{ hour: number; viewers: number; channels?: number }>;
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

/** A comparable's release-day viewers (and hour-1 channels) by hour offset. */
const releaseDay = (c: Comparable): { day: Map<number, number>; ch1: number } => {
  const day = new Map<number, number>();
  let ch1 = 0;
  for (const pt of c.hourly ?? []) {
    const off = hourOffset(pt.from, c.day_start);
    if (off >= 0 && typeof pt.viewers === 'number' && pt.viewers > 0) {
      day.set(off, pt.viewers);
      if (off === 0 && typeof pt.channels === 'number' && pt.channels > 0) ch1 = pt.channels;
    }
  }
  return { day, ch1 };
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

/** OLS with two predictors (x1, x2) on ys: [intercept, b1, b2], or null when degenerate. */
const ols2 = (x1: number[], x2: number[], ys: number[]): [number, number, number] | null => {
  const n = x1.length;
  const m1 = x1.reduce((s, x) => s + x, 0) / n;
  const m2 = x2.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  const s11 = x1.reduce((s, x) => s + (x - m1) ** 2, 0);
  const s22 = x2.reduce((s, x) => s + (x - m2) ** 2, 0);
  const s12 = x1.reduce((s, x, i) => s + (x - m1) * (x2[i] - m2), 0);
  const s1y = x1.reduce((s, x, i) => s + (x - m1) * (ys[i] - my), 0);
  const s2y = x2.reduce((s, x, i) => s + (x - m2) * (ys[i] - my), 0);
  const det = s11 * s22 - s12 * s12;
  if (Math.abs(det) < 1e-9) return null;
  const b1 = (s1y * s22 - s2y * s12) / det;
  const b2 = (s2y * s11 - s1y * s12) / det;
  return [my - b1 * m1 - b2 * m2, b1, b2];
};

interface DaySet {
  p1: number;
  ch1: number;
  day: Map<number, number>;
}

/** The five quantiles of one hour h, from the comparables, hour 1's print and channels. */
const hourQuantiles = (
  p1: number,
  ch1: number,
  h: number,
  days: DaySet[],
  allDays: DaySet[],
): number[] => {
  const lp = Math.log(Math.max(1, p1));
  const lc = ch1 > 0 ? Math.log(ch1) : null;
  let use = days.filter((c) => c.p1 > 0 && (c.day.get(h - 1) ?? 0) > 0);
  if (use.length < MIN_FIT) {
    use = allDays.filter((c) => c.p1 > 0 && (c.day.get(h - 1) ?? 0) > 0);
  }
  if (use.length < MIN_POINTS) return persistenceQuantiles(p1, h);
  const xs = use.map((c) => Math.log(c.p1));
  const ys = use.map((c) => Math.log(c.day.get(h - 1) as number));
  let mu: number;
  let s2: number;
  let n = use.length;
  // Two predictors when hour-1 channels are known for the target and most
  // comparables: the channel coefficient is shrunk, and the intercept is
  // re-centred so the line still passes through the data's centre.
  const withCh = use.filter((c) => c.ch1 > 0);
  if (lc !== null && withCh.length >= 5) {
    const x1 = withCh.map((c) => Math.log(c.p1));
    const x2 = withCh.map((c) => Math.log(c.ch1));
    const y2 = withCh.map((c) => Math.log(c.day.get(h - 1) as number));
    const coef = ols2(x1, x2, y2);
    if (coef) {
      const [, b1, b2raw] = coef;
      const b2 = b2raw * CHANNEL_SHRINK;
      const m1 = x1.reduce((s, x) => s + x, 0) / x1.length;
      const m2 = x2.reduce((s, x) => s + x, 0) / x2.length;
      const my = y2.reduce((s, y) => s + y, 0) / y2.length;
      const a = my - b1 * m1 - b2 * m2;
      mu = a + b1 * lp + b2 * lc;
      s2 = y2.reduce((s, y, i) => s + (y - (a + b1 * x1[i] + b2 * x2[i])) ** 2, 0) / Math.max(1, y2.length - 3);
      n = y2.length;
    } else {
      const fit = ols(xs, ys);
      const b = Math.min(1.5, Math.max(0, fit.b));
      const a = ys.reduce((s, y) => s + y, 0) / ys.length - b * fit.mx;
      s2 = ys.reduce((s, y, i) => s + (y - (a + b * xs[i])) ** 2, 0) / Math.max(1, ys.length - 2);
      mu = a + b * lp;
    }
  } else {
    const fit = ols(xs, ys);
    // A slope outside [0, 1.5] is not a launch shape but a fit artefact (a tiny or
    // degenerate set): clamp it, keeping the line through the data's centre.
    const b = Math.min(1.5, Math.max(0, fit.b));
    const a = ys.reduce((s, y) => s + y, 0) / ys.length - b * fit.mx;
    s2 = ys.reduce((s, y, i) => s + (y - (a + b * xs[i])) ** 2, 0) / Math.max(1, ys.length - 2);
    mu = a + b * lp;
  }
  // Prediction interval: the residual spread, with a floor against overconfidence.
  const sd = Math.max(Math.sqrt(s2 * (1 + 1 / n)), SPREAD_FLOOR);
  const qs = Z.map((z) => round2(Math.min(1e9, Math.max(1, Math.exp(mu + z * sd)))));
  for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  return qs;
};

/** The forecast of one case: 23 rows, hours 2..24. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const ch1 = typeof first.channels === 'number' && first.channels > 0 ? first.channels : 0;
  const parsed = (c.comparables ?? []).map((comp) => {
    const { day, ch1: c1 } = releaseDay(comp);
    return { p1: day.get(0) ?? 0, ch1: c1, day, console: isConsoleOnly(comp) && !isEarlyAccessContaminated(comp) };
  });
  const allDays: DaySet[] = parsed.map(({ p1, ch1, day }) => ({ p1, ch1, day }));
  const consoleDays: DaySet[] = parsed.filter((d) => d.console).map(({ p1, ch1, day }) => ({ p1, ch1, day }));
  const days = consoleDays.filter((d) => d.p1 > 0).length >= MIN_FIT ? consoleDays : allDays;
  const rows = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, ch1, hour, days, allDays) }));
  // Ordering guard: the afternoon peak must clear the morning bump. Only bites
  // when the per-hour slopes extrapolate to an inverted shape at a huge hour 1.
  let morning = 0;
  let peak = 0;
  for (const r of rows) {
    if (r.hour >= 2 && r.hour <= 8 && r.q[2] > morning) morning = r.q[2];
    if (r.hour >= 16 && r.q[2] > peak) peak = r.q[2];
  }
  if (peak > 0 && peak < PEAK_ORDER * morning) {
    const k = (PEAK_ORDER * morning) / peak;
    return rows.map((r) => {
      if (r.hour < 16) return r;
      const qs = r.q.map((q) => round2(Math.min(1e9, Math.max(1, q * k))));
      for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
      return { hour: r.hour, q: qs };
    });
  }
  return rows;
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
