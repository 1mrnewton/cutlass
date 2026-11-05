//! The spatially coded image of a VP8L stream (RFC 9649 section 5):
//! backward references, the color cache, groups of prefix codes chosen
//! per tile, and the codes themselves.

use super::bits::BitWriter;
use super::codes::{self, Code, write_code};
use super::decode::{
    CACHE_HASH, DISTANCE_MAP, MAX_CACHE_BITS, NUM_DISTANCE_CODES, NUM_LENGTH_CODES, NUM_LITERALS,
    subsample,
};
use super::huffman::MAX_CODE_LENGTH;

/// Longest backward reference: 24 length codes reach 4096.
const MAX_LENGTH: usize = 4096;

/// Farthest backward reference: a distance code reaches 2^20, and the
/// first 120 codes are the neighbourhood map.
const MAX_DISTANCE: usize = (1 << 20) - 120;

/// Most tiles of the group image; past this the tiles grow.
const MAX_GROUP_TILES: usize = 2600;

/// How hard the entropy coder works.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Effort {
    /// Hash-chain candidates per position; 0 looks at the left pixel and
    /// the pixel above only.
    pub chain: usize,
    /// Distance limit of the chain search.
    pub window: usize,
    /// Look one pixel ahead before taking a reference.
    pub lazy: bool,
    /// Try each color cache size and keep the cheapest.
    pub cache_search: bool,
    /// Tile size of the group image, `None` for one group.
    pub group_bits: Option<u32>,
    /// Rounds of cost-based parsing after the greedy one.
    pub optimal: usize,
}

impl Effort {
    /// Small images inside a stream: transform data, palettes, groups.
    pub(crate) const SUB_IMAGE: Self = Self {
        chain: 16,
        window: 1 << 16,
        lazy: false,
        cache_search: true,
        group_bits: None,
        optimal: 0,
    };
}

#[derive(Clone, Copy, Debug)]
enum Token {
    Literal(u32),
    Cache(u16),
    Copy { length: u16, code: u32 },
}

/// A length or distance code: the prefix symbol, and its extra bits.
#[inline]
fn prefix(value: u32) -> (usize, u32, u32) {
    let d = value - 1;
    if d < 4 {
        return (d as usize, 0, 0);
    }
    let high = 31 - d.leading_zeros();
    let second = (d >> (high - 1)) & 1;
    let extra = high - 1;
    ((2 * high + second) as usize, extra, d & ((1 << extra) - 1))
}

/// Distance code of a pixel distance: the neighbourhood map when it holds
/// the distance, otherwise the distance plus 120.
struct DistanceCodes {
    /// Code of each short distance, 0 where the map has none.
    near: Vec<u32>,
}

impl DistanceCodes {
    fn new(width: usize) -> Self {
        let mut near = Vec::new();
        for (index, &(dx, dy)) in DISTANCE_MAP.iter().enumerate() {
            let distance = i64::from(dy) * width as i64 + i64::from(dx);
            let distance = distance.max(1) as usize;
            if distance >= near.len() {
                near.resize(distance + 1, 0);
            }
            if near[distance] == 0 {
                near[distance] = index as u32 + 1;
            }
        }
        Self { near }
    }

    #[inline]
    fn code(&self, distance: usize) -> u32 {
        match self.near.get(distance) {
            Some(&code) if code != 0 => code,
            _ => distance as u32 + 120,
        }
    }
}

/// One group's symbol counts.
#[derive(Clone)]
pub(crate) struct Histogram {
    green: Vec<u32>,
    red: [u32; 256],
    blue: [u32; 256],
    alpha: [u32; 256],
    distance: [u32; NUM_DISTANCE_CODES],
}

impl Histogram {
    fn new(cache_bits: u32) -> Self {
        let cache = if cache_bits > 0 { 1 << cache_bits } else { 0 };
        Self {
            green: vec![0; NUM_LITERALS + NUM_LENGTH_CODES + cache],
            red: [0; 256],
            blue: [0; 256],
            alpha: [0; 256],
            distance: [0; NUM_DISTANCE_CODES],
        }
    }

    #[inline]
    fn add(&mut self, token: Token) {
        match token {
            Token::Literal(argb) => {
                self.green[((argb >> 8) & 0xff) as usize] += 1;
                self.red[((argb >> 16) & 0xff) as usize] += 1;
                self.blue[(argb & 0xff) as usize] += 1;
                self.alpha[(argb >> 24) as usize] += 1;
            }
            Token::Cache(key) => {
                self.green[NUM_LITERALS + NUM_LENGTH_CODES + usize::from(key)] += 1
            }
            Token::Copy { length, code } => {
                self.green[NUM_LITERALS + prefix(u32::from(length)).0] += 1;
                self.distance[prefix(code).0] += 1;
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        for (a, b) in self.green.iter_mut().zip(&other.green) {
            *a += b;
        }
        for (a, b) in self.red.iter_mut().zip(&other.red) {
            *a += b;
        }
        for (a, b) in self.blue.iter_mut().zip(&other.blue) {
            *a += b;
        }
        for (a, b) in self.alpha.iter_mut().zip(&other.alpha) {
            *a += b;
        }
        for (a, b) in self.distance.iter_mut().zip(&other.distance) {
            *a += b;
        }
    }

    fn is_empty(&self) -> bool {
        self.green.iter().all(|&c| c == 0)
    }

    /// Estimated bits of the five codes and the symbols they code.
    pub(crate) fn cost(&self) -> f64 {
        codes::cost(&self.green)
            + codes::cost(&self.red)
            + codes::cost(&self.blue)
            + codes::cost(&self.alpha)
            + codes::cost(&self.distance)
    }
}

const HASH_BITS: u32 = 18;

/// Finds earlier runs of pixels equal to those at a position: the pixel
/// to the left, the one above, and hash chains over pixel pairs.
struct Matcher<'a> {
    pixels: &'a [u32],
    width: usize,
    head: Vec<usize>,
    chain: Vec<usize>,
    inserted: usize,
    window: usize,
    tries: usize,
    codes: &'a DistanceCodes,
}

impl<'a> Matcher<'a> {
    fn new(pixels: &'a [u32], width: usize, effort: &Effort, codes: &'a DistanceCodes) -> Self {
        let use_chain = effort.chain > 0;
        Self {
            pixels,
            width,
            head: if use_chain {
                vec![usize::MAX; 1 << HASH_BITS]
            } else {
                Vec::new()
            },
            chain: if use_chain {
                vec![usize::MAX; pixels.len()]
            } else {
                Vec::new()
            },
            inserted: 0,
            window: effort.window.min(MAX_DISTANCE),
            tries: effort.chain,
            codes,
        }
    }

    #[inline]
    fn hash(&self, at: usize) -> usize {
        let a = self.pixels[at].wrapping_mul(0x9e37_79b1);
        let b = self.pixels[at + 1].wrapping_mul(0x85eb_ca6b);
        ((a ^ b.rotate_left(13)) >> (32 - HASH_BITS)) as usize
    }

    fn insert_to(&mut self, upto: usize) {
        if self.tries == 0 {
            return;
        }
        while self.inserted < upto && self.inserted + 1 < self.pixels.len() {
            let h = self.hash(self.inserted);
            self.chain[self.inserted] = self.head[h];
            self.head[h] = self.inserted;
            self.inserted += 1;
        }
    }

    #[inline]
    fn match_length(&self, at: usize, from: usize, limit: usize) -> usize {
        self.pixels[at..at + limit]
            .iter()
            .zip(&self.pixels[from..from + limit])
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// `(length, distance)` of the pixel to the left, the one above, and
    /// the longest chain match (the cheaper distance code on a tie), each
    /// when it matches at least one pixel. Earlier positions must have been
    /// inserted.
    fn candidates(&mut self, at: usize, out: &mut Vec<(usize, usize)>) {
        out.clear();
        self.insert_to(at);
        let n = self.pixels.len();
        let limit = MAX_LENGTH.min(n - at);
        let mut longest = 0;
        for distance in [1, self.width] {
            if distance <= at && !out.iter().any(|&(_, d)| d == distance) {
                let length = self.match_length(at, at - distance, limit);
                if length > 0 {
                    out.push((length, distance));
                    longest = longest.max(length);
                }
            }
        }
        if self.tries == 0 || at + 1 >= n || longest >= limit {
            return;
        }
        let mut best = (longest, 0usize);
        let mut candidate = self.head[self.hash(at)];
        let mut tries = self.tries;
        while candidate != usize::MAX && tries > 0 {
            if candidate >= at {
                candidate = self.chain[candidate];
                continue;
            }
            let distance = at - candidate;
            if distance > self.window {
                break;
            }
            tries -= 1;
            let probe = best.0.min(limit - 1);
            if self.pixels[candidate + probe] == self.pixels[at + probe] {
                let length = self.match_length(at, candidate, limit);
                if length > best.0
                    || (length == best.0
                        && best.1 != 0
                        && self.codes.code(distance) < self.codes.code(best.1))
                {
                    best = (length, distance);
                    if length == limit {
                        break;
                    }
                }
            }
            candidate = self.chain[candidate];
        }
        if best.1 != 0 && !out.iter().any(|&(_, d)| d == best.1) {
            out.push(best);
        }
    }

    /// The single best reference: the longest, then the cheapest code.
    fn best(&mut self, at: usize, scratch: &mut Vec<(usize, usize)>) -> (usize, usize) {
        self.candidates(at, scratch);
        let mut best = (0, 0);
        for &(length, distance) in scratch.iter() {
            if length > best.0
                || (length == best.0 && self.codes.code(distance) < self.codes.code(best.1))
            {
                best = (length, distance);
            }
        }
        best
    }
}

/// Backward references, greedy with optional one-pixel lookahead.
fn references(pixels: &[u32], width: usize, effort: &Effort, codes: &DistanceCodes) -> Vec<Token> {
    let n = pixels.len();
    let mut tokens = Vec::with_capacity(n / 2);
    let mut matcher = Matcher::new(pixels, width, effort, codes);
    let mut scratch = Vec::with_capacity(4);
    let min_length = 3;
    let mut at = 0;
    while at < n {
        let (length, distance) = matcher.best(at, &mut scratch);
        let mut take = length >= min_length;
        if take && effort.lazy && length < 64 && at + 1 < n {
            let (next, _) = matcher.best(at + 1, &mut scratch);
            if next > length + 1 {
                take = false;
            }
        }
        if take {
            tokens.push(Token::Copy {
                length: length as u16,
                code: codes.code(distance),
            });
            at += length;
        } else {
            tokens.push(Token::Literal(pixels[at]));
            at += 1;
        }
    }
    tokens
}

/// Bits per symbol under a histogram, for the parse below. Unseen symbols
/// cost a little more than the rarest seen one.
fn symbol_costs(counts: &[u32]) -> Vec<f32> {
    let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return vec![8.0; counts.len()];
    }
    let log_total = (total as f64).log2();
    counts
        .iter()
        .map(|&c| {
            if c == 0 {
                (log_total + 2.0) as f32
            } else {
                (log_total - f64::from(c).log2()).max(0.0) as f32
            }
        })
        .collect()
}

/// The cheapest parse under the cost model of `model`, by dynamic
/// programming over positions. Every pixel enters the color cache whatever
/// codes it, so the cache contents at a position do not depend on the
/// parse, and a cache hit is just a cheaper literal.
fn optimal(
    pixels: &[u32],
    width: usize,
    effort: &Effort,
    codes: &DistanceCodes,
    model: &Histogram,
    cache_bits: u32,
) -> Vec<Token> {
    let n = pixels.len();
    let green = symbol_costs(&model.green);
    let red = symbol_costs(&model.red);
    let blue = symbol_costs(&model.blue);
    let alpha = symbol_costs(&model.alpha);
    let distance = symbol_costs(&model.distance);
    let length_cost = |length: usize| {
        let (symbol, extra, _) = prefix(length as u32);
        green[NUM_LITERALS + symbol] + extra as f32
    };
    let distance_cost = |code: u32| {
        let (symbol, extra, _) = prefix(code);
        distance[symbol] + extra as f32
    };
    // Lengths up to this are each tried; a longer match is tried whole.
    const SHORT: usize = 16;
    let short_lengths: Vec<f32> = (0..=SHORT)
        .map(|l| if l == 0 { 0.0 } else { length_cost(l) })
        .collect();
    let mut cost = vec![f32::INFINITY; n + 1];
    // (length, distance) of the step into each position; length 1 with
    // distance 0 is a literal.
    let mut step = vec![(0u16, 0u32); n + 1];
    cost[0] = 0.0;
    let mut cache = if cache_bits > 0 {
        vec![0u32; 1 << cache_bits]
    } else {
        Vec::new()
    };
    let mut matcher = Matcher::new(pixels, width, effort, codes);
    let mut found = Vec::with_capacity(4);
    for at in 0..n {
        let here = cost[at];
        let argb = pixels[at];
        let mut literal = green[((argb >> 8) & 0xff) as usize]
            + red[((argb >> 16) & 0xff) as usize]
            + blue[(argb & 0xff) as usize]
            + alpha[(argb >> 24) as usize];
        if cache_bits > 0 {
            let key = cache_key(argb, cache_bits);
            if cache[key] == argb {
                literal = literal.min(green[NUM_LITERALS + NUM_LENGTH_CODES + key]);
            }
            cache[key] = argb;
        }
        if here + literal < cost[at + 1] {
            cost[at + 1] = here + literal;
            step[at + 1] = (1, 0);
        }
        matcher.candidates(at, &mut found);
        for &(length, dist) in &found {
            let code = codes.code(dist);
            let base = here + distance_cost(code);
            for l in 1..=length.min(SHORT) {
                let total = base + short_lengths[l];
                if total < cost[at + l] {
                    cost[at + l] = total;
                    step[at + l] = (l as u16, dist as u32);
                }
            }
            if length > SHORT {
                let total = base + length_cost(length);
                if total < cost[at + length] {
                    cost[at + length] = total;
                    step[at + length] = (length as u16, dist as u32);
                }
            }
        }
    }
    let mut tokens = Vec::new();
    let mut at = n;
    while at > 0 {
        let (length, dist) = step[at];
        let length = usize::from(length.max(1));
        at -= length;
        if dist == 0 {
            tokens.push(Token::Literal(pixels[at]));
        } else {
            tokens.push(Token::Copy {
                length: length as u16,
                code: codes.code(dist as usize),
            });
        }
    }
    tokens.reverse();
    tokens
}

#[inline]
fn cache_key(argb: u32, bits: u32) -> usize {
    (argb.wrapping_mul(CACHE_HASH) >> (32 - bits)) as usize
}

/// Replace literals that the color cache already holds.
fn apply_cache(tokens: &mut [Token], pixels: &[u32], bits: u32) {
    if bits == 0 {
        return;
    }
    let mut cache = vec![0u32; 1 << bits];
    let mut at = 0;
    for token in tokens.iter_mut() {
        match *token {
            Token::Literal(argb) => {
                let key = cache_key(argb, bits);
                if cache[key] == argb {
                    *token = Token::Cache(key as u16);
                } else {
                    cache[key] = argb;
                }
                at += 1;
            }
            Token::Cache(_) => at += 1,
            Token::Copy { length, .. } => {
                for &argb in &pixels[at..at + usize::from(length)] {
                    cache[cache_key(argb, bits)] = argb;
                }
                at += usize::from(length);
            }
        }
    }
}

/// The cache size with the cheapest single-group cost.
fn choose_cache_bits(tokens: &[Token], pixels: &[u32], search: bool) -> u32 {
    let candidates: Vec<u32> = if search {
        (0..=MAX_CACHE_BITS).collect()
    } else {
        vec![0, 6, 10]
    };
    let mut best = (f64::INFINITY, 0);
    for bits in candidates {
        let mut trial = tokens.to_vec();
        apply_cache(&mut trial, pixels, bits);
        let mut histogram = Histogram::new(bits);
        for &token in &trial {
            histogram.add(token);
        }
        let cost = histogram.cost();
        if cost < best.0 {
            best = (cost, bits);
        }
    }
    best.1
}

/// Tile of every token start, for the group image.
fn token_tiles(tokens: &[Token], width: usize, bits: u32) -> Vec<usize> {
    let tiles_wide = subsample(width as u32, bits) as usize;
    let mut at = 0;
    tokens
        .iter()
        .map(|token| {
            let tile = ((at / width) >> bits) * tiles_wide + ((at % width) >> bits);
            at += match token {
                Token::Copy { length, .. } => usize::from(*length),
                _ => 1,
            };
            tile
        })
        .collect()
}

/// Group the tiles: merge histograms while merging lowers the estimated
/// size, then send each tile to the group that codes it cheapest.
fn cluster(tiles: &[Histogram]) -> Vec<usize> {
    let mut groups: Vec<(Histogram, f64)> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (tile, histogram) in tiles.iter().enumerate() {
        if !histogram.is_empty() {
            groups.push((histogram.clone(), histogram.cost()));
            members.push(vec![tile]);
        }
    }
    if groups.is_empty() {
        return vec![0; tiles.len()];
    }
    // Tiles alike in entropy go to one bin first, so the pairwise search
    // runs on a few hundred groups at most.
    const BINS: usize = 192;
    if groups.len() > BINS {
        let mut binned: Vec<Option<usize>> = vec![None; BINS];
        let mut merged: Vec<(Histogram, f64)> = Vec::new();
        let mut merged_members: Vec<Vec<usize>> = Vec::new();
        let costs: Vec<f64> = groups.iter().map(|(_, cost)| *cost).collect();
        let (low, high) = costs
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(l, h), &c| (l.min(c), h.max(c)));
        for ((histogram, cost), tile_members) in groups.into_iter().zip(members) {
            let bin = if high > low {
                (((cost - low) / (high - low)) * (BINS - 1) as f64) as usize
            } else {
                0
            };
            match binned[bin] {
                Some(index) => {
                    merged[index].0.merge(&histogram);
                    merged[index].1 = merged[index].0.cost();
                    merged_members[index].extend(tile_members);
                }
                None => {
                    binned[bin] = Some(merged.len());
                    merged.push((histogram, cost));
                    merged_members.push(tile_members);
                }
            }
        }
        groups = merged;
        members = merged_members;
    }
    // Greedy pairwise merging. `gains[i][j]` (i < j) is the estimated
    // saving of merging groups i and j; only the merged group's row and
    // column change after a merge.
    let pair_gain = |a: &(Histogram, f64), b: &(Histogram, f64)| {
        let mut combined = a.0.clone();
        combined.merge(&b.0);
        a.1 + b.1 - combined.cost()
    };
    let k = groups.len();
    let mut gains = vec![vec![f64::NEG_INFINITY; k]; k];
    for i in 0..k {
        for j in i + 1..k {
            gains[i][j] = pair_gain(&groups[i], &groups[j]);
        }
    }
    let mut alive = vec![true; k];
    loop {
        let mut best = (0.0f64, usize::MAX, usize::MAX);
        for i in 0..k {
            if !alive[i] {
                continue;
            }
            for j in i + 1..k {
                if alive[j] && gains[i][j] > best.0 {
                    best = (gains[i][j], i, j);
                }
            }
        }
        let (_, i, j) = best;
        if i == usize::MAX {
            break;
        }
        let other = groups[j].0.clone();
        groups[i].0.merge(&other);
        groups[i].1 = groups[i].0.cost();
        let moved = std::mem::take(&mut members[j]);
        members[i].extend(moved);
        alive[j] = false;
        for m in 0..k {
            if m != i && alive[m] {
                let gain = pair_gain(&groups[i], &groups[m]);
                if m < i {
                    gains[m][i] = gain;
                } else {
                    gains[i][m] = gain;
                }
            }
        }
    }
    let groups: Vec<(Histogram, f64)> = groups
        .into_iter()
        .zip(&alive)
        .filter_map(|(group, &alive)| alive.then_some(group))
        .collect();
    let members: Vec<Vec<usize>> = members
        .into_iter()
        .zip(&alive)
        .filter_map(|(group, &alive)| alive.then_some(group))
        .collect();
    let mut assignment = vec![0usize; tiles.len()];
    for (group, tile_members) in members.iter().enumerate() {
        for &tile in tile_members {
            assignment[tile] = group;
        }
    }
    // Send each tile where it adds the least.
    if groups.len() > 1 {
        for (tile, histogram) in tiles.iter().enumerate() {
            if histogram.is_empty() {
                continue;
            }
            let mut best = (f64::INFINITY, assignment[tile]);
            for (group, (group_histogram, group_cost)) in groups.iter().enumerate() {
                let mut combined = group_histogram.clone();
                combined.merge(histogram);
                let added = combined.cost() - group_cost;
                if added < best.0 {
                    best = (added, group);
                }
            }
            assignment[tile] = best.1;
        }
    }
    // Number the groups that are used, in order of first use.
    let mut renumber = vec![usize::MAX; groups.len()];
    let mut next = 0;
    for slot in &mut assignment {
        if renumber[*slot] == usize::MAX {
            renumber[*slot] = next;
            next += 1;
        }
        *slot = renumber[*slot];
    }
    assignment
}

/// Tile bits of the group image for an effort's choice, grown until the
/// image has at most [`MAX_GROUP_TILES`] tiles.
fn group_bits(width: usize, height: usize, bits: u32) -> u32 {
    let mut bits = bits.clamp(2, 9);
    while bits < 9
        && (subsample(width as u32, bits) as usize) * (subsample(height as u32, bits) as usize)
            > MAX_GROUP_TILES
    {
        bits += 1;
    }
    bits
}

struct GroupCodes {
    green: Code,
    red: Code,
    blue: Code,
    alpha: Code,
    distance: Code,
}

fn write_group(writer: &mut BitWriter, histogram: &Histogram) -> GroupCodes {
    let limit = MAX_CODE_LENGTH as u8;
    GroupCodes {
        green: write_code(writer, &codes::lengths(&histogram.green, limit)),
        red: write_code(writer, &codes::lengths(&histogram.red, limit)),
        blue: write_code(writer, &codes::lengths(&histogram.blue, limit)),
        alpha: write_code(writer, &codes::lengths(&histogram.alpha, limit)),
        distance: write_code(writer, &codes::lengths(&histogram.distance, limit)),
    }
}

/// Write one entropy-coded image. Only the main image may use a group
/// image (`main`).
pub(crate) fn write_image(
    writer: &mut BitWriter,
    pixels: &[u32],
    width: usize,
    height: usize,
    effort: &Effort,
    main: bool,
) {
    let codes = DistanceCodes::new(width);
    let mut tokens = references(pixels, width, effort, &codes);
    let cache_bits = choose_cache_bits(&tokens, pixels, effort.cache_search);
    apply_cache(&mut tokens, pixels, cache_bits);
    for _ in 0..effort.optimal {
        let mut model = Histogram::new(cache_bits);
        for &token in &tokens {
            model.add(token);
        }
        let mut refined = optimal(pixels, width, effort, &codes, &model, cache_bits);
        apply_cache(&mut refined, pixels, cache_bits);
        tokens = refined;
    }
    if cache_bits > 0 {
        writer.write(1, 1);
        writer.write(cache_bits, 4);
    } else {
        writer.write(0, 1);
    }

    let mut assignment = vec![0usize; tokens.len()];
    let mut group_count = 1;
    if main {
        let grouped = effort.group_bits.and_then(|bits| {
            let bits = group_bits(width, height, bits);
            let tiles_wide = subsample(width as u32, bits) as usize;
            let tiles_high = subsample(height as u32, bits) as usize;
            let token_tile = token_tiles(&tokens, width, bits);
            let mut tiles = vec![Histogram::new(cache_bits); tiles_wide * tiles_high];
            for (&token, &tile) in tokens.iter().zip(&token_tile) {
                tiles[tile].add(token);
            }
            let tile_group = cluster(&tiles);
            let count = tile_group.iter().max().map_or(1, |&max| max + 1);
            (count > 1).then_some((bits, tiles_wide, tiles_high, token_tile, tile_group, count))
        });
        if let Some((bits, tiles_wide, tiles_high, token_tile, tile_group, count)) = grouped {
            writer.write(1, 1);
            writer.write(bits - 2, 3);
            let image: Vec<u32> = tile_group
                .iter()
                .map(|&group| (group as u32) << 8)
                .collect();
            write_image(
                writer,
                &image,
                tiles_wide,
                tiles_high,
                &Effort::SUB_IMAGE,
                false,
            );
            for (slot, &tile) in assignment.iter_mut().zip(&token_tile) {
                *slot = tile_group[tile];
            }
            group_count = count;
        } else {
            writer.write(0, 1);
        }
    }

    let mut histograms = vec![Histogram::new(cache_bits); group_count];
    for (&token, &group) in tokens.iter().zip(&assignment) {
        histograms[group].add(token);
    }
    let codes: Vec<GroupCodes> = histograms
        .iter()
        .map(|histogram| write_group(writer, histogram))
        .collect();
    for (&token, &group) in tokens.iter().zip(&assignment) {
        let group = &codes[group];
        match token {
            Token::Literal(argb) => {
                group.green.put(writer, ((argb >> 8) & 0xff) as usize);
                group.red.put(writer, ((argb >> 16) & 0xff) as usize);
                group.blue.put(writer, (argb & 0xff) as usize);
                group.alpha.put(writer, (argb >> 24) as usize);
            }
            Token::Cache(key) => {
                group
                    .green
                    .put(writer, NUM_LITERALS + NUM_LENGTH_CODES + usize::from(key));
            }
            Token::Copy { length, code } => {
                let (symbol, extra, value) = prefix(u32::from(length));
                group.green.put(writer, NUM_LITERALS + symbol);
                writer.write(value, extra);
                let (symbol, extra, value) = prefix(code);
                group.distance.put(writer, symbol);
                writer.write(value, extra);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::prefix;

    #[test]
    fn prefix_codes_invert_the_reader() {
        for value in 1..5000u32 {
            let (symbol, extra, bits) = prefix(value);
            let symbol = symbol as u32;
            let decoded = if symbol < 4 {
                symbol + 1
            } else {
                let count = (symbol - 2) >> 1;
                assert_eq!(count, extra);
                ((2 + (symbol & 1)) << count) + bits + 1
            };
            assert_eq!(decoded, value);
        }
    }
}
