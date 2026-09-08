// The sampler's kernels: the gather that takes a row's token from what the device last sampled
// for its request slot, and the sample that draws each row's next token from its logits under
// the slot's record and leaves it on the device. Every launcher takes the caller's stream and
// returns the launch status. One block serves one row, and every reduction inside it runs in a
// fixed order, so a row's token depends on its logits, its record and nothing else: not on the
// batch it sits in, not on the slot it occupies, and not on which launch computed it.
//
// The sample is bounded by a device-resident count of the batch's live rows rather than by its
// launch: a row at or past that count returns before it reads anything of its own, leaving that
// row's token, its slot's last sampled token and its slot's draw counter as they are. A launch
// over more rows than the batch has live therefore samples the live rows and no others, and one
// over a count of zero samples nothing at all.
//
// The sample kernel is the host reference (atoma-engine's sampling::reference) step for step:
// logits are ordered by a key monotone in their value with a not-a-number last; a greedy record
// takes the largest and draws nothing; a drawn record admits the top_k largest with every tie of
// the k-th, weights each admitted token by exp((logit - max) / temperature) truncated to 32-bit
// fixed point, keeps the heaviest tokens whose mass reaches top_p of the total with every tie of
// the last, and picks a token in index order by its share of what is left under one 64-bit
// Philox uniform. The selections are radix walks over histograms where the reference sorts.

#include <cuda_runtime.h>
#include <math.h>

#include <cstddef>
#include <cstdint>

// One request slot's record, as atoma-engine's sampling::record lays it out.
struct SlotRecord {
    float temperature;
    float top_p;
    uint32_t top_k;
    uint32_t draws;
    uint64_t seed;
};
static_assert(sizeof(SlotRecord) == 24, "the record is 24 bytes, as the host declares it");

// The sample launch's arguments, as atoma-kernels' ffi::SampleArgs lays them out.
struct SampleArgs {
    const float* logits;
    const int32_t* row_slots;
    SlotRecord* records;
    uint32_t* sampled;
    uint32_t* out;
    const uint32_t* live_rows;
    int64_t vocab;
    int64_t n_rows;
};
static_assert(sizeof(SampleArgs) == 64, "the arguments are 64 bytes, as the host declares them");
// Every field at the byte the host holds it to, so a reordering on this side stops the build
// rather than reading one argument as another.
static_assert(offsetof(SampleArgs, logits) == 0, "logits is the host's first argument");
static_assert(offsetof(SampleArgs, row_slots) == 8, "row_slots is the host's second argument");
static_assert(offsetof(SampleArgs, records) == 16, "records is the host's third argument");
static_assert(offsetof(SampleArgs, sampled) == 24, "sampled is the host's fourth argument");
static_assert(offsetof(SampleArgs, out) == 32, "out is the host's fifth argument");
static_assert(offsetof(SampleArgs, live_rows) == 40, "live_rows is the host's sixth argument");
static_assert(offsetof(SampleArgs, vocab) == 48, "vocab is the host's seventh argument");
static_assert(offsetof(SampleArgs, n_rows) == 56, "n_rows is the host's eighth argument");

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
// The block reductions finish in warp zero, which reads one entry per warp with one lane each.
static_assert(kWarps == 32, "one block of kThreads threads must hold exactly one warp of warps");
constexpr int kBins = 256;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
// The weight of the largest logit: one, in 32-bit fixed point.
constexpr double kUnitWeight = 4294967296.0;

// One row of logits.
struct Row {
    const float* logits;
    int64_t vocab;
};

// A candidate for the row's largest logit: its order key and its index.
struct Largest {
    uint32_t key;
    long long index;
};

// What a drawn row keeps: a token is admitted when its key is at least `admit_key`, weighs
// exp((logit - max) / temperature) in fixed point, and is kept when that weight is at least
// `keep_at_least`.
struct Kept {
    uint32_t admit_key;
    float max;
    float temperature;
    unsigned long long keep_at_least;
};

// The block's shared memory: the histogram the radix walks bin into, the per-warp partials of
// the reductions and the scan's total, and the digit thread zero chose.
struct Scratch {
    unsigned long long hist[kBins];
    unsigned long long totals[kWarps + 1];
    Largest largest[kWarps];
    uint32_t chosen;
};

// Philox4x32-10, as Random123 defines it and the host computes it.
__device__ __forceinline__ void philox_round(uint32_t c[4], const uint32_t k[2]) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c[0]);
    const uint32_t lo0 = 0xD2511F53u * c[0];
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c[2]);
    const uint32_t lo1 = 0xCD9E8D57u * c[2];
    const uint32_t next0 = hi1 ^ c[1] ^ k[0];
    const uint32_t next2 = hi0 ^ c[3] ^ k[1];
    c[0] = next0;
    c[1] = lo1;
    c[2] = next2;
    c[3] = lo0;
}

// The 64-bit uniform for draw number `draw` under `seed`: the block's first two words for the
// counter (draw, 0, 0, 0) under the seed's two words, the first low.
__device__ uint64_t philox_draw(uint64_t seed, uint32_t draw) {
    uint32_t c[4] = {draw, 0u, 0u, 0u};
    uint32_t k[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    for (int round = 0; round < 10; ++round) {
        philox_round(c, k);
        if (round + 1 < 10) {
            k[0] += 0x9E3779B9u;
            k[1] += 0xBB67AE85u;
        }
    }
    return static_cast<uint64_t>(c[0]) | (static_cast<uint64_t>(c[1]) << 32);
}

// A logit's order among its row: monotone in the value, with a not-a-number below every number.
__device__ __forceinline__ uint32_t order_key(float logit) {
    if (isnan(logit)) {
        logit = -INFINITY;
    }
    const uint32_t bits = __float_as_uint(logit);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// An admitted token's weight in 32-bit fixed point; a not-a-number weighs nothing.
__device__ __forceinline__ unsigned long long weight_q(float logit, float max, float temperature) {
    if (isnan(logit)) {
        return 0ull;
    }
    const float w = expf((logit - max) / temperature);
    return static_cast<unsigned long long>(static_cast<double>(w) * kUnitWeight);
}

// The weight of `logit` when its token is kept, and zero otherwise.
__device__ __forceinline__ unsigned long long kept_weight(float logit, const Kept& kept) {
    if (order_key(logit) < kept.admit_key) {
        return 0ull;
    }
    const unsigned long long weight = weight_q(logit, kept.max, kept.temperature);
    return weight >= kept.keep_at_least ? weight : 0ull;
}

// The larger of two candidates: keys descending, then indices ascending, so on a tie the
// earlier index wins.
struct Larger {
    __device__ __forceinline__ Largest operator()(Largest a, Largest b) const {
        return (b.key > a.key || (b.key == a.key && b.index < a.index)) ? b : a;
    }
};

struct Sum {
    __device__ __forceinline__ unsigned long long operator()(unsigned long long a,
                                                            unsigned long long b) const {
        return a + b;
    }
};

__device__ __forceinline__ unsigned long long shuffle_down(unsigned long long value, int offset) {
    return __shfl_down_sync(kFullMask, value, offset);
}

__device__ __forceinline__ Largest shuffle_down(Largest value, int offset) {
    return {__shfl_down_sync(kFullMask, value.key, offset),
            __shfl_down_sync(kFullMask, value.index, offset)};
}

// Every thread's value combined across the block in a fixed order: within each warp, then across
// the warps' results in warp zero, one per lane. Every thread returns the result.
template <typename T, typename Combine>
__device__ T block_reduce(T value, Combine combine, T* per_warp) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = combine(value, shuffle_down(value, offset));
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        per_warp[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = per_warp[lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = combine(value, shuffle_down(value, offset));
        }
        if (lane == 0) {
            per_warp[0] = value;
        }
    }
    __syncthreads();
    value = per_warp[0];
    __syncthreads();
    return value;
}

// Every thread's exclusive prefix over the block's values in thread order, and the total.
__device__ unsigned long long block_exclusive_scan(unsigned long long value, Scratch& scratch,
                                                   unsigned long long& total) {
    unsigned long long* totals = scratch.totals;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned long long inclusive = value;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned long long other = __shfl_up_sync(kFullMask, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        totals[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        unsigned long long warp_inclusive = totals[lane];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const unsigned long long other = __shfl_up_sync(kFullMask, warp_inclusive, offset);
            if (lane >= offset) {
                warp_inclusive += other;
            }
        }
        totals[lane] = warp_inclusive - totals[lane];
        if (lane == 31) {
            totals[kWarps] = warp_inclusive;
        }
    }
    __syncthreads();
    const unsigned long long exclusive = totals[warp] + inclusive - value;
    total = totals[kWarps];
    __syncthreads();
    return exclusive;
}

// The digit of the bin the walk from the top crosses `remaining` in; `remaining` is what is left
// to reach once the bins above are taken off. Thread zero walks and keeps `remaining` for the
// next digit; the other threads' copies are never read again, and every thread reads the digit
// from shared memory.
__device__ uint32_t crossing_bin(Scratch& scratch, unsigned long long& remaining) {
    if (threadIdx.x == 0) {
        uint32_t digit = 0;
        for (int bin = kBins - 1; bin >= 0; --bin) {
            if (remaining <= scratch.hist[bin]) {
                digit = static_cast<uint32_t>(bin);
                break;
            }
            remaining -= scratch.hist[bin];
        }
        scratch.chosen = digit;
    }
    __syncthreads();
    const uint32_t digit = scratch.chosen;
    __syncthreads();
    return digit;
}

__device__ __forceinline__ void clear_histogram(Scratch& scratch) {
    for (int bin = threadIdx.x; bin < kBins; bin += kThreads) {
        scratch.hist[bin] = 0;
    }
    __syncthreads();
}

// The k-th largest key of the row: after four digit walks the prefix is that key exactly. Every
// token whose key is at least it is admitted, which is the k largest and every tie of the k-th.
__device__ uint32_t kth_largest_key(const Row& row, uint32_t k, Scratch& scratch) {
    uint32_t prefix = 0;
    unsigned long long remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        clear_histogram(scratch);
        const uint32_t prefix_mask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int64_t i = threadIdx.x; i < row.vocab; i += kThreads) {
            const uint32_t key = order_key(row.logits[i]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&scratch.hist[(key >> shift) & 0xFFu], 1ull);
            }
        }
        __syncthreads();
        prefix |= crossing_bin(scratch, remaining) << shift;
    }
    return prefix;
}

// The key every admitted token's key is at least: the top_k-th largest when top_k filters, and
// zero, which admits every token, when it is unset or covers the row.
__device__ uint32_t admit_key(const Row& row, const SlotRecord& record, Scratch& scratch) {
    const uint32_t vocab =
        row.vocab > 0xFFFFFFFFll ? 0xFFFFFFFFu : static_cast<uint32_t>(row.vocab);
    const bool filters = record.top_k != 0 && record.top_k < vocab;
    return filters ? kth_largest_key(row, record.top_k, scratch) : 0u;
}

// The mass of what the row keeps.
__device__ unsigned long long kept_mass(const Row& row, const Kept& kept, Scratch& scratch) {
    unsigned long long local = 0;
    for (int64_t i = threadIdx.x; i < row.vocab; i += kThreads) {
        local += kept_weight(row.logits[i], kept);
    }
    return block_reduce(local, Sum{}, scratch.totals);
}

// The mass top_p asks to keep of `mass`: that share rounded up, and at least one.
__device__ __forceinline__ unsigned long long target_mass(float top_p, unsigned long long mass) {
    const double target = ceil(static_cast<double>(top_p) * static_cast<double>(mass));
    const unsigned long long rounded = static_cast<unsigned long long>(target);
    return rounded < 1 ? 1 : rounded;
}

// The weight of the token the walk down the kept weights crosses `target` at: after five digit
// walks over the 33-bit weights the prefix is that weight exactly. Every kept token weighing at
// least it stays kept, which is the heaviest reaching the target and every tie of the last.
__device__ unsigned long long crossing_weight(const Row& row, const Kept& kept,
                                              unsigned long long target, Scratch& scratch) {
    unsigned long long prefix = 0;
    unsigned long long remaining = target;
    for (int shift = 32; shift >= 0; shift -= 8) {
        clear_histogram(scratch);
        const unsigned long long prefix_mask = shift == 32 ? 0ull : (~0ull << (shift + 8));
        for (int64_t i = threadIdx.x; i < row.vocab; i += kThreads) {
            const unsigned long long weight = kept_weight(row.logits[i], kept);
            if ((weight & prefix_mask) == prefix) {
                atomicAdd(&scratch.hist[(weight >> shift) & 0xFFull], weight);
            }
        }
        __syncthreads();
        prefix |= static_cast<unsigned long long>(crossing_bin(scratch, remaining)) << shift;
    }
    return prefix;
}

// The first largest logit of the row. A thread's run is in index order, so on a tie the earlier
// index wins by staying; the sentinel index sits past the row and loses every tie.
__device__ Largest row_largest(const Row& row, Scratch& scratch) {
    Largest best = {order_key(-INFINITY), row.vocab};
    for (int64_t i = threadIdx.x; i < row.vocab; i += kThreads) {
        best = Larger{}(best, Largest{order_key(row.logits[i]), i});
    }
    return block_reduce(best, Larger{}, scratch.largest);
}

// The tokens this thread owns for the pick: a contiguous run, so the block's threads in order
// cover the row in index order.
__device__ __forceinline__ void owned_range(int64_t vocab, int64_t& begin, int64_t& end) {
    const int64_t thread = threadIdx.x;
    begin = (thread * vocab) / kThreads;
    end = ((thread + 1) * vocab) / kThreads;
}

// The kept token the uniform's point falls on, walking the row in index order by weight, from
// the one thread whose run holds it; every other thread returns a negative index.
__device__ long long picked(const Row& row, const Kept& kept, uint64_t uniform, Scratch& scratch) {
    int64_t begin;
    int64_t end;
    owned_range(row.vocab, begin, end);
    unsigned long long local = 0;
    for (int64_t i = begin; i < end; ++i) {
        local += kept_weight(row.logits[i], kept);
    }
    unsigned long long total;
    const unsigned long long exclusive = block_exclusive_scan(local, scratch, total);
    // The uniform's share of the total: a total a few units apart from the host's moves it by no
    // more than those units, where a remainder would move it by the quotient.
    const unsigned long long point = __umul64hi(uniform, total);
    if (point < exclusive || exclusive + local <= point) {
        return -1;
    }
    unsigned long long cumulative = exclusive;
    for (int64_t i = begin; i < end; ++i) {
        cumulative += kept_weight(row.logits[i], kept);
        if (cumulative > point) {
            return i;
        }
    }
    return -1;
}

__global__ void __launch_bounds__(kThreads) sample_kernel(SampleArgs args) {
    __shared__ Scratch scratch;
    const int64_t row_index = blockIdx.x;
    // The whole block shares the row, so it returns whole, before the first reduction and
    // before the row's slot is read: only a live row's slot is the caller's to have made valid.
    if (row_index >= static_cast<int64_t>(*args.live_rows)) {
        return;
    }
    const Row row = {args.logits + row_index * args.vocab, args.vocab};
    const int32_t slot = args.row_slots[row_index];
    const SlotRecord record = args.records[slot];

    const Largest largest = row_largest(row, scratch);
    const float max = row.logits[largest.index];
    // A greedy record takes the largest; so does a row with no finite logit to draw from.
    if (record.temperature == 0.0f || !isfinite(max)) {
        if (threadIdx.x == 0) {
            const uint32_t token = static_cast<uint32_t>(largest.index);
            args.out[row_index] = token;
            args.sampled[slot] = token;
        }
        return;
    }

    Kept kept = {admit_key(row, record, scratch), max, record.temperature, 1ull};
    if (record.top_p < 1.0f) {
        const unsigned long long target = target_mass(record.top_p, kept_mass(row, kept, scratch));
        kept.keep_at_least = crossing_weight(row, kept, target, scratch);
    }
    const long long token = picked(row, kept, philox_draw(record.seed, record.draws), scratch);
    if (token >= 0) {
        args.out[row_index] = static_cast<uint32_t>(token);
        args.sampled[slot] = static_cast<uint32_t>(token);
        args.records[slot].draws = record.draws + 1;
    }
}

__global__ void gather_kernel(uint32_t* token_ids, const int32_t* __restrict__ gather_slots,
                              const uint32_t* __restrict__ sampled, int64_t n_rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int32_t slot = gather_slots[row];
    if (slot < 0) {
        return;
    }
    token_ids[row] = sampled[slot];
}

}  // namespace

extern "C" cudaError_t sampler_sample_f32(const SampleArgs* args, cudaStream_t stream) {
    if (args->n_rows == 0 || args->vocab == 0) {
        return cudaSuccess;
    }
    sample_kernel<<<static_cast<unsigned int>(args->n_rows), kThreads, 0, stream>>>(*args);
    return cudaGetLastError();
}

extern "C" cudaError_t sampler_gather_u32(void* token_ids, const void* gather_slots,
                                          const void* sampled, int64_t n_rows,
                                          cudaStream_t stream) {
    if (n_rows == 0) {
        return cudaSuccess;
    }
    constexpr int kGatherThreads = 256;
    const int64_t blocks = (n_rows + kGatherThreads - 1) / kGatherThreads;
    gather_kernel<<<static_cast<unsigned int>(blocks), kGatherThreads, 0, stream>>>(
        static_cast<uint32_t*>(token_ids), static_cast<const int32_t*>(gather_slots),
        static_cast<const uint32_t*>(sampled), n_rows);
    return cudaGetLastError();
}
