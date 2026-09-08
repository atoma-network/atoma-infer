//! The device sampler against its host reference, on a device.
//!
//! Needs a device and the CUDA toolkit; no checkpoint and no model. The sampler is built over
//! synthetic logits uploaded to a buffer of its own, so what is measured is the kernels and
//! nothing else: greedy rows must match the reference exactly; drawn rows must match it token for
//! token but for a bounded few that rounding at a cutoff can move, and those must still be tokens
//! the reference keeps; a seeded request must produce the same tokens whatever batch it is
//! sampled in, wherever in that batch it sits and whatever slot it occupies; the gather must
//! overwrite a decoding row's token with what its slot sampled, leave a fresh slot's row to the
//! host, and cover the token rows the caller stated rather than the rows the step samples; a
//! step staged where the caller says must sample under the row slots uploaded from
//! there; the eager upload must refuse a step staged where the caller says; a sample with no
//! copy described behind it must leave nothing to wait on; and a view of other rows than the
//! staged step's, or of another dtype, must be refused by name.
//!
//! Run through `scripts/sampler-parity.sh`.

#![cfg(feature = "cuda")]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout)]

use std::slice;
use std::sync::Arc;

use atoma_core::dispatch::{DispatchDecision, EagerReason};
use atoma_core::request::SamplingParams;
use atoma_core::step::{CommandEntry, StepCommand};
use atoma_core::types::{
    BlockId, RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount,
};
use atoma_engine::batch::BatchLayout;
use atoma_engine::decode::staging::SamplerArrays;
use atoma_engine::device::sampler::{ArraysIn, DeviceSampler, SamplerError};
use atoma_engine::readback::ReadbackError;
use atoma_engine::sampling::record::SlotRecord;
use atoma_engine::sampling::reference;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::session::{Allocation, Descriptor};
use atoma_runtime::tensor::{Dtype, Layout, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

/// Small enough to upload per step, wide enough for the kernel's block-wide reductions to span
/// several strides.
const VOCAB: usize = 4096;
const SLOTS: usize = 16;
const MAX_ROWS: RequestCount = RequestCount::new(8).expect("nonzero");
const BLOCK_SIZE: TokenCount = TokenCount::new(16).expect("nonzero");
/// Draws per distribution when a frequency is measured.
const DRAWS: usize = 4096;

/// A small deterministic generator, so a run is reproducible from its seed alone.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A logit in [-8, 8).
    fn logit(&mut self) -> f32 {
        let thousandths = u16::try_from(self.next() % 16_000).expect("below sixteen thousand");
        f32::from(thousandths) / 1000.0 - 8.0
    }
}

/// The device, its stream and the sampler under test, with a buffer for the logits, one for the
/// token ids a step would copy in, and one for each of the sampler's two per-step arrays where a
/// decode step's copy-in would put them, each viewed at its full rows as a decode step's views
/// are minted.
struct Rig {
    sampler: DeviceSampler,
    stream: Arc<CudaStream>,
    logits: CudaSlice<f32>,
    token_ids: CudaSlice<u32>,
    row_slots: CudaSlice<i32>,
    gather_slots: CudaSlice<i32>,
    views: Views,
    _allocation: Allocation,
}

/// The views a decode step would hand the sampler, at the rig's full rows: the decode step
/// narrows them to the live rows, and so does the rig.
struct Views {
    /// f32 `[MAX_ROWS, VOCAB]`.
    logits: Tensor,
    /// u32 `[MAX_ROWS]`.
    token_ids: Tensor,
    /// i32 `[MAX_ROWS]`.
    row_slots: Tensor,
    /// i32 `[MAX_ROWS]`.
    gather_slots: Tensor,
}

impl Rig {
    fn open() -> Self {
        let context = RuntimeContext::new(0).expect("device 0 opens");
        let allocation = Allocation::new(&context).expect("the session opens");
        let stream = context.cuda().default_stream();
        let sampler = DeviceSampler::new(&allocation, &stream, SLOTS, MAX_ROWS, VOCAB)
            .expect("the sampler allocates");
        let logits = stream
            .alloc_zeros::<f32>(MAX_ROWS.get() * VOCAB)
            .expect("the logits allocate");
        let token_ids = stream
            .alloc_zeros::<u32>(MAX_ROWS.get())
            .expect("the token ids allocate");
        let row_slots = stream
            .alloc_zeros::<i32>(MAX_ROWS.get())
            .expect("the row slots allocate");
        let gather_slots = stream
            .alloc_zeros::<i32>(MAX_ROWS.get())
            .expect("the gather slots allocate");
        let rows = MAX_ROWS.get();
        let views = Views {
            logits: view(&allocation, &stream, &logits, &[rows, VOCAB], Dtype::F32),
            token_ids: view(&allocation, &stream, &token_ids, &[rows], Dtype::U32),
            row_slots: view(&allocation, &stream, &row_slots, &[rows], Dtype::I32),
            gather_slots: view(&allocation, &stream, &gather_slots, &[rows], Dtype::I32),
        };
        Self {
            sampler,
            stream,
            logits,
            token_ids,
            row_slots,
            gather_slots,
            views,
            _allocation: allocation,
        }
    }

    /// Uploads `rows` of logits for a step.
    fn upload_logits(&mut self, rows: &[Vec<f32>]) {
        let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * VOCAB);
        for row in rows {
            flat.extend_from_slice(row);
        }
        self.stream
            .memcpy_htod(&flat, &mut self.logits)
            .expect("the logits upload");
    }

    /// Stages `layout` where the caller says, covering its token rows with the gather, uploads
    /// the two arrays to the rig's buffers as a decode step's block copy-in would, and enqueues
    /// the record upload.
    fn stage_as_decode_step(&mut self, layout: &BatchLayout, gather_rows: usize) {
        let mut row_slots = vec![0; MAX_ROWS.get()];
        let mut gather_slots = vec![0; MAX_ROWS.get()];
        self.sampler
            .stage(
                layout,
                gather_rows,
                SamplerArrays {
                    row_slots: &mut row_slots,
                    gather_slots: &mut gather_slots,
                    // A decode step's block holds its live-row count beside the two arrays;
                    // nothing here reads it, so it is staged into a word of its own.
                    live_rows: &mut 0,
                },
            )
            .expect("the layout stages");
        self.stream
            .memcpy_htod(&row_slots, &mut self.row_slots)
            .expect("the row slots upload");
        self.stream
            .memcpy_htod(&gather_slots, &mut self.gather_slots)
            .expect("the gather slots upload");
        // SAFETY: the stream is the sampler's, and every address is the sampler's own.
        unsafe {
            self.sampler
                .upload_records()
                .expect("a step is staged")
                .enqueue(self.stream.cu_stream())
                .expect("the record upload enqueues");
        }
    }

    /// Stages `layout` as a decode step would, uploads `token_ids` as the host would, runs the
    /// gather over every one of them, and returns the token ids the gather left.
    fn gather(&mut self, layout: &BatchLayout, token_ids: &[u32]) -> Vec<u32> {
        self.gather_covering(layout, token_ids, token_ids.len())
    }

    /// Stages `layout` as a decode step would with the gather covering `gather_rows` leading
    /// token rows, uploads `token_ids` as the host would, runs the gather, and returns the token
    /// ids it left. The two views are the stated rows exactly, as the sampler requires; the
    /// readback covers every uploaded row, so a gather reaching past the rows it was given shows
    /// up in the tail.
    fn gather_covering(
        &mut self,
        layout: &BatchLayout,
        token_ids: &[u32],
        gather_rows: usize,
    ) -> Vec<u32> {
        self.stream
            .memcpy_htod(token_ids, &mut self.token_ids)
            .expect("the token ids upload");
        self.stage_as_decode_step(layout, gather_rows);
        let ids = self
            .views
            .token_ids
            .narrow(0, 0, gather_rows)
            .expect("within the rig's rows");
        let slots = self
            .views
            .gather_slots
            .narrow(0, 0, gather_rows)
            .expect("within the rig's rows");
        // SAFETY: the stream is the sampler's, and the token ids and the gather slots are live on
        // its device.
        unsafe {
            self.sampler
                .gather(&ids, &slots)
                .expect("a step is staged")
                .enqueue(self.stream.cu_stream())
                .expect("the gather enqueues");
        }
        self.stream.synchronize().expect("the stream drains");
        self.stream
            .clone_dtoh(&self.token_ids.slice(0..token_ids.len()))
            .expect("the token ids read back")
    }

    /// Samples `rows` under `layout` as an eager step, and returns the tokens.
    fn sample(&mut self, layout: &BatchLayout, rows: &[Vec<f32>]) -> Vec<u32> {
        self.upload_logits(rows);
        self.sampler.stage_eager(layout).expect("the layout stages");
        let view = self.logits.slice(0..rows.len() * VOCAB);
        self.sampler
            .run_on(&self.stream, &view)
            .expect("the sampler runs")
            .to_vec()
    }

    /// Samples `rows` under `layout` as a decode step would, staged where the caller says and
    /// uploaded from there, with the readback enqueued behind the sample as the decode step
    /// enqueues it, and returns the tokens.
    fn sample_as_decode_step(&mut self, layout: &BatchLayout, rows: &[Vec<f32>]) -> Vec<u32> {
        self.upload_logits(rows);
        self.stage_as_decode_step(layout, rows.len());
        let (logits, row_slots) = self.live_views(rows.len());
        // SAFETY: the stream is the sampler's, and the logits and the row slots were uploaded to
        // its device.
        unsafe {
            self.sampler
                .sample(&logits, &row_slots)
                .expect("a step is staged")
                .enqueue(self.stream.cu_stream())
                .expect("the sample enqueues");
            self.sampler
                .read_tokens()
                .expect("a step is staged")
                .enqueue(self.stream.cu_stream())
                .expect("the readback enqueues");
        }
        self.sampler.wait().expect("the tokens come back").to_vec()
    }

    /// The logits and row slots views narrowed to `live` rows, as the decode step hands them to
    /// the sample.
    fn live_views(&self, live: usize) -> (Tensor, Tensor) {
        (
            self.views
                .logits
                .narrow(0, 0, live)
                .expect("within the rig's rows"),
            self.views
                .row_slots
                .narrow(0, 0, live)
                .expect("within the rig's rows"),
        )
    }
}

/// The view of `dims` of `dtype` over `buffer`, minted at Allocation as a decode step mints its
/// views.
fn view<T>(
    allocation: &Allocation,
    stream: &Arc<CudaStream>,
    buffer: &CudaSlice<T>,
    dims: &[usize],
    dtype: Dtype,
) -> Tensor {
    let (address, _reads) = buffer.device_ptr(stream);
    let layout = Layout::contiguous(dims, dtype).expect("rank one or two");
    Tensor::new(allocation, address, layout).expect("a device allocation is aligned")
}

/// One decode entry for `request` in `slot`, sampling under `params`.
fn entry(request: u64, slot: u32, params: SamplingParams) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(request),
        slot: RequestSlot::new(slot),
        sequence: SequenceIndex::new(0),
        context_len: 0,
        input_tokens: vec![1],
        block_table: vec![BlockId::new(0)],
        sampling: Some(params),
    }
}

/// The layout of a batch of `entries`, run eagerly so no bucket is implied.
fn layout(entries: Vec<CommandEntry>) -> BatchLayout {
    let command = StepCommand {
        step: StepId::new(1),
        entries,
        padding_count: 0,
        dispatch: DispatchDecision::Eager(EagerReason::NotUniformDecode {
            token_count: TokenCount::new(1).expect("nonzero"),
            request_count: RequestCount::new(1).expect("nonzero"),
        }),
    };
    BatchLayout::lay_out(&command, BLOCK_SIZE).expect("the command lays out")
}

fn drawn(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        do_sample: true,
        seed,
    }
}

/// A row of random logits.
fn random_row(random: &mut Lcg) -> Vec<f32> {
    (0..VOCAB).map(|_| random.logit()).collect()
}

/// `count` out of `total`, as a fraction.
fn ratio(count: usize, total: usize) -> f32 {
    let count = u16::try_from(count).expect("a draw count fits u16");
    let total = u16::try_from(total).expect("a draw count fits u16");
    f32::from(count) / f32::from(total)
}

/// What the reference samples for the `draw`-th draw of `params` over `row`.
fn expected(row: &[f32], params: &SamplingParams, draw: u32) -> u32 {
    let mut record = SlotRecord::new(params);
    record.draws = draw;
    reference::sample(row, &record)
}

/// Whether the reference keeps each token of `row` under `params`: admitted by top-k, and
/// weighing something after top-p.
fn kept_by_reference(row: &[f32], params: &SamplingParams) -> Vec<bool> {
    let record = SlotRecord::new(params);
    let (_, max) = reference::argmax(row);
    let admitted = reference::admitted_by_top_k(row, record.top_k);
    let weights = reference::weights(row, &admitted, max, record.temperature);
    reference::admitted_by_top_p(&weights, record.top_p)
        .iter()
        .map(|&weight| weight > 0)
        .collect()
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn the_device_sampler_matches_its_host_reference() {
    let mut rig = Rig::open();
    let mut random = Lcg(0x5EED_2026_0904);
    let mut greedy_rows = 0;
    let mut drawn_rows = 0;
    let mut disagreements = 0;

    // Greedy rows, exactly: the largest logit, ties to the first index.
    for step in 0..16 {
        let rows: Vec<Vec<f32>> = (0..4).map(|_| random_row(&mut random)).collect();
        let entries = (0..4u32)
            .map(|index| entry(100 + u64::from(index), index, SamplingParams::default()))
            .collect();
        let sampled = rig.sample(&layout(entries), &rows);
        for (row, token) in rows.iter().zip(&sampled) {
            let want = expected(row, &SamplingParams::default(), 0);
            greedy_rows += 1;
            if *token != want {
                disagreements += 1;
                println!("greedy step {step}: sampled {token}, the reference {want}");
            }
        }
    }
    assert_eq!(disagreements, 0, "greedy sampling matches the reference");
    let mut cutoff_disagreements = 0;

    // A row with tied maxima: the first index wins on both sides.
    let mut tied = vec![0.0f32; VOCAB];
    tied[7] = 5.0;
    tied[9] = 5.0;
    let sampled = rig.sample(
        &layout(vec![entry(1, 0, SamplingParams::default())]),
        &[tied.clone()],
    );
    assert_eq!(sampled, [7], "the first of two equal maxima");

    // Drawn rows, against the reference draw for draw, under several filters.
    for (temperature, top_k, top_p) in [
        (1.0, 0, 1.0),
        (0.7, 40, 1.0),
        (1.0, 0, 0.9),
        (0.5, 64, 0.95),
        (1.3, 8, 0.5),
    ] {
        let row = random_row(&mut random);
        let params = drawn(temperature, top_k, top_p, 0x5EED);
        for draw in 0..32u32 {
            // A new request each time, under an id no earlier row used, so its slot changes
            // hands and the seed under test reaches the record; its counter is then zero, which
            // is the draw compared.
            let slot = draw % u32::try_from(SLOTS).expect("the slot count fits u32");
            let mut params = params;
            params.seed = 0x5EED + u64::from(draw);
            let sampled = rig.sample(
                &layout(vec![entry(1000 + u64::from(draw), slot, params)]),
                slice::from_ref(&row),
            );
            let want = expected(&row, &params, 0);
            drawn_rows += 1;
            if sampled[0] != want {
                // The device rounds the exponential within an ulp or two of the host, which can
                // move a cutoff or the pick by one token; whatever it picks is still kept.
                let kept = kept_by_reference(&row, &params);
                assert!(
                    kept[sampled[0] as usize],
                    "drawn (T {temperature}, k {top_k}, p {top_p}) draw {draw}: sampled {}, \
                     which the reference does not keep",
                    sampled[0]
                );
                cutoff_disagreements += 1;
                println!(
                    "drawn (T {temperature}, k {top_k}, p {top_p}) draw {draw}: sampled {}, the \
                     reference {want}; a kept token, so rounding at a cutoff",
                    sampled[0]
                );
            }
        }
    }

    println!("=============== sampler parity evidence ===============");
    println!("vocabulary:           {VOCAB}");
    println!("greedy rows:          {greedy_rows}");
    println!("drawn rows:           {drawn_rows}");
    println!("cutoff disagreements: {cutoff_disagreements}");
    assert!(
        cutoff_disagreements * 100 <= drawn_rows,
        "{cutoff_disagreements} drawn rows of {drawn_rows} differ from the reference, more than \
         rounding at a cutoff explains"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn a_seeded_request_draws_the_same_tokens_in_any_batch_and_any_slot() {
    let mut rig = Rig::open();
    let mut random = Lcg(0x11ED_2026_0904);
    let row = random_row(&mut random);
    let params = drawn(0.8, 50, 0.95, 4242);
    let steps = 24;

    // Alone, in slot zero.
    let alone: Vec<u32> = (0..steps)
        .map(|_| rig.sample(&layout(vec![entry(1, 0, params)]), slice::from_ref(&row))[0])
        .collect();

    // The same request in a different slot, sharing every batch with other requests whose rows
    // are different and whose count changes step to step, and sitting at a different row of
    // the batch each step.
    let slot = 11;
    let mut together = Vec::with_capacity(steps);
    for step in 0..steps {
        let others = step % 3;
        let mut entries: Vec<CommandEntry> = (0..others)
            .map(|other| {
                let other = u32::try_from(other).expect("below three");
                entry(
                    200 + u64::from(other),
                    other + 1,
                    drawn(1.0, 0, 1.0, 7 + u64::from(other)),
                )
            })
            .collect();
        let mut rows: Vec<Vec<f32>> = (0..others).map(|_| random_row(&mut random)).collect();
        let position = step % (others + 1);
        entries.insert(position, entry(1, slot, params));
        rows.insert(position, row.clone());
        let sampled = rig.sample(&layout(entries), &rows);
        together.push(sampled[position]);
    }

    println!("=============== seeded reproducibility ===============");
    println!("steps:                {steps}");
    println!("alone, slot 0:        {alone:?}");
    println!("in company, slot {slot}:  {together:?}");
    assert_eq!(
        alone, together,
        "a seeded request's tokens do not depend on its batch, its row in it or its slot"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn drawn_tokens_follow_the_distribution_the_filters_leave() {
    let mut rig = Rig::open();
    // Four tokens carry all the mass; the rest are far below and top_p drops them.
    let mut row = vec![-30.0f32; VOCAB];
    let heavy = [11usize, 222, 3333, 4000];
    let probabilities = [0.5f32, 0.3, 0.15, 0.05];
    for (&token, &probability) in heavy.iter().zip(&probabilities) {
        row[token] = probability.ln();
    }
    let params = drawn(1.0, 0, 0.999, 9090);

    // One request in one slot throughout, so its record is written once and the kernel advances
    // its draw counter: what is measured is one seeded request's stream, which is what a client
    // gets.
    let mut counts = [0usize; 4];
    for draw in 0..DRAWS {
        let token = rig.sample(&layout(vec![entry(1, 0, params)]), slice::from_ref(&row))[0];
        let index = heavy
            .iter()
            .position(|&heavy| heavy == token as usize)
            .unwrap_or_else(|| {
                panic!("draw {draw} sampled {token}, which top_p should have dropped")
            });
        counts[index] += 1;
    }

    println!("=============== draw frequencies ===============");
    println!("draws:                {DRAWS}");
    for (index, (&count, &probability)) in counts.iter().zip(&probabilities).enumerate() {
        let frequency = ratio(count, DRAWS);
        println!(
            "token {}: {frequency:.4} against {probability:.4}",
            heavy[index]
        );
        assert!(
            (frequency - probability).abs() < 0.03,
            "token {} drawn {frequency:.4} of the time, not {probability:.4}",
            heavy[index]
        );
    }
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn the_gather_takes_a_decoding_rows_token_from_its_slot_and_leaves_a_fresh_slots_row() {
    let mut rig = Rig::open();
    let mut random = Lcg(0x6A7E_2026_0904);
    let rows: Vec<Vec<f32>> = (0..2).map(|_| random_row(&mut random)).collect();
    let decoding = vec![
        entry(1, 3, SamplingParams::default()),
        entry(2, 5, SamplingParams::default()),
    ];
    let sampled = rig.sample(&layout(decoding.clone()), &rows);

    // The next step: the same two requests decoding, and a third whose slot has sampled nothing.
    let mut next = decoding;
    next.push(entry(3, 7, SamplingParams::default()));
    let uploaded = [999, 999, 999];
    let gathered = rig.gather(&layout(next), &uploaded);

    println!("=============== gather ===============");
    println!("sampled:              {sampled:?}");
    println!("uploaded:             {uploaded:?}");
    println!("gathered:             {gathered:?}");
    assert_eq!(
        gathered,
        [sampled[0], sampled[1], 999],
        "the decoding rows take their slots' tokens; the fresh slot's row keeps the host's"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn the_gather_covers_the_token_rows_the_caller_stated_and_not_the_rows_the_step_samples() {
    let mut rig = Rig::open();
    let mut random = Lcg(0x6A7E_2026_0907);
    let rows: Vec<Vec<f32>> = (0..3).map(|_| random_row(&mut random)).collect();
    let decoding = vec![
        entry(1, 3, SamplingParams::default()),
        entry(2, 5, SamplingParams::default()),
        entry(3, 7, SamplingParams::default()),
    ];
    let sampled = rig.sample(&layout(decoding.clone()), &rows);

    // All three slots sampled last step, so all three would gather. The caller states a gather
    // over the first two token rows, as one holding fewer token rows than the batch samples
    // does, which is the only step where the rows the gather covers and the rows the step
    // samples are different numbers.
    let uploaded = [999, 999, 999];
    let gathered = rig.gather_covering(&layout(decoding), &uploaded, 2);

    println!("=============== gather over the stated rows ===============");
    println!("sampled:              {sampled:?}");
    println!("uploaded:             {uploaded:?}");
    println!("gathered:             {gathered:?}");
    assert_eq!(
        gathered,
        [sampled[0], sampled[1], 999],
        "the two token rows the caller stated take their slots' tokens and the row past them \
         keeps the host's: a gather running over the rows the step samples instead would reach \
         a third row it was given no slot for"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn a_step_staged_where_the_caller_says_samples_under_the_row_slots_uploaded_from_there() {
    let mut rig = Rig::open();
    let mut random = Lcg(0xCA11_2026_0905);
    let rows: Vec<Vec<f32>> = (0..3).map(|_| random_row(&mut random)).collect();
    // Slots out of row order and none of them slot zero, so a row sampling under the wrong
    // slot's record would be greedy under a drawn one, or the reverse. A fresh device's row
    // slots all name slot zero, whose zeroed record is greedy, so the drawn row is held to its
    // exact first draw: the greedy token it would sample without the upload is a kept token too.
    let params = drawn(0.7, 40, 1.0, 77);
    let entries = vec![
        entry(1, 9, SamplingParams::default()),
        entry(2, 2, params),
        entry(3, 6, SamplingParams::default()),
    ];
    let sampled = rig.sample_as_decode_step(&layout(entries), &rows);

    let greedy: Vec<u32> = [0, 2]
        .iter()
        .map(|&row| expected(&rows[row], &SamplingParams::default(), 0))
        .collect();
    let drawn_expected = expected(&rows[1], &params, 0);
    println!("=============== caller-staged sample ===============");
    println!("sampled:              {sampled:?}");
    println!("greedy rows expected: {greedy:?}");
    println!("drawn row expected:   {drawn_expected}");
    assert_eq!(
        [sampled[0], sampled[2]],
        [greedy[0], greedy[1]],
        "the greedy rows sample under their own slots' records"
    );
    assert_eq!(
        sampled[1], drawn_expected,
        "the drawn row samples its slot's first draw"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn the_eager_upload_refuses_a_step_staged_where_the_caller_says() {
    let mut rig = Rig::open();
    let step = layout(vec![entry(1, 0, SamplingParams::default())]);

    // A decode step's arrays are the caller's; the eager upload would carry stale rows.
    let mut row_slots = [0; 1];
    let mut gather_slots = [0; 1];
    rig.sampler
        .stage(
            &step,
            1,
            SamplerArrays {
                row_slots: &mut row_slots,
                gather_slots: &mut gather_slots,
                live_rows: &mut 0,
            },
        )
        .expect("the layout stages");
    let view = rig.logits.slice(0..VOCAB);
    assert!(
        matches!(
            rig.sampler.run_on(&rig.stream, &view),
            Err(SamplerError::ArraysElsewhere(ArraysIn::Caller))
        ),
        "the eager upload refuses a step staged in the caller's memory"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn a_view_of_other_rows_than_the_staged_step_or_of_another_dtype_is_refused_by_name() {
    let mut rig = Rig::open();
    let step = layout(vec![
        entry(1, 0, SamplingParams::default()),
        entry(2, 1, SamplingParams::default()),
    ]);
    rig.stage_as_decode_step(&step, 2);

    // Two rows are staged: a three-row logits view is not the step's rows, and the row slots
    // view handed as the logits is not f32.
    let (logits, row_slots) = rig.live_views(2);
    let (three, _) = rig.live_views(3);
    assert!(
        matches!(
            rig.sampler.sample(&three, &row_slots),
            Err(SamplerError::ViewShape {
                what: "logits",
                ref held,
                ref expected
            }) if *held == [3, VOCAB] && *expected == [2, VOCAB]
        ),
        "a logits view of other rows than the staged step's is refused"
    );
    assert!(
        matches!(
            rig.sampler.sample(&row_slots, &logits),
            Err(SamplerError::ViewDtype {
                what: "logits",
                held: Dtype::I32,
                expected: Dtype::F32
            })
        ),
        "the row slots handed as the logits are refused"
    );

    // The gather's two views swapped: the gather slots are not u32 token ids.
    let ids = rig
        .views
        .token_ids
        .narrow(0, 0, 2)
        .expect("within the rig's rows");
    let slots = rig
        .views
        .gather_slots
        .narrow(0, 0, 2)
        .expect("within the rig's rows");
    assert!(
        matches!(
            rig.sampler.gather(&slots, &ids),
            Err(SamplerError::ViewDtype {
                what: "token ids",
                held: Dtype::I32,
                expected: Dtype::U32
            })
        ),
        "the gather slots handed as the token ids are refused"
    );
    assert!(
        rig.sampler.gather(&ids, &slots).is_ok(),
        "the right views are taken"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn a_sample_with_no_copy_described_behind_it_leaves_nothing_to_wait_on() {
    let mut rig = Rig::open();
    let step = layout(vec![entry(1, 0, SamplingParams::default())]);
    rig.stage_as_decode_step(&step, 1);

    // The rig's zeroed logits: what the row samples does not matter here, only that the sample
    // describes the launch and nothing else, so the tokens stay on the device until a copy is
    // described behind it.
    let (logits, row_slots) = rig.live_views(1);
    // SAFETY: the stream is the sampler's, and the logits and the row slots are live on its
    // device.
    unsafe {
        rig.sampler
            .sample(&logits, &row_slots)
            .expect("a step is staged")
            .enqueue(rig.stream.cu_stream())
            .expect("the sample enqueues");
    }
    let refused = rig.sampler.wait().expect_err("no copy was described");
    assert!(
        matches!(
            refused,
            SamplerError::Readback(ReadbackError::NoCopyPending)
        ),
        "a sample with no copy described behind it leaves nothing to wait on: {refused}"
    );
    rig.stream.synchronize().expect("the stream drains");
}
