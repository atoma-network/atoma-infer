# atoma-infer

Shared language for the engine core and the CUDA-graph execution work: one thread scheduling
requests over engine-owned KV, and decode steps captured as replayable graphs. This file fixes
the vocabulary used in tracked identifiers.

## Language

**Capture**:
Recording a step's device work into a CUDA graph instead of executing it.
_Avoid_: trace, record (as nouns)

**Replay**:
Launching a previously captured graph executable for one step.
_Avoid_: playback, re-run

**Capture check**:
Recording a step under capture over a dummy run to show the driver accepts it: the recording is
not invalidated, and no node of the graph it produces allocates or frees memory. It needs no live
batch, since the dummy run fills the bucket's rows. The warmup pass every recording consumes runs
inside it and is not another name for it.
_Avoid_: capture cleanliness, capture smoke test

**Graph set**:
Which graph serves each bucket: the recordings one capture of the bucket ladder made, one per
bucket at the bucket's index, held by the forward beside the session that indexes them and looked
up by the bucket the key names. `GraphSet` in the engine; the runtime's graph set is the
session's own list of graph entries, which the engine's indexes into.
_Avoid_: graph cache, graph table, graph map

**Capture report**:
What capturing the bucket ladder cost at startup: each graph's cost, `GraphCost` — its bucket,
the time its warmup and recording took, the drop in free memory across them and the free memory
after, which the next bucket's drop is read from — and the whole capture's time and drop, the
one warmup ahead of every recording included. Logged on the executor thread as it is measured;
`CaptureReport`.
_Avoid_: capture stats, capture metrics, startup profile

**Graph memory**:
A graph set's device memory as a fixed term plus a marginal term per graph, fitted by least
squares over what each capture used: a report of what a capture that has run cost, and never a
ceiling for one that has not, since the fit bounds no single point. `GraphMemory`.
_Avoid_: graph overhead, memory budget (for the fit), per-graph cost (for the marginal term)

**Free-memory reading**:
One query of the device's free memory through the runtime context, as `DeviceBytes`: what the
device has free, not what this process holds, so anything else on the device moves it, and the
drop between two readings is a cost and never below zero. A reading can be the first call to
check the context after a driver status was deferred onto it, and then fails under that status's
own classification, so a reading is propagated and never read as a number in its place.
_Avoid_: memory probe, mem info, usage sample

**Capture session**:
The value that carries one graph set through the three session phases, with consuming
transitions between them. It lives and dies on the executor thread that runs it; the executor
is the thread, never this value.
_Avoid_: executor (for this value), executor session

**Session phase**:
One of the capture session's three — Allocation, Capture, Replay — each named for the
operation it permits. Always written qualified: "phase" alone is ambiguous with request phase.
_Avoid_: phase (unqualified), stage, mode

**Allocation**:
The first session phase. Allocation fixes every device address and binds every stream and
communicator. Nothing is captured in it.
_Avoid_: setup, init, startup

**Baked address**:
A device address a step's descriptors pass to the device unchanged, fixed in the Allocation
phase: a recording bakes it into the graph's nodes, so a replay reads exactly the memory the step
was built over. The set is every weight and cache view candle owns, the device block a copy-in
lands in, the arena, the step's fixed buffers and the sampler's arrays. The forward reads it by
name when it is built, and a debug build reads it again before each keyed step and panics naming
the first address that moved.
_Avoid_: frozen address, cached pointer, pinned address (which is pinned host memory)

**Descriptor**:
A description of device work the capture session enqueues onto the capture stream. Its one
implementation per backend — the descriptor seam — is the only place a raw stream handle
appears.
_Avoid_: launcher, enqueue callback, work item

**Op launcher**:
Where a step's walk hands each op once its operands are resolved: the CUDA launcher issues the
op's kernel call on the stream the enclosing descriptor was handed, passing the handle through
to the C-ABI launch and keeping it nowhere; a recording launcher lists the addresses the op
would touch. Lives inside a descriptor and is never one.
_Avoid_: descriptor (for this), backend, executor

**Weight reload**:
The explicit reload of model weights, and the one transition from Replay back to Allocation:
new weights mean new baked addresses, so the graph set is torn down with the phase.
_Avoid_: hot swap, weight refresh

**Bucket**:
One captured batch size. A live batch is padded up to the nearest bucket.
_Avoid_: shape, size class

**Bucket ladder**:
The ordered list of buckets the engine captures. Always written qualified — "ladder" alone is
ambiguous with the rung ladder.
_Avoid_: capture sizes, batch-size list

**Live batch**:
The shape of one step's scheduled set — its token and request counts — before padding.
_Avoid_: batch shape

**Dispatch**:
Choosing, for a live batch, the captured graph to replay or the eager fallback. Engine-thread
work; the executor never re-derives it.
_Avoid_: admission (for this), graph lookup, routing

**Graph key**:
The value that selects one captured graph for a padded batch.
_Avoid_: signature, shape id, cache key

**Rung**:
A numbered project milestone, used in `rungN-MM` commit-message prefixes. Not a bucket.
_Avoid_: ladder (unqualified), phase

**Arena**:
The engine-owned device allocation from which every captured step's activations are addressed.
_Avoid_: pool — the arena is neither a CUDA memory pool nor the KV block pool; its slots are
fixed, role-addressed extents, not interchangeable blocks

**Role**:
One tensor the step produces, declared with a per-token width and a lifetime.
_Avoid_: tensor name, buffer kind

**Lifetime**:
The half-open range of a layer's op order in which a role's slot holds live data.
_Avoid_: liveness, span

**Slot**:
The arena extent reserved for one tensor role in one layer.
_Avoid_: buffer (for arena extents), region

**Op timeline**:
The step's launches indexed as the arena counts them, which is where a poison fill's place is
stated: layer `l`'s op `i` at `l * ops_per_layer + i`, so the embedding gather, which writes the
first layer's residual, sits at -1, and the final norm and the head projection sit at the index
after the last layer's last op and the one after that, where the arena's extra row holds their
slots.
_Avoid_: global op index, step index (for this), op order (which is one layer's)

**Poison fill**:
One write of the poison byte over the whole of one slot, scheduled by the arena's poison layout
before one index of the op timeline: before a role's first use, so a read ahead of it sees
not-a-number, and after its last use, so a read behind it does too and the pattern stands for
the next replay. Resolved with the bucket's tables to the view of the slot it writes, and
launched by the step's walk as an op of its own through the op launcher, as an async memset on
the capture stream; the greedy and no-reuse layouts schedule none, so a serving step launches
none. `ScheduledPoisonFill` while it is an offset and a length in the arena's schedule,
`PoisonFill` once a bucket's tables have resolved it to the slot it writes.
_Avoid_: poison write, canary, memset (as the name), fill (unqualified)

**Activation**:
An intermediate tensor produced and consumed within a single step.
_Avoid_: scratch, temporary

**Tensor view**:
A device address with a layout — shape, element strides and dtype — over runtime-owned memory,
minted in the Allocation phase and never an owner. Sub-views narrow, select and reshape inside
the extent their root declared.
_Avoid_: tensor (where candle's owning tensor could be meant), buffer handle, pointer

**Layer class**:
The linear op order every layer of a class runs, with the roles each op reads and writes: the
coordinate system the roles' lifetimes index. Llama declares one class of fifteen ops.
_Avoid_: layer type, block definition, op list

**Slot table**:
Every address one bucket's step touches, resolved once at Allocation: a tensor view per role per
arena row, the column views of the fused row, the weights, the cache halves, and the leading rows
of the fixed buffers.
_Avoid_: pointer table, address map

**Segment**:
One captured graph within a forward pass that is split around eager operations.
_Avoid_: piece, partial graph

**Break point**:
The place in a forward pass where one segment ends and the next begins.
_Avoid_: split, boundary, cut

**Declarer**:
The backend or the model that states a break point. A backend states a capability — an op it
cannot capture, or one that is rank-coupled. A model states a policy — where its eager region
lies. The two are independent and the dispatcher takes their union.
_Avoid_: owner, source, author

**Bridge buffer**:
A fixed-address buffer through which an eager operation passes data between two segments.
_Avoid_: intermediate buffer, staging buffer

**Support level**:
What a backend declares its captured routine stays valid for: always, any uniform batch, uniform
single-token decode, or never. A statement about the recorded routine, never about what the
kernels can compute — which is why a backend's break points, which are capability statements, are
a separate declaration.
_Avoid_: capability (as a name for the level), kernel support, capture support

**Graph mode**:
The support level every captured routine runs under: the minimum across the active backends,
settled once at startup and never raised. Always written qualified: "mode" alone is ambiguous with
the driver's capture-interaction mode.
_Avoid_: capture mode, graph level

**Workspace**:
Caller-owned bytes a kernel is handed because it may not allocate its own. Captured and eager
execution each own one; they are never the same bytes.
_Avoid_: scratch buffer, temp buffer, kernel arena

**Lease**:
The value that holds one pool block for exactly one owner. While it exists the block cannot be
evicted or handed out again; surrendering it is the block's only way back to the pool.
_Avoid_: ref count, handle

**Chain hash**:
The digest of one block-sized token run chained through its parent run's digest, so equal hashes
mean equal prefixes. The tier-agnostic identity of cached KV.
_Avoid_: content hash, prefix hash

**Dummy**:
A padding request filling one bucket slot. Each dummy occupies a request slot and owns its own
KV block, both held from startup for the process lifetime; it never finishes and never enters
admission.
_Avoid_: filler, fake request

**Layer group**:
A set of layers with a common cache kind and geometry. A model's cache is declared group by
group, and a group can share another group's cache instead of writing its own.
_Avoid_: cache group, attention group

**Prefix index**:
The radix tree over chains of block hashes that answers longest-prefix match. It answers in
block hashes, never slot ids; which slot holds a hash's bytes is the pool's residence lookup.
_Avoid_: radix cache, prefix tree

**Tier**:
Where cached KV bytes live — device or host memory. A tier is where bytes live, never a
preemption mechanism.
_Avoid_: swap, cache level

**Residence**:
Which tier and slot currently hold a block hash's bytes. Identity is the chain hash and never
carries residence; residence is always a separate lookup.
_Avoid_: placement (for cached KV bytes; arena slot placement is unrelated and stays legal)

**Request phase**:
Where a request is in its lifecycle: Waiting, Running, Preempted, Finished, or Padding for a
dummy. Always written qualified; illegal transitions between phases are unrepresentable.
_Avoid_: phase (unqualified), status, state, stage

**Prefilling**:
A running request whose computed count is below its prompt length. A derived property of the
Running phase, never a phase of its own; a chunked prefill is a prefilling request that computes
fewer tokens this step than remain.
_Avoid_: prefill stage, prompt phase, is_prompt

**Decoding**:
A running request whose computed count has reached its prompt length. A derived property of the
Running phase, never a phase of its own.
_Avoid_: decode stage, generation phase

**Computed**:
The count of a request's tokens whose KV is resident. Resets to zero on preemption; whatever
the prefix index still holds is rediscovered when the request next enters Running.
_Avoid_: num computed tokens, processed, cached length

**Total**:
A request's prompt tokens plus every token it has generated so far.
_Avoid_: length, sequence length (for a request)

**Query length**:
The tokens an entry computes in one step.
_Avoid_: chunk size, token chunk size, num new tokens

**Context length**:
The tokens an entry already holds in KV before the step — its computed count when the step
command is built — in the sense attention kernels use. Never the model's maximum.
_Avoid_: context window, context size, num computed (in a step command)

**Sequence length**:
An entry's context length plus its query length: the tokens its KV holds after the step, in the
sense attention kernels use.
_Avoid_: seq len (in prose), total (in a step command)

**Max model length**:
The longest sequence the model serves. The only meaning "context window" carries here.
_Avoid_: context length (for the maximum), max context

**Intake**:
Taking a submitted request off ingress into Waiting: minting its request id, giving it a slab
slot and queueing it — or finishing it on the spot when its prompt can never run. The one
transition into Waiting, and never a word for admission.
_Avoid_: enqueue, accept, submit (for this transition), admission (for this)

**Admission**:
Moving a request from Waiting or Preempted into Running under the admission policy, which
examines a bounded window of candidates per pass and offers preempted requests first, last-in
first-out. The one transition into Running; nothing else is called admission.
_Avoid_: scheduling (for this transition), pickup, intake (which is the transition into
Waiting), resume

**Priority**:
How urgently a request admits, as its client submits it. Higher admits first, and the default is
the lowest there is, so traffic that asks for nothing shares one priority and is ordered by
arrival alone. An input to admission, never to preemption: the victim is always the newest
running request.
_Avoid_: weight, rank, nice, urgency (as the field)

**Preemption**:
Releasing a running request's KV and returning it to run again from whatever the prefix index
still holds. Never a swap: nothing is written out to bring back.
_Avoid_: swap, evict (the prefix index's word for cached blocks), recompute (as the noun)

**Request**:
The client unit: one prompt, one set of sampling parameters, one priority, one egress sink. What
admission admits and preemption displaces, as a whole.
_Avoid_: sequence group, job, query (for a request)

**Sequence**:
One token stream inside a request, with its own block table, computed count, total and finish.
A request is born with one sequence and forks at its first sample when it asks for more.
_Avoid_: beam, candidate, hypothesis, stream

**Block table**:
The ordered block ids a sequence's KV occupies. Host-native: a step command is built from it
with no device read.
_Avoid_: block list, slot mapping (for the table)

**Step**:
One engine iteration: a scheduling pass, the step command it yields, and the step result that
comes back. Numbered by step id.
_Avoid_: iteration, tick, cycle

**Step deadline**:
How long a step command may be out with the executor before the engine treats the executor as
lost: live requests fail as executor lost and the engine thread returns. An executor held inside
a step — a leader kept in a collective by a rank that died mid-step — never drops its rings, so
the deadline is what ends the wait.
_Avoid_: step timeout, watchdog, forward timeout

**Scheduled**:
The output of one scheduling pass: the step it is for, the entries this step runs and the slots
it preempted. Indices and counts, never copied request state. There are no block deltas: a step
command carries each entry's whole block table, since preemption releases KV rather than moving
it.
_Avoid_: schedule, scheduler output, scheduler step, plan

**Entry**:
One row of a Scheduled: a sequence, its query length, and whether it samples. An entry samples
only when its query reaches the sequence's total; a non-final prefill chunk does not. Unqualified,
the term is this one: the staging ring's is always a staging entry, and the runtime's graph set
has graph entries of its own.
_Avoid_: scheduled tokens, scheduled sequence, batch item, staging entry (which is where a step's
copy-in is written)

**Batch layout**:
A step command laid out as the arrays the model forward takes: prefills first, then decodes,
with every entry's tokens, positions and KV slots flattened in that order, the per-entry lengths
and cumulative starts, the padded block tables, and the logits rows to select. Pure host
arithmetic from the command; the forward re-derives nothing.
_Avoid_: input metadata, model input, batch tensors (for the host arrays)

**Uniform decode**:
A Scheduled whose every entry has query length one. The condition full-graph replay requires.
_Avoid_: decode-only, pure decode, all-decode

**Token budget**:
The per-step cap on query tokens summed over entries, plus a request cap equal to the largest
bucket. Spent by running requests first; the remainder is offered to admission.
_Avoid_: scheduling budget, batch budget, max batched tokens (as the name of the budget)

**Window**:
The count of admission candidates one pass examines. Scheduler-wide: every admission policy
sees the same window and differs only in how it orders it.
_Avoid_: lookahead, scan depth, sliding window (a KV geometry term)

**Engine thread**:
The one thread that owns all engine state — the request slab, block pool, prefix index,
scheduler and dispatch — and every transition on it. No lock sits on its step path.
_Avoid_: scheduler thread, engine core, driver

**Executor thread**:
The pinned per-rank thread that owns the device and runs the session. It acts on a step command
and re-derives nothing in it.
_Avoid_: worker, model thread, runner

**Executor handoff**:
What the engine hands the executor's ranks at startup, and nothing else: their ends of the rings,
and the padding dummies' blocks as block ids in reservation order, for the capture of the bucket
ladder to fill every dummy run from. Minted by the engine when it is built, consumed when the
ranks are spawned; `ExecutorHandoff`.
_Avoid_: executor config, startup bundle, rings (for the whole)

**Follower**:
An executor rank other than zero. Rank zero, the leader, owns the engine's rings and feeds each
step command to every follower over a ring of its own; a follower runs the forward for it and
produces nothing, since the leader alone holds a sampler. A follower going ends the leader, and
the leader going ends every follower.
_Avoid_: worker rank, replica, secondary

**Feed**:
The single-producer single-consumer ring from the leader to one follower, carrying each step
command. A push wakes the follower and a pop wakes the leader, so the leader can wait on a full
feed; either end dropping wakes the far side, which is how a rank's death is seen.
_Avoid_: follower queue, broadcast channel, command channel

**Ingress**:
The bounded channel that carries requests into the engine thread. A refused send is overload.
_Avoid_: request queue, inbound, input channel

**Overload**:
The condition an ingress refusal signals: the engine cannot take another request right now. The
API's 429.
_Avoid_: backpressure (as the condition), rejection, throttling

**Control**:
The bounded channel the engine thread drains before ingress on every pass. Carries drain,
shutdown and state queries — never cancels and never requests.
_Avoid_: command channel, admin channel

**Egress**:
The per-request channel that carries a request's output to its client. Its receiver dropping is
the one and only cancel; a failed send returns nothing to ignore.
_Avoid_: response channel, output stream, sink (for the channel)

**Backlog**:
Events a client has left unread on its egress channel. A client that keeps up leaves none; one
that leaves more than the scheduler allows has its request retired, keeping every event already
queued behind it. What bounds an unbounded channel.
_Avoid_: lag, buffer depth, backpressure (which is what the channel does not apply)

**Ring**:
One of the two single-producer single-consumer rings between the engine thread and the
executor thread: step commands one way, step results the other. Rings are not channels.
Unqualified, the term is this one; the staging ring is always written qualified.
_Avoid_: channel (for these), queue, pipe, staging ring (which holds a step's staging entries)

**Step command**:
Everything the executor acts on for one step: entries with context and sequence lengths, block
tables, the padding dummies inserted, and the dispatch decision. Built with zero device reads.
_Avoid_: execute model request, model input, batch (for the command)

**Step result**:
What the executor returns for one step: each sampling entry's token and whatever the engine
needs to advance request state.
_Avoid_: model output, step output, sampler output

**Copy-in**:
The one copy per step that carries the host's per-step arrays to the device: the model's five
inputs, the sampler's two and its live-row count, packed into one block and copied from a staging
entry into the device block in front of the step. Named against the readback, which is the one
copy the other way. A slot's sampling record is not in it — records are written when a slot
changes hands, not per step, and go up as sparse copies of their own in front of it.
_Avoid_: input upload (for this in the engine), host-to-device transfer, staging copy

**Packed block**:
One bucket's eight staged arrays laid consecutively, each at the alignment CUDA guarantees for
a device allocation: what one copy-in carries, at a length that follows the bucket rather than the
largest one. A staging entry's pinned block and the one device block are each allocated at the
largest bucket's packed length, and every bucket reads the device block through views minted at
its own offsets.
_Avoid_: KV block, staging buffer, input buffer (each for this)

**Staging ring**:
The staging entries a step's copy-in is written into, handed out in turn: a cursor names the one
handed out next, and an acquire waits on that staging entry's fence before handing it out and
moving the cursor on. Its non-blocking form leaves the cursor where it is while the copy that
last read the staging entry is still in flight. Written qualified in prose, where "ring" alone is
one of the two between the engine thread and the executor thread; the module that holds it is
`ring`.
_Avoid_: ring (in prose, for this), staging queue, double buffer, buffer pool

**Staging entry**:
One place in the staging ring: a pinned packed block and the fence that guards it, named by the
token an acquire hands out and a copy-in spends. Whoever owns the staging memory keeps each
staging entry's block indexed by that token. Written qualified in prose, where the bare word is
Entry's; the parameter and binding that carry one are named `entry`.
_Avoid_: entry (in prose, for this), staging slot, staging buffer

**Staging fence**:
What says when the host may write a staging entry again: one event, signaled through the
descriptor seam behind the copy that reads the staging entry, and waited on by the host with or
without blocking. A fence nobody has signaled is passed, so a staging entry no copy has read is
written without a wait; no wait reaches a stream, so the capture stream keeps its no-synchronize
rule. The fence, and not the host wait that ends a step, is what makes the reuse safe: it is what
still holds when the host runs ahead of the device.
_Avoid_: barrier, sync point, semaphore

**Staging depth**:
How many staging entries the staging ring holds: two unless configured, never zero, and at most
eight, since each staging entry pins host memory sized for the largest bucket. What bounds how
many copy-ins can be in flight at once, and so how far a host could run ahead of the device
before an acquire has to wait.
_Avoid_: ring size, queue depth, staging count

**Keyed step**:
The device work of a keyed batch: the gather that takes each decoding row's token from what the
device sampled for its slot, the model step over the bucket's rows, and the sample of the rows'
logits, in that order and composed into one descriptor, `KeyedStep`, since a recording takes
one. It is what a bucket's graph holds and what a replay launches. The wait on candle's stream,
the sampler's record upload and the copy-in are enqueued ahead of it and the readback behind it,
and none of them is in the graph; "before each keyed step" is before any of that is enqueued.
_Avoid_: graph step, kernel sequence, decode kernels (for the whole)

**Live-row count**:
How many leading rows of a keyed step are live: one `u32`, the eighth array of the packed block,
carried by the copy-in beside the sampler's two and read on the device by the sample, which
returns for a row at or past it. One graph captured at a bucket samples exactly the live rows of
any batch it serves; a dummy run's count is zero, so a recording samples nothing. The candle path
passes a word of the sampler's own, written with the records.
_Avoid_: valid count, non-padded count, active rows, live tokens

**Dummy run**:
A bucket's rows filled as padding rows over one KV block each, staged and copied in through the
same acquire and fence as a live step and then run with nothing read back: what a capture check,
a warmup or a recording runs when there is no live batch. Every row is what a dummy's row is in a
live step, so the only cache it writes is each block's first KV slot. Its sampler arrays are
written too — the two naming no request slot and the live-row count zero — so its copy-in carries
nothing stale, and a sample launched over its rows returns for every one of them: no slot's
sampling record or draw counter moves.
_Avoid_: padding batch, fake batch, dummy step

**Readback**:
The one device-to-host copy per step that brings the sampled tokens to the host: into a pinned
buffer sized for the largest batch, enqueued on the forward's stream, and waited on through the
buffer's own event and nothing else. Logits never cross it in serving; a harness that compares
them reads them through a readback of its own.
_Avoid_: download, sync (for this copy), logits fetch

**Sampled witness**:
That a sample of the staged step reached the stream, so the readback of its tokens can be
described behind it: minted by a `Sample` once it has been enqueued and by the decode step once
it has replayed a graph that holds one, and nowhere else, so a readback cannot be asked for
tokens no sample wrote; `Sampled`.
_Avoid_: receipt, sample proof, sampled flag

**Sampling record**:
What one request slot holds on the device for the request in it: its temperature, top-k, top-p,
seed and draw counter. Written once when the slot changes hands, never per step, and read by the
sampler every step the slot samples.
_Avoid_: sampling params (which is what a request asks for), sampler state

**Draw counter**:
How many draws a slot's request has made, kept in its sampling record and advanced by the sampler
alone. It is what a seeded request's next draw is numbered by, so its tokens follow from its seed
and its own history and never from the batch it sits in or the slot it occupies.
_Avoid_: offset, rng state, step count

**Gather**:
Taking a decoding row's input token from what the sampler last drew for its request slot, on the
device, instead of from the host's copy-in. What removes the host from between a replay and the
next step's input.
_Avoid_: scatter, copy-back, token fetch

**Drain**:
The control message that stops admission and lets running requests finish. The engine is
drained when no step is in flight and nothing runs; that is the point at which control is
honoured.
_Avoid_: quiesce, pause, stop-the-world

**Heartbeat**:
The pass counter and timestamp the engine thread publishes every pass, so liveness is read from
the thread that could wedge rather than from the API in front of it.
_Avoid_: health check (for the signal), liveness probe, ping

**Spike**:
A time-boxed experiment that answers a design question with measurements on real hardware.
_Avoid_: prototype, proof of concept

**Stint**:
A scheduled block of GPU-rig time during which spikes and verifications run.
_Avoid_: session, rental
