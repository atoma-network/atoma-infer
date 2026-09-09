//! Graph memory as a fixed term plus a marginal term per graph, fitted over what each capture
//! used.
//!
//! A bucket ladder's graphs do not cost one constant each, and no single capture's reading is that
//! constant either. The driver reserves device memory in 2 MiB chunks, so most captures read as
//! nothing at all and the occasional one reads as a whole chunk, while the first capture carries a
//! one-time context and pool reservation on top of its own state. The 2026-08-12 H100 spike
//! measured exactly that shape over 96 buckets — 34 MiB on the first graph, eleven 2 MiB chunks
//! spread over the rest, nothing on the other 84 — and quoted what survives it as an affine rule:
//! a bucket ladder of `n` graphs costs a fixed term plus `n` marginal terms.
//!
//! [`GraphMemory::fit`] puts an ordinary least-squares line through those readings' running total
//! and reports its two terms. They are not the two the spike quoted, and a fit of those readings
//! cannot be: it charges the first capture's one-time reservation to every point of the line
//! rather than to graph one alone, so the spike's own readings report 31.3 MiB fixed and 266.6 KiB
//! a graph where its two-point split named 34 MiB and 237 KiB. What the fit reports is what a
//! capture that has run cost, and never a ceiling for one that has not —
//! [`GraphMemory::bytes_for`] carries that boundary.
//!
//! The arithmetic reaches no device: the readings are numbers the caller has already taken, so
//! every boundary it has is settled here rather than on a rig.

use thiserror::Error;

use crate::context::DeviceBytes;

/// Rejected fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GraphMemoryError {
    #[error(
        "a fixed term and a marginal one cannot be separated from fewer than two graphs' used \
         bytes, and this fit was given {graphs}; report what the captures used in total instead"
    )]
    FitWithoutTwoGraphs { graphs: usize },
}

/// A graph set's device memory split into the part that does not grow with the number of graphs
/// and the part each graph adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphMemory {
    fixed: DeviceBytes,
    marginal: DeviceBytes,
}

impl GraphMemory {
    /// Fits `used` — one reading per graph, in capture order — to a fixed term plus a marginal
    /// term per graph.
    ///
    /// Reading `i` contributes the point `(i, what the first i graphs used between them)`, and the
    /// two terms are the intercept and the slope of the ordinary least-squares line through those
    /// points. Fitting the running total rather than the readings themselves is what makes the two
    /// terms answer for a whole bucket ladder rather than for one graph of it: the line puts `n`
    /// graphs at `fixed + marginal * n`, which is [`GraphMemory::bytes_for`].
    ///
    /// Both terms are whole bytes, each the nearest to the line's own value, ties away from zero;
    /// rounding up to the driver's 2 MiB granularity is the caller's to do. Readings that
    /// grow can put the line below zero at no graphs at all, and a fixed term there reports as
    /// none, since memory has no quantity below zero. Only the fixed term reaches it: a running
    /// total of byte counts never falls, so the fitted slope is never negative.
    ///
    /// # Errors
    ///
    /// [`GraphMemoryError::FitWithoutTwoGraphs`] where `used` holds fewer than two readings: one
    /// point names no line, and two terms need two. Nothing else is rejected. Every point sits at
    /// its own graph count, so no two share an x and the slope's denominator is positive for every
    /// `used` of two or more.
    pub fn fit(used: &[DeviceBytes]) -> Result<Self, GraphMemoryError> {
        let graphs = used.len();
        if graphs < 2 {
            return Err(GraphMemoryError::FitWithoutTwoGraphs { graphs });
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "a graph count is exact in f64 far past any number of captures a device holds"
        )]
        let count = graphs as f64;
        let mean_graph = points(used).map(|(graph, _)| graph).sum::<f64>() / count;
        let mean_total = points(used).map(|(_, total)| total).sum::<f64>() / count;
        let covariance: f64 = points(used)
            .map(|(graph, total)| (graph - mean_graph) * (total - mean_total))
            .sum();
        let spread: f64 = points(used)
            .map(|(graph, _)| (graph - mean_graph).powi(2))
            .sum();

        let marginal = covariance / spread;
        Ok(Self {
            fixed: whole_bytes(mean_total - marginal * mean_graph),
            marginal: whole_bytes(marginal),
        })
    }

    /// Where the fitted line puts a set of `graphs` graphs: the fixed term plus one marginal term
    /// each, saturating at the largest byte count there is rather than wrapping.
    ///
    /// A report of what a capture that has run cost, and no ceiling for one that has not. Least
    /// squares minimises the squared residual, which bounds no single point, so the line can sit
    /// below what a bucket ladder of that many graphs went on to use: keeping the 2026-08-12
    /// spike's own size, its first reading and its eleven driver chunks, and moving only which
    /// graphs the chunks land on, puts the line 16.87 MiB — eight whole chunks — under what those
    /// graphs used. A caller sizing device memory from these terms needs headroom of its own.
    #[must_use]
    pub const fn bytes_for(self, graphs: usize) -> DeviceBytes {
        let marginal = self.marginal.get().saturating_mul(graphs);
        DeviceBytes::new(marginal.saturating_add(self.fixed.get()))
    }

    /// The part that does not grow with the number of graphs.
    #[must_use]
    pub const fn fixed(self) -> DeviceBytes {
        self.fixed
    }

    /// What each further graph adds.
    #[must_use]
    pub const fn marginal(self) -> DeviceBytes {
        self.marginal
    }
}

/// One point per reading: how many graphs have been captured, and what they used between them.
///
/// The running total is carried as `f64` rather than as a `usize`, since `used` is a slice of byte
/// counts and nothing in its type bounds their sum: two maxima already overflow a `usize`. `f64`
/// does not bound that sum either — it trades the overflow for inexactness, and is exact only
/// while the running total stays under 2^53 bytes, 8 PiB. Past there a reading smaller than the
/// total's own ulp adds nothing to it at all and the marginal term comes back short: the readings
/// `[2^53, 1, 1, 1]` fit a marginal term of none where one byte a graph is right. Device readings
/// run to some 1e11 bytes, so a hundred of them total 1e13 and clear the boundary by a factor of
/// 900.
#[expect(
    clippy::cast_precision_loss,
    reason = "a byte count and its running total are exact in f64 below the 2^53 bytes above"
)]
fn points(used: &[DeviceBytes]) -> impl Iterator<Item = (f64, f64)> + '_ {
    used.iter()
        .scan(0.0_f64, |total, reading| {
            *total += reading.get() as f64;
            Some(*total)
        })
        .enumerate()
        .map(|(index, total)| ((index + 1) as f64, total))
}

/// The whole byte nearest `value`, ties away from zero.
///
/// Rust's float-to-integer cast saturates in both directions, which is where a fit below zero and
/// a fit past `usize::MAX` land.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the saturating cast is the clamp: a line below zero reports no memory at all, and \
              one past the largest byte count reports that"
)]
fn whole_bytes(value: f64) -> DeviceBytes {
    DeviceBytes::new(value.round() as usize)
}

#[cfg(test)]
mod tests {
    use super::{GraphMemory, GraphMemoryError};
    use crate::context::DeviceBytes;

    const KIB: usize = 1024;
    const MIB: usize = 1024 * KIB;

    fn readings(bytes: &[usize]) -> Vec<DeviceBytes> {
        bytes.iter().copied().map(DeviceBytes::new).collect()
    }

    /// The shape the 2026-08-12 H100 run measured over its 96 captures: 34 MiB on the first
    /// graph, a 2 MiB driver chunk on the eleven graphs `chunks` names, and nothing at all on the
    /// other 84. The driver reserves device memory in 2 MiB chunks, so one graph's own state
    /// reads as nothing until enough of them have accumulated to take the next chunk.
    fn spike_shape(chunks: [usize; 11]) -> Vec<DeviceBytes> {
        let mut used = vec![DeviceBytes::new(0); 96];
        used[0] = DeviceBytes::new(34 * MIB);
        for graph in chunks {
            used[graph] = DeviceBytes::new(2 * MIB);
        }
        used
    }

    /// What that run's captures used graph by graph, chunks and all.
    fn spike_readings() -> Vec<DeviceBytes> {
        spike_shape([13, 26, 32, 40, 49, 54, 56, 64, 68, 82, 95])
    }

    #[test]
    fn readings_an_affine_rule_generates_fit_back_to_its_two_terms() {
        // The rule the 2026-08-12 H100 run quoted over its bucket ladder of 96: 34 MiB fixed and
        // 237 KiB per graph. A bucket ladder whose readings follow it — the first graph paying
        // the fixed term as well as its own — has to fit back to the two terms that generated it.
        let (fixed, marginal) = (34 * MIB, 237 * KIB);
        let mut used = vec![DeviceBytes::new(marginal); 96];
        used[0] = DeviceBytes::new(fixed + marginal);

        let memory = GraphMemory::fit(&used).unwrap();

        assert_eq!(memory.fixed(), DeviceBytes::new(fixed));
        assert_eq!(memory.marginal(), DeviceBytes::new(marginal));
    }

    #[test]
    fn the_spikes_own_readings_report_less_fixed_and_more_per_graph_than_it_quoted() {
        let memory = GraphMemory::fit(&spike_readings()).unwrap();

        // Least squares charges the first graph's one-time reservation to every point of the
        // line rather than to graph one alone, so the split it reports is not the two-point one
        // the spike quoted: under 34 MiB fixed, and over 237 KiB a graph to make up for it.
        assert!(memory.fixed() < DeviceBytes::new(34 * MIB));
        assert!(memory.marginal() > DeviceBytes::new(237 * KIB));

        // And the marginal term amortises the driver's chunks rather than reporting one: no
        // graph of the bucket ladder costs a whole reservation of its own.
        assert!(memory.marginal() < DeviceBytes::new(2 * MIB));
    }

    #[test]
    fn a_report_is_the_fixed_term_and_one_marginal_term_a_graph() {
        // Running totals 10 and 12 MiB against graph counts 1 and 2: a line rising 2 MiB a graph
        // from 8 MiB at no graphs at all.
        let memory = GraphMemory::fit(&readings(&[10 * MIB, 2 * MIB])).unwrap();

        // No graphs at all is the fixed term on its own, one graph is the first reading back, and
        // a bucket ladder of 96 is that 8 MiB and ninety-six 2 MiB terms: 200 MiB. The report is
        // the line's value at any number of graphs, not only at the ones it was fitted over.
        assert_eq!(memory.bytes_for(0), DeviceBytes::new(8 * MIB));
        assert_eq!(memory.bytes_for(1), DeviceBytes::new(10 * MIB));
        assert_eq!(memory.bytes_for(96), DeviceBytes::new(200 * MIB));
    }

    #[test]
    fn a_bucket_ladder_of_the_spikes_shape_can_report_less_than_its_graphs_used() {
        // The spike's own shape — its 96 graphs, its 34 MiB first reading, its eleven 2 MiB
        // driver chunks — with only the graphs the chunks land on moved, here to the last eleven.
        // Least squares minimises the squared residual over all 96 points and bounds none of
        // them, so the line this shape fits reports eight whole driver chunks less than these
        // graphs used. The two terms report what a capture cost; they are no ceiling.
        let used = spike_shape([85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]);
        let total = DeviceBytes::new(used.iter().map(|reading| reading.get()).sum());

        let memory = GraphMemory::fit(&used).unwrap();

        let reported = memory.bytes_for(used.len());
        assert!(
            total.saturating_sub(reported) > DeviceBytes::new(8 * 2 * MIB),
            "reported {reported} for {} graphs against {total} used",
            used.len()
        );
    }

    #[test]
    fn where_a_reading_falls_moves_both_terms_and_not_just_what_they_total() {
        // Same first reading, same last, same total: only the graph the 4 MiB lands on differs.
        // The rule the spike quoted — the first reading is the fixed term, the rest average into
        // the marginal one — cannot tell these apart. Least squares over all four points can.
        let early = GraphMemory::fit(&readings(&[20 * MIB, 4 * MIB, 0, 0])).unwrap();
        let late = GraphMemory::fit(&readings(&[20 * MIB, 0, 4 * MIB, 0])).unwrap();

        // Running totals 20, 24, 24, 24 MiB against graph counts 1 to 4: a slope of 6/5 MiB —
        // 1258291.2 bytes, nearest whole byte 1258291 — through the means (2.5, 23 MiB).
        assert_eq!(early.fixed(), DeviceBytes::new(20 * MIB));
        assert_eq!(early.marginal(), DeviceBytes::new(1_258_291));
        // Running totals 20, 20, 24, 24 MiB: a slope of 8/5 MiB — 1677721.6 bytes, nearest whole
        // byte 1677722 — through the means (2.5, 22 MiB), which puts the line 18 MiB up at zero.
        assert_eq!(late.fixed(), DeviceBytes::new(18 * MIB));
        assert_eq!(late.marginal(), DeviceBytes::new(1_677_722));
    }

    #[test]
    fn a_fixed_term_the_readings_put_below_zero_is_no_memory_at_all() {
        // Nothing measurable until the last graph: running totals 0, 0, 30 MiB against graph
        // counts 1 to 3 rise 15 MiB a graph, which puts the line 20 MiB below zero at no graphs.
        // Memory has no quantity below zero, so the fixed term is none at all and the marginal
        // term stands.
        let memory = GraphMemory::fit(&readings(&[0, 0, 30 * MIB])).unwrap();

        assert_eq!(memory.fixed(), DeviceBytes::new(0));
        assert_eq!(memory.marginal(), DeviceBytes::new(15 * MIB));
    }

    #[test]
    fn a_term_that_falls_between_two_bytes_takes_the_nearer_and_ties_round_up() {
        // Running totals 0, 1, 1 bytes against graph counts 1 to 3: a slope of exactly half a
        // byte, which is the one value the nearest whole byte does not name on its own.
        let memory = GraphMemory::fit(&readings(&[0, 1, 0])).unwrap();

        assert_eq!(memory.marginal(), DeviceBytes::new(1));
    }

    #[test]
    fn readings_that_measured_nothing_fit_no_memory_at_all() {
        // The driver reserves in 2 MiB chunks, so a bucket ladder short enough to fit in one reads
        // as nothing at every graph. That is a line — the flat one through the origin — and not
        // a refusal.
        let memory = GraphMemory::fit(&readings(&[0, 0, 0, 0])).unwrap();

        assert_eq!(memory.fixed(), DeviceBytes::new(0));
        assert_eq!(memory.marginal(), DeviceBytes::new(0));
    }

    #[test]
    fn readings_that_total_past_a_usize_still_fit() {
        // `used` is a slice of byte counts and nothing in its type bounds their sum, so the
        // running total is carried in floating point rather than in a usize two maxima overflow.
        // Two equal readings rise by one of them a graph from nothing at no graphs; f64 carries
        // that slope as 2^64, and the cast back saturates to the largest usize there is.
        let memory = GraphMemory::fit(&readings(&[usize::MAX, usize::MAX])).unwrap();

        assert_eq!(memory.fixed(), DeviceBytes::new(0));
        assert_eq!(memory.marginal(), DeviceBytes::new(usize::MAX));
    }

    #[test]
    fn a_report_past_what_a_usize_names_saturates_rather_than_wrapping() {
        // The largest terms a fit can return: two maxima rise by one of them a graph, and the
        // report over the two graphs they were fitted over is twice what a usize names.
        let maxima = GraphMemory::fit(&readings(&[usize::MAX, usize::MAX])).unwrap();

        assert_eq!(maxima.bytes_for(2), DeviceBytes::new(usize::MAX));

        // Each term can sit inside a usize with only their sum outside it: two maxima and then
        // nothing rise 2^63 a graph from two thirds of 2^64 at no graphs at all.
        let sum = GraphMemory::fit(&readings(&[usize::MAX, usize::MAX, 0])).unwrap();

        assert_eq!(sum.marginal(), DeviceBytes::new(1 << 63));
        assert!(sum.fixed() < DeviceBytes::new(usize::MAX));
        assert_eq!(sum.bytes_for(1), DeviceBytes::new(usize::MAX));
    }

    #[test]
    fn a_fit_needs_two_graphs_to_have_two_terms() {
        assert_eq!(
            GraphMemory::fit(&[]),
            Err(GraphMemoryError::FitWithoutTwoGraphs { graphs: 0 })
        );
        assert_eq!(
            GraphMemory::fit(&readings(&[34 * MIB])),
            Err(GraphMemoryError::FitWithoutTwoGraphs { graphs: 1 })
        );

        // Two readings name one line exactly: the points (1, 34 MiB) and (2, 35 MiB) rise 1 MiB
        // per graph from 33 MiB at no graphs at all.
        let two = GraphMemory::fit(&readings(&[34 * MIB, MIB])).unwrap();
        assert_eq!(two.fixed(), DeviceBytes::new(33 * MIB));
        assert_eq!(two.marginal(), DeviceBytes::new(MIB));
    }
}
