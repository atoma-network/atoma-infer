use thiserror::Error;
use tokenizers::{tokenizer::Tokenizer, Encoding, Error};
use tokio::sync::{
    mpsc::{self, error::SendError},
    oneshot,
};
use tracing::{error, info, info_span, instrument, trace, Span};

/// Represents a request to encode a string input into tokens.
pub struct EncodeTokenizerRequest {
    /// The input string to be tokenized.
    pub input: String,
    /// Optional maximum length to truncate the input. If provided, the input will be
    /// truncated to this number of characters from the end.
    pub truncate: Option<usize>,
    /// Channel for sending the tokenization result, which includes:
    /// - The `Encoding` containing the tokenized output
    /// - The original (potentially truncated) input string
    pub sender: oneshot::Sender<Result<(Encoding, String), TokenizerError>>,
    /// The current tracing span for context propagation and logging.
    pub span: Span,
}

/// `DecodeTokenizerRequest` - A request to decode a single token ID into text
pub struct DecodeTokenizerRequest {
    /// The token ID to be decoded
    pub token_id: u32,
    /// Channel for sending the decoding result
    pub sender: oneshot::Sender<Result<String, TokenizerError>>,
    /// The current tracing span for context propagation
    pub span: Span,
}

/// `TokenizerWorker` - A struct responsible for managing tokenization tasks
///
/// This struct provides functionality to start and manage multiple tokenizer
/// workers, which handle encoding requests in parallel. It uses a round-robin
/// approach to distribute tasks among workers for efficient processing.
pub struct TokenizerWorker;

impl TokenizerWorker {
    /// Starts the tokenizer workers
    pub async fn start(
        tokenizer: Tokenizer,
        receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
        workers: usize,
    ) -> Result<(), TokenizerError> {
        let mut senders = Vec::with_capacity(workers);

        for i in 0..workers {
            let tokenizer_clone = tokenizer.clone();
            let (sender, worker_receiver) = mpsc::unbounded_channel();
            senders.push(sender);

            // Spawning the worker
            let span = info_span!("tokenizer-worker");
            tokio::task::spawn_blocking(move || {
                let _span = span.clone();
                let _enter = _span.enter();
                info!("Starting {i}-th tokenizer task");
                start_tokenizer_task(tokenizer_clone, worker_receiver, span)?;
                Ok::<_, TokenizerError>(())
            });
        }

        // Create tokenization round robin task
        tokio::spawn(round_robin_task(receiver, senders));

        Ok(())
    }
}

/// Starts a new tokenizer task that processes encoding requests.
///
/// This function runs in a loop, continuously receiving and processing
/// `EncodeTokenizerRequest`s. It uses the provided tokenizer to encode
/// input strings and sends the results back through a oneshot channel.
///
/// # Arguments
///
/// * `tokenizer` - The tokenizer used for encoding input strings.
/// * `receiver` - An unbounded receiver for `EncodeTokenizerRequest`s.
/// * `span` - A tracing span for logging and debugging purposes.
///
/// # Returns
///
/// Returns `Ok(())` if the task completes successfully, or a `TokenizerError`
/// if there's an error during processing.
///
/// # Note
///
/// This function is designed to be run in a blocking task, as it uses
/// `blocking_recv()` to receive requests.
#[instrument(skip_all)]
fn start_tokenizer_task(
    tokenizer: Tokenizer,
    mut receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
    span: Span,
) -> Result<(), TokenizerError> {
    let _enter = span.enter();
    trace!("Starting tokenizer task..");

    // Loops over requests
    while let Some(request) = receiver.blocking_recv() {
        info!("Received new `EncodeTokenizerRequest`");
        let EncodeTokenizerRequest {
            input,
            truncate,
            sender,
            span,
        } = request;
        span.in_scope(|| {
            let prepared_inputs = prepare_inputs(&tokenizer, input, truncate);
            sender.send(prepared_inputs).unwrap_or(())
        });
    }
    Ok(())
}

/// Distributes tokenization requests among multiple workers using a round-robin algorithm.
///
/// This function implements a simple round-robin scheduling strategy to balance the workload
/// across multiple tokenizer workers. It continuously receives requests from a central receiver
/// and distributes them to the workers in a circular order.
///
/// # Arguments
///
/// * `receiver` - An unbounded receiver for `EncodeTokenizerRequest`s.
/// * `senders` - A vector of unbounded senders, each corresponding to a tokenizer worker.
///
/// # Returns
///
/// Returns `Ok(())` if the task completes successfully, or a `TokenizerError` if there's an error
/// sending a request to a worker.
///
/// # Behavior
///
/// The function runs in an infinite loop, cycling through the list of senders. For each sender:
/// 1. It attempts to receive a request from the central receiver.
/// 2. If a request is received, it's forwarded to the current worker.
/// 3. If `None` is received (indicating the channel was closed), the function logs an error and exits.
///
/// This ensures that requests are distributed evenly among all available workers.
///
/// Check https://en.wikipedia.org/wiki/Round-robin_scheduling
/// for more details.
async fn round_robin_task(
    mut receiver: mpsc::UnboundedReceiver<EncodeTokenizerRequest>,
    senders: Vec<mpsc::UnboundedSender<EncodeTokenizerRequest>>,
) -> Result<(), TokenizerError> {
    loop {
        for sender in &senders {
            match receiver.recv().await {
                None => {
                    error!("Received None from the tokenizer receiver");
                    return Ok(());
                }
                Some(request) => {
                    trace!("Received a new request from the tokenizer receiver");
                    sender.send(request)?;
                }
            }
        }
    }
}

/// Prepares and tokenizes the input string, optionally truncating it.
///
/// # Arguments
///
/// * `tokenizer` - The tokenizer to use for encoding the input.
/// * `input` - The input string to be tokenized.
/// * `truncate` - Optional maximum number of characters to keep from the end of the input.
///
/// # Returns
///
/// A Result containing a tuple of:
/// - The tokenized `Encoding`
/// - The (potentially truncated) input string
///
/// # Errors
///
/// Returns a `TokenizerError` if the tokenization process fails.
fn prepare_inputs(
    tokenizer: &Tokenizer,
    input: String,
    truncate: Option<usize>,
) -> Result<(Encoding, String), TokenizerError> {
    let input = if let Some(truncate) = truncate {
        if truncate > input.chars().count() {
            input
        } else {
            let start = input
                .char_indices()
                .nth(input.chars().count() - truncate)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            input[start..].to_string()
        }
    } else {
        input
    };
    let encoding = tokenizer.encode(input.clone(), true)?;
    Ok((encoding, input))
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Oneshot sender error: `{0}`")]
    OneshotSenderError(String),
    #[error("Tokenizer error: `{0}`")]
    Tokenizer(#[from] Error),
    #[error("Send error: `{0}`")]
    SendError(#[from] SendError<EncodeTokenizerRequest>),
}
