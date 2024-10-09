use crate::api::chat_completions::ChatCompletionChunk;
#[cfg(feature = "vllm")]
use atoma_backends::GenerateStreamingOutput;
use axum::{response::sse::Event, Error};
use flume::Receiver;
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A structure for streaming chat completion chunks.
///
/// `Streamer` manages the reception of `ChatCompletionChunk`s and tracks the current status
/// of the streaming process.
pub struct Streamer {
    /// The receiver end of a channel for incoming `ChatCompletionChunk`s.
    receiver: Receiver<GenerateStreamingOutput>,
    /// The current status of the streaming process.
    status: StreamStatus,
    /// The model used for generating the output.
    model: String,
}

impl Streamer {
    /// Creates a new `Streamer` with the specified receiver and model.
    pub fn new(receiver: Receiver<GenerateStreamingOutput>, model: String) -> Self {
        Self {
            receiver,
            status: StreamStatus::NotStarted,
            model,
        }
    }
}

/// Represents the various states of a streaming process.
///
/// This enum is used to track and communicate the current state of a `Streamer`,
/// allowing for proper handling of different scenarios during streaming.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamStatus {
    /// Indicates that the streaming process has not started yet.
    NotStarted,
    /// Indicates that the streaming process has started and is actively receiving data.
    ///
    /// This is the initial state when a stream begins and is ready to process incoming chunks.
    Started,
    /// Indicates that the streaming process has completed successfully.
    ///
    /// This state is reached when all data has been received and processed without errors.
    Completed,
    /// Indicates that the streaming process has failed, with an associated error message.
    ///
    /// This state is used when an error occurs during streaming, providing context about the failure.
    Failed {
        /// A descriptive error message explaining the reason for the failure.
        error: String,
    },
    /// Indicates that the streaming process was interrupted before completion.
    ///
    /// This state is used when the stream is stopped prematurely, either by user action or system events.
    Interrupted {
        /// A description of why the stream was interrupted.
        reason: String,
    },
}

/// Represents the different types of responses that can be received during streaming.
///
/// This enum is used to encapsulate the various outcomes of a streaming operation,
/// allowing for proper handling of successful chunks, errors, and stream completion.
pub enum StreamResponse {
    /// Represents a successful chunk of generated output.
    ///
    /// This variant is used when a new piece of generated content is available for streaming.
    Chunk(GenerateStreamingOutput),
    /// Represents an error that occurred during the streaming process.
    ///
    /// This variant is used when an error is encountered, allowing for proper error handling and reporting.
    Error(String),
    /// Indicates that the streaming process has finished successfully.
    ///
    /// This variant is used to signal the end of the stream when all data has been processed without errors.
    Finished,
}

impl Stream for Streamer {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamStatus::Completed {
            return Poll::Ready(None);
        }
        match self.receiver.try_recv() {
            Ok(chunk) => match chunk {
                StreamResponse::Chunk(chunk) => {
                    if self.status != StreamStatus::Started {
                        self.status = StreamStatus::Started;
                    }
                    let response = ChatCompletionChunk::try_from((self.model.clone(), chunk))
                        .map_err(|e| Error::new(e))?;
                    Poll::Ready(Some(Ok(Event::default().json_data(response))))
                }
                StreamResponse::Finished => {
                    self.status = StreamStatus::Completed;
                    Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
                }
                StreamResponse::Error(error) => {
                    self.status = StreamStatus::Failed { error };
                    Poll::Ready(Some(Ok(Event::default().data(e))))
                }
            },
            Err(error) => {
                if self.status == StreamStatus::Started
                    && error == flume::TryRecvError::Disconnected
                {
                    self.status = StreamStatus::Interrupted {
                        reason: "Stream disconnected".to_string(),
                    };
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}
