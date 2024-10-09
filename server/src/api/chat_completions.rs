//! Responsible for creating the json schema associated with the AtomaAPI, which is modeled after OpenAI's own API.

#[cfg(feature = "vllm")]
use atoma_backends::{
    GenerateParameters, GenerateRequest, GenerateRequestOutput, GenerateStreamingOutput,
};
use std::{
    collections::HashMap,
    time::{Instant, SystemTime},
};
use utoipa::ToSchema;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

// TODO: fields that are named `r#type` should have values that represent
// actual expected types that are deserializable from a string instead of
// just `String` since a user could input anything if we allow them to.
// On our end, it's also beneficial since we will want to match on that
// type. For now a naive version of this is OK, but may want to do this
// before deploying v1 of the schema to avoid misuse.

/// ID of the model to use.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename(serialize = "model", deserialize = "model"))]
pub enum Model {
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-8B",
        deserialize = "meta-llama/Meta-Llama-3-8B"
    ))]
    Llama38b,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-8B-Instruct",
        deserialize = "meta-llama/Meta-Llama-3-8B-Instruct"
    ))]
    Llama38bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-70B",
        deserialize = "meta-llama/Meta-Llama-3-70B"
    ))]
    Llama370b,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-70B-Instruct",
        deserialize = "meta-llama/Meta-Llama-3-70B-Instruct"
    ))]
    Llama370bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-8B",
        deserialize = "meta-llama/Llama-3.1-8B"
    ))]
    Llama318b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-8B-Instruct",
        deserialize = "meta-llama/Llama-3.1-8B-Instruct"
    ))]
    Llama318bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-70B",
        deserialize = "meta-llama/Llama-3.1-70B"
    ))]
    Llama3170b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-70B-Instruct",
        deserialize = "meta-llama/Llama-3.1-70B-Instruct"
    ))]
    Llama3170bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-405B",
        deserialize = "meta-llama/Llama-3.1-405B"
    ))]
    Llama31405b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-405B-Instruct",
        deserialize = "meta-llama/Llama-3.1-405B-Instruct"
    ))]
    Llama31405bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-1B",
        deserialize = "meta-llama/Llama-3.2-1B"
    ))]
    Llama321b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-1B-Instruct",
        deserialize = "meta-llama/Llama-3.2-1B-Instruct"
    ))]
    Llama321bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-3B",
        deserialize = "meta-llama/Llama-3.2-3B"
    ))]
    Llama323b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-3B-Instruct",
        deserialize = "meta-llama/Llama-3.2-3B-Instruct"
    ))]
    Llama323bInstruct,
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Llama38b => write!(f, "meta-llama/Meta-Llama-3-8B"),
            Model::Llama38bInstruct => write!(f, "meta-llama/Meta-Llama-3-8B-Instruct"),
            Model::Llama370b => write!(f, "meta-llama/Meta-Llama-3-70B"),
            Model::Llama370bInstruct => write!(f, "meta-llama/Meta-Llama-3-70B-Instruct"),
            Model::Llama318b => write!(f, "meta-llama/Llama-3.1-8B"),
            Model::Llama318bInstruct => write!(f, "meta-llama/Llama-3.1-8B-Instruct"),
            Model::Llama3170b => write!(f, "meta-llama/Llama-3.1-70B"),
            Model::Llama3170bInstruct => write!(f, "meta-llama/Llama-3.1-70B-Instruct"),
            Model::Llama31405b => write!(f, "meta-llama/Llama-3.1-405B"),
            Model::Llama31405bInstruct => write!(f, "meta-llama/Llama-3.1-405B-Instruct"),
            Model::Llama321b => write!(f, "meta-llama/Llama-3.2-1B"),
            Model::Llama321bInstruct => write!(f, "meta-llama/Llama-3.2-1B-Instruct"),
            Model::Llama323b => write!(f, "meta-llama/Llama-3.2-3B"),
            Model::Llama323bInstruct => write!(f, "meta-llama/Llama-3.2-3B-Instruct"),
        }
    }
}

/// A message that is part of a conversation which is based on the role
/// of the author of the message.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// The role of the messages author, in this case system.
    System {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// The role of the messages author, in this case user.
    User {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// The role of the messages author, in this case assistant.
    Assistant {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// The refusal message by the assistant.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        /// The tool calls generated by the model, such as function calls.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    /// The role of the messages author, in this case tool.
    Tool {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// Tool call that this message is responding to.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        tool_call_id: String,
    },
}

impl Message {
    pub fn to_prompt_string(&self) -> String {
        match self {
            Message::System { content, name } => {
                let role = "System";
                messages::format_message(role, content, name)
            }
            Message::User { content, name } => {
                let role = "User";
                messages::format_message(role, content, name)
            }
            Message::Assistant {
                content,
                name,
                refusal,
                tool_calls,
            } => {
                let role = "Assistant";
                let mut prompt = messages::format_message(role, content, name);
                if let Some(refusal_message) = refusal {
                    prompt.push_str(&format!("\nRefusal: {}", refusal_message));
                }
                if !tool_calls.is_empty() {
                    prompt.push_str(&format!("\nTool Calls: {:?}", tool_calls));
                }
                prompt
            }
            Message::Tool {
                content,
                tool_call_id,
            } => {
                let role = "Tool";
                let mut prompt = messages::format_message(role, content, &None);
                prompt.push_str(&format!("\nTool Call ID: {}", tool_call_id));
                prompt
            }
        }
    }
}

pub(crate) mod messages {
    use super::{Message, MessageContent};

    /// Helper function to format a message
    pub(crate) fn format_message(
        role: &str,
        content: &Option<MessageContent>,
        name: &Option<String>,
    ) -> String {
        let name_str = if let Some(name) = name {
            format!(" ({})", name)
        } else {
            String::new()
        };

        let content_str = if let Some(content) = content {
            format!("{}", content)
        } else {
            String::new()
        };

        format!("{}{}: {}", role, name_str, content_str)
    }

    /// Function to convert a list of messages to a prompt string
    pub(crate) fn messages_to_prompt(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|message| message.to_prompt_string())
            .collect::<Vec<_>>()
            .join("\n\n") // Separate each message by two newlines
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum MessageContent {
    /// The text contents of the message.
    #[serde(rename(serialize = "text", deserialize = "text"))]
    Text(String),
    /// An array of content parts with a defined type, each can be of type text or image_url when passing in images.
    /// You can pass multiple images by adding multiple image_url content parts. Image input is only supported when using the gpt-4o model.
    #[serde(rename(serialize = "array", deserialize = "array"))]
    Array(Vec<MessageContentPart>),
}

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContent::Text(text) => write!(f, "{}", text),
            MessageContent::Array(parts) => {
                let mut content = String::new();
                for part in parts {
                    content.push_str(&format!("{}\n", part))
                }
                write!(f, "{}", content)
            }
        }
    }
}

// We manually implement Deserialize here for more control.
impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Value = Value::deserialize(deserializer)?;

        if let Some(s) = value.as_str() {
            return Ok(MessageContent::Text(s.to_string()));
        }

        if let Some(arr) = value.as_array() {
            let parts: Result<Vec<MessageContentPart>, _> = arr
                .iter()
                .map(|v| serde_json::from_value(v.clone()).map_err(serde::de::Error::custom))
                .collect();
            return Ok(MessageContent::Array(parts?));
        }

        Err(serde::de::Error::custom(
            "Expected a string or an array of content parts",
        ))
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum MessageContentPart {
    #[serde(rename(serialize = "text", deserialize = "text"))]
    Text {
        /// The type of the content part.
        #[serde(rename(serialize = "type", deserialize = "type"))]
        r#type: String,
        /// The text content.
        text: String,
    },
    #[serde(rename(serialize = "image", deserialize = "image"))]
    Image {
        /// The type of the content part.
        #[serde(rename(serialize = "type", deserialize = "type"))]
        r#type: String,
        image_url: MessageContentPartImageUrl,
    },
}

impl std::fmt::Display for MessageContentPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContentPart::Text { r#type, text } => {
                write!(f, "{}: {}", r#type, text)
            }
            MessageContentPart::Image { r#type, image_url } => {
                write!(f, "{}: [Image URL: {}]", r#type, image_url)
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename(serialize = "image_url", deserialize = "image_url"))]
pub struct MessageContentPartImageUrl {
    /// Either a URL of the image or the base64 encoded image data.
    url: String,
    /// Specifies the detail level of the image.
    detail: Option<String>,
}

/// Implementing Display for MessageContentPartImageUrl
impl std::fmt::Display for MessageContentPartImageUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(detail) => write!(f, "Image URL: {}, Detail: {}", self.url, detail),
            None => write!(f, "Image URL: {}", self.url),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename(serialize = "tool_call", deserialize = "tool_call"))]
pub struct ToolCall {
    /// The ID of the tool call.
    id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename(serialize = "type", deserialize = "type"))]
    r#type: String,
    /// The function that the model called.
    function: ToolCallFunction,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCallFunction {
    /// The name of the function to call.
    name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    /// Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema.
    /// Validate the arguments in your code before calling your function.
    arguments: Value,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename(serialize = "tool", deserialize = "tool"))]
pub struct Tool {
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename(serialize = "type", deserialize = "type"))]
    r#type: String,
    /// The function that the model called.
    function: ToolFunction,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolFunction {
    /// Description of the function to call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    /// The name of the function to call.
    name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    /// Whether to enable strict schema adherence when generating the function call. If set to true, the
    /// model will follow the exact schema defined in the parameters field. Only a subset of JSON Schema is supported when strict is true
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// The stop condition for the chat completion.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename(serialize = "stop", deserialize = "stop"))]
#[serde(untagged)]
pub enum StopCondition {
    Array(Vec<String>),
    String(String),
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema, ToSchema)]
#[serde(rename(serialize = "requestBody", deserialize = "requestBody"))]
pub struct RequestBody {
    /// A list of messages comprising the conversation so far.
    messages: Vec<Message>,
    /// ID of the model to use.
    model: Model,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a JSON object that maps tokens (specified as their token ID in the tokenizer) to an associated bias value from -100 to 100.
    logit_bias: Option<HashMap<String, f32>>,
    /// Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.
    /// logprobs must be set to true if this parameter is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<i32>,
    /// An upper bound for the number of tokens that can be generated for a completion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    /// How many chat completion choices to generate for each input message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    /// A seed to use for random number generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<StopCondition>,
    /// If set, the server will stream the results as they come in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
    /// of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    /// A unique identifier representing your end-user, which can help the system to monitor and detect abuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl RequestBody {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn messages(&self) -> &Vec<Message> {
        &self.messages
    }

    pub fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }

    pub fn logit_bias(&self) -> Option<HashMap<String, f32>> {
        self.logit_bias.clone()
    }

    pub fn logprobs(&self) -> Option<bool> {
        self.logprobs
    }

    pub fn top_logprobs(&self) -> Option<i32> {
        self.top_logprobs
    }

    pub fn max_completion_tokens(&self) -> Option<u32> {
        self.max_completion_tokens
    }

    pub fn n(&self) -> Option<usize> {
        self.n
    }

    pub fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }

    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    pub fn stop(&self) -> Option<&StopCondition> {
        self.stop.as_ref()
    }

    pub fn stream(&self) -> Option<bool> {
        self.stream
    }

    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    pub fn tools(&self) -> Option<&Vec<Tool>> {
        self.tools.as_ref()
    }

    pub fn user(&self) -> Option<&String> {
        self.user.as_ref()
    }

    /// The control structure for testing the rust to json api, and schema.
    /// Represents all possible values for serialization.
    #[cfg(test)]
    pub(crate) fn control() -> Self {
        use serde_json::json;

        Self {
            model: Model::Llama38b,
            messages: vec![
                Message::System {
                    content: Some(MessageContent::Text("test".into())),
                    name: Some("test".into()),
                },
                Message::User {
                    content: Some(MessageContent::Array(vec![
                        MessageContentPart::Text {
                            r#type: "test".into(),
                            text: "test".into(),
                        },
                        MessageContentPart::Image {
                            r#type: "test".into(),
                            image_url: MessageContentPartImageUrl {
                                url: "https://imgur.com/m6eWDSz".into(),
                                detail: Some("high".into()),
                            },
                        },
                    ])),
                    name: Some("test".into()),
                },
                Message::Assistant {
                    content: Some(MessageContent::Text("test".into())),
                    name: Some("test".into()),
                    refusal: None,
                    tool_calls: vec![ToolCall {
                        id: "chatcmpl-123".into(),
                        r#type: "function".into(),
                        function: ToolCallFunction {
                            name: "myFunction".into(),
                            arguments: serde_json::json!({"key": "value"}),
                        },
                    }],
                },
                Message::Tool {
                    content: Some(MessageContent::Text("test".into())),
                    tool_call_id: "0".into(),
                },
            ],
            frequency_penalty: Some(0.5),
            logit_bias: Some(HashMap::from_iter(vec![(String::from("test"), 0.5)])),
            logprobs: Some(true),
            top_logprobs: Some(1),
            max_completion_tokens: Some(100),
            n: Some(1),
            presence_penalty: Some(0.5),
            seed: Some(1),
            stop: Some(StopCondition::String("test".into())),
            stream: Some(true),
            temperature: Some(0.5),
            top_p: Some(0.5),
            tools: Some(vec![Tool {
                r#type: "function".into(),
                function: ToolFunction {
                    name: "myFunction".into(),
                    description: Some("This is a test function".into()),
                    parameters: Some(json!({"key": "value"})),
                    strict: Some(true),
                },
            }]),
            user: Some("test".into()),
        }
    }
}

impl RequestBody {
    pub fn to_generate_request(self, request_id: String) -> GenerateRequest {
        let inputs = messages::messages_to_prompt(self.messages());
        let frequency_penalty = self.frequency_penalty();
        let max_new_tokens = self.max_completion_tokens();
        let decoder_input_details = self.logprobs().unwrap_or_default();
        let repetition_penalty = self.presence_penalty();
        let stop = match self.stop() {
            Some(StopCondition::Array(stop_tokens)) => stop_tokens.clone(),
            Some(StopCondition::String(stop_token)) => vec![stop_token.clone()],
            None => Vec::new(),
        };
        let seed = self.seed();
        let temperature = self.temperature();
        let top_p = self.top_p();
        let _user = self.user();
        let n = self.n.unwrap_or(1);
        let parameters = GenerateParameters {
            best_of: None,
            temperature,
            repetition_penalty,
            frequency_penalty,
            max_new_tokens,
            repeat_last_n: None,
            top_k: None,
            top_p,
            typical_p: None,
            do_sample: true,
            return_full_text: Some(false),
            stop,
            truncate: None,
            decoder_input_details,
            random_seed: seed,
            top_n_tokens: None,
            n,
        };
        GenerateRequest {
            request_id,
            inputs,
            parameters,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub logprobs: Option<Value>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stopped,
    LengthCapped,
    ContentFilter,
}

impl TryFrom<Option<&str>> for FinishReason {
    type Error = String;

    fn try_from(value: Option<&str>) -> Result<Self, Self::Error> {
        match value {
            Some("stopped") => Ok(FinishReason::Stopped),
            Some("length_capped") => Ok(FinishReason::LengthCapped),
            Some("content_filter") => Ok(FinishReason::ContentFilter),
            None => Ok(FinishReason::Stopped),
            _ => Err(format!("Invalid finish reason: {}", value.unwrap())),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TryFrom<(String, GenerateRequestOutput)> for ChatCompletionResponse {
    type Error = String;

    fn try_from((model, value): (String, GenerateRequestOutput)) -> Result<Self, Self::Error> {
        let inference_outputs = value.inference_outputs;
        let choices = inference_outputs
            .iter()
            .map(|output| {
                Ok(Choice {
                    index: output.index as u32,
                    message: Message::Assistant {
                        content: Some(MessageContent::Text(output.output_text.clone())),
                        name: None,
                        refusal: None,
                        tool_calls: vec![],
                    },
                    logprobs: Some(
                        serde_json::to_value(&output.logprobs)
                            .map_err(|e| format!("Failed to convert logprobs to JSON: {}", e))?,
                    ),
                    finish_reason: FinishReason::try_from(output.finish_reason.as_deref())?,
                })
            })
            .collect::<Result<Vec<Choice>, Self::Error>>()?;
        let prompt_tokens = value.prompt_token_ids.len() as u32;
        let completion_tokens = inference_outputs
            .iter()
            .map(|o| o.token_ids.len())
            .sum::<usize>() as u32;
        let total_tokens = prompt_tokens + completion_tokens;

        let now = Instant::now();
        let finished_time = value
            .metrics
            .read()
            .expect("Failed to read metrics from the inference output response")
            .finished_time
            .ok_or("Finished time not found")?;
        let duration = now.duration_since(finished_time);
        let system_now = SystemTime::now();
        let system_duration = system_now
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Failed to get system duration");
        let created = system_duration.as_millis() as u64 - duration.as_millis() as u64;

        Ok(ChatCompletionResponse {
            id: value.request_id,
            object: "chat.completions".into(),
            created,
            model,
            system_fingerprint: "vllm".into(),
            choices,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
        })
    }
}

/// Represents a chunk of a streaming chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChunk {
    /// A unique identifier for this chat completion.
    pub id: String,
    /// The object type, which is "chat.completion" for this struct.
    pub object: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for this chat completion.
    pub model: String,
    /// A unique identifier for the model's configuration and version.
    pub system_fingerprint: String,
    /// An array of chat completion choices. Each choice represents a possible completion for the input.
    pub choices: Vec<StreamChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

/// Represents a single choice in a streaming chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamChoice {
    /// The index of this choice in the list of choices.
    pub index: u32,
    /// The delta (incremental update) for this choice.
    pub delta: Delta,
    /// Log probabilities for the tokens in this choice, if requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    /// The reason why the model stopped generating tokens, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Represents the delta (incremental update) in a streaming chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    /// The role of the message author (e.g., "assistant").
    pub role: String,
    /// The content of the message, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// A refusal message, if the assistant refuses to respond.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// A list of tool calls made by the assistant, if any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl TryFrom<(String, GenerateStreamingOutput)> for ChatCompletionChunk {
    type Error = String;

    fn try_from((model, value): (String, GenerateStreamingOutput)) -> Result<Self, Self::Error> {
        let id = value.request_id;
        let created = value.created;
        let usage = Usage {
            prompt_tokens: value.num_prompt_tokens as u32,
            completion_tokens: value.num_completion_tokens as u32,
            total_tokens: value.num_prompt_tokens as u32 + value.num_completion_tokens as u32,
        };
        let choices = vec![StreamChoice {
            index: 0,
            delta: Delta {
                role: "assistant".into(),
                content: Some(value.output_text),
                refusal: None,
                tool_calls: vec![],
            },
            logprobs: Some(
                serde_json::to_value(&value.logprobs)
                    .map_err(|e| format!("Failed to convert logprobs to JSON: {}", e))?,
            ),
            finish_reason: value.finish_reason,
        }];
        let chunk = ChatCompletionChunk {
            id,
            object: "chat.completion.chunk".into(),
            created,
            model,
            system_fingerprint: "vllm".into(),
            choices,
            usage,
        };
        Ok(chunk)
    }
}

#[cfg(test)]
pub mod json_schema_tests {
    // TODO: Move check functions to a test utils module.
    //! Note: These tests make use of the `expect_test` crate, and can be updated by
    //! setting `UPDATE_EXPECT=1`.
    use std::{fs::File, io::BufReader};

    use expect_test::{expect_file, ExpectFile};
    use schemars::schema_for;
    use serde_json::json;

    use super::{
        messages, ChatCompletionChunk, ChatCompletionResponse, Choice, Delta, FinishReason,
        Message, MessageContent, MessageContentPart, MessageContentPartImageUrl, RequestBody,
        StreamChoice, ToolCall, ToolCallFunction, Usage,
    };
    use crate::api::validate_with_schema;

    /// Used in tandem with a schema file, this will check if there are
    /// changes to the JSON API schema, and show a diff if so.
    /// If there are changes, running the test with `UPDATE_EXPECT=1`
    /// will update the json schema file.
    fn check_schema(schema: &str, expect_file: ExpectFile) {
        expect_file.assert_eq(schema);
    }

    #[test]
    /// Used in tandem with a schema file, this will check if there are
    /// changes to the JSON API schema, and show a diff if so.
    /// If there are changes, running the test with `UPDATE_EXPECT=1`
    /// will update the json schema file.
    fn verify_request_schema() {
        let request_schema = schema_for!(RequestBody);
        let json_request_schema = serde_json::to_string_pretty(&request_schema)
            .expect("failed to parse json schema into str while verifying request schema");
        check_schema(
            &json_request_schema,
            expect_file!["../../request_schema.json"],
        );
    }

    // TODO: Add the above test for response_schema

    #[test]
    fn request_schema_control() {
        let schema_path = concat!(env!("CARGO_MANIFEST_DIR"), "/request_schema.json");
        let schema_file = File::open(schema_path).expect("request_schema.json not found, try running the verify_request_schema test with UPDATE_EXPECT=1 and try again.");
        let reader = BufReader::new(schema_file);
        let schema: serde_json::Value =
            serde_json::from_reader(reader).expect("failed to read request schema");
        let validator = jsonschema::draft7::new(&schema)
            .expect("failed to create validator from request schema");
        let request = serde_json::json!(RequestBody::control());
        assert!(
            validate_with_schema(validator, request).is_none(),
            "Failed to validate control from request schema.\nThe AtomaAPI JSON schema is auto generated by running 'UPDATE_EXPECT=1 cargo test verify_request_schema' and is located in 'server/request_schema.json'."
        );
    }

    #[test]
    fn deserialize_request_body_basic() {
        let json_request_body = r#"
            {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant"
                    }
                ]
            }
        "#;

        let request_body: Result<RequestBody, serde_json::Error> =
            serde_json::from_str(json_request_body);
        assert!(request_body.is_ok());
    }

    #[test]
    fn deserialize_system_message() {
        let json_system_message_text = r#"
            {
                "role": "system",
                "content": "Hello, World!"
            }
        "#;

        let system_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_system_message_text);
        assert_eq!(
            system_message.unwrap(),
            Message::System {
                content: Some(MessageContent::Text("Hello, World!".to_string())),
                name: None
            }
        );
    }

    #[test]
    fn deserialize_user_message() {
        let json_user_message_text = r#"
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, World!"
                    },
                    {
                        "type": "image",
                        "image_url": {
                            "url": "http://example.com/image.png",
                            "detail": "high"
                        }
                    }
                ]
            }
        "#;

        let user_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_user_message_text);
        assert_eq!(
            user_message.unwrap(),
            Message::User {
                content: Some(MessageContent::Array(vec![
                    MessageContentPart::Text {
                        r#type: "text".into(),
                        text: "Hello, World!".into(),
                    },
                    MessageContentPart::Image {
                        r#type: "image".into(),
                        image_url: MessageContentPartImageUrl {
                            url: "http://example.com/image.png".into(),
                            detail: Some("high".into()),
                        },
                    },
                ])),
                name: None,
            }
        );
    }

    #[test]
    fn deserialize_assistant_message() {
        let json_assistant_message_text = r#"
            {
                "role": "assistant",
                "content": "Sure! Here is your answer: ...",
                "refusal": null,
                "tool_calls": [{
                    "id": "chatcmpl-123",
                    "type": "function",
                    "function": {
                        "name": "myFunction",
                        "arguments": {
                            "key": "value"
                        }
                    }
                }]
            }
        "#;

        let assistant_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_assistant_message_text);
        assert_eq!(
            assistant_message.unwrap(),
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Sure! Here is your answer: ...".to_string()
                )),
                name: None,
                refusal: None,
                tool_calls: vec![ToolCall {
                    id: "chatcmpl-123".into(),
                    r#type: "function".into(),
                    function: ToolCallFunction {
                        name: "myFunction".into(),
                        arguments: json!({"key": "value"}),
                    },
                }],
            }
        );
    }

    #[test]
    fn deserialize_tool_message() {
        let json_tool_message_text = r#"
            {
                "role": "tool",
                "content": "Using tool ...",
                "tool_call_id": "123"
            }
        "#;

        let tool_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_tool_message_text);
        assert_eq!(
            tool_message.unwrap(),
            Message::Tool {
                content: Some(MessageContent::Text("Using tool ...".to_string())),
                tool_call_id: "123".into(),
            }
        );
    }

    #[test]
    fn deserialize_message_content_text() {
        let json_message_content_text = r#"
            "Hello, World!"
        "#;

        let message_content: Result<MessageContent, serde_json::Error> =
            serde_json::from_str(json_message_content_text);
        assert!(message_content.is_ok());
    }

    #[test]
    fn deserialize_message_content_array() {
        let json_message_content_array = r#"
            [
                {
                    "type": "text",
                    "text": "Hello, World!"
                },
                {
                    "type": "image",
                    "image_url": {
                        "url": "http://example.com/image.png",
                        "detail": "high"
                    }
                }
            ]
        "#;

        let message_content: Result<MessageContent, serde_json::Error> =
            serde_json::from_str(json_message_content_array);
        assert!(message_content.is_ok());
    }

    #[test]
    fn test_messages_to_prompt() {
        // Create some sample messages
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are a helpful assistant.".to_string(),
                )),
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text(
                    "What is the capital of France?".to_string(),
                )),
                name: Some("User1".to_string()),
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "The capital of France is Paris.".to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];

        let expected_prompt = "\
            System: You are a helpful assistant.\n\n\
            User (User1): What is the capital of France?\n\n\
            Assistant: The capital of France is Paris.";

        let prompt = messages::messages_to_prompt(&messages);
        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_empty_messages() {
        let messages: Vec<Message> = vec![];
        let expected_prompt = "";

        let prompt = messages::messages_to_prompt(&messages);
        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_message_with_no_content() {
        let messages = vec![
            Message::System {
                content: None,
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text("What is 2 + 2?".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("2 + 2 is 4.".to_string())),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];

        let expected_prompt = "\
            System: \n\n\
            User: What is 2 + 2?\n\n\
            Assistant: 2 + 2 is 4.";

        let prompt = messages::messages_to_prompt(&messages);
        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_message_to_prompt_without_sytem() {
        let messages = vec![
            Message::User {
                content: Some(MessageContent::Text("What is 2 + 2?".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("2 + 2 is 4.".to_string())),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];

        let expected_prompt = "\
            User: What is 2 + 2?\n\n\
            Assistant: 2 + 2 is 4.";

        let prompt = messages::messages_to_prompt(&messages);
        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_message_to_prompt_without_user() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are a helpful assistant.".to_string(),
                )),
                name: Some("Me".to_string()),
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "The capital of France is Paris.".to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];

        let expected_prompt = "\
            System (Me): You are a helpful assistant.\n\n\
            Assistant: The capital of France is Paris.";

        let prompt = messages::messages_to_prompt(&messages);
        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_deserialize_chat_completion_response() {
        let json = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "finish_reason": "stopped"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        let response: ChatCompletionResponse = serde_json::from_value(json).unwrap();

        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1677652288);
        assert_eq!(response.model, "llama");
        assert_eq!(response.system_fingerprint, "fp_44709d6fcb");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(
            response.choices[0].message,
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Hello, how can I help you today?".to_string()
                )),
                name: None,
                refusal: None,
                tool_calls: vec![],
            }
        );
        assert_eq!(response.choices[0].finish_reason, FinishReason::Stopped);
        assert_eq!(response.usage.prompt_tokens, 9);
        assert_eq!(response.usage.completion_tokens, 12);
        assert_eq!(response.usage.total_tokens, 21);
    }

    #[test]
    fn test_deserialize_choice() {
        let json = json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you today?"
            },
            "finish_reason": "stopped"
        });

        let choice: Choice = serde_json::from_value(json).unwrap();

        assert_eq!(choice.index, 0);
        assert!(matches!(choice.message, Message::Assistant { .. }));
        assert!(matches!(choice.finish_reason, FinishReason::Stopped));
    }

    #[test]
    fn test_deserialize_finish_reason() {
        assert_eq!(
            serde_json::from_str::<FinishReason>("\"stopped\"").unwrap(),
            FinishReason::Stopped
        );
        assert_eq!(
            serde_json::from_str::<FinishReason>("\"length_capped\"").unwrap(),
            FinishReason::LengthCapped
        );
        assert_eq!(
            serde_json::from_str::<FinishReason>("\"content_filter\"").unwrap(),
            FinishReason::ContentFilter
        );
    }

    #[test]
    fn test_deserialize_usage() {
        let json = json!({
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        });

        let usage: Usage = serde_json::from_value(json).unwrap();

        assert_eq!(usage.prompt_tokens, 9);
        assert_eq!(usage.completion_tokens, 12);
        assert_eq!(usage.total_tokens, 21);
    }

    #[test]
    fn test_deserialize_choice_with_logprobs() {
        let json = json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you today?"
            },
            "logprobs": {
                "token_logprobs": [-0.5, -0.2, -0.3],
                "top_logprobs": [
                    {"Hello": -0.5, "Hi": -0.7},
                    {"how": -0.2, "what": -0.4},
                    {"can": -0.3, "may": -0.5}
                ]
            },
            "finish_reason": "stopped"
        });

        let choice: Choice = serde_json::from_value(json).unwrap();

        assert_eq!(choice.index, 0);
        assert!(matches!(choice.message, Message::Assistant { .. }));
        assert!(choice.logprobs.is_some());
        assert_eq!(
            choice.logprobs.unwrap(),
            json!({
                "token_logprobs": [-0.5, -0.2, -0.3],
                "top_logprobs": [
                    {"Hello": -0.5, "Hi": -0.7},
                    {"how": -0.2, "what": -0.4},
                    {"can": -0.3, "may": -0.5}
                ]
            })
        );
        assert!(matches!(choice.finish_reason, FinishReason::Stopped));
    }

    #[test]
    fn test_deserialize_delta() {
        let json = json!({
            "role": "assistant",
            "content": "Hello, how can I help you today?",
            "tool_calls": [],
            "refusal": "refusal"
        });

        let delta: Delta = serde_json::from_value(json).unwrap();

        assert_eq!(delta.role, "assistant");
        assert_eq!(
            delta.content,
            Some("Hello, how can I help you today?".to_string())
        );
        assert_eq!(delta.tool_calls, vec![]);
        assert_eq!(delta.refusal, Some("refusal".to_string()));
    }

    #[test]
    fn test_deserialize_stream_choice() {
        let json = json!({
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "Hello, how can I help you today?"
            }
        });

        let choice: StreamChoice = serde_json::from_value(json).unwrap();

        assert_eq!(choice.index, 0);
        assert_eq!(choice.delta.role, "assistant");
        assert_eq!(
            choice.delta.content,
            Some("Hello, how can I help you today?".to_string())
        );
    }

    #[test]
    fn test_deserialize_chat_completion_chunk() {
        let json = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                }
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();

        assert_eq!(chunk.id, "chatcmpl-123");
        assert_eq!(chunk.object, "chat.completion");
        assert_eq!(chunk.created, 1677652288);
        assert_eq!(chunk.model, "llama");
        assert_eq!(chunk.system_fingerprint, "fp_44709d6fcb");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(
            chunk.choices[0].delta,
            Delta {
                role: "assistant".to_string(),
                content: Some("Hello, how can I help you today?".to_string()),
                tool_calls: vec![],
                refusal: None,
            }
        );
        assert_eq!(chunk.usage.prompt_tokens, 9);
        assert_eq!(chunk.usage.completion_tokens, 12);
        assert_eq!(chunk.usage.total_tokens, 21);
    }
}
