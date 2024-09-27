//! Responsible for creating the json schema associated with the AtomaAPI, which is modeled after OpenAI's own API.

use std::collections::HashMap;

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
    #[serde(rename(serialize = "llama3", deserialize = "llama3"))]
    Llama3,
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

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename(serialize = "image_url", deserialize = "image_url"))]
pub struct MessageContentPartImageUrl {
    /// Either a URL of the image or the base64 encoded image data.
    url: String,
    /// Specifies the detail level of the image.
    detail: Option<String>,
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

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
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
    max_completion_tokens: Option<i32>,
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

    pub fn max_completion_tokens(&self) -> Option<i32> {
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
            model: Model::Llama3,
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
        Message, MessageContent, MessageContentPart, MessageContentPartImageUrl, ToolCall,
        ToolCallFunction,
    };
    use crate::{validate_with_schema, RequestBody};

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
                "model": "llama3",
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
}
