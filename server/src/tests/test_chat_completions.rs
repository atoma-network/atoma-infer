use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, Role,
    },
    Client,
};

use crate::tests::common;

#[tokio::test]
async fn test_chat_completions() {
    let server_api_base = common::spawn_server().await;

    // Create a custom client for async-openai that points to our server
    let config = OpenAIConfig::new()
        .with_api_key("sk-anything")
        .with_api_base(server_api_base);

    let client = Client::with_config(config);
    // Create request using builder pattern
    let request = CreateChatCompletionRequestArgs::default()
        .model("llama3")
        .max_tokens(512u32)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Who won the world series in 2020?")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("The Los Angeles Dodgers won the World Series in 2020.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Where was it played?")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();

    // Call API
    let response = client
        .chat()
        .create(request)
        .await
        .expect("Failed to get chat completion");

    // Assert the response structure and content
    assert!(!response.choices.is_empty());
    let choice = &response.choices[0];
    assert_eq!(choice.index, 0);
    assert!(choice.message.content.is_some());
    assert_eq!(choice.message.role, Role::Assistant);
    assert!(response.usage.is_some());

    // Print response details
    println!("\nResponse:\n");
    for choice in response.choices {
        println!(
            "{}: Role: {}  Content: {:?}",
            choice.index, choice.message.role, choice.message.content
        );
    }

    // Test error case: invalid model
    let invalid_request = CreateChatCompletionRequestArgs::default()
        .model("invalid_model")
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Hello")
            .build()
            .unwrap()
            .into()])
        .build()
        .unwrap();

    let error_response = client.chat().create(invalid_request).await;
    assert!(error_response.is_err());
}
