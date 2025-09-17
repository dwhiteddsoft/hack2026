use uocvr::UniversalSession;

#[tokio::test]
async fn test_session_creation() {
    let session = UniversalSession::builder()
        .model_file("test.onnx")
        .build()
        .await;
    
    // This should fail for now since it's not implemented
    assert!(session.is_err());
}

#[test]
fn test_error_types() {
    use uocvr::error::UocvrError;
    
    let error = UocvrError::runtime("test error");
    assert!(error.to_string().contains("test error"));
}