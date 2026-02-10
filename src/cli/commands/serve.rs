//! HTTP Server command for Cloud Run deployment
//!
//! Starts an HTTP server exposing all LLM-Latency-Lens agents:
//! - POST /analyze - Latency Analysis Agent
//! - POST /inspect - Inspect previous analysis
//! - POST /replay - Replay analysis with different config
//! - GET /health - Health check
//! - POST /cold-start/measure - Cold Start Mitigation Agent
//! - POST /cold-start/characterize - Cold start characterization
//!
//! # Agentics Execution Graph
//!
//! All operation endpoints (except /health) require an execution context
//! via JSON body or HTTP headers (X-Execution-Id, X-Parent-Span-Id).
//! Responses are wrapped in ExecutionResult with repo and agent spans.

use crate::agents::{
    edge_function::{EdgeFunctionHandler, EdgeFunctionRequest, EdgeOperation},
    execution_graph::{
        validate_execution_context, Artifact, ExecutionContext, RepoExecution,
    },
    ruvector::{RuVectorClient, RuVectorConfig},
};
use crate::cli::ServeArgs;
use anyhow::Result;
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tracing::{error, info};

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    service: String,
    version: String,
    agents: Vec<AgentStatus>,
}

#[derive(Debug, Serialize)]
struct AgentStatus {
    name: String,
    classification: String,
    endpoints: Vec<String>,
}

/// Run the HTTP server
pub async fn run(args: ServeArgs) -> Result<()> {
    info!(
        host = %args.host,
        port = %args.port,
        "Starting LLM-Latency-Lens HTTP server"
    );

    // Create RuVector client
    let ruvector_config = RuVectorConfig {
        endpoint: args.ruvector_url.unwrap_or_else(|| {
            std::env::var("RUVECTOR_SERVICE_URL")
                .unwrap_or_else(|_| "https://ruvector-service.run.app".to_string())
        }),
        api_key: args.ruvector_key.or_else(|| std::env::var("RUVECTOR_API_KEY").ok()),
        timeout_ms: 30000,
        max_retries: 3,
        compression: true,
        batch_size: 100,
    };

    let ruvector_client = Arc::new(
        RuVectorClient::new(ruvector_config)
            .map_err(|e| anyhow::anyhow!("Failed to create RuVector client: {}", e))?,
    );

    let handler = Arc::new(EdgeFunctionHandler::new(Arc::clone(&ruvector_client)));

    // Build the router
    let app = build_router(handler, args.timeout);

    // Bind to address
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))?;

    info!("Listening on http://{}", addr);
    info!("Endpoints:");
    info!("  POST /analyze       - Latency Analysis Agent");
    info!("  POST /inspect       - Inspect previous analysis");
    info!("  POST /replay        - Replay with different config");
    info!("  GET  /health        - Health check");
    info!("  POST /cold-start/*  - Cold Start Mitigation Agent");

    // Start the server
    let listener = TcpListener::bind(addr).await?;

    // Simple HTTP server loop
    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let handler = Arc::clone(&app);

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, handler, remote_addr).await {
                error!("Connection error from {}: {}", remote_addr, e);
            }
        });
    }
}

/// Simple HTTP request/response types
#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    status_text: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpResponse {
    fn ok(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            status_text: "OK".to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body,
        }
    }

    fn bad_request(body: Vec<u8>) -> Self {
        Self {
            status: 400,
            status_text: "Bad Request".to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body,
        }
    }

    fn not_found() -> Self {
        Self {
            status: 404,
            status_text: "Not Found".to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: b"{\"error\":\"Not Found\"}".to_vec(),
        }
    }

    fn method_not_allowed() -> Self {
        Self {
            status: 405,
            status_text: "Method Not Allowed".to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: b"{\"error\":\"Method Not Allowed\"}".to_vec(),
        }
    }

    fn internal_error(msg: &str) -> Self {
        Self {
            status: 500,
            status_text: "Internal Server Error".to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: format!("{{\"error\":\"{}\"}}", msg).into_bytes(),
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {} {}\r\n",
            self.status, self.status_text
        );

        for (key, value) in &self.headers {
            response.push_str(&format!("{}: {}\r\n", key, value));
        }

        response.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        response.push_str("Connection: close\r\n");
        response.push_str("\r\n");

        let mut bytes = response.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

/// Router wrapper
struct Router {
    handler: Arc<EdgeFunctionHandler>,
    timeout: Duration,
}

fn build_router(handler: Arc<EdgeFunctionHandler>, timeout_secs: u64) -> Arc<Router> {
    Arc::new(Router {
        handler,
        timeout: Duration::from_secs(timeout_secs),
    })
}

/// Handle a single connection
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    router: Arc<Router>,
    remote_addr: SocketAddr,
) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Read request
    let mut buffer = vec![0u8; 65536];
    let n = stream.read(&mut buffer).await?;
    buffer.truncate(n);

    // Parse request
    let request = parse_http_request(&buffer)?;

    info!(
        method = %request.method,
        path = %request.path,
        remote = %remote_addr,
        "Incoming request"
    );

    // Route request
    let response = route_request(&router, request).await;

    // Write response
    let response_bytes = response.to_bytes();
    stream.write_all(&response_bytes).await?;
    stream.flush().await?;

    Ok(())
}

/// Parse HTTP request from bytes
fn parse_http_request(buffer: &[u8]) -> Result<HttpRequest> {
    let text = String::from_utf8_lossy(buffer);
    let mut lines = text.lines();

    // Parse request line
    let request_line = lines.next().ok_or_else(|| anyhow::anyhow!("Empty request"))?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();

    if parts.len() < 2 {
        return Err(anyhow::anyhow!("Invalid request line"));
    }

    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Parse headers
    let mut headers = Vec::new();
    let mut content_length = 0usize;

    for line in lines.by_ref() {
        if line.is_empty() {
            break;
        }

        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_lowercase();
            let value = value.trim().to_string();

            if key == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }

            headers.push((key, value));
        }
    }

    // Find body start
    let header_end = text.find("\r\n\r\n").or_else(|| text.find("\n\n"));
    let body = if let Some(pos) = header_end {
        let body_start = if text[pos..].starts_with("\r\n\r\n") {
            pos + 4
        } else {
            pos + 2
        };

        buffer[body_start..].to_vec()
    } else {
        Vec::new()
    };

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

/// Extract execution context from HTTP headers or JSON body.
///
/// Tries JSON body field "execution_context" first, then falls back to
/// X-Execution-Id and X-Parent-Span-Id headers.
fn extract_execution_context(
    headers: &[(String, String)],
    body: &serde_json::Value,
) -> Option<ExecutionContext> {
    // First try: look in body.execution_context
    if let Some(ctx) = body.get("execution_context") {
        if let Ok(parsed) = serde_json::from_value::<ExecutionContext>(ctx.clone()) {
            return Some(parsed);
        }
    }

    // Second try: look in HTTP headers
    let execution_id = headers
        .iter()
        .find(|(k, _)| k == "x-execution-id")
        .and_then(|(_, v)| uuid::Uuid::parse_str(v).ok())?;
    let parent_span_id = headers
        .iter()
        .find(|(k, _)| k == "x-parent-span-id")
        .and_then(|(_, v)| uuid::Uuid::parse_str(v).ok())?;
    let trace_id = headers
        .iter()
        .find(|(k, _)| k == "x-trace-id")
        .and_then(|(_, v)| uuid::Uuid::parse_str(v).ok());

    Some(ExecutionContext {
        execution_id,
        parent_span_id,
        trace_id,
    })
}

/// Route request to appropriate handler
async fn route_request(router: &Router, request: HttpRequest) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        // Health check (no execution context required)
        ("GET", "/health") => handle_health().await,

        // Latency Analysis Agent endpoints (execution context required)
        ("POST", "/analyze") => {
            handle_edge_operation(&router.handler, EdgeOperation::Analyze, &request.headers, &request.body).await
        }
        ("POST", "/inspect") => {
            handle_edge_operation(&router.handler, EdgeOperation::Inspect, &request.headers, &request.body).await
        }
        ("POST", "/replay") => {
            handle_edge_operation(&router.handler, EdgeOperation::Replay, &request.headers, &request.body).await
        }

        // Cold Start Mitigation Agent endpoints (execution context required)
        ("POST", "/cold-start/measure") => {
            handle_cold_start_measure(&request.headers, &request.body).await
        }
        ("POST", "/cold-start/characterize") => {
            handle_cold_start_characterize(&request.headers, &request.body).await
        }

        // Generic edge function endpoint (unified)
        ("POST", "/") => handle_edge_function(&router.handler, &request.body).await,

        // Method not allowed
        ("GET", "/analyze") | ("GET", "/inspect") | ("GET", "/replay") => {
            HttpResponse::method_not_allowed()
        }

        // Not found
        _ => HttpResponse::not_found(),
    }
}

/// Handle health check
async fn handle_health() -> HttpResponse {
    let health = HealthResponse {
        status: "healthy".to_string(),
        service: "llm-latency-lens".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        agents: vec![
            AgentStatus {
                name: "latency-analysis-agent".to_string(),
                classification: "analysis".to_string(),
                endpoints: vec![
                    "/analyze".to_string(),
                    "/inspect".to_string(),
                    "/replay".to_string(),
                ],
            },
            AgentStatus {
                name: "cold-start-mitigation-agent".to_string(),
                classification: "measurement".to_string(),
                endpoints: vec![
                    "/cold-start/measure".to_string(),
                    "/cold-start/characterize".to_string(),
                ],
            },
        ],
    };

    let body = serde_json::to_vec(&health).unwrap_or_default();
    HttpResponse::ok(body)
}

/// Handle edge function operation with execution context extraction
async fn handle_edge_operation(
    handler: &EdgeFunctionHandler,
    operation: EdgeOperation,
    headers: &[(String, String)],
    body: &[u8],
) -> HttpResponse {
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return HttpResponse::bad_request(
                format!("{{\"error\":\"Invalid JSON: {}\"}}", e).into_bytes(),
            );
        }
    };

    // Extract execution context from headers or body
    let execution_context = extract_execution_context(headers, &payload);

    let request = EdgeFunctionRequest {
        operation,
        payload,
        trace_context: None,
        execution_context,
    };

    let response = handler.handle(request).await;
    let body = serde_json::to_vec(&response).unwrap_or_default();

    if response.success {
        HttpResponse::ok(body)
    } else {
        HttpResponse::bad_request(body)
    }
}

/// Handle unified edge function endpoint
async fn handle_edge_function(handler: &EdgeFunctionHandler, body: &[u8]) -> HttpResponse {
    let request: EdgeFunctionRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return HttpResponse::bad_request(
                format!("{{\"error\":\"Invalid request format: {}\"}}", e).into_bytes(),
            );
        }
    };

    let response = handler.handle(request).await;
    let body = serde_json::to_vec(&response).unwrap_or_default();

    if response.success {
        HttpResponse::ok(body)
    } else {
        HttpResponse::bad_request(body)
    }
}

/// Handle cold start measurement with execution context
async fn handle_cold_start_measure(
    headers: &[(String, String)],
    body: &[u8],
) -> HttpResponse {
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return HttpResponse::bad_request(
                format!("{{\"error\":\"Invalid JSON: {}\"}}", e).into_bytes(),
            );
        }
    };

    // Extract and validate execution context
    let execution_context = extract_execution_context(headers, &payload);
    let ctx = match validate_execution_context(execution_context) {
        Ok(ctx) => ctx,
        Err(err_result) => {
            let body = serde_json::to_vec(&err_result).unwrap_or_default();
            return HttpResponse::bad_request(body);
        }
    };

    let mut repo_exec = RepoExecution::begin(ctx);
    let agent_span_id =
        repo_exec.begin_agent("cold-start-mitigation-agent", "measurement");

    // Placeholder: cold start measurement integration
    let result = serde_json::json!({
        "message": "Cold start measurement endpoint",
        "classification": "measurement",
        "note": "Full implementation integrates with ColdStartMitigationAgent"
    });

    let artifacts = vec![Artifact::from_json_output(
        "cold_start_measurement",
        "placeholder",
        &result,
    )];
    repo_exec.complete_agent(agent_span_id, artifacts);

    let dummy_ctx = ExecutionContext {
        execution_id: uuid::Uuid::nil(),
        parent_span_id: uuid::Uuid::nil(),
        trace_id: None,
    };
    let taken = std::mem::replace(&mut repo_exec, RepoExecution::begin(dummy_ctx));
    let exec_result = taken.finalize(Some(result));

    let body = serde_json::to_vec(&exec_result).unwrap_or_default();
    if exec_result.success {
        HttpResponse::ok(body)
    } else {
        HttpResponse::bad_request(body)
    }
}

/// Handle cold start characterization with execution context
async fn handle_cold_start_characterize(
    headers: &[(String, String)],
    body: &[u8],
) -> HttpResponse {
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return HttpResponse::bad_request(
                format!("{{\"error\":\"Invalid JSON: {}\"}}", e).into_bytes(),
            );
        }
    };

    // Extract and validate execution context
    let execution_context = extract_execution_context(headers, &payload);
    let ctx = match validate_execution_context(execution_context) {
        Ok(ctx) => ctx,
        Err(err_result) => {
            let body = serde_json::to_vec(&err_result).unwrap_or_default();
            return HttpResponse::bad_request(body);
        }
    };

    let mut repo_exec = RepoExecution::begin(ctx);
    let agent_span_id =
        repo_exec.begin_agent("cold-start-mitigation-agent", "measurement");

    // Placeholder: cold start characterization integration
    let result = serde_json::json!({
        "message": "Cold start characterization endpoint",
        "classification": "measurement",
        "note": "Full implementation integrates with ColdStartMitigationAgent"
    });

    let artifacts = vec![Artifact::from_json_output(
        "cold_start_characterization",
        "placeholder",
        &result,
    )];
    repo_exec.complete_agent(agent_span_id, artifacts);

    let dummy_ctx = ExecutionContext {
        execution_id: uuid::Uuid::nil(),
        parent_span_id: uuid::Uuid::nil(),
        trace_id: None,
    };
    let taken = std::mem::replace(&mut repo_exec, RepoExecution::begin(dummy_ctx));
    let exec_result = taken.finalize(Some(result));

    let body = serde_json::to_vec(&exec_result).unwrap_or_default();
    if exec_result.success {
        HttpResponse::ok(body)
    } else {
        HttpResponse::bad_request(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response() {
        let health = HealthResponse {
            status: "healthy".to_string(),
            service: "llm-latency-lens".to_string(),
            version: "0.1.0".to_string(),
            agents: vec![],
        };

        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("healthy"));
    }

    #[test]
    fn test_http_response_to_bytes() {
        let response = HttpResponse::ok(b"test".to_vec());
        let bytes = response.to_bytes();
        let text = String::from_utf8_lossy(&bytes);

        assert!(text.contains("HTTP/1.1 200 OK"));
        assert!(text.contains("Content-Type: application/json"));
        assert!(text.contains("test"));
    }

    #[test]
    fn test_extract_execution_context_from_body() {
        let eid = uuid::Uuid::new_v4();
        let psid = uuid::Uuid::new_v4();
        let body = serde_json::json!({
            "execution_context": {
                "execution_id": eid.to_string(),
                "parent_span_id": psid.to_string()
            }
        });
        let ctx = extract_execution_context(&[], &body);
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.execution_id, eid);
        assert_eq!(ctx.parent_span_id, psid);
    }

    #[test]
    fn test_extract_execution_context_from_headers() {
        let eid = uuid::Uuid::new_v4();
        let psid = uuid::Uuid::new_v4();
        let headers = vec![
            ("x-execution-id".to_string(), eid.to_string()),
            ("x-parent-span-id".to_string(), psid.to_string()),
        ];
        let body = serde_json::json!({});
        let ctx = extract_execution_context(&headers, &body);
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.execution_id, eid);
        assert_eq!(ctx.parent_span_id, psid);
    }

    #[test]
    fn test_extract_execution_context_missing() {
        let body = serde_json::json!({});
        let ctx = extract_execution_context(&[], &body);
        assert!(ctx.is_none());
    }
}
