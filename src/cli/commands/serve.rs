//! HTTP Server command for Cloud Run deployment
//!
//! Starts an HTTP server exposing all LLM-Latency-Lens agents:
//! - POST /analyze - Latency Analysis Agent
//! - POST /inspect - Inspect previous analysis
//! - POST /replay - Replay analysis with different config
//! - GET /health - Health check
//! - POST /cold-start/measure - Cold Start Mitigation Agent
//! - POST /cold-start/characterize - Cold start characterization

use crate::agents::{
    edge_function::{EdgeFunctionHandler, EdgeFunctionRequest, EdgeFunctionResponse, EdgeOperation},
    ruvector::{RuVectorClient, RuVectorConfig},
};
use crate::cli::ServeArgs;
use anyhow::Result;
use serde::{Deserialize, Serialize};
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

/// Route request to appropriate handler
async fn route_request(router: &Router, request: HttpRequest) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        // Health check
        ("GET", "/health") => handle_health().await,

        // Latency Analysis Agent endpoints
        ("POST", "/analyze") => {
            handle_edge_operation(&router.handler, EdgeOperation::Analyze, &request.body).await
        }
        ("POST", "/inspect") => {
            handle_edge_operation(&router.handler, EdgeOperation::Inspect, &request.body).await
        }
        ("POST", "/replay") => {
            handle_edge_operation(&router.handler, EdgeOperation::Replay, &request.body).await
        }

        // Cold Start Mitigation Agent endpoints
        ("POST", "/cold-start/measure") => handle_cold_start_measure(&request.body).await,
        ("POST", "/cold-start/characterize") => handle_cold_start_characterize(&request.body).await,

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

/// Handle edge function operation
async fn handle_edge_operation(
    handler: &EdgeFunctionHandler,
    operation: EdgeOperation,
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

    let request = EdgeFunctionRequest {
        operation,
        payload,
        trace_context: None,
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

/// Handle cold start measurement (placeholder - calls into cold_start module)
async fn handle_cold_start_measure(body: &[u8]) -> HttpResponse {
    // This would integrate with the ColdStartMitigationAgent
    // For now, return a placeholder response indicating the endpoint exists
    let response = serde_json::json!({
        "success": true,
        "message": "Cold start measurement endpoint",
        "classification": "measurement",
        "note": "Full implementation integrates with ColdStartMitigationAgent"
    });

    HttpResponse::ok(serde_json::to_vec(&response).unwrap_or_default())
}

/// Handle cold start characterization (placeholder)
async fn handle_cold_start_characterize(body: &[u8]) -> HttpResponse {
    let response = serde_json::json!({
        "success": true,
        "message": "Cold start characterization endpoint",
        "classification": "measurement",
        "note": "Full implementation integrates with ColdStartMitigationAgent"
    });

    HttpResponse::ok(serde_json::to_vec(&response).unwrap_or_default())
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
}
