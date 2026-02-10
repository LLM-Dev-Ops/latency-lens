//! Google Cloud Edge Function Handler
//!
//! Entry point for deploying the Latency Analysis Agent as a Google Cloud Edge Function.
//! This module provides the HTTP handler that:
//!
//! 1. Receives JSON input conforming to LatencyAnalysisInput schema
//! 2. Validates input against agentics-contracts
//! 3. Validates execution context (parent_span_id required)
//! 4. Executes the agent within a tracked execution span
//! 5. Returns ExecutionResult with repo and agent spans
//!
//! # Deployment Model
//!
//! - Deployed as part of LLM-Latency-Lens unified Google Cloud service
//! - Stateless execution
//! - No local persistence
//! - All data persisted via ruvector-service
//!
//! # Agentics Execution Graph Contract
//!
//! Every operation (except Health) MUST:
//! - Accept an ExecutionContext with a valid parent_span_id
//! - Emit a repo-level span and at least one agent-level span
//! - Return an ExecutionResult containing all spans
//! - Reject execution if parent_span_id is missing

use crate::agents::{
    contracts::{DecisionEvent, LatencyAnalysisConfig, LatencyAnalysisInput, LatencyAnalysisOutput},
    execution_graph::{
        validate_execution_context, Artifact, ExecutionContext, ExecutionResult, RepoExecution,
    },
    latency_analysis::LatencyAnalysisAgent,
    ruvector::{EventQuery, RuVectorClient, RuVectorConfig},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, instrument, warn};

/// Edge function request envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFunctionRequest {
    /// Operation to perform
    pub operation: EdgeOperation,
    /// Request payload
    pub payload: serde_json::Value,
    /// Optional trace context (legacy, still supported)
    pub trace_context: Option<TraceContext>,
    /// Agentics execution context. REQUIRED for span-emitting operations.
    #[serde(default)]
    pub execution_context: Option<ExecutionContext>,
}

/// Edge function operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeOperation {
    /// Perform latency analysis
    Analyze,
    /// Inspect a previous analysis
    Inspect,
    /// Replay analysis with different config
    Replay,
    /// Health check
    Health,
}

/// Trace context for distributed tracing (legacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
}

/// Edge function response envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFunctionResponse {
    /// Whether operation succeeded
    pub success: bool,
    /// Operation result (if successful)
    pub result: Option<serde_json::Value>,
    /// Error information (if failed)
    pub error: Option<EdgeFunctionError>,
    /// DecisionEvent ID (for audit trail)
    pub decision_event_id: Option<String>,
    /// Trace context
    pub trace_context: Option<TraceContext>,
}

/// Edge function error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFunctionError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Additional details
    pub details: Option<serde_json::Value>,
}

/// Inspect request payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectRequest {
    /// Analysis ID to inspect
    pub analysis_id: Option<uuid::Uuid>,
    /// Decision event ID to inspect
    pub event_id: Option<uuid::Uuid>,
    /// Query by agent ID and time range
    pub query: Option<EventQuery>,
}

/// Replay request payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRequest {
    /// Original analysis ID to replay
    pub original_analysis_id: uuid::Uuid,
    /// New configuration to apply
    pub new_config: Option<LatencyAnalysisConfig>,
}

/// Edge Function Handler
///
/// Handles HTTP requests for the Latency Analysis Agent.
/// Deployed as a Google Cloud Edge Function.
///
/// All operations (except Health) are wrapped in the Agentics execution
/// graph contract, producing repo-level and agent-level spans.
pub struct EdgeFunctionHandler {
    /// RuVector client
    ruvector_client: Arc<RuVectorClient>,
}

impl EdgeFunctionHandler {
    /// Create a new handler
    pub fn new(ruvector_client: Arc<RuVectorClient>) -> Self {
        Self { ruvector_client }
    }

    /// Create handler from environment configuration
    pub fn from_env() -> Result<Self, String> {
        let config = RuVectorConfig::from_env();
        let client = RuVectorClient::new(config)
            .map_err(|e| format!("Failed to create RuVector client: {}", e))?;
        Ok(Self::new(Arc::new(client)))
    }

    /// Handle incoming request
    ///
    /// For operations that require execution context (Analyze, Inspect, Replay),
    /// validates the context and wraps execution in repo/agent spans.
    /// Health checks bypass the execution graph.
    #[instrument(skip(self, request))]
    pub async fn handle(&self, request: EdgeFunctionRequest) -> ExecutionResult {
        info!(
            operation = ?request.operation,
            "Handling edge function request"
        );

        // Health checks bypass the execution graph
        if request.operation == EdgeOperation::Health {
            return self.handle_health_as_execution_result().await;
        }

        // Validate execution context (REQUIRED for all non-health operations)
        let ctx = match validate_execution_context(request.execution_context.clone()) {
            Ok(ctx) => ctx,
            Err(err_result) => return err_result,
        };

        let mut repo_exec = RepoExecution::begin(ctx);

        match request.operation {
            EdgeOperation::Analyze => {
                self.handle_analyze(request, &mut repo_exec).await
            }
            EdgeOperation::Inspect => {
                self.handle_inspect(request, &mut repo_exec).await
            }
            EdgeOperation::Replay => {
                self.handle_replay(request, &mut repo_exec).await
            }
            EdgeOperation::Health => unreachable!(),
        }
    }

    /// Handle analyze operation with execution graph tracking
    async fn handle_analyze(
        &self,
        request: EdgeFunctionRequest,
        repo_exec: &mut RepoExecution,
    ) -> ExecutionResult {
        // Parse input
        let input: LatencyAnalysisInput = match serde_json::from_value(request.payload) {
            Ok(input) => input,
            Err(e) => {
                return repo_exec_taken(repo_exec, vec![
                    format!("Failed to parse input: {}", e),
                ]);
            }
        };

        // Begin agent span
        let agent_span_id =
            repo_exec.begin_agent("latency-analysis-agent", "analysis");

        // Create agent with execution context
        let mut agent = LatencyAnalysisAgent::new(Arc::clone(&self.ruvector_client));
        agent = agent.with_execution_context(repo_exec.context());

        if let Some(ref trace) = request.trace_context {
            agent = agent.with_trace_context(&trace.trace_id, &trace.span_id);
        }

        // Execute agent
        match agent.execute(input).await {
            Ok((output, event)) => {
                let result = serde_json::to_value(&output).ok();

                // Build artifacts
                let artifacts = vec![
                    Artifact::from_decision_event(&event),
                    Artifact::from_json_output(
                        "analysis_output",
                        &output.analysis_id.to_string(),
                        &output,
                    ),
                ];
                repo_exec.complete_agent(agent_span_id, artifacts);

                // Build the inner response for the result payload
                let inner_response = EdgeFunctionResponse {
                    success: true,
                    result,
                    error: None,
                    decision_event_id: Some(event.event_id.to_string()),
                    trace_context: request.trace_context,
                };

                take_repo_exec(repo_exec, Some(serde_json::to_value(&inner_response).unwrap_or_default()))
            }
            Err(e) => {
                error!(error = %e, "Analyze operation failed");
                repo_exec.fail_agent(
                    agent_span_id,
                    vec![e.to_string()],
                );

                let inner_response = EdgeFunctionResponse {
                    success: false,
                    result: None,
                    error: Some(EdgeFunctionError {
                        code: "ANALYSIS_FAILED".to_string(),
                        message: e.to_string(),
                        details: None,
                    }),
                    decision_event_id: None,
                    trace_context: request.trace_context,
                };

                take_repo_exec(repo_exec, Some(serde_json::to_value(&inner_response).unwrap_or_default()))
            }
        }
    }

    /// Handle inspect operation with execution graph tracking
    async fn handle_inspect(
        &self,
        request: EdgeFunctionRequest,
        repo_exec: &mut RepoExecution,
    ) -> ExecutionResult {
        let inspect_req: InspectRequest = match serde_json::from_value(request.payload) {
            Ok(req) => req,
            Err(e) => {
                return repo_exec_taken(repo_exec, vec![
                    format!("Failed to parse inspect request: {}", e),
                ]);
            }
        };

        // Begin agent span for the inspect operation
        let agent_span_id =
            repo_exec.begin_agent("latency-analysis-agent", "analysis");

        // Query by event ID if provided
        if let Some(event_id) = inspect_req.event_id {
            match self.ruvector_client.get_event(event_id).await {
                Ok(Some(event)) => {
                    let result = serde_json::json!({
                        "success": true,
                        "event": event,
                        "decision_event_id": event_id.to_string()
                    });
                    let artifacts = vec![Artifact::from_json_output(
                        "inspect_result",
                        &event_id.to_string(),
                        &result,
                    )];
                    repo_exec.complete_agent(agent_span_id, artifacts);
                    return take_repo_exec(repo_exec, Some(result));
                }
                Ok(None) => {
                    repo_exec.fail_agent(
                        agent_span_id,
                        vec![format!("Event {} not found", event_id)],
                    );
                    return take_repo_exec(repo_exec, None);
                }
                Err(e) => {
                    repo_exec.fail_agent(
                        agent_span_id,
                        vec![format!("Query failed: {}", e)],
                    );
                    return take_repo_exec(repo_exec, None);
                }
            }
        }

        // Query by analysis ID
        if let Some(analysis_id) = inspect_req.analysis_id {
            let query = EventQuery::new()
                .agent_id(crate::agents::latency_analysis::AGENT_ID)
                .decision_type("latency_analysis");

            match self.ruvector_client.query_events(query).await {
                Ok(result) => {
                    let filtered: Vec<_> = result
                        .events
                        .into_iter()
                        .filter(|e| {
                            e.get("metadata")
                                .and_then(|m| m.get("analysis_id"))
                                .and_then(|a| a.as_str())
                                .map(|a| a == analysis_id.to_string())
                                .unwrap_or(false)
                        })
                        .collect();

                    let result_val = serde_json::json!({
                        "events": filtered,
                        "total_count": filtered.len()
                    });
                    let artifacts = vec![Artifact::from_json_output(
                        "inspect_result",
                        &analysis_id.to_string(),
                        &result_val,
                    )];
                    repo_exec.complete_agent(agent_span_id, artifacts);
                    return take_repo_exec(repo_exec, Some(result_val));
                }
                Err(e) => {
                    repo_exec.fail_agent(
                        agent_span_id,
                        vec![format!("Query failed: {}", e)],
                    );
                    return take_repo_exec(repo_exec, None);
                }
            }
        }

        // Query with custom query
        if let Some(query) = inspect_req.query {
            match self.ruvector_client.query_events(query).await {
                Ok(result) => {
                    let result_val = serde_json::to_value(result).unwrap_or_default();
                    let artifacts = vec![Artifact::from_json_output(
                        "inspect_result",
                        "custom_query",
                        &result_val,
                    )];
                    repo_exec.complete_agent(agent_span_id, artifacts);
                    return take_repo_exec(repo_exec, Some(result_val));
                }
                Err(e) => {
                    repo_exec.fail_agent(
                        agent_span_id,
                        vec![format!("Query failed: {}", e)],
                    );
                    return take_repo_exec(repo_exec, None);
                }
            }
        }

        repo_exec.fail_agent(
            agent_span_id,
            vec!["Must provide event_id, analysis_id, or query".to_string()],
        );
        take_repo_exec(repo_exec, None)
    }

    /// Handle replay operation with execution graph tracking
    async fn handle_replay(
        &self,
        request: EdgeFunctionRequest,
        repo_exec: &mut RepoExecution,
    ) -> ExecutionResult {
        let replay_req: ReplayRequest = match serde_json::from_value(request.payload) {
            Ok(req) => req,
            Err(e) => {
                return repo_exec_taken(repo_exec, vec![
                    format!("Failed to parse replay request: {}", e),
                ]);
            }
        };

        // Begin agent span for the replay operation
        let agent_span_id =
            repo_exec.begin_agent("latency-analysis-agent", "analysis");

        // Fetch original analysis event
        let query = EventQuery::new()
            .agent_id(crate::agents::latency_analysis::AGENT_ID)
            .decision_type("latency_analysis")
            .limit(1);

        let result = match self.ruvector_client.query_events(query).await {
            Ok(r) => r,
            Err(e) => {
                repo_exec.fail_agent(
                    agent_span_id,
                    vec![format!("Query failed: {}", e)],
                );
                return take_repo_exec(repo_exec, None);
            }
        };

        // Find the original event
        let original_event = result
            .events
            .into_iter()
            .find(|e| {
                e.get("metadata")
                    .and_then(|m| m.get("analysis_id"))
                    .and_then(|a| a.as_str())
                    .map(|a| a == replay_req.original_analysis_id.to_string())
                    .unwrap_or(false)
            });

        let original = match original_event {
            Some(e) => e,
            None => {
                repo_exec.fail_agent(
                    agent_span_id,
                    vec![format!(
                        "Original analysis {} not found",
                        replay_req.original_analysis_id
                    )],
                );
                return take_repo_exec(repo_exec, None);
            }
        };

        warn!(
            original_id = %replay_req.original_analysis_id,
            "Replay requested but original inputs not available"
        );

        repo_exec.fail_agent(
            agent_span_id,
            vec![
                "Replay requires original input metrics to be stored. Original analysis found but inputs are not persisted.".to_string(),
            ],
        );

        let details = serde_json::json!({
            "original_event": original,
            "hint": "To enable replay, persist input metrics along with DecisionEvent"
        });

        take_repo_exec(repo_exec, Some(details))
    }

    /// Handle health check (no execution graph, returns a self-contained ExecutionResult)
    async fn handle_health_as_execution_result(&self) -> ExecutionResult {
        // Health checks get a self-generated context since they don't participate
        // in the execution graph hierarchy
        let ctx = ExecutionContext {
            execution_id: uuid::Uuid::new_v4(),
            parent_span_id: uuid::Uuid::new_v4(),
            trace_id: None,
        };
        let mut repo_exec = RepoExecution::begin(ctx);
        let agent_span_id = repo_exec.begin_agent("health-check", "infrastructure");

        match self.ruvector_client.health_check().await {
            Ok(health) => {
                let result = serde_json::json!({
                    "agent_id": crate::agents::latency_analysis::AGENT_ID,
                    "agent_version": crate::agents::latency_analysis::AGENT_VERSION,
                    "ruvector_status": health,
                    "classification": "analysis"
                });
                repo_exec.complete_agent(agent_span_id, vec![]);
                take_repo_exec(&mut repo_exec, Some(result))
            }
            Err(e) => {
                repo_exec.fail_agent(
                    agent_span_id,
                    vec![format!("Health check failed: {}", e)],
                );
                take_repo_exec(&mut repo_exec, None)
            }
        }
    }
}

/// Helper: finalize a RepoExecution by taking ownership via std::mem::replace.
///
/// This works around the borrow checker since `finalize` consumes self,
/// but we receive `&mut RepoExecution`.
fn take_repo_exec(
    repo_exec: &mut RepoExecution,
    result: Option<serde_json::Value>,
) -> ExecutionResult {
    // Create a dummy context for the replacement
    let dummy_ctx = ExecutionContext {
        execution_id: uuid::Uuid::nil(),
        parent_span_id: uuid::Uuid::nil(),
        trace_id: None,
    };
    let taken = std::mem::replace(repo_exec, RepoExecution::begin(dummy_ctx));
    taken.finalize(result)
}

/// Helper: finalize with error before any agent started.
fn repo_exec_taken(
    repo_exec: &mut RepoExecution,
    reasons: Vec<String>,
) -> ExecutionResult {
    let dummy_ctx = ExecutionContext {
        execution_id: uuid::Uuid::nil(),
        parent_span_id: uuid::Uuid::nil(),
        trace_id: None,
    };
    let taken = std::mem::replace(repo_exec, RepoExecution::begin(dummy_ctx));
    taken.finalize_error(reasons)
}

/// HTTP entry point for Google Cloud Functions
///
/// This function is the entry point when deployed as a Cloud Function.
/// It handles the HTTP request/response lifecycle.
#[cfg(feature = "cloud-function")]
pub async fn http_entry_point(
    req: hyper::Request<hyper::Body>,
) -> Result<hyper::Response<hyper::Body>, std::convert::Infallible> {
    use hyper::{Body, Response, StatusCode};

    // Parse request body
    let body_bytes = match hyper::body::to_bytes(req.into_body()).await {
        Ok(b) => b,
        Err(e) => {
            let response = serde_json::json!({
                "success": false,
                "error": format!("Failed to read request body: {}", e)
            });
            let json = serde_json::to_string(&response).unwrap_or_default();
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(json))
                .unwrap());
        }
    };

    // Parse request
    let request: EdgeFunctionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            let response = serde_json::json!({
                "success": false,
                "error": format!("Failed to parse request: {}", e)
            });
            let json = serde_json::to_string(&response).unwrap_or_default();
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(json))
                .unwrap());
        }
    };

    // Create handler and process
    let handler = match EdgeFunctionHandler::from_env() {
        Ok(h) => h,
        Err(e) => {
            let response = serde_json::json!({
                "success": false,
                "error": e
            });
            let json = serde_json::to_string(&response).unwrap_or_default();
            return Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(json))
                .unwrap());
        }
    };

    let response = handler.handle(request).await;
    let status = if response.success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };

    let json = serde_json::to_string(&response).unwrap_or_default();
    Ok(Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_operation_serialization() {
        let op = EdgeOperation::Analyze;
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, "\"analyze\"");

        let parsed: EdgeOperation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, EdgeOperation::Analyze);
    }

    #[test]
    fn test_edge_function_request() {
        let request = EdgeFunctionRequest {
            operation: EdgeOperation::Health,
            payload: serde_json::Value::Null,
            trace_context: Some(TraceContext {
                trace_id: "trace-123".to_string(),
                span_id: "span-456".to_string(),
                parent_span_id: None,
            }),
            execution_context: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: EdgeFunctionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.operation, EdgeOperation::Health);
        assert!(parsed.trace_context.is_some());
    }

    #[test]
    fn test_edge_function_request_with_execution_context() {
        let request = EdgeFunctionRequest {
            operation: EdgeOperation::Analyze,
            payload: serde_json::Value::Null,
            trace_context: None,
            execution_context: Some(ExecutionContext {
                execution_id: uuid::Uuid::new_v4(),
                parent_span_id: uuid::Uuid::new_v4(),
                trace_id: None,
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: EdgeFunctionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.operation, EdgeOperation::Analyze);
        assert!(parsed.execution_context.is_some());
    }
}
