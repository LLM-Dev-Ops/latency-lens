//! Agentics Execution Graph - Foundational Execution Unit
//!
//! This module implements the execution span contract required by the Agentics
//! execution system. Every externally-invoked operation in this repository MUST
//! produce a hierarchical span structure:
//!
//! ```text
//! Core (external caller)
//!   └─ Repo Span (type="repo", repo_name="llm-latency-lens")
//!       └─ Agent Span (type="agent", agent_name="<agent>")
//!             └─ artifacts: [DecisionEvent, AnalysisOutput, ...]
//! ```
//!
//! # Invariants
//!
//! - Every operation MUST accept an [`ExecutionContext`] with a valid `parent_span_id`
//! - Every operation MUST produce at least one agent-level span
//! - Agents MUST NOT execute without emitting a span
//! - On failure, all emitted spans are still returned
//! - Span structure is append-only and causally ordered via `parent_span_id`

use crate::agents::contracts::DecisionEvent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Repository name constant for span identification.
pub const REPO_NAME: &str = "llm-latency-lens";

// ---------------------------------------------------------------------------
// Execution Context
// ---------------------------------------------------------------------------

/// Execution context provided by the Core/caller.
///
/// Every externally-invoked operation MUST receive this context.
/// Operations MUST reject execution if `parent_span_id` is missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// The execution_id assigned by the Core for this entire execution graph.
    pub execution_id: Uuid,
    /// The parent_span_id from the caller (Core's repo-level span).
    /// This is REQUIRED -- operations MUST reject if missing/nil.
    pub parent_span_id: Uuid,
    /// Optional trace_id for distributed tracing correlation.
    /// Defaults to `execution_id` if not provided.
    #[serde(default)]
    pub trace_id: Option<Uuid>,
}

impl ExecutionContext {
    /// Get the effective trace_id, falling back to execution_id.
    pub fn effective_trace_id(&self) -> Uuid {
        self.trace_id.unwrap_or(self.execution_id)
    }
}

// ---------------------------------------------------------------------------
// Span Types
// ---------------------------------------------------------------------------

/// Discriminates repo-level from agent-level spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanType {
    /// Repository-level span, parented to the Core's span.
    Repo,
    /// Agent-level span, parented to the repo span.
    Agent,
}

/// Outcome status of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    /// Span completed successfully.
    Success,
    /// Span failed.
    Failed,
    /// Span started but was never completed (e.g., panic/timeout).
    Incomplete,
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// A reference to evidence produced by an agent, attached to the agent's span.
///
/// Artifacts are references (not inline data). The actual content lives in
/// the response `result` field and/or in ruvector-service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Unique artifact identifier.
    pub artifact_id: Uuid,
    /// Artifact type (e.g., "decision_event", "analysis_output").
    pub artifact_type: String,
    /// Stable reference: the ID/URI of the artifact.
    pub reference_id: String,
    /// SHA-256 hash of the artifact content for verification.
    pub content_hash: String,
    /// MIME type or format description.
    pub format: String,
    /// Size in bytes (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

impl Artifact {
    /// Create an artifact reference from a DecisionEvent.
    pub fn from_decision_event(event: &DecisionEvent) -> Self {
        let json = serde_json::to_vec(event).unwrap_or_default();
        let hash = Sha256::digest(&json);
        Artifact {
            artifact_id: Uuid::new_v4(),
            artifact_type: "decision_event".to_string(),
            reference_id: event.event_id.to_string(),
            content_hash: format!("{:x}", hash),
            format: "application/json".to_string(),
            size_bytes: Some(json.len() as u64),
        }
    }

    /// Create an artifact reference from a JSON-serializable output.
    pub fn from_json_output<T: Serialize>(
        artifact_type: &str,
        reference_id: &str,
        output: &T,
    ) -> Self {
        let json = serde_json::to_vec(output).unwrap_or_default();
        let hash = Sha256::digest(&json);
        Artifact {
            artifact_id: Uuid::new_v4(),
            artifact_type: artifact_type.to_string(),
            reference_id: reference_id.to_string(),
            content_hash: format!("{:x}", hash),
            format: "application/json".to_string(),
            size_bytes: Some(json.len() as u64),
        }
    }
}

// ---------------------------------------------------------------------------
// Execution Span
// ---------------------------------------------------------------------------

/// Universal span structure for both repo and agent spans.
///
/// The span structure is append-only and causally ordered via `parent_span_id`.
/// It is fully JSON-serializable without loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSpan {
    /// Unique identifier for this span.
    pub span_id: Uuid,
    /// Parent span ID. For repo span: the caller's span. For agent span: the repo span.
    pub parent_span_id: Uuid,
    /// Span type discriminator.
    pub span_type: SpanType,
    /// Repository name (always "llm-latency-lens").
    pub repo_name: String,
    /// Agent name (only set for SpanType::Agent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    /// Agent classification (only set for SpanType::Agent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_classification: Option<String>,
    /// Execution ID from the Core.
    pub execution_id: Uuid,
    /// Start time (UTC).
    pub start_time: DateTime<Utc>,
    /// End time (UTC). None if still running.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<DateTime<Utc>>,
    /// Span status.
    pub status: SpanStatus,
    /// Failure reasons (populated when status is Failed).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failure_reasons: Vec<String>,
    /// Artifacts attached to this span (only for agent spans).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
}

// ---------------------------------------------------------------------------
// Execution Result
// ---------------------------------------------------------------------------

/// The output contract for every externally-invoked operation.
///
/// Contains the repo-level span, all nested agent-level spans, and the
/// original operation result. This structure satisfies the Agentics
/// ExecutionGraph invariant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// The execution ID echoed back to the caller.
    pub execution_id: Uuid,
    /// The repo-level span.
    pub repo_span: ExecutionSpan,
    /// All agent-level spans (nested under repo_span).
    pub agent_spans: Vec<ExecutionSpan>,
    /// The original operation result (the payload the caller cares about).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Whether the overall operation succeeded.
    pub success: bool,
}

// ---------------------------------------------------------------------------
// RepoExecution - Runtime Guard
// ---------------------------------------------------------------------------

/// Runtime guard that manages the lifecycle of repo and agent spans.
///
/// Create with `begin()`, register agents with `begin_agent()`,
/// complete or fail them, then call `finalize()` to enforce invariants
/// and produce the final `ExecutionResult`.
pub struct RepoExecution {
    execution_ctx: ExecutionContext,
    repo_span: ExecutionSpan,
    agent_spans: Vec<ExecutionSpan>,
}

impl RepoExecution {
    /// Create a new repo-level execution. Called at each entry point.
    pub fn begin(ctx: ExecutionContext) -> Self {
        let repo_span = ExecutionSpan {
            span_id: Uuid::new_v4(),
            parent_span_id: ctx.parent_span_id,
            span_type: SpanType::Repo,
            repo_name: REPO_NAME.to_string(),
            agent_name: None,
            agent_classification: None,
            execution_id: ctx.execution_id,
            start_time: Utc::now(),
            end_time: None,
            status: SpanStatus::Incomplete,
            failure_reasons: Vec::new(),
            artifacts: Vec::new(),
        };

        Self {
            execution_ctx: ctx,
            repo_span,
            agent_spans: Vec::new(),
        }
    }

    /// Get the repo span_id (for agents to use as parent).
    pub fn repo_span_id(&self) -> Uuid {
        self.repo_span.span_id
    }

    /// Get the execution context.
    pub fn context(&self) -> &ExecutionContext {
        &self.execution_ctx
    }

    /// Start an agent-level span. Returns the span_id for the agent.
    pub fn begin_agent(&mut self, agent_name: &str, classification: &str) -> Uuid {
        let span_id = Uuid::new_v4();
        let agent_span = ExecutionSpan {
            span_id,
            parent_span_id: self.repo_span.span_id,
            span_type: SpanType::Agent,
            repo_name: REPO_NAME.to_string(),
            agent_name: Some(agent_name.to_string()),
            agent_classification: Some(classification.to_string()),
            execution_id: self.execution_ctx.execution_id,
            start_time: Utc::now(),
            end_time: None,
            status: SpanStatus::Incomplete,
            failure_reasons: Vec::new(),
            artifacts: Vec::new(),
        };
        self.agent_spans.push(agent_span);
        span_id
    }

    /// Complete an agent-level span with success, attaching artifacts.
    pub fn complete_agent(&mut self, span_id: Uuid, artifacts: Vec<Artifact>) {
        if let Some(span) = self.agent_spans.iter_mut().find(|s| s.span_id == span_id) {
            span.end_time = Some(Utc::now());
            span.status = SpanStatus::Success;
            span.artifacts = artifacts;
        }
    }

    /// Fail an agent-level span with reason(s).
    pub fn fail_agent(&mut self, span_id: Uuid, reasons: Vec<String>) {
        if let Some(span) = self.agent_spans.iter_mut().find(|s| s.span_id == span_id) {
            span.end_time = Some(Utc::now());
            span.status = SpanStatus::Failed;
            span.failure_reasons = reasons;
        }
    }

    /// Finalize the entire repo execution. Returns ExecutionResult.
    ///
    /// Enforces invariants:
    /// - At least one agent span must have been emitted
    /// - All agent spans must be completed (success or failed)
    /// - If any agent failed, repo status = Failed
    /// - All spans are always returned (even on failure)
    pub fn finalize(mut self, result: Option<serde_json::Value>) -> ExecutionResult {
        self.repo_span.end_time = Some(Utc::now());

        // ENFORCE: At least one agent span must exist
        if self.agent_spans.is_empty() {
            self.repo_span.status = SpanStatus::Failed;
            self.repo_span.failure_reasons.push(
                "No agent-level spans emitted. At least one agent must execute.".to_string(),
            );
        }

        // ENFORCE: Every agent span must be completed
        for span in &self.agent_spans {
            if span.end_time.is_none() {
                self.repo_span.status = SpanStatus::Failed;
                self.repo_span.failure_reasons.push(format!(
                    "Agent span '{}' was not completed",
                    span.agent_name.as_deref().unwrap_or("unknown")
                ));
            }
        }

        // ENFORCE: If any agent failed, repo fails
        let any_agent_failed = self
            .agent_spans
            .iter()
            .any(|s| s.status == SpanStatus::Failed);
        if any_agent_failed {
            self.repo_span.status = SpanStatus::Failed;
        }

        // If no failures detected, mark success
        if self.repo_span.failure_reasons.is_empty() && !any_agent_failed {
            self.repo_span.status = SpanStatus::Success;
        }

        let success = self.repo_span.status == SpanStatus::Success;

        ExecutionResult {
            execution_id: self.execution_ctx.execution_id,
            repo_span: self.repo_span,
            agent_spans: self.agent_spans,
            result: if success { result } else { None },
            success,
        }
    }

    /// Emergency finalize: for when we need to return an error before any agents ran.
    pub fn finalize_error(mut self, reasons: Vec<String>) -> ExecutionResult {
        self.repo_span.end_time = Some(Utc::now());
        self.repo_span.status = SpanStatus::Failed;
        self.repo_span.failure_reasons = reasons;

        ExecutionResult {
            execution_id: self.execution_ctx.execution_id,
            repo_span: self.repo_span,
            agent_spans: self.agent_spans,
            result: None,
            success: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate an incoming ExecutionContext.
///
/// Returns an error ExecutionResult (suitable for returning to the caller)
/// if the context is missing or has a nil parent_span_id.
pub fn validate_execution_context(
    ctx: Option<ExecutionContext>,
) -> Result<ExecutionContext, ExecutionResult> {
    let ctx = ctx.ok_or_else(|| {
        // Create a minimal error result when no context is provided at all
        let nil_id = Uuid::nil();
        ExecutionResult {
            execution_id: nil_id,
            repo_span: ExecutionSpan {
                span_id: Uuid::new_v4(),
                parent_span_id: nil_id,
                span_type: SpanType::Repo,
                repo_name: REPO_NAME.to_string(),
                agent_name: None,
                agent_classification: None,
                execution_id: nil_id,
                start_time: Utc::now(),
                end_time: Some(Utc::now()),
                status: SpanStatus::Failed,
                failure_reasons: vec![
                    "Missing execution_context. All operations require execution_context with execution_id and parent_span_id.".to_string(),
                ],
                artifacts: Vec::new(),
            },
            agent_spans: Vec::new(),
            result: None,
            success: false,
        }
    })?;

    if ctx.parent_span_id.is_nil() {
        let repo_exec = RepoExecution::begin(ctx.clone());
        return Err(repo_exec.finalize_error(vec![
            "Invalid parent_span_id: must not be nil UUID.".to_string(),
        ]));
    }

    Ok(ctx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> ExecutionContext {
        ExecutionContext {
            execution_id: Uuid::new_v4(),
            parent_span_id: Uuid::new_v4(),
            trace_id: None,
        }
    }

    #[test]
    fn test_repo_execution_success_with_agent() {
        let ctx = test_context();
        let mut exec = RepoExecution::begin(ctx.clone());

        let agent_span_id = exec.begin_agent("latency-analysis-agent", "analysis");
        exec.complete_agent(agent_span_id, vec![]);

        let result = exec.finalize(Some(serde_json::json!({"status": "ok"})));

        assert!(result.success);
        assert_eq!(result.execution_id, ctx.execution_id);
        assert_eq!(result.repo_span.span_type, SpanType::Repo);
        assert_eq!(result.repo_span.repo_name, REPO_NAME);
        assert_eq!(result.repo_span.status, SpanStatus::Success);
        assert_eq!(result.repo_span.parent_span_id, ctx.parent_span_id);
        assert_eq!(result.agent_spans.len(), 1);
        assert_eq!(
            result.agent_spans[0].agent_name,
            Some("latency-analysis-agent".to_string())
        );
        assert_eq!(result.agent_spans[0].span_type, SpanType::Agent);
        assert_eq!(result.agent_spans[0].status, SpanStatus::Success);
        assert_eq!(
            result.agent_spans[0].parent_span_id,
            result.repo_span.span_id
        );
        assert!(result.result.is_some());
    }

    #[test]
    fn test_repo_execution_fails_without_agents() {
        let ctx = test_context();
        let exec = RepoExecution::begin(ctx);

        let result = exec.finalize(Some(serde_json::json!({"status": "ok"})));

        assert!(!result.success);
        assert_eq!(result.repo_span.status, SpanStatus::Failed);
        assert!(result
            .repo_span
            .failure_reasons
            .iter()
            .any(|r| r.contains("No agent-level spans emitted")));
        assert!(result.result.is_none());
    }

    #[test]
    fn test_repo_execution_fails_when_agent_fails() {
        let ctx = test_context();
        let mut exec = RepoExecution::begin(ctx);

        let agent_span_id = exec.begin_agent("latency-analysis-agent", "analysis");
        exec.fail_agent(agent_span_id, vec!["Analysis error".to_string()]);

        let result = exec.finalize(Some(serde_json::json!({"status": "ok"})));

        assert!(!result.success);
        assert_eq!(result.repo_span.status, SpanStatus::Failed);
        assert_eq!(result.agent_spans[0].status, SpanStatus::Failed);
        assert!(result.agent_spans[0]
            .failure_reasons
            .contains(&"Analysis error".to_string()));
        // All spans are still returned even on failure
        assert_eq!(result.agent_spans.len(), 1);
    }

    #[test]
    fn test_repo_execution_fails_with_incomplete_agent() {
        let ctx = test_context();
        let mut exec = RepoExecution::begin(ctx);

        // Start agent but never complete it
        exec.begin_agent("latency-analysis-agent", "analysis");

        let result = exec.finalize(None);

        assert!(!result.success);
        assert_eq!(result.repo_span.status, SpanStatus::Failed);
        assert!(result
            .repo_span
            .failure_reasons
            .iter()
            .any(|r| r.contains("was not completed")));
    }

    #[test]
    fn test_multiple_agent_spans() {
        let ctx = test_context();
        let mut exec = RepoExecution::begin(ctx);

        let agent1 = exec.begin_agent("latency-analysis-agent", "analysis");
        let agent2 = exec.begin_agent("cold-start-mitigation-agent", "measurement");
        exec.complete_agent(agent1, vec![]);
        exec.complete_agent(agent2, vec![]);

        let result = exec.finalize(Some(serde_json::json!({})));

        assert!(result.success);
        assert_eq!(result.agent_spans.len(), 2);
    }

    #[test]
    fn test_validate_execution_context_missing() {
        let result = validate_execution_context(None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.success);
        assert!(err
            .repo_span
            .failure_reasons
            .iter()
            .any(|r| r.contains("Missing execution_context")));
    }

    #[test]
    fn test_validate_execution_context_nil_parent() {
        let ctx = ExecutionContext {
            execution_id: Uuid::new_v4(),
            parent_span_id: Uuid::nil(),
            trace_id: None,
        };
        let result = validate_execution_context(Some(ctx));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.success);
        assert!(err
            .repo_span
            .failure_reasons
            .iter()
            .any(|r| r.contains("nil UUID")));
    }

    #[test]
    fn test_validate_execution_context_valid() {
        let ctx = test_context();
        let result = validate_execution_context(Some(ctx.clone()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().execution_id, ctx.execution_id);
    }

    #[test]
    fn test_finalize_error() {
        let ctx = test_context();
        let exec = RepoExecution::begin(ctx);

        let result = exec.finalize_error(vec!["Something went wrong".to_string()]);

        assert!(!result.success);
        assert_eq!(result.repo_span.status, SpanStatus::Failed);
        assert!(result
            .repo_span
            .failure_reasons
            .contains(&"Something went wrong".to_string()));
    }

    #[test]
    fn test_artifact_from_json_output() {
        let output = serde_json::json!({"analysis": "result"});
        let artifact = Artifact::from_json_output("analysis_output", "analysis-123", &output);

        assert_eq!(artifact.artifact_type, "analysis_output");
        assert_eq!(artifact.reference_id, "analysis-123");
        assert_eq!(artifact.format, "application/json");
        assert!(artifact.size_bytes.is_some());
        assert_eq!(artifact.content_hash.len(), 64);
    }

    #[test]
    fn test_execution_result_serialization() {
        let ctx = test_context();
        let mut exec = RepoExecution::begin(ctx);

        let agent1 = exec.begin_agent("latency-analysis-agent", "analysis");
        let artifact = Artifact::from_json_output("analysis_output", "test-id", &"test");
        exec.complete_agent(agent1, vec![artifact]);

        let result = exec.finalize(Some(serde_json::json!({"data": "test"})));

        // Verify it serializes to JSON without loss
        let json = serde_json::to_string_pretty(&result).unwrap();
        let deserialized: ExecutionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.execution_id, result.execution_id);
        assert_eq!(deserialized.success, result.success);
        assert_eq!(deserialized.repo_span.span_id, result.repo_span.span_id);
        assert_eq!(deserialized.agent_spans.len(), 1);
        assert_eq!(deserialized.agent_spans[0].artifacts.len(), 1);
    }

    #[test]
    fn test_execution_context_effective_trace_id() {
        let ctx = ExecutionContext {
            execution_id: Uuid::new_v4(),
            parent_span_id: Uuid::new_v4(),
            trace_id: None,
        };
        assert_eq!(ctx.effective_trace_id(), ctx.execution_id);

        let trace = Uuid::new_v4();
        let ctx_with_trace = ExecutionContext {
            trace_id: Some(trace),
            ..ctx
        };
        assert_eq!(ctx_with_trace.effective_trace_id(), trace);
    }

    #[test]
    fn test_span_causal_ordering() {
        let ctx = test_context();
        let mut exec = RepoExecution::begin(ctx.clone());

        let agent1 = exec.begin_agent("agent-a", "analysis");
        let agent2 = exec.begin_agent("agent-b", "measurement");
        exec.complete_agent(agent1, vec![]);
        exec.complete_agent(agent2, vec![]);

        let result = exec.finalize(None);

        // Repo span parent is the Core's span
        assert_eq!(result.repo_span.parent_span_id, ctx.parent_span_id);

        // All agent spans parent to the repo span
        for agent_span in &result.agent_spans {
            assert_eq!(agent_span.parent_span_id, result.repo_span.span_id);
        }
    }
}
