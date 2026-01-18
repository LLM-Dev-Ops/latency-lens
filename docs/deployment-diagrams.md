# LLM-Latency-Lens: Architecture Diagrams

## Overview Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT TOPOLOGY OVERVIEW                     │
└─────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
     │  Developer   │         │   CI/CD      │         │  Production  │
     │  Workstation │         │   Pipeline   │         │   Service    │
     └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
            │                        │                        │
            │                        │                        │
     ┌──────▼───────┐         ┌─────▼────────┐         ┌─────▼────────┐
     │ Standalone   │         │   GitHub     │         │  Embedded    │
     │     CLI      │         │   Actions    │         │   Library    │
     │              │         │   /GitLab    │         │              │
     │ • Local runs │         │ • Auto tests │         │ • In-process │
     │ • Quick test │         │ • PR checks  │         │ • Real-time  │
     │ • Dev debug  │         │ • Scheduled  │         │ • Production │
     └──────────────┘         └──────────────┘         └──────────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │                                 │
            ┌────────▼─────────┐           ┌──────────▼─────────┐
            │   Distributed    │           │    Observatory     │
            │    Execution     │           │    Integration     │
            │                  │           │                    │
            │ • Multi-region   │           │ • Prometheus       │
            │ • High scale     │           │ • Grafana          │
            │ • Coordinated    │           │ • Alerts           │
            └──────────────────┘           │ • Tracing          │
                                           └────────────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                               │
└─────────────────────────────────────────────────────────────────┘

  Configuration      Execution         Metrics          Storage
  ─────────────      ─────────         ───────          ───────

┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│              │   │              │   │              │   │            │
│ YAML/TOML    │──▶│   Profiler   │──▶│   Metrics    │──▶│   Files    │
│   Config     │   │    Engine    │   │  Collector   │   │  (JSON/CSV)│
│              │   │              │   │              │   │            │
└──────────────┘   └──────┬───────┘   └──────┬───────┘   └────────────┘
                          │                  │
                          │                  │           ┌────────────┐
                          │                  │           │            │
                   ┌──────▼───────┐          └──────────▶│  Database  │
                   │              │                      │ (SQLite/PG)│
                   │  LLM APIs    │                      │            │
                   │              │                      └────────────┘
                   │ • OpenAI     │
                   │ • Anthropic  │                      ┌────────────┐
                   │ • Google     │                      │            │
                   │ • Custom     │                      │ Prometheus │
                   │              │          ┌──────────▶│  Metrics   │
                   └──────┬───────┘          │           │            │
                          │                  │           └────────────┘
                          │           ┌──────┴───────┐
                          │           │              │   ┌────────────┐
                          │           │   Exporters  │   │            │
                          └──────────▶│              │──▶│   Jaeger   │
                                      │ • Prometheus │   │   Traces   │
                                      │ • OpenTel    │   │            │
                                      │ • Logs       │   └────────────┘
                                      │ • Callbacks  │
                                      └──────────────┘
```

## Distributed Architecture Detail

```
┌───────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED EXECUTION ARCHITECTURE                 │
└───────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────┐
                         │   Load Balancer     │
                         │   (Kubernetes)      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Coordinator       │
                         │   ┌─────────────┐   │
                         │   │Work Splitter│   │
                         │   └──────┬──────┘   │
                         │          │          │
                         │   ┌──────▼──────┐   │
                         │   │  Scheduler  │   │
                         │   └──────┬──────┘   │
                         │          │          │
                         │   ┌──────▼──────┐   │
                         │   │ Aggregator  │   │
                         │   └─────────────┘   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │     Message Queue (Redis)     │
                    │                               │
                    │  ┌─────────┐  ┌─────────┐   │
                    │  │ Task Q  │  │Result Q │   │
                    │  └─────────┘  └─────────┘   │
                    └───┬────────┬────────┬────────┘
                        │        │        │
        ┌───────────────┘        │        └───────────────┐
        │                        │                        │
┌───────▼────────┐      ┌────────▼────────┐      ┌───────▼────────┐
│   Worker Pool  │      │   Worker Pool   │      │   Worker Pool  │
│   (Region 1)   │      │   (Region 2)    │      │   (Region 3)   │
│                │      │                 │      │                │
│  ┌──────────┐  │      │  ┌──────────┐   │      │  ┌──────────┐  │
│  │ Worker 1 │  │      │  │ Worker 1 │   │      │  │ Worker 1 │  │
│  ├──────────┤  │      │  ├──────────┤   │      │  ├──────────┤  │
│  │ Worker 2 │  │      │  │ Worker 2 │   │      │  │ Worker 2 │  │
│  ├──────────┤  │      │  ├──────────┤   │      │  ├──────────┤  │
│  │ Worker N │  │      │  │ Worker N │   │      │  │ Worker N │  │
│  └──────────┘  │      │  └──────────┘   │      │  └──────────┘  │
│                │      │                 │      │                │
│  Each Worker:  │      │  Each Worker:   │      │  Each Worker:  │
│  • Task fetch  │      │  • Task fetch   │      │  • Task fetch  │
│  • LLM call    │      │  • LLM call     │      │  • LLM call    │
│  • Metrics     │      │  • Metrics      │      │  • Metrics     │
│  • Result push │      │  • Result push  │      │  • Result push │
│                │      │                 │      │                │
│ us-east-1      │      │  eu-west-1      │      │ ap-south-1     │
└────────────────┘      └─────────────────┘      └────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                         ┌───────▼────────┐
                         │ Result Storage │
                         │                │
                         │  • S3/GCS      │
                         │  • PostgreSQL  │
                         │  • TimescaleDB │
                         └────────────────┘
```

## CI/CD Pipeline Integration

```
┌───────────────────────────────────────────────────────────────┐
│                    CI/CD PIPELINE FLOW                        │
└───────────────────────────────────────────────────────────────┘

   Git Push/PR
       │
       ▼
┌──────────────┐
│   Trigger    │
│   Pipeline   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Build     │
│    Stage     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Test      │
│    Stage     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│    Benchmark Stage                   │
│                                      │
│  ┌────────────────────────────────┐ │
│  │ 1. Download Baseline           │ │
│  └────────────┬───────────────────┘ │
│               ▼                     │
│  ┌────────────────────────────────┐ │
│  │ 2. Run llm-lens Benchmark      │ │
│  │    • Quick run for PRs (60s)   │ │
│  │    • Full run for main (600s)  │ │
│  └────────────┬───────────────────┘ │
│               ▼                     │
│  ┌────────────────────────────────┐ │
│  │ 3. Analyze Results             │ │
│  │    • Compare vs baseline       │ │
│  │    • Detect regressions        │ │
│  │    • Generate report           │ │
│  └────────────┬───────────────────┘ │
│               ▼                     │
│  ┌────────────────────────────────┐ │
│  │ 4. Quality Gate Check          │ │
│  │    • Latency threshold         │ │
│  │    • Error rate check          │ │
│  │    • Cost budget               │ │
│  └────────────┬───────────────────┘ │
└───────────────┼──────────────────────┘
                │
        ┌───────┴───────┐
        │               │
     Pass            Fail
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│  Proceed to  │  │  Block PR    │
│  Deploy      │  │  Notify Team │
└──────┬───────┘  └──────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Post-Pipeline Actions           │
│                                  │
│  • Upload results to S3          │
│  • Comment on PR                 │
│  • Update dashboard              │
│  • Send notifications            │
│  • Update baseline (if main)     │
└──────────────────────────────────┘
```

## Observatory Integration Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                  OBSERVATORY ARCHITECTURE                         │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Service  │  │ Service  │  │ Service  │  │   CLI    │       │
│  │    A     │  │    B     │  │    C     │  │          │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                           │                                     │
│               ┌───────────▼───────────┐                        │
│               │  llm-latency-lens     │                        │
│               │  Library/Profiler     │                        │
│               └───────────┬───────────┘                        │
└───────────────────────────┼─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │ Metrics      │ │  Traces    │ │   Logs     │
    │ (Prometheus) │ │  (OTLP)    │ │ (Loki)     │
    └───────┬──────┘ └─────┬──────┘ └─────┬──────┘
            │              │              │
            │      ┌───────▼──────┐       │
            │      │   Jaeger     │       │
            │      │   Backend    │       │
            │      └──────────────┘       │
            │                             │
            └──────────────┬──────────────┘
                           │
                 ┌─────────▼──────────┐
                 │    Grafana         │
                 │    Dashboards      │
                 │                    │
                 │  ┌──────────────┐  │
                 │  │   Overview   │  │
                 │  ├──────────────┤  │
                 │  │   Latency    │  │
                 │  ├──────────────┤  │
                 │  │   Cost       │  │
                 │  ├──────────────┤  │
                 │  │   Errors     │  │
                 │  ├──────────────┤  │
                 │  │   Traces     │  │
                 │  └──────────────┘  │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Alert Manager     │
                 │                    │
                 │  • Grouping        │
                 │  • Routing         │
                 │  • Deduplication   │
                 └─────────┬──────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
    ┌───────▼──────┐ ┌────▼──────┐ ┌────▼──────┐
    │    Slack     │ │ PagerDuty │ │   Email   │
    │ Notifications│ │ Incidents │ │  Alerts   │
    └──────────────┘ └───────────┘ └───────────┘
```

## Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     NETWORK TOPOLOGY                            │
└─────────────────────────────────────────────────────────────────┘

                          Internet
                              │
                    ┌─────────▼─────────┐
                    │  Load Balancer    │
                    │  (Ingress)        │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│  Application   │   │   Coordinator   │   │   Monitoring   │
│   Namespace    │   │   Namespace     │   │   Namespace    │
│                │   │                 │   │                │
│ ┌────────────┐ │   │ ┌────────────┐  │   │ ┌────────────┐ │
│ │  Service   │ │   │ │Coordinator │  │   │ │ Prometheus │ │
│ │   Pods     │ │   │ │    Pod     │  │   │ │            │ │
│ └────────────┘ │   │ └─────┬──────┘  │   │ └────────────┘ │
│                │   │       │         │   │                │
│                │   │ ┌─────▼──────┐  │   │ ┌────────────┐ │
│                │   │ │   Redis    │  │   │ │  Grafana   │ │
│                │   │ │  Cluster   │  │   │ │            │ │
│                │   │ └────────────┘  │   │ └────────────┘ │
└────────────────┘   └──────┬──────────┘   │                │
                            │              │ ┌────────────┐ │
                            │              │ │   Jaeger   │ │
        ┌───────────────────┼──────────┐   │ │            │ │
        │                   │          │   │ └────────────┘ │
┌───────▼────────┐   ┌──────▼──────┐  │   └────────────────┘
│ Worker Pool 1  │   │Worker Pool 2│  │
│  (us-east-1)   │   │ (eu-west-1) │  │   ┌────────────────┐
│                │   │             │  │   │   Storage      │
│ ┌────────────┐ │   │┌──────────┐ │  │   │                │
│ │Worker Pods │ │   ││Worker Pod│ │  └──▶│ ┌────────────┐ │
│ │(StatefulSet│ │   ││          │ │      │ │    S3      │ │
│ │  or Deploy)│ │   ││          │ │      │ │  Bucket    │ │
│ └────────────┘ │   │└──────────┘ │      │ └────────────┘ │
│                │   │             │      │                │
│                │   │             │      │ ┌────────────┐ │
│                │   │             │      │ │ PostgreSQL │ │
│                │   │             │      │ │  Database  │ │
│                │   │             │      │ └────────────┘ │
└────────────────┘   └─────────────┘      └────────────────┘
        │                   │                      │
        └───────────────────┼──────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   LLM APIs     │
                    │   (External)   │
                    │                │
                    │ • OpenAI       │
                    │ • Anthropic    │
                    │ • Google       │
                    └────────────────┘
```

## Scaling Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                    HORIZONTAL SCALING                           │
└─────────────────────────────────────────────────────────────────┘

Low Load (5 RPS)          Medium Load (50 RPS)      High Load (500 RPS)
─────────────────         ─────────────────────      ────────────────────

┌────────────┐            ┌────────────┐             ┌────────────┐
│Coordinator │            │Coordinator │             │Coordinator │
│   (1)      │            │   (1-2)    │             │   (3-5)    │
└─────┬──────┘            └─────┬──────┘             └─────┬──────┘
      │                         │                          │
┌─────▼──────┐            ┌─────▼──────┐             ┌─────▼──────┐
│  Workers   │            │  Workers   │             │  Workers   │
│   (2-5)    │            │  (10-20)   │             │ (50-100)   │
└────────────┘            └────────────┘             └────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    VERTICAL SCALING                             │
└─────────────────────────────────────────────────────────────────┘

Development              Staging                    Production
───────────              ───────                    ──────────

Worker Resources:        Worker Resources:          Worker Resources:
• CPU: 0.5 cores         • CPU: 2 cores             • CPU: 4 cores
• Memory: 512MB          • Memory: 2GB              • Memory: 8GB
• Concurrent: 10         • Concurrent: 50           • Concurrent: 100

Coordinator:             Coordinator:               Coordinator:
• CPU: 1 core            • CPU: 2 cores             • CPU: 4 cores
• Memory: 1GB            • Memory: 4GB              • Memory: 8GB

┌─────────────────────────────────────────────────────────────────┐
│                   AUTO-SCALING TRIGGERS                         │
└─────────────────────────────────────────────────────────────────┘

Metric                  Scale Up Threshold      Scale Down Threshold
──────                  ──────────────────      ────────────────────
CPU Usage               > 70%                   < 30%
Memory Usage            > 80%                   < 40%
Queue Depth             > 100 tasks             < 10 tasks
Response Time           > 5s (p95)              < 2s (p95)
Request Rate            > 80% capacity          < 50% capacity

Scaling Behavior:
• Cool-down period: 5 minutes
• Scale up: Immediate (within 30s)
• Scale down: Gradual (5 min intervals)
• Min replicas: 2 (HA)
• Max replicas: 100 (cost limit)
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Network Security                                      │
│                                                                 │
│  • Network Policies (K8s)                                       │
│  • Security Groups (Cloud)                                      │
│  • TLS/HTTPS only                                               │
│  • Private subnets for workers                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Authentication & Authorization                        │
│                                                                 │
│  • Service Accounts (K8s)                                       │
│  • RBAC policies                                                │
│  • IAM roles (Cloud)                                            │
│  • API key rotation                                             │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Secret Management                                     │
│                                                                 │
│  • Vault / Cloud Secret Manager                                │
│  • Encrypted at rest                                            │
│  • Encrypted in transit                                         │
│  • No secrets in logs/configs                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Data Protection                                       │
│                                                                 │
│  • PII detection & redaction                                    │
│  • Encryption for stored results                                │
│  • Data retention policies                                      │
│  • GDPR/CCPA compliance                                         │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: Monitoring & Audit                                    │
│                                                                 │
│  • Access logging                                               │
│  • Anomaly detection                                            │
│  • Security alerts                                              │
│  • Compliance reporting                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Disaster Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│                  DISASTER RECOVERY ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────────┘

Primary Region (us-east-1)          Backup Region (us-west-2)
──────────────────────────          ─────────────────────────

┌────────────────────────┐          ┌────────────────────────┐
│   Active Cluster       │          │   Standby Cluster      │
│                        │          │                        │
│  ┌──────────────────┐  │          │  ┌──────────────────┐  │
│  │   Coordinator    │  │          │  │   Coordinator    │  │
│  │    (Active)      │  │          │  │   (Standby)      │  │
│  └──────────────────┘  │          │  └──────────────────┘  │
│                        │          │                        │
│  ┌──────────────────┐  │          │  ┌──────────────────┐  │
│  │   Worker Pool    │  │          │  │   Worker Pool    │  │
│  │    (Active)      │  │          │  │   (Standby)      │  │
│  └──────────────────┘  │          │  └──────────────────┘  │
└────────┬───────────────┘          └─────────┬──────────────┘
         │                                    │
         └────────────┬───────────────────────┘
                      │
         ┌────────────▼───────────┐
         │  Cross-Region          │
         │  Replication           │
         │                        │
         │  • Database (Active-   │
         │    Passive)            │
         │  • Object Storage      │
         │    (Multi-region)      │
         │  • Config backup       │
         └────────────────────────┘

Recovery Scenarios:
──────────────────────

1. Regional Failure
   • RTO: 15 minutes
   • RPO: 5 minutes
   • Action: Failover to backup region

2. Service Failure
   • RTO: 2 minutes
   • RPO: 0 (no data loss)
   • Action: Kubernetes auto-restart

3. Data Corruption
   • RTO: 1 hour
   • RPO: 1 hour (backup interval)
   • Action: Restore from backup

4. Configuration Error
   • RTO: 5 minutes
   • RPO: 0
   • Action: Rollback to previous config
```

## Cost Optimization Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   COST OPTIMIZATION LAYERS                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Compute Optimization                                           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Spot      │  │  Reserved   │  │ On-Demand   │            │
│  │ Instances   │  │ Instances   │  │ Instances   │            │
│  │             │  │             │  │             │            │
│  │ Workers     │  │ Coordinator │  │  Critical   │            │
│  │ (70%)       │  │  (20%)      │  │   (10%)     │            │
│  │             │  │             │  │             │            │
│  │ Save 70%    │  │  Save 40%   │  │  Full cost  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Storage Optimization                                           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Hot       │  │   Warm      │  │   Cold      │            │
│  │  Storage    │  │  Storage    │  │  Storage    │            │
│  │             │  │             │  │             │            │
│  │  < 7 days   │  │ 7-90 days   │  │  > 90 days  │            │
│  │             │  │             │  │             │            │
│  │ S3 Standard │  │ S3 IA       │  │ S3 Glacier  │            │
│  │ $0.023/GB   │  │ $0.0125/GB  │  │ $0.004/GB   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Auto-Scaling Policies                                          │
│                                                                 │
│  Business Hours (9am-5pm)     Off-Hours (5pm-9am)              │
│  ─────────────────────────    ────────────────────             │
│  Min Workers: 10              Min Workers: 2                   │
│  Max Workers: 100             Max Workers: 20                  │
│  Target CPU: 70%              Target CPU: 50%                  │
│                                                                 │
│  Estimated Savings: 60% on compute costs                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  LLM API Cost Controls                                          │
│                                                                 │
│  • Request caching (dedupe identical requests)                  │
│  • Rate limiting (prevent runaway costs)                        │
│  • Budget alerts (notify at 50%, 80%, 100%)                     │
│  • Model selection (prefer cost-effective models)               │
│  • Token optimization (minimize prompt tokens)                  │
└─────────────────────────────────────────────────────────────────┘
```

This comprehensive set of architecture diagrams complements the deployment strategy document and provides visual representations of all major deployment patterns, data flows, and operational considerations.
