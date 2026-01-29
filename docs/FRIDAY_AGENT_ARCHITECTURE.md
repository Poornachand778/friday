# Friday AI - Agent Architecture Design

**Created**: January 28, 2026
**Purpose**: Design document for deploying Friday as an agentic AI system on AWS

---

## Executive Summary

Friday is not just a fine-tuned LLM - it's an **agent** that can:
1. Understand context and intent
2. Decide when to use tools
3. Execute tools via MCP
4. Process results and respond naturally

This document covers the complete architecture for deploying Friday as a production agent.

---

## Current State Analysis

### What Already Exists (from codebase exploration)

| Component | Location | Status |
|-----------|----------|--------|
| **Orchestrator** | `orchestrator/core.py` | Implemented |
| **Agent Loop** | `orchestrator/core.py:_handle_tool_calls()` | Implemented (max 5 iterations) |
| **Tool Registry** | `orchestrator/tools/registry.py` | 10 tools registered |
| **MCP Servers** | `mcp/scene_manager/`, `mcp/gmail/`, `mcp/voice/` | Implemented |
| **Context System** | `orchestrator/context/contexts.py` | 4 contexts (Writers Room, Kitchen, etc.) |
| **LLM Client** | `orchestrator/inference/local_llm.py` | OpenAI/Anthropic format support |
| **Training Data** | `data/instructions/iteration2_tool_examples.jsonl` | 30+ tool calling examples |

### Current Agent Flow (Already Implemented!)

```
User Message
    ↓
FastAPI /chat endpoint
    ↓
FridayOrchestrator.chat()
    ↓
ContextDetector → Detect room/context
    ↓
ContextBuilder → Build full prompt + available tools
    ↓
LLMClient.chat() → Get response with potential tool_calls
    ↓
_handle_tool_calls() [AGENT LOOP]
    ├─ Extract tool calls from response
    ├─ Execute via ToolRegistry
    ├─ Add results to conversation
    ├─ Call LLM again with results
    └─ Repeat until no more tool calls (max 5)
    ↓
Return final response
```

**Key Finding**: The agent architecture is ALREADY built. What's missing is:
1. **AWS Deployment** of the orchestrator
2. **Fine-tuned model** that knows WHEN to call tools
3. **MCP servers** deployed and accessible

---

## AWS Deployment Architecture

### Option A: Full Serverless (Recommended for Cost)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AWS CLOUD                                      │
│                                                                          │
│  ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐  │
│  │   API Gateway   │────▶│  Lambda Function │────▶│   SageMaker     │  │
│  │   (WebSocket)   │     │  (Orchestrator)  │     │   Endpoint      │  │
│  │                 │     │                  │     │  (Friday LLM)   │  │
│  └─────────────────┘     └────────┬─────────┘     └─────────────────┘  │
│                                   │                                      │
│                    ┌──────────────┼──────────────┐                      │
│                    ▼              ▼              ▼                       │
│          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│          │   Lambda    │  │   Lambda    │  │   Lambda    │              │
│          │  MCP Scene  │  │  MCP Gmail  │  │  MCP Voice  │              │
│          │   Manager   │  │             │  │             │              │
│          └──────┬──────┘  └─────────────┘  └─────────────┘              │
│                 │                                                        │
│                 ▼                                                        │
│          ┌─────────────┐                                                │
│          │    RDS      │  (PostgreSQL - scenes, projects)               │
│          │  PostgreSQL │                                                │
│          └─────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Cost Estimate**: ~$50-100/month (pay-per-use)

### Option B: Container-Based (Recommended for Performance)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AWS CLOUD                                      │
│                                                                          │
│  ┌─────────────────┐                                                    │
│  │       ALB       │  (Application Load Balancer)                       │
│  │   (HTTPS/WSS)   │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────┐                            │
│  │              ECS Fargate                │                            │
│  │  ┌─────────────────────────────────┐   │                            │
│  │  │     Friday Orchestrator         │   │                            │
│  │  │     (FastAPI Container)         │───┼───▶ SageMaker Endpoint     │
│  │  │                                 │   │     (Friday LLM)            │
│  │  │  - Agent Loop                   │   │                            │
│  │  │  - Context Detection            │   │                            │
│  │  │  - Tool Registry                │   │                            │
│  │  └─────────────────────────────────┘   │                            │
│  │                                         │                            │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐│                            │
│  │  │MCP Scene │ │MCP Gmail │ │MCP Voice ││                            │
│  │  │(Sidecar) │ │(Sidecar) │ │(Sidecar) ││                            │
│  │  └────┬─────┘ └──────────┘ └──────────┘│                            │
│  │       │                                 │                            │
│  └───────┼─────────────────────────────────┘                            │
│          │                                                              │
│          ▼                                                              │
│  ┌─────────────┐     ┌─────────────┐                                   │
│  │    RDS      │     │ ElastiCache │  (Session/Conversation Memory)    │
│  │  PostgreSQL │     │   (Redis)   │                                   │
│  └─────────────┘     └─────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Cost Estimate**: ~$150-300/month (always-on)

### Option C: Hybrid (Best of Both)

- **SageMaker Endpoint**: Always-on for LLM (with auto-scaling)
- **Lambda**: MCP tools (serverless, event-driven)
- **ECS**: Orchestrator (containerized, scales with demand)

---

## Component Deep Dive

### 1. SageMaker Endpoint (LLM with Tool Calling)

**Current**: `friday-iter3` endpoint with LoRA adapters

**Challenge**: Fine-tuned LLaMA 3.1 8B needs to learn tool calling format

**Solution**: Train with tool calling examples in the dataset

```json
{
  "messages": [
    {"role": "system", "content": "You are Friday..."},
    {"role": "user", "content": "Find the court scene"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {
        "name": "scene_search",
        "arguments": "{\"query\": \"court scene\"}"
      }}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "[{\"scene_code\": \"SC045\"...}]"},
    {"role": "assistant", "content": "Court scene dorikindi - SC045..."}
  ]
}
```

**Key Insight from AWS Research**:
> "Tool calling support varies by model. Models like Mistral-Small-24B-Instruct-2501 have demonstrated reliable tool calling capabilities."

LLaMA 3.1 8B Instruct DOES support tool calling natively - we just need to fine-tune it with our tool definitions.

### 2. MCP Server Deployment Options

**Option A: AWS Lambda (Recommended)**

Using [awslabs/run-model-context-protocol-servers-with-aws-lambda](https://github.com/awslabs/run-model-context-protocol-servers-with-aws-lambda):

```python
# Example: Wrapping scene_manager as Lambda
from run_mcp_servers_with_aws_lambda import create_lambda_handler
from mcp.scene_manager.server import SceneManagerMCPServer

handler = create_lambda_handler(SceneManagerMCPServer)
```

**Pros**:
- Pay-per-invocation ($0.0000002 per request)
- Auto-scaling
- No server management

**Cons**:
- Cold starts (1-3 seconds)
- 15-minute timeout limit

**Option B: ECS Sidecar**

Run MCP servers as sidecar containers alongside orchestrator:

```yaml
# ECS Task Definition
containers:
  - name: friday-orchestrator
    image: friday-orchestrator:latest
    portMappings: [{containerPort: 8000}]

  - name: mcp-scene-manager
    image: friday-mcp-scene:latest
    command: ["python", "-m", "mcp.scene_manager.server"]
```

**Pros**:
- Low latency (same network)
- No cold starts

**Cons**:
- Always-on cost

### 3. Orchestrator Deployment

**Dockerfile** (already exists at `orchestrator/Dockerfile`):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY orchestrator/ ./orchestrator/
COPY config/ ./config/
EXPOSE 8000
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**ECS Service Configuration**:
- Instance: 2 vCPU, 4 GB RAM (Fargate)
- Auto-scaling: 1-10 tasks based on CPU
- Health check: `/health` endpoint

### 4. Database (RDS PostgreSQL)

**Current**: Local PostgreSQL with pgvector

**Migration to RDS**:
- Instance: db.t4g.medium (2 vCPU, 4 GB)
- Storage: 20 GB gp3
- Enable pgvector extension for embeddings
- Multi-AZ for production

---

## Agent Loop Design (Enhanced)

### Current Implementation Analysis

From `orchestrator/core.py:_handle_tool_calls()`:

```python
async def _handle_tool_calls(self, response, built_context, max_iterations=5):
    """Handle tool calls from LLM response - agentic loop"""

    while current_response.has_tool_calls and iteration < max_iterations:
        # Execute tools
        for tc in current_response.tool_calls:
            result = self._tool_registry.execute(tc.name, tc.arguments)

        # Get next LLM response with tool results
        current_response = await self._llm_client.chat(
            messages=messages_with_tool_results,
            tools=available_tools,
        )

    return final_response
```

### Enhanced Agent Loop (Proposed)

```python
async def enhanced_agent_loop(self, user_message, context):
    """
    Enhanced agent loop with:
    - Parallel tool execution
    - Error recovery
    - Streaming support
    - Observation/reflection pattern
    """

    MAX_ITERATIONS = 5
    iteration = 0

    while iteration < MAX_ITERATIONS:
        # 1. THINK: Get LLM reasoning + tool calls
        response = await self._llm_client.chat(
            messages=conversation,
            tools=context.available_tools,
            tool_choice="auto",  # Let model decide
        )

        if not response.has_tool_calls:
            # No tools needed, return response
            return response

        # 2. ACT: Execute tools (parallel when possible)
        tool_results = await asyncio.gather(*[
            self._execute_tool_async(tc)
            for tc in response.tool_calls
        ])

        # 3. OBSERVE: Add results to conversation
        conversation.extend([
            {"role": "assistant", "tool_calls": response.tool_calls},
            *[{"role": "tool", "tool_call_id": r.id, "content": r.content}
              for r in tool_results]
        ])

        # 4. REFLECT: Check if we should continue
        iteration += 1

        # Optional: Add reflection step
        # "Given these results, do I need more information?"

    # Safety: Max iterations reached
    return await self._generate_summary_response(conversation)
```

---

## Training Data Requirements for Agent

### Current Tool Examples: 30+ (from iteration2_tool_examples.jsonl)

### Required for Reliable Agent Behavior: 100-150 examples

**Categories to Cover**:

| Category | Examples Needed | Current |
|----------|-----------------|---------|
| Single tool call | 30 | ~15 |
| Multi-tool sequence | 25 | ~5 |
| Tool + conversation | 20 | ~5 |
| No tool needed (rejection) | 20 | ~3 |
| Error handling | 15 | ~2 |
| Clarification before tool | 15 | 0 |
| Telugu tool calls | 15 | 0 |
| **Total** | **140** | **~30** |

### Example: Clarification Before Tool Call

```json
{
  "messages": [
    {"role": "user", "content": "Update the scene"},
    {"role": "assistant", "content": "Which scene? Scene number or description cheppu, update chestha."}
  ],
  "metadata": {"task": "clarification", "no_tool_call": true}
}
```

### Example: No Tool Needed

```json
{
  "messages": [
    {"role": "user", "content": "What makes a good screenplay?"},
    {"role": "assistant", "content": "Three pillars: engaging characters with clear wants, escalating conflict, and pacing that keeps tension. Structure matters, but story soul matters more."}
  ],
  "metadata": {"task": "conversation", "no_tool_call": true}
}
```

---

## Deployment Roadmap

### Phase 1: Local Agent Testing (Current)

```
Local Machine
├── FastAPI Orchestrator (localhost:8000)
├── MCP Servers (stdio processes)
├── PostgreSQL (localhost:5432)
└── SageMaker Endpoint (cloud, friday-iter3)
```

**Test Command**:
```bash
cd orchestrator && uvicorn main:app --reload
# Test: curl -X POST localhost:8000/chat -d '{"message": "Find court scene"}'
```

### Phase 2: Cloud Infrastructure Setup

1. **RDS PostgreSQL**
   - Create instance with pgvector
   - Migrate schema and data
   - Update connection strings

2. **ECR Repositories**
   - friday-orchestrator
   - friday-mcp-scene-manager
   - friday-mcp-gmail

3. **VPC & Security Groups**
   - Private subnets for services
   - Public subnet for ALB
   - Security groups for RDS, ECS, Lambda

### Phase 3: Service Deployment

1. **Deploy MCP Servers to Lambda**
   ```bash
   # Using AWS SAM
   sam deploy --template mcp-lambda-template.yaml
   ```

2. **Deploy Orchestrator to ECS**
   ```bash
   # Build and push image
   docker build -t friday-orchestrator .
   aws ecr push ...

   # Create ECS service
   aws ecs create-service --service-name friday-orchestrator ...
   ```

3. **Configure ALB + Route 53**
   - HTTPS termination
   - Custom domain: friday.kalvanamu.com

### Phase 4: Integration & Testing

1. **Update orchestrator to use Lambda MCP**
   ```python
   # Instead of stdio MCP
   class LambdaMCPClient:
       async def call_tool(self, name, args):
           response = lambda_client.invoke(
               FunctionName=f"friday-mcp-{name}",
               Payload=json.dumps({"tool": name, "arguments": args})
           )
           return json.loads(response["Payload"].read())
   ```

2. **End-to-end tests**
   - Voice → STT → Orchestrator → MCP → LLM → TTS → Voice
   - Web chat interface
   - Tool execution latency

### Phase 5: Production Hardening

- CloudWatch alarms
- X-Ray tracing
- Auto-scaling policies
- Cost monitoring
- Backup & recovery

---

## Cost Analysis

### Development/Testing Phase

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| SageMaker Endpoint | ml.g5.2xlarge, 8hr/day | ~$200 |
| RDS PostgreSQL | db.t4g.micro | ~$15 |
| Lambda (MCP) | Pay-per-use | ~$5 |
| **Total** | | **~$220/month** |

### Production Phase

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| SageMaker Endpoint | ml.g5.2xlarge, always-on | ~$600 |
| ECS Fargate | 2 vCPU, 4 GB, 2 tasks | ~$150 |
| RDS PostgreSQL | db.t4g.medium, Multi-AZ | ~$100 |
| Lambda (MCP) | ~100K invocations | ~$10 |
| ALB | Standard | ~$25 |
| **Total** | | **~$885/month** |

### Cost Optimization Strategies

1. **SageMaker Inference Components**: Share endpoint across multiple models
2. **Spot Instances**: For non-critical workloads
3. **Reserved Capacity**: 1-year commitment = 30% savings
4. **Auto-scaling**: Scale to zero during off-hours

---

## Security Considerations

### Authentication

1. **User → ALB**: Cognito or API Keys
2. **Orchestrator → MCP Lambda**: IAM roles
3. **Orchestrator → SageMaker**: IAM roles
4. **MCP → RDS**: IAM database authentication

### Data Protection

- Encryption at rest (RDS, S3)
- Encryption in transit (TLS everywhere)
- VPC isolation
- No PII in logs

### Rate Limiting

- API Gateway throttling
- SageMaker invocation limits
- Cost alerts

---

## Key Decisions Needed

### 1. MCP Deployment Strategy

| Option | Latency | Cost | Complexity |
|--------|---------|------|------------|
| **Lambda** | 100-500ms (cold: 1-3s) | Low | Medium |
| **ECS Sidecar** | 10-50ms | Medium | Low |
| **Hybrid** | Depends | Medium | High |

**Recommendation**: Start with ECS sidecar for low latency, migrate to Lambda for cost optimization later.

### 2. Training Data Priority

| Option | Effort | Impact |
|--------|--------|--------|
| A. Finish Phase 1 (117→300 pairs) | Medium | High persona quality |
| B. Add 100+ tool examples | Medium | Better agent behavior |
| C. Both in parallel | High | Best overall |

**Recommendation**: Do both. Phase 1 for persona, tool examples for agent capability.

### 3. LLM Hosting

| Option | Cost | Latency | Control |
|--------|------|---------|---------|
| SageMaker (current) | $600/mo | 200-500ms | Full |
| Bedrock (managed) | Pay-per-token | 100-300ms | Limited |
| Local (RTX 4090) | $0 (after hw) | 50-150ms | Full |

**Recommendation**: Keep SageMaker for now, migrate to local hardware when production-ready.

---

## Next Steps

1. **Immediate**: Finish Phase 1 review (117 pairs → validate quality)
2. **This Week**: Add 50+ tool calling examples to training data
3. **Next Week**: Train Iteration 4 with tool calling support
4. **Week 3**: Test agent loop locally with new model
5. **Week 4**: Deploy to AWS (ECS + Lambda)

---

## References

- [AWS SageMaker + MCP Integration](https://aws.amazon.com/blogs/machine-learning/extend-large-language-models-powered-by-amazon-sagemaker-ai-using-model-context-protocol/)
- [MCP Servers on AWS Lambda](https://github.com/awslabs/run-model-context-protocol-servers-with-aws-lambda)
- [AWS Guidance for MCP Deployment](https://aws.amazon.com/solutions/guidance/deploying-model-context-protocol-servers-on-aws/)
- [Building AI Agents on AWS 2025](https://dev.to/aws-builders/building-ai-agents-on-aws-in-2025-a-practitioners-guide-to-bedrock-agentcore-and-beyond-4efn)
- [Strands Agents SDK](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/sagemaker/)

---

*Last updated: January 28, 2026*
