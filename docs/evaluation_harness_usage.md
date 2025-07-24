# FounderForge AI Evaluation Harness

The evaluation harness provides comprehensive testing capabilities for the FounderForge AI system, including accuracy measurement, confidence threshold validation, and fallback testing.

## Features

### 1. Predefined Test Scenarios
- **10 default scenarios** covering different business advice types:
  - Funding advice (Series A preparation, valuation guidance)
  - Business planning (business plan creation, go-to-market strategy)
  - Strategy guidance (market entry, product-market fit)
  - Operational support (team building, scaling operations)
  - General queries (terminology, equity distribution)

### 2. Comprehensive Evaluation Metrics
- **Accuracy scoring** based on keyword matching and response quality
- **Confidence threshold validation** (80% minimum target)
- **Execution time monitoring** (30-second maximum per scenario)
- **Fallback usage tracking** for low-confidence responses

### 3. Multiple Testing Modes
- **Full evaluation**: Run all or specific scenarios
- **Confidence threshold validation**: Test different confidence levels
- **Fallback mechanism testing**: Verify fallback triggers work correctly

## Usage

### Command Line Interface

```bash
# Run full evaluation with mock responses (no API calls)
python scripts/run_evaluation.py full --mock-mode

# Run specific scenarios
python scripts/run_evaluation.py full --scenarios funding_001 planning_001

# Generate detailed report
python scripts/run_evaluation.py full --generate-report

# Validate confidence thresholds
python scripts/run_evaluation.py confidence --mock-mode

# Test fallback mechanisms
python scripts/run_evaluation.py fallback --mock-mode
```

### Programmatic Usage

```python
from src.evaluation_harness import EvaluationHarness
from src.agents import AgentOrchestrator
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager
from src.gemini_client import GeminiClient, MockMode

# Initialize components
gemini_client = GeminiClient(api_key="your_key", mock_mode=MockMode.SUCCESS)
context_manager = ContextAssembler(db_manager=db_manager)
confidence_manager = ConfidenceManager()
agent_orchestrator = AgentOrchestrator(...)

# Create evaluation harness
harness = EvaluationHarness(
    gemini_client=gemini_client,
    context_manager=context_manager,
    confidence_manager=confidence_manager,
    agent_orchestrator=agent_orchestrator
)

# Run evaluation
summary = await harness.run_evaluation(mock_mode=True)
print(f"Accuracy: {summary.overall_accuracy:.3f}")
print(f"Passed: {summary.passed}/{summary.total_scenarios}")
```

## Performance Targets

The evaluation harness validates against these targets:

- **90% accuracy rate** on predefined test scenarios
- **80% confidence threshold** for responses
- **Sub-30 second execution time** per scenario
- **Proper fallback usage** for low-confidence or unsafe content

## Test Scenarios

### Scenario Structure
Each test scenario includes:
- **Query**: The input question/request
- **Expected keywords**: Terms that should appear in good responses
- **Success criteria**: Minimum requirements (keyword count, response length)
- **Context**: Business context for the query

### Example Scenario
```json
{
  "id": "funding_001",
  "name": "Basic Funding Advice",
  "query": "How should I prepare for a Series A funding round?",
  "expected_keywords": ["series a", "funding", "preparation", "investors", "pitch deck"],
  "success_criteria": {"min_keywords": 3, "response_length": 100}
}
```

## Output and Reporting

### Evaluation Summary
- Total scenarios tested
- Pass/fail/partial/error counts
- Overall accuracy and confidence metrics
- Average execution time
- Fallback usage rate

### Detailed Report
- Individual scenario results
- Keyword matching analysis
- Confidence scores
- Execution times
- Target achievement status

### File Outputs
- `evaluation_results_YYYYMMDD_HHMMSS.json`: Raw results data
- `evaluation_report_YYYYMMDD_HHMMSS.md`: Human-readable report
- `evaluation_logs.log`: Detailed execution logs

## Customization

### Adding New Scenarios
1. Edit `data/evaluation/test_scenarios.json`
2. Add new scenario with required fields
3. Run evaluation to test new scenarios

### Adjusting Targets
Modify targets in `EvaluationHarness.__init__()`:
```python
self.target_accuracy = 0.9  # 90% pass rate
self.target_confidence = 0.8  # 80% confidence
self.max_execution_time = 30.0  # 30 seconds max
```

## Integration with CI/CD

The evaluation harness returns appropriate exit codes:
- `0`: All targets met
- `1`: Targets missed or errors occurred

This allows integration with automated testing pipelines to ensure AI quality standards are maintained.