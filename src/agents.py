"""
Core agent types and orchestration system using LangGraph.
Implements Orchestrator, Validator, Planner, Tool-Caller, and Coach agents.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union
from enum import Enum
import uuid

from langgraph.graph import StateGraph, END

from src.models import (
    AgentLog, WorkflowResult, UserContext, Response, TokenUsage
)
from src.gemini_client import GeminiClient, GeminiResponse, MockMode
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager


class AgentType(Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    VALIDATOR = "validator"
    PLANNER = "planner"
    TOOL_CALLER = "tool_caller"
    COACH = "coach"


class WorkflowState(TypedDict):
    """State structure for LangGraph workflows."""
    user_id: str
    user_context: Dict[str, Any]
    current_task: str
    task_data: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    workflow_result: Dict[str, Any]
    confidence_scores: Dict[str, float]
    execution_logs: List[Dict[str, Any]]
    error_messages: List[str]
    next_agent: Optional[str]
    completed: bool


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    name: str
    agent_type: AgentType
    max_retries: int = 3
    timeout_seconds: float = 30.0
    confidence_threshold: float = 0.8
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        config: AgentConfig,
        gemini_client: GeminiClient,
        context_manager: ContextAssembler,
        confidence_manager: ConfidenceManager
    ):
        """Initialize base agent.
        
        Args:
            config: Agent configuration
            gemini_client: Gemini API client
            context_manager: Context management system
            confidence_manager: Confidence scoring system
        """
        self.config = config
        self.gemini_client = gemini_client
        self.context_manager = context_manager
        self.confidence_manager = confidence_manager
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # Initialize execution tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        
    @abstractmethod
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute the agent's main functionality.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary with agent output and updated state
        """
        pass
    
    def _create_agent_log(
        self,
        action: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        success: bool = True,
        error_message: str = ""
    ) -> AgentLog:
        """Create an agent log entry."""
        return AgentLog(
            agent_name=self.config.name,
            action=action,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            success=success,
            error_message=error_message
        )
    
    def _log_execution(self, log: AgentLog) -> None:
        """Log agent execution to file."""
        try:
            log_entry = {
                "timestamp": log.timestamp.isoformat(),
                "agent_name": log.agent_name,
                "action": log.action,
                "execution_time": log.execution_time,
                "success": log.success,
                "error_message": log.error_message,
                "input_data_size": len(str(log.input_data)),
                "output_data_size": len(str(log.output_data))
            }
            
            # Write to local log file
            import os
            os.makedirs("data/agent_logs", exist_ok=True)
            
            log_file = f"data/agent_logs/{self.config.name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log execution: {e}")
    
    def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update agent performance metrics."""
        self.execution_count += 1
        self.total_execution_time += execution_time
        if success:
            self.success_count += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "agent_name": self.config.name,
            "execution_count": self.execution_count,
            "success_rate": self.success_count / max(1, self.execution_count),
            "average_execution_time": self.total_execution_time / max(1, self.execution_count),
            "total_execution_time": self.total_execution_time
        }


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates multi-step processes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow_templates = self._load_workflow_templates()
    
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute orchestration logic."""
        start_time = time.time()
        
        try:
            # Determine workflow type and next steps
            workflow_type = state.get("current_task", "general")
            user_context = state.get("user_context", {})
            
            # Analyze task requirements
            analysis_result = await self._analyze_task_requirements(
                workflow_type, state.get("task_data", {})
            )
            
            # Determine agent sequence
            agent_sequence = self._determine_agent_sequence(workflow_type, analysis_result)
            
            # Create execution plan
            execution_plan = {
                "workflow_type": workflow_type,
                "agent_sequence": agent_sequence,
                "estimated_time": self._estimate_execution_time(agent_sequence),
                "confidence": analysis_result.get("confidence", 0.8),
                "next_agent": agent_sequence[0] if agent_sequence else None
            }
            
            execution_time = time.time() - start_time
            
            # Create log entry
            log = self._create_agent_log(
                action="orchestrate_workflow",
                input_data={"workflow_type": workflow_type, "task_data": state.get("task_data", {})},
                output_data=execution_plan,
                execution_time=execution_time
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, True)
            
            return {
                "orchestrator_output": execution_plan,
                "next_agent": execution_plan["next_agent"],
                "confidence_scores": {self.config.name: execution_plan["confidence"]},
                "execution_logs": [log.to_dict()]
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Orchestration failed: {str(e)}"
            
            log = self._create_agent_log(
                action="orchestrate_workflow",
                input_data={"workflow_type": state.get("current_task", "unknown")},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, False)
            
            return {
                "orchestrator_output": {"error": error_msg},
                "next_agent": None,
                "confidence_scores": {self.config.name: 0.0},
                "execution_logs": [log.to_dict()],
                "error_messages": [error_msg]
            }
    
    async def _analyze_task_requirements(self, workflow_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements to determine processing approach."""
        prompt = f"""
        Analyze the following task requirements:
        
        Workflow Type: {workflow_type}
        Task Data: {json.dumps(task_data, indent=2)}
        
        Provide analysis including:
        1. Complexity level (low/medium/high)
        2. Required agent types
        3. Estimated processing time
        4. Confidence in successful completion
        5. Potential risks or challenges
        
        Respond in JSON format.
        """
        
        try:
            response = self.gemini_client.generate_content(prompt, temperature=0.3)
            
            # Parse JSON response
            analysis = json.loads(response.content)
            analysis["confidence"] = response.confidence
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Task analysis failed: {e}")
            return {
                "complexity": "medium",
                "required_agents": ["validator", "planner"],
                "estimated_time": 30.0,
                "confidence": 0.5,
                "risks": [f"Analysis failed: {str(e)}"]
            }
    
    def _determine_agent_sequence(self, workflow_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Determine the sequence of agents needed for the workflow."""
        # Default sequences for different workflow types
        sequences = {
            "funding_form": ["validator", "planner", "tool_caller"],
            "daily_planning": ["planner", "tool_caller", "coach"],
            "general": ["validator", "planner"],
            "complex_analysis": ["validator", "planner", "tool_caller", "coach"]
        }
        
        base_sequence = sequences.get(workflow_type, sequences["general"])
        
        # Modify sequence based on analysis
        complexity = analysis.get("complexity", "medium")
        if complexity == "high":
            # Add coach for high complexity tasks
            if "coach" not in base_sequence:
                base_sequence.append("coach")
        elif complexity == "low":
            # Simplify for low complexity
            base_sequence = [agent for agent in base_sequence if agent != "coach"]
        
        return base_sequence
    
    def _estimate_execution_time(self, agent_sequence: List[str]) -> float:
        """Estimate total execution time for agent sequence."""
        # Base time estimates per agent type (in seconds)
        agent_times = {
            "validator": 5.0,
            "planner": 10.0,
            "tool_caller": 8.0,
            "coach": 7.0
        }
        
        return sum(agent_times.get(agent, 5.0) for agent in agent_sequence)
    
    def _load_workflow_templates(self) -> Dict[str, Any]:
        """Load workflow templates from configuration."""
        # This would typically load from a config file
        return {
            "funding_form": {
                "description": "Process and validate funding applications",
                "required_fields": ["company_name", "funding_amount", "business_plan"],
                "validation_rules": ["amount_positive", "plan_complete"]
            },
            "daily_planning": {
                "description": "Generate personalized daily action plans",
                "required_context": ["user_goals", "business_stage", "recent_activities"],
                "output_format": "structured_plan"
            }
        }


class ValidatorAgent(BaseAgent):
    """Validator agent that validates data and responses."""
    
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute validation logic."""
        start_time = time.time()
        
        try:
            task_data = state.get("task_data", {})
            workflow_type = state.get("current_task", "general")
            
            # Perform validation based on workflow type
            validation_result = await self._validate_data(workflow_type, task_data)
            
            execution_time = time.time() - start_time
            
            log = self._create_agent_log(
                action="validate_data",
                input_data={"workflow_type": workflow_type, "data_size": len(str(task_data))},
                output_data=validation_result,
                execution_time=execution_time
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, validation_result["is_valid"])
            
            return {
                "validator_output": validation_result,
                "confidence_scores": {self.config.name: validation_result["confidence"]},
                "execution_logs": [log.to_dict()]
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Validation failed: {str(e)}"
            
            log = self._create_agent_log(
                action="validate_data",
                input_data={"workflow_type": state.get("current_task", "unknown")},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, False)
            
            return {
                "validator_output": {"is_valid": False, "error": error_msg},
                "confidence_scores": {self.config.name: 0.0},
                "execution_logs": [log.to_dict()],
                "error_messages": [error_msg]
            }
    
    async def _validate_data(self, workflow_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data based on workflow type and rules."""
        validation_rules = self._get_validation_rules(workflow_type)
        
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "errors": [],
            "warnings": [],
            "validated_fields": []
        }
        
        # Apply validation rules
        for rule in validation_rules:
            try:
                rule_result = await self._apply_validation_rule(rule, data)
                
                if not rule_result["passed"]:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(rule_result["message"])
                
                validation_result["confidence"] *= rule_result.get("confidence", 1.0)
                validation_result["validated_fields"].append(rule["field"])
                
            except Exception as e:
                validation_result["warnings"].append(f"Rule {rule['name']} failed: {str(e)}")
                validation_result["confidence"] *= 0.9
        
        return validation_result
    
    def _get_validation_rules(self, workflow_type: str) -> List[Dict[str, Any]]:
        """Get validation rules for specific workflow type."""
        rules = {
            "funding_form": [
                {
                    "name": "company_name_required",
                    "field": "company_name",
                    "type": "required",
                    "message": "Company name is required"
                },
                {
                    "name": "funding_amount_positive",
                    "field": "funding_amount",
                    "type": "numeric_positive",
                    "message": "Funding amount must be positive"
                },
                {
                    "name": "business_plan_length",
                    "field": "business_plan",
                    "type": "min_length",
                    "params": {"min_length": 100},
                    "message": "Business plan must be at least 100 characters"
                }
            ],
            "daily_planning": [
                {
                    "name": "goals_present",
                    "field": "goals",
                    "type": "required",
                    "message": "User goals are required for planning"
                }
            ]
        }
        
        return rules.get(workflow_type, [])
    
    async def _apply_validation_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single validation rule to data."""
        field = rule["field"]
        rule_type = rule["type"]
        value = data.get(field)
        
        if rule_type == "required":
            passed = value is not None and str(value).strip() != ""
            
        elif rule_type == "numeric_positive":
            try:
                passed = float(value) > 0
            except (ValueError, TypeError):
                passed = False
                
        elif rule_type == "min_length":
            min_length = rule.get("params", {}).get("min_length", 0)
            passed = value is not None and len(str(value)) >= min_length
            
        else:
            # Unknown rule type
            passed = True
        
        return {
            "passed": passed,
            "message": rule["message"],
            "confidence": 1.0 if passed else 0.0
        }


class PlannerAgent(BaseAgent):
    """Planner agent that creates action plans and strategies."""
    
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute planning logic."""
        start_time = time.time()
        
        try:
            user_context = state.get("user_context", {})
            task_data = state.get("task_data", {})
            workflow_type = state.get("current_task", "general")
            
            # Generate plan using Gemini
            plan = await self._generate_plan(workflow_type, user_context, task_data)
            
            execution_time = time.time() - start_time
            
            log = self._create_agent_log(
                action="generate_plan",
                input_data={"workflow_type": workflow_type, "context_size": len(str(user_context))},
                output_data={"plan_items": len(plan.get("action_items", []))},
                execution_time=execution_time
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, True)
            
            return {
                "planner_output": plan,
                "confidence_scores": {self.config.name: plan.get("confidence", 0.8)},
                "execution_logs": [log.to_dict()]
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Planning failed: {str(e)}"
            
            log = self._create_agent_log(
                action="generate_plan",
                input_data={"workflow_type": state.get("current_task", "unknown")},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, False)
            
            return {
                "planner_output": {"error": error_msg},
                "confidence_scores": {self.config.name: 0.0},
                "execution_logs": [log.to_dict()],
                "error_messages": [error_msg]
            }
    
    async def _generate_plan(
        self,
        workflow_type: str,
        user_context: Dict[str, Any],
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate action plan using Gemini API."""
        
        # Build context-aware prompt
        prompt = self._build_planning_prompt(workflow_type, user_context, task_data)
        
        # Generate plan using Gemini
        response = self.gemini_client.generate_content(
            prompt,
            temperature=0.7,
            max_output_tokens=1500
        )
        
        try:
            # Try to parse as JSON first
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            # If not JSON, create structured plan from text
            plan = self._parse_text_plan(response.content)
        
        # Add metadata
        plan["confidence"] = response.confidence
        plan["generated_at"] = datetime.now().isoformat()
        plan["workflow_type"] = workflow_type
        
        return plan
    
    def _build_planning_prompt(
        self,
        workflow_type: str,
        user_context: Dict[str, Any],
        task_data: Dict[str, Any]
    ) -> str:
        """Build context-aware prompt for planning."""
        
        base_prompt = f"""
        You are an AI business planning assistant. Create a detailed action plan for the following:
        
        Workflow Type: {workflow_type}
        
        User Context:
        - Business Stage: {user_context.get('business_info', {}).get('stage', 'unknown')}
        - Industry: {user_context.get('business_info', {}).get('industry', 'unknown')}
        - Goals: {user_context.get('goals', [])}
        
        Task Data:
        {json.dumps(task_data, indent=2)}
        
        Create a structured plan with:
        1. Executive summary
        2. Action items with priorities and timelines
        3. Success metrics
        4. Potential risks and mitigation strategies
        5. Next steps
        
        Respond in JSON format with the following structure:
        {{
            "executive_summary": "Brief overview",
            "action_items": [
                {{
                    "title": "Action title",
                    "description": "Detailed description",
                    "priority": "high/medium/low",
                    "timeline": "timeframe",
                    "resources_needed": ["resource1", "resource2"]
                }}
            ],
            "success_metrics": ["metric1", "metric2"],
            "risks": [
                {{
                    "risk": "Risk description",
                    "mitigation": "Mitigation strategy"
                }}
            ],
            "next_steps": ["step1", "step2"]
        }}
        """
        
        return base_prompt
    
    def _parse_text_plan(self, text_content: str) -> Dict[str, Any]:
        """Parse text content into structured plan format."""
        # Simple text parsing - in production, this would be more sophisticated
        lines = text_content.split('\n')
        
        plan = {
            "executive_summary": "Generated from text response",
            "action_items": [],
            "success_metrics": [],
            "risks": [],
            "next_steps": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if "action" in line.lower() and "item" in line.lower():
                current_section = "action_items"
            elif "metric" in line.lower():
                current_section = "success_metrics"
            elif "risk" in line.lower():
                current_section = "risks"
            elif "next" in line.lower() and "step" in line.lower():
                current_section = "next_steps"
            elif line.startswith('-') or line.startswith('•'):
                # List item
                item = line[1:].strip()
                if current_section == "action_items":
                    plan["action_items"].append({
                        "title": item,
                        "description": item,
                        "priority": "medium",
                        "timeline": "TBD",
                        "resources_needed": []
                    })
                elif current_section in plan:
                    plan[current_section].append(item)
        
        return plan


class ToolCallerAgent(BaseAgent):
    """Tool-Caller agent that executes local tools and APIs."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_tools = self._initialize_tools()
    
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute tool calling logic."""
        start_time = time.time()
        
        try:
            task_data = state.get("task_data", {})
            planner_output = state.get("agent_outputs", {}).get("planner_output", {})
            
            # Determine which tools to call based on plan
            tool_calls = self._determine_tool_calls(planner_output, task_data)
            
            # Execute tool calls
            tool_results = await self._execute_tool_calls(tool_calls)
            
            execution_time = time.time() - start_time
            
            log = self._create_agent_log(
                action="execute_tools",
                input_data={"tool_calls": len(tool_calls)},
                output_data={"results": len(tool_results)},
                execution_time=execution_time
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, True)
            
            return {
                "tool_caller_output": {
                    "tool_results": tool_results,
                    "execution_summary": f"Executed {len(tool_calls)} tools successfully"
                },
                "confidence_scores": {self.config.name: 0.9},
                "execution_logs": [log.to_dict()]
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution failed: {str(e)}"
            
            log = self._create_agent_log(
                action="execute_tools",
                input_data={},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, False)
            
            return {
                "tool_caller_output": {"error": error_msg},
                "confidence_scores": {self.config.name: 0.0},
                "execution_logs": [log.to_dict()],
                "error_messages": [error_msg]
            }
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools."""
        return {
            "data_analyzer": {
                "description": "Analyze business data and metrics",
                "function": self._analyze_data
            },
            "report_generator": {
                "description": "Generate reports and summaries",
                "function": self._generate_report
            },
            "file_processor": {
                "description": "Process and validate files",
                "function": self._process_file
            },
            "priority_optimizer": {
                "description": "Optimize task priorities based on business goals",
                "function": self._optimize_priorities
            },
            "time_estimator": {
                "description": "Estimate time requirements for tasks",
                "function": self._estimate_task_times
            },
            "resource_checker": {
                "description": "Check resource availability and requirements",
                "function": self._check_resources
            }
        }
    
    def _determine_tool_calls(
        self,
        planner_output: Dict[str, Any],
        task_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Determine which tools to call based on the plan."""
        tool_calls = []
        
        action_items = planner_output.get("action_items", [])
        
        for item in action_items:
            # Simple heuristics to determine tool needs
            title = item.get("title", "").lower()
            description = item.get("description", "").lower()
            
            if "analyze" in title or "data" in description:
                tool_calls.append({
                    "tool": "data_analyzer",
                    "params": {"data": task_data, "analysis_type": "general"}
                })
            
            if "report" in title or "summary" in description:
                tool_calls.append({
                    "tool": "report_generator",
                    "params": {"content": item, "format": "text"}
                })
        
        return tool_calls
    
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute the determined tool calls."""
        results = []
        
        for call in tool_calls:
            tool_name = call["tool"]
            params = call.get("params", {})
            
            if tool_name in self.available_tools:
                try:
                    result = await self.available_tools[tool_name]["function"](params)
                    results.append({
                        "tool": tool_name,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "success": False,
                        "error": str(e)
                    })
            else:
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool {tool_name} not available"
                })
        
        return results
    
    async def _analyze_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data tool implementation."""
        data = params.get("data", {})
        analysis_type = params.get("analysis_type", "general")
        
        # Simple data analysis
        analysis = {
            "data_size": len(str(data)),
            "fields_count": len(data) if isinstance(data, dict) else 0,
            "analysis_type": analysis_type,
            "summary": f"Analyzed {analysis_type} data with {len(data) if isinstance(data, dict) else 0} fields"
        }
        
        return analysis
    
    async def _generate_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report tool implementation."""
        content = params.get("content", {})
        format_type = params.get("format", "text")
        
        report = {
            "format": format_type,
            "content_summary": str(content)[:200] + "..." if len(str(content)) > 200 else str(content),
            "generated_at": datetime.now().isoformat(),
            "word_count": len(str(content).split())
        }
        
        return report
    
    async def _process_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process file tool implementation."""
        file_path = params.get("file_path", "")
        
        # Mock file processing
        result = {
            "file_path": file_path,
            "processed": True,
            "size": "unknown",
            "type": "unknown"
        }
        
        return result
    
    async def _optimize_priorities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize task priorities based on business goals."""
        tasks = params.get("tasks", [])
        business_goals = params.get("business_goals", [])
        
        # Simple priority optimization logic
        optimized_tasks = []
        for task in tasks:
            priority_score = 0.5  # Default medium priority
            
            # Increase priority if task aligns with business goals
            task_text = str(task).lower()
            for goal in business_goals:
                if any(word in task_text for word in str(goal).lower().split()):
                    priority_score += 0.2
            
            # Cap at 1.0
            priority_score = min(1.0, priority_score)
            
            # Convert to priority level
            if priority_score >= 0.8:
                priority = "high"
            elif priority_score >= 0.6:
                priority = "medium"
            else:
                priority = "low"
            
            optimized_tasks.append({
                "task": task,
                "priority": priority,
                "priority_score": priority_score
            })
        
        return {
            "optimized_tasks": optimized_tasks,
            "optimization_method": "goal_alignment",
            "processed_count": len(tasks)
        }
    
    async def _estimate_task_times(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate time requirements for tasks."""
        tasks = params.get("tasks", [])
        
        time_estimates = []
        for task in tasks:
            task_text = str(task).lower()
            
            # Simple heuristics for time estimation
            if any(word in task_text for word in ["meeting", "call", "discussion"]):
                estimated_time = "30-60 minutes"
                time_minutes = 45
            elif any(word in task_text for word in ["analyze", "research", "review"]):
                estimated_time = "1-2 hours"
                time_minutes = 90
            elif any(word in task_text for word in ["create", "build", "develop", "write"]):
                estimated_time = "2-4 hours"
                time_minutes = 180
            elif any(word in task_text for word in ["plan", "strategy", "design"]):
                estimated_time = "1-3 hours"
                time_minutes = 120
            else:
                estimated_time = "30 minutes"
                time_minutes = 30
            
            time_estimates.append({
                "task": task,
                "estimated_time": estimated_time,
                "time_minutes": time_minutes
            })
        
        total_time = sum(est["time_minutes"] for est in time_estimates)
        
        return {
            "time_estimates": time_estimates,
            "total_estimated_minutes": total_time,
            "total_estimated_hours": round(total_time / 60, 1),
            "estimation_method": "heuristic_analysis"
        }
    
    async def _check_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check resource availability and requirements."""
        tasks = params.get("tasks", [])
        available_resources = params.get("available_resources", [])
        
        resource_analysis = []
        for task in tasks:
            task_text = str(task).lower()
            
            # Determine required resources based on task content
            required_resources = []
            if any(word in task_text for word in ["meeting", "call"]):
                required_resources.extend(["calendar", "communication_tools"])
            if any(word in task_text for word in ["analyze", "data"]):
                required_resources.extend(["data_access", "analysis_tools"])
            if any(word in task_text for word in ["create", "write", "develop"]):
                required_resources.extend(["development_tools", "focused_time"])
            if any(word in task_text for word in ["team", "collaborate"]):
                required_resources.extend(["team_availability", "collaboration_tools"])
            
            # Check availability
            available = all(res in available_resources for res in required_resources)
            
            resource_analysis.append({
                "task": task,
                "required_resources": required_resources,
                "resources_available": available,
                "missing_resources": [res for res in required_resources if res not in available_resources]
            })
        
        return {
            "resource_analysis": resource_analysis,
            "total_tasks_ready": sum(1 for analysis in resource_analysis if analysis["resources_available"]),
            "total_tasks": len(tasks),
            "readiness_percentage": round(sum(1 for analysis in resource_analysis if analysis["resources_available"]) / max(1, len(tasks)) * 100, 1)
        }


class CoachAgent(BaseAgent):
    """Coach agent that provides motivational and strategic guidance."""
    
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute coaching logic."""
        start_time = time.time()
        
        try:
            user_context = state.get("user_context", {})
            agent_outputs = state.get("agent_outputs", {})
            
            # Generate coaching response
            coaching_response = await self._generate_coaching_response(user_context, agent_outputs)
            
            execution_time = time.time() - start_time
            
            log = self._create_agent_log(
                action="provide_coaching",
                input_data={"context_size": len(str(user_context))},
                output_data={"response_length": len(coaching_response.get("message", ""))},
                execution_time=execution_time
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, True)
            
            return {
                "coach_output": coaching_response,
                "confidence_scores": {self.config.name: coaching_response.get("confidence", 0.8)},
                "execution_logs": [log.to_dict()]
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Coaching failed: {str(e)}"
            
            log = self._create_agent_log(
                action="provide_coaching",
                input_data={},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            
            self._log_execution(log)
            self._update_metrics(execution_time, False)
            
            return {
                "coach_output": {"error": error_msg},
                "confidence_scores": {self.config.name: 0.0},
                "execution_logs": [log.to_dict()],
                "error_messages": [error_msg]
            }
    
    async def _generate_coaching_response(
        self,
        user_context: Dict[str, Any],
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate personalized coaching response."""
        
        # Build coaching prompt
        prompt = self._build_coaching_prompt(user_context, agent_outputs)
        
        # Generate response using Gemini
        response = self.gemini_client.generate_content(
            prompt,
            temperature=0.8,  # Higher temperature for more creative coaching
            max_output_tokens=800
        )
        
        coaching_response = {
            "message": response.content,
            "confidence": response.confidence,
            "coaching_type": "motivational",
            "generated_at": datetime.now().isoformat(),
            "personalization_factors": self._extract_personalization_factors(user_context)
        }
        
        return coaching_response
    
    def _build_coaching_prompt(
        self,
        user_context: Dict[str, Any],
        agent_outputs: Dict[str, Any]
    ) -> str:
        """Build personalized coaching prompt."""
        
        business_info = user_context.get("business_info", {})
        goals = user_context.get("goals", [])
        preferences = user_context.get("preferences", {})
        
        # Extract key information from other agents
        plan_summary = ""
        if "planner_output" in agent_outputs:
            plan_summary = agent_outputs["planner_output"].get("executive_summary", "")
        
        prompt = f"""
        You are an experienced business coach and mentor. Provide personalized, motivational guidance to an entrepreneur.
        
        Entrepreneur Profile:
        - Business Stage: {business_info.get('stage', 'unknown')}
        - Industry: {business_info.get('industry', 'unknown')}
        - Company: {business_info.get('company_name', 'their venture')}
        - Team Size: {business_info.get('team_size', 'unknown')}
        - Goals: {', '.join(goals) if goals else 'Not specified'}
        - Communication Style: {preferences.get('communication_style', 'professional')}
        
        Current Plan Summary: {plan_summary}
        
        Provide coaching that:
        1. Acknowledges their current situation and progress
        2. Offers specific, actionable encouragement
        3. Addresses potential challenges with optimism
        4. Reinforces their goals and vision
        5. Provides strategic insights relevant to their stage and industry
        
        Keep the tone {preferences.get('communication_style', 'professional')} and inspiring.
        Focus on building confidence while being realistic about challenges.
        """
        
        return prompt
    
    def _extract_personalization_factors(self, user_context: Dict[str, Any]) -> List[str]:
        """Extract factors used for personalization."""
        factors = []
        
        business_info = user_context.get("business_info", {})
        if business_info.get("stage"):
            factors.append(f"business_stage:{business_info['stage']}")
        if business_info.get("industry"):
            factors.append(f"industry:{business_info['industry']}")
        
        preferences = user_context.get("preferences", {})
        if preferences.get("communication_style"):
            factors.append(f"communication_style:{preferences['communication_style']}")
        
        goals = user_context.get("goals", [])
        if goals:
            factors.append(f"goals_count:{len(goals)}")
        
        return factors


class AgentOrchestrator:
    """Main orchestrator for multi-agent workflows using LangGraph."""
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        context_manager: ContextAssembler,
        confidence_manager: ConfidenceManager,
        checkpoint_db_path: str = "data/checkpoints.db"
    ):
        """Initialize the agent orchestrator.
        
        Args:
            gemini_client: Gemini API client
            context_manager: Context management system
            confidence_manager: Confidence scoring system
            checkpoint_db_path: Path to SQLite database for checkpoints
        """
        self.gemini_client = gemini_client
        self.context_manager = context_manager
        self.confidence_manager = confidence_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Build workflow graph
        self.workflow_graph = self._build_workflow_graph()
        
        # Compile the graph (without checkpointing for now)
        self.compiled_graph = self.workflow_graph.compile()
        
        self.logger.info("Agent orchestrator initialized with LangGraph workflow")
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all agent instances."""
        agents = {}
        
        # Create agent configurations
        agent_configs = {
            "orchestrator": AgentConfig(
                name="orchestrator",
                agent_type=AgentType.ORCHESTRATOR,
                timeout_seconds=30.0,
                confidence_threshold=0.7
            ),
            "validator": AgentConfig(
                name="validator",
                agent_type=AgentType.VALIDATOR,
                timeout_seconds=15.0,
                confidence_threshold=0.8
            ),
            "planner": AgentConfig(
                name="planner",
                agent_type=AgentType.PLANNER,
                timeout_seconds=45.0,
                confidence_threshold=0.7
            ),
            "tool_caller": AgentConfig(
                name="tool_caller",
                agent_type=AgentType.TOOL_CALLER,
                timeout_seconds=60.0,
                confidence_threshold=0.8
            ),
            "coach": AgentConfig(
                name="coach",
                agent_type=AgentType.COACH,
                timeout_seconds=30.0,
                confidence_threshold=0.7
            )
        }
        
        # Initialize agent instances
        for name, config in agent_configs.items():
            if name == "orchestrator":
                agents[name] = OrchestratorAgent(
                    config, self.gemini_client, self.context_manager, self.confidence_manager
                )
            elif name == "validator":
                agents[name] = ValidatorAgent(
                    config, self.gemini_client, self.context_manager, self.confidence_manager
                )
            elif name == "planner":
                agents[name] = PlannerAgent(
                    config, self.gemini_client, self.context_manager, self.confidence_manager
                )
            elif name == "tool_caller":
                agents[name] = ToolCallerAgent(
                    config, self.gemini_client, self.context_manager, self.confidence_manager
                )
            elif name == "coach":
                agents[name] = CoachAgent(
                    config, self.gemini_client, self.context_manager, self.confidence_manager
                )
        
        return agents
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("tool_caller", self._tool_caller_node)
        workflow.add_node("coach", self._coach_node)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add conditional edges based on orchestrator output
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "validator": "validator",
                "planner": "planner",
                "tool_caller": "tool_caller",
                "coach": "coach",
                "end": END
            }
        )
        
        # Add edges from validator
        workflow.add_conditional_edges(
            "validator",
            self._route_from_validator,
            {
                "planner": "planner",
                "end": END
            }
        )
        
        # Add edges from planner
        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "tool_caller": "tool_caller",
                "coach": "coach",
                "finalizer": "finalizer"
            }
        )
        
        # Add edges from tool_caller
        workflow.add_conditional_edges(
            "tool_caller",
            self._route_from_tool_caller,
            {
                "coach": "coach",
                "finalizer": "finalizer"
            }
        )
        
        # Add edges from coach
        workflow.add_edge("coach", "finalizer")
        
        # Add edge from finalizer to end
        workflow.add_edge("finalizer", END)
        
        return workflow
    
    async def _orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """Execute orchestrator agent node."""
        result = await self.agents["orchestrator"].execute(state)
        
        # Update state with orchestrator output
        state["agent_outputs"]["orchestrator_output"] = result.get("orchestrator_output", {})
        state["next_agent"] = result.get("next_agent")
        state["confidence_scores"].update(result.get("confidence_scores", {}))
        state["execution_logs"].extend(result.get("execution_logs", []))
        
        if result.get("error_messages"):
            state["error_messages"].extend(result["error_messages"])
        
        return state
    
    async def _validator_node(self, state: WorkflowState) -> WorkflowState:
        """Execute validator agent node."""
        result = await self.agents["validator"].execute(state)
        
        # Update state with validator output
        state["agent_outputs"]["validator_output"] = result.get("validator_output", {})
        state["confidence_scores"].update(result.get("confidence_scores", {}))
        state["execution_logs"].extend(result.get("execution_logs", []))
        
        if result.get("error_messages"):
            state["error_messages"].extend(result["error_messages"])
        
        return state
    
    async def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """Execute planner agent node."""
        result = await self.agents["planner"].execute(state)
        
        # Update state with planner output
        state["agent_outputs"]["planner_output"] = result.get("planner_output", {})
        state["confidence_scores"].update(result.get("confidence_scores", {}))
        state["execution_logs"].extend(result.get("execution_logs", []))
        
        if result.get("error_messages"):
            state["error_messages"].extend(result["error_messages"])
        
        return state
    
    async def _tool_caller_node(self, state: WorkflowState) -> WorkflowState:
        """Execute tool caller agent node."""
        result = await self.agents["tool_caller"].execute(state)
        
        # Update state with tool caller output
        state["agent_outputs"]["tool_caller_output"] = result.get("tool_caller_output", {})
        state["confidence_scores"].update(result.get("confidence_scores", {}))
        state["execution_logs"].extend(result.get("execution_logs", []))
        
        if result.get("error_messages"):
            state["error_messages"].extend(result["error_messages"])
        
        return state
    
    async def _coach_node(self, state: WorkflowState) -> WorkflowState:
        """Execute coach agent node."""
        result = await self.agents["coach"].execute(state)
        
        # Update state with coach output
        state["agent_outputs"]["coach_output"] = result.get("coach_output", {})
        state["confidence_scores"].update(result.get("confidence_scores", {}))
        state["execution_logs"].extend(result.get("execution_logs", []))
        
        if result.get("error_messages"):
            state["error_messages"].extend(result["error_messages"])
        
        return state
    
    async def _finalizer_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow execution and prepare results."""
        
        # Calculate overall confidence
        confidence_scores = state.get("confidence_scores", {})
        overall_confidence = sum(confidence_scores.values()) / max(1, len(confidence_scores))
        
        # Prepare final result
        workflow_result = {
            "success": len(state.get("error_messages", [])) == 0,
            "overall_confidence": overall_confidence,
            "agent_outputs": state.get("agent_outputs", {}),
            "execution_summary": self._create_execution_summary(state),
            "total_execution_time": sum(
                log.get("execution_time", 0) for log in state.get("execution_logs", [])
            )
        }
        
        state["workflow_result"] = workflow_result
        state["completed"] = True
        
        return state
    
    def _create_execution_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Create execution summary from state."""
        logs = state.get("execution_logs", [])
        
        summary = {
            "total_agents_executed": len(set(log.get("agent_name", "") for log in logs)),
            "total_actions": len(logs),
            "successful_actions": sum(1 for log in logs if log.get("success", True)),
            "total_time": sum(log.get("execution_time", 0) for log in logs),
            "error_count": len(state.get("error_messages", []))
        }
        
        return summary
    
    def _route_from_orchestrator(self, state: WorkflowState) -> str:
        """Route from orchestrator based on next_agent."""
        next_agent = state.get("next_agent")
        
        if next_agent in ["validator", "planner", "tool_caller", "coach"]:
            return next_agent
        
        return "end"
    
    def _route_from_validator(self, state: WorkflowState) -> str:
        """Route from validator based on validation result."""
        validator_output = state.get("agent_outputs", {}).get("validator_output", {})
        
        if validator_output.get("is_valid", False):
            return "planner"
        
        return "end"
    
    def _route_from_planner(self, state: WorkflowState) -> str:
        """Route from planner based on plan complexity and workflow type."""
        workflow_type = state.get("current_task", "")
        planner_output = state.get("agent_outputs", {}).get("planner_output", {})
        action_items = planner_output.get("action_items", [])
        
        # For daily planning workflows, always chain through Tool-Caller
        if "daily_planning" in workflow_type.lower():
            return "tool_caller"
        
        # For other workflows, check if tools are needed
        needs_tools = any(
            "analyze" in item.get("title", "").lower() or 
            "process" in item.get("description", "").lower() or
            "data" in item.get("description", "").lower()
            for item in action_items
        )
        
        if needs_tools:
            return "tool_caller"
        
        # Check if coaching is needed (high complexity or many action items)
        if len(action_items) > 3:
            return "coach"
        
        return "finalizer"
    
    def _route_from_tool_caller(self, state: WorkflowState) -> str:
        """Route from tool caller based on workflow type."""
        workflow_type = state.get("current_task", "")
        
        # Daily planning workflows should always include coaching
        if "daily_planning" in workflow_type.lower():
            return "coach"
        
        # Other planning workflows should also include coaching
        if "planning" in workflow_type.lower():
            return "coach"
        
        return "finalizer"
    
    async def execute_workflow(
        self,
        workflow_type: str,
        user_id: str,
        task_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute a multi-agent workflow.
        
        Args:
            workflow_type: Type of workflow to execute
            user_id: User identifier
            task_data: Data for the task
            user_context: Optional user context
            
        Returns:
            WorkflowResult with execution details
        """
        start_time = time.time()
        
        try:
            # Prepare initial state
            initial_state: WorkflowState = {
                "user_id": user_id,
                "user_context": user_context or {},
                "current_task": workflow_type,
                "task_data": task_data,
                "agent_outputs": {},
                "workflow_result": {},
                "confidence_scores": {},
                "execution_logs": [],
                "error_messages": [],
                "next_agent": None,
                "completed": False
            }
            
            # Execute workflow with checkpointing
            config = {"configurable": {"thread_id": f"{user_id}_{workflow_type}_{int(time.time())}"}}
            
            final_state = None
            async for state in self.compiled_graph.astream(initial_state, config):
                final_state = state
                self.logger.debug(f"Workflow state update: {list(state.keys())}")
            
            if not final_state:
                raise Exception("Workflow execution failed - no final state")
            
            # Extract workflow result
            workflow_result_data = final_state.get("workflow_result", {})
            execution_time = time.time() - start_time
            
            # Create WorkflowResult object
            workflow_result = WorkflowResult(
                success=workflow_result_data.get("success", False),
                result_data={
                    "agent_outputs": workflow_result_data.get("agent_outputs", {}),
                    "execution_summary": workflow_result_data.get("execution_summary", {}),
                    "task_data": final_state.get("task_data", {})
                },
                execution_time=execution_time,
                confidence_score=workflow_result_data.get("overall_confidence", 0.0)
            )
            
            # Add agent logs
            for log_data in final_state.get("execution_logs", []):
                agent_log = AgentLog(
                    agent_name=log_data.get("agent_name", "unknown"),
                    action=log_data.get("action", "unknown"),
                    input_data=log_data.get("input_data", {}),
                    output_data=log_data.get("output_data", {}),
                    execution_time=log_data.get("execution_time", 0.0),
                    success=log_data.get("success", True),
                    error_message=log_data.get("error_message", "")
                )
                workflow_result.add_agent_log(agent_log)
            
            # Log workflow execution
            self._log_workflow_execution(workflow_type, user_id, workflow_result)
            
            return workflow_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Workflow execution failed: {str(e)}"
            
            self.logger.error(error_msg)
            
            # Create failed workflow result
            workflow_result = WorkflowResult(
                success=False,
                result_data={"error": error_msg},
                execution_time=execution_time,
                confidence_score=0.0
            )
            
            # Add error log
            error_log = AgentLog(
                agent_name="orchestrator",
                action="execute_workflow",
                input_data={"workflow_type": workflow_type, "user_id": user_id},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            workflow_result.add_agent_log(error_log)
            
            return workflow_result
    
    def _log_workflow_execution(
        self,
        workflow_type: str,
        user_id: str,
        result: WorkflowResult
    ) -> None:
        """Log workflow execution to file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "workflow_id": result.workflow_id,
                "workflow_type": workflow_type,
                "user_id": user_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "confidence_score": result.confidence_score,
                "agent_count": len(result.agent_logs),
                "total_agent_time": result.get_total_execution_time()
            }
            
            # Write to workflow log file
            import os
            os.makedirs("data/workflow_logs", exist_ok=True)
            
            log_file = f"data/workflow_logs/workflows_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log workflow execution: {e}")
    
    def get_agent_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents."""
        metrics = {}
        
        for name, agent in self.agents.items():
            metrics[name] = agent.get_performance_metrics()
        
        return metrics
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow types."""
        return [
            "funding_form",
            "daily_planning",
            "general",
            "complex_analysis"
        ]
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all agents."""
        health_status = {}
        
        for name, agent in self.agents.items():
            try:
                # Simple health check - verify agent can be initialized
                health_status[name] = agent.config.enabled
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                health_status[name] = False
        
        # Check Gemini client
        health_status["gemini_client"] = self.gemini_client.health_check()
        
        return health_status