"""
Daily planning agent workflow for FounderForge AI Cofounder.
Chains Planner, Tool-Caller, and Coach agents for personalized action plan generation.
"""

import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

try:
    from .agents import AgentOrchestrator, WorkflowState
    from .models import WorkflowResult, AgentLog
    from .gemini_client import GeminiClient
    from .context_manager import ContextAssembler
    from .confidence_manager import ConfidenceManager
except ImportError:
    from agents import AgentOrchestrator, WorkflowState
    from models import WorkflowResult, AgentLog
    from gemini_client import GeminiClient
    from context_manager import ContextAssembler
    from confidence_manager import ConfidenceManager


@dataclass
class DailyPlanningInput:
    """Input data for daily planning workflow."""
    user_id: str
    current_priorities: List[str] = field(default_factory=list)
    available_time: str = "8 hours"  # e.g., "8 hours", "4 hours", "full day"
    energy_level: str = "medium"  # low, medium, high
    upcoming_deadlines: List[str] = field(default_factory=list)
    recent_accomplishments: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)
    meeting_schedule: List[Dict[str, Any]] = field(default_factory=list)
    personal_goals: List[str] = field(default_factory=list)
    business_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "current_priorities": self.current_priorities,
            "available_time": self.available_time,
            "energy_level": self.energy_level,
            "upcoming_deadlines": self.upcoming_deadlines,
            "recent_accomplishments": self.recent_accomplishments,
            "focus_areas": self.focus_areas,
            "meeting_schedule": self.meeting_schedule,
            "personal_goals": self.personal_goals,
            "business_context": self.business_context
        }


@dataclass
class ActionItem:
    """Individual action item in daily plan."""
    title: str
    description: str
    priority: str  # high, medium, low
    estimated_time: str  # e.g., "30 minutes", "2 hours"
    category: str  # e.g., "product", "business", "team", "personal"
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "estimated_time": self.estimated_time,
            "category": self.category,
            "deadline": self.deadline,
            "dependencies": self.dependencies,
            "tools_needed": self.tools_needed,
            "success_criteria": self.success_criteria
        }


@dataclass
class DailyPlan:
    """Complete daily action plan."""
    plan_id: str
    user_id: str
    date: str
    action_items: List[ActionItem] = field(default_factory=list)
    time_blocks: List[Dict[str, Any]] = field(default_factory=list)
    motivational_message: str = ""
    success_metrics: List[str] = field(default_factory=list)
    contingency_plans: List[str] = field(default_factory=list)
    coaching_insights: List[str] = field(default_factory=list)
    estimated_completion_time: str = ""
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "date": self.date,
            "action_items": [item.to_dict() for item in self.action_items],
            "time_blocks": self.time_blocks,
            "motivational_message": self.motivational_message,
            "success_metrics": self.success_metrics,
            "contingency_plans": self.contingency_plans,
            "coaching_insights": self.coaching_insights,
            "estimated_completion_time": self.estimated_completion_time,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat()
        }


class DailyPlanningWorkflow:
    """Daily planning workflow orchestrator."""
    
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        data_sources_path: str = "data/business_data"
    ):
        """Initialize daily planning workflow.
        
        Args:
            orchestrator: Agent orchestrator instance
            data_sources_path: Path to local data sources
        """
        self.orchestrator = orchestrator
        self.data_sources_path = Path(data_sources_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        self.data_sources_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.planning_count = 0
        self.total_planning_time = 0.0
        self.completion_rates = []
        
        # Load planning templates and configurations
        self.planning_templates = self._load_planning_templates()
        
    def _load_planning_templates(self) -> Dict[str, Any]:
        """Load planning templates and configurations."""
        templates = {
            "time_blocks": {
                "morning_focus": {
                    "name": "Morning Focus Block",
                    "duration": "2-3 hours",
                    "best_for": ["deep work", "strategic thinking", "complex problem solving"],
                    "energy_required": "high"
                },
                "afternoon_execution": {
                    "name": "Afternoon Execution Block",
                    "duration": "2-4 hours",
                    "best_for": ["meetings", "communication", "administrative tasks"],
                    "energy_required": "medium"
                },
                "evening_review": {
                    "name": "Evening Review Block",
                    "duration": "30-60 minutes",
                    "best_for": ["planning", "reflection", "preparation"],
                    "energy_required": "low"
                }
            },
            "priority_frameworks": {
                "eisenhower_matrix": {
                    "urgent_important": "Do first",
                    "important_not_urgent": "Schedule",
                    "urgent_not_important": "Delegate",
                    "neither": "Eliminate"
                },
                "founder_priorities": {
                    "product_development": 0.3,
                    "customer_acquisition": 0.25,
                    "team_building": 0.2,
                    "fundraising": 0.15,
                    "operations": 0.1
                }
            },
            "coaching_prompts": [
                "What would make today a success?",
                "What's the most important thing you can accomplish today?",
                "How can you move closer to your long-term goals?",
                "What obstacles might you face and how will you overcome them?",
                "How will you celebrate your wins today?"
            ]
        }
        
        return templates
    
    async def generate_daily_plan(
        self,
        planning_input: DailyPlanningInput,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[DailyPlan, WorkflowResult]:
        """Generate comprehensive daily action plan.
        
        Chains Planner, Tool-Caller, and Coach agents for personalized action plan generation.
        Integrates with local data sources and Gemini API for personalized coaching.
        Supports parallel task processing under 1-minute completion time.
        
        Args:
            planning_input: Daily planning input data
            user_context: Optional user context
            
        Returns:
            Tuple of (DailyPlan, WorkflowResult)
        """
        start_time = time.time()
        
        try:
            # Prepare enhanced task data with local data sources
            task_data = await self._prepare_task_data(planning_input, user_context)
            
            # Execute multi-agent workflow with proper chaining
            workflow_result = await self.orchestrator.execute_workflow(
                workflow_type="daily_planning",
                user_id=planning_input.user_id,
                task_data=task_data,
                user_context=user_context
            )
            
            # Validate that required agents were executed
            agent_outputs = workflow_result.result_data.get("agent_outputs", {})
            if not self._validate_agent_chain_execution(agent_outputs):
                self.logger.warning("Not all required agents executed in daily planning workflow")
            
            # Process workflow results into daily plan
            daily_plan = await self._process_workflow_results(
                planning_input, workflow_result
            )
            
            # Track performance metrics
            execution_time = time.time() - start_time
            self.planning_count += 1
            self.total_planning_time += execution_time
            
            # Validate performance requirement (under 1 minute)
            if execution_time >= 60.0:
                self.logger.warning(f"Daily planning exceeded 1-minute target: {execution_time:.2f}s")
            
            # Log successful planning
            self.logger.info(
                f"Daily plan generated for user {planning_input.user_id} in {execution_time:.2f}s"
            )
            
            return daily_plan, workflow_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Daily planning failed for user {planning_input.user_id}: {e}")
            
            # Return fallback plan
            fallback_plan = self._create_fallback_plan(planning_input)
            fallback_result = WorkflowResult(
                success=False,
                result_data={"error": str(e)},
                execution_time=execution_time,
                agent_logs=[],
                confidence_score=0.0
            )
            
            return fallback_plan, fallback_result
    
    def _validate_agent_chain_execution(self, agent_outputs: Dict[str, Any]) -> bool:
        """Validate that the required agent chain was executed for daily planning.
        
        Daily planning workflow should chain: Planner -> Tool-Caller -> Coach
        
        Args:
            agent_outputs: Dictionary of agent outputs from workflow execution
            
        Returns:
            True if required agents were executed, False otherwise
        """
        required_agents = ["planner_output"]  # Planner is mandatory
        recommended_agents = ["tool_caller_output", "coach_output"]  # These should be present for full workflow
        
        # Check if planner was executed (mandatory)
        has_planner = "planner_output" in agent_outputs
        if not has_planner:
            self.logger.error("Daily planning workflow missing required Planner agent")
            return False
        
        # Check for tool caller and coach (recommended for complete workflow)
        has_tool_caller = "tool_caller_output" in agent_outputs
        has_coach = "coach_output" in agent_outputs
        
        if not has_tool_caller:
            self.logger.warning("Daily planning workflow missing Tool-Caller agent")
        
        if not has_coach:
            self.logger.warning("Daily planning workflow missing Coach agent")
        
        # Log successful agent chain execution
        executed_agents = list(agent_outputs.keys())
        self.logger.info(f"Daily planning agents executed: {executed_agents}")
        
        # Return True if at least planner executed, but log warnings for missing agents
        return has_planner
    
    async def _prepare_task_data(
        self,
        planning_input: DailyPlanningInput,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare enhanced task data with local data sources."""
        
        # Load local data sources in parallel
        tasks = [
            self._load_user_business_data(planning_input.user_id),
            self._load_industry_insights(),
            self._load_planning_templates_data(),
            self._load_recent_activities(planning_input.user_id)
        ]
        
        business_data, industry_insights, templates, recent_activities = await asyncio.gather(*tasks)
        
        # Combine all data sources
        task_data = {
            "planning_input": planning_input.to_dict(),
            "user_context": user_context or {},
            "business_data": business_data,
            "industry_insights": industry_insights,
            "planning_templates": templates,
            "recent_activities": recent_activities,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "day_of_week": datetime.now().strftime("%A"),
            "time_blocks": self.planning_templates["time_blocks"],
            "priority_frameworks": self.planning_templates["priority_frameworks"]
        }
        
        return task_data
    
    async def _load_user_business_data(self, user_id: str) -> Dict[str, Any]:
        """Load user-specific business data from local files."""
        business_data = {}
        
        try:
            # Load user business profile
            business_file = self.data_sources_path / f"{user_id}_business.json"
            if business_file.exists():
                with open(business_file, 'r', encoding='utf-8') as f:
                    business_data = json.load(f)
            
            # Load user goals and objectives
            goals_file = self.data_sources_path / f"{user_id}_goals.json"
            if goals_file.exists():
                with open(goals_file, 'r', encoding='utf-8') as f:
                    goals_data = json.load(f)
                    business_data["goals"] = goals_data
            
            # Load user metrics and KPIs
            metrics_file = self.data_sources_path / f"{user_id}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                    business_data["metrics"] = metrics_data
                    
        except Exception as e:
            self.logger.warning(f"Failed to load business data for {user_id}: {e}")
        
        return business_data
    
    async def _load_industry_insights(self) -> Dict[str, Any]:
        """Load industry insights and best practices."""
        insights = {}
        
        try:
            insights_file = self.data_sources_path / "industry_insights.json"
            if insights_file.exists():
                with open(insights_file, 'r', encoding='utf-8') as f:
                    insights = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load industry insights: {e}")
        
        return insights
    
    async def _load_planning_templates_data(self) -> Dict[str, Any]:
        """Load planning templates and frameworks."""
        return self.planning_templates
    
    async def _load_recent_activities(self, user_id: str) -> List[Dict[str, Any]]:
        """Load recent user activities and accomplishments."""
        activities = []
        
        try:
            activities_file = self.data_sources_path / f"{user_id}_activities.json"
            if activities_file.exists():
                with open(activities_file, 'r', encoding='utf-8') as f:
                    activities_data = json.load(f)
                    activities = activities_data.get("recent_activities", [])
        except Exception as e:
            self.logger.warning(f"Failed to load recent activities for {user_id}: {e}")
        
        return activities
    
    async def _process_workflow_results(
        self,
        planning_input: DailyPlanningInput,
        workflow_result: WorkflowResult
    ) -> DailyPlan:
        """Process workflow results into structured daily plan."""
        
        # Extract agent outputs
        agent_outputs = workflow_result.result_data.get("agent_outputs", {})
        planner_output = agent_outputs.get("planner_output", {})
        tool_caller_output = agent_outputs.get("tool_caller_output", {})
        coach_output = agent_outputs.get("coach_output", {})
        
        # Create action items from planner output
        action_items = []
        for item in planner_output.get("action_items", []):
            action_item = ActionItem(
                title=item.get("title", ""),
                description=item.get("description", ""),
                priority=item.get("priority", "medium"),
                estimated_time=item.get("timeline", "30 minutes"),
                category=self._categorize_action_item(item),
                tools_needed=item.get("resources_needed", []),
                success_criteria=self._generate_success_criteria(item)
            )
            action_items.append(action_item)
        
        # Generate time blocks
        time_blocks = self._generate_time_blocks(action_items, planning_input)
        
        # Create daily plan
        daily_plan = DailyPlan(
            plan_id=f"plan_{planning_input.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=planning_input.user_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            action_items=action_items,
            time_blocks=time_blocks,
            motivational_message=coach_output.get("message", ""),
            success_metrics=planner_output.get("success_metrics", []),
            contingency_plans=self._generate_contingency_plans(action_items),
            coaching_insights=self._extract_coaching_insights(coach_output),
            estimated_completion_time=self._calculate_total_time(action_items),
            confidence_score=workflow_result.confidence_score
        )
        
        # Save plan to local storage
        await self._save_daily_plan(daily_plan)
        
        return daily_plan
    
    def _categorize_action_item(self, item: Dict[str, Any]) -> str:
        """Categorize action item based on content."""
        title = item.get("title", "").lower()
        description = item.get("description", "").lower()
        
        # Simple categorization logic
        if any(word in title + description for word in ["product", "develop", "build", "code"]):
            return "product"
        elif any(word in title + description for word in ["customer", "sales", "marketing", "user"]):
            return "business"
        elif any(word in title + description for word in ["team", "hire", "meeting", "people"]):
            return "team"
        elif any(word in title + description for word in ["fund", "investor", "pitch", "finance"]):
            return "funding"
        else:
            return "operations"
    
    def _generate_success_criteria(self, item: Dict[str, Any]) -> List[str]:
        """Generate success criteria for action item."""
        criteria = []
        
        # Basic success criteria based on action type
        title = item.get("title", "").lower()
        
        if "meeting" in title:
            criteria.extend([
                "Meeting completed on time",
                "Key decisions documented",
                "Next steps clearly defined"
            ])
        elif "analyze" in title or "research" in title:
            criteria.extend([
                "Analysis completed with clear findings",
                "Recommendations documented",
                "Data sources validated"
            ])
        elif "create" in title or "build" in title:
            criteria.extend([
                "Deliverable completed to specification",
                "Quality standards met",
                "Stakeholder approval obtained"
            ])
        else:
            criteria.extend([
                "Task completed successfully",
                "Objectives achieved",
                "Documentation updated"
            ])
        
        return criteria
    
    def _generate_time_blocks(
        self,
        action_items: List[ActionItem],
        planning_input: DailyPlanningInput
    ) -> List[Dict[str, Any]]:
        """Generate optimized time blocks for action items."""
        time_blocks = []
        
        # Group items by priority and energy requirements
        high_priority = [item for item in action_items if item.priority == "high"]
        medium_priority = [item for item in action_items if item.priority == "medium"]
        low_priority = [item for item in action_items if item.priority == "low"]
        
        # Morning focus block (high energy tasks)
        if high_priority and planning_input.energy_level in ["medium", "high"]:
            morning_block = {
                "name": "Morning Focus Block",
                "start_time": "09:00",
                "end_time": "11:30",
                "duration": "2.5 hours",
                "energy_level": "high",
                "action_items": [item.title for item in high_priority[:2]],
                "description": "Deep work on highest priority items"
            }
            time_blocks.append(morning_block)
        
        # Afternoon execution block
        if medium_priority:
            afternoon_block = {
                "name": "Afternoon Execution Block",
                "start_time": "13:00",
                "end_time": "16:00",
                "duration": "3 hours",
                "energy_level": "medium",
                "action_items": [item.title for item in medium_priority[:3]],
                "description": "Meetings, communication, and collaborative work"
            }
            time_blocks.append(afternoon_block)
        
        # Evening wrap-up block
        if low_priority:
            evening_block = {
                "name": "Evening Wrap-up Block",
                "start_time": "16:30",
                "end_time": "17:30",
                "duration": "1 hour",
                "energy_level": "low",
                "action_items": [item.title for item in low_priority[:2]],
                "description": "Administrative tasks and planning"
            }
            time_blocks.append(evening_block)
        
        return time_blocks
    
    def _generate_contingency_plans(self, action_items: List[ActionItem]) -> List[str]:
        """Generate contingency plans for potential issues."""
        contingencies = [
            "If running behind schedule, defer low-priority items to tomorrow",
            "If energy is low, switch to administrative tasks",
            "If meetings run long, have backup 15-minute versions of agenda items",
            "Keep buffer time between high-focus tasks for context switching"
        ]
        
        # Add specific contingencies based on action items
        has_meetings = any("meeting" in item.title.lower() for item in action_items)
        if has_meetings:
            contingencies.append("Have backup agenda for shortened meetings")
        
        has_creative_work = any(
            word in item.title.lower() 
            for item in action_items 
            for word in ["design", "create", "brainstorm", "strategy"]
        )
        if has_creative_work:
            contingencies.append("If creativity is blocked, switch to research or planning tasks")
        
        return contingencies
    
    def _extract_coaching_insights(self, coach_output: Dict[str, Any]) -> List[str]:
        """Extract coaching insights from coach agent output."""
        insights = []
        
        message = coach_output.get("message", "")
        if message:
            # Simple extraction of key insights
            sentences = message.split('. ')
            insights = [s.strip() + '.' for s in sentences if len(s.strip()) > 20][:3]
        
        # Add default insights if none extracted
        if not insights:
            insights = [
                "Focus on your highest-impact activities first",
                "Take breaks between intensive tasks to maintain energy",
                "Celebrate small wins throughout the day"
            ]
        
        return insights
    
    def _calculate_total_time(self, action_items: List[ActionItem]) -> str:
        """Calculate total estimated time for all action items."""
        total_minutes = 0
        
        for item in action_items:
            time_str = item.estimated_time.lower()
            
            # Simple time parsing
            if "hour" in time_str:
                hours = 1
                if any(char.isdigit() for char in time_str):
                    hours = int(''.join(filter(str.isdigit, time_str.split('hour')[0])))
                total_minutes += hours * 60
            elif "minute" in time_str:
                minutes = 30  # default
                if any(char.isdigit() for char in time_str):
                    minutes = int(''.join(filter(str.isdigit, time_str.split('minute')[0])))
                total_minutes += minutes
        
        # Convert back to hours and minutes
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0 and minutes > 0:
            return f"{hours} hours {minutes} minutes"
        elif hours > 0:
            return f"{hours} hours"
        else:
            return f"{minutes} minutes"
    
    async def _save_daily_plan(self, daily_plan: DailyPlan) -> bool:
        """Save daily plan to local storage."""
        try:
            plans_dir = self.data_sources_path / "daily_plans"
            plans_dir.mkdir(exist_ok=True)
            
            plan_file = plans_dir / f"{daily_plan.plan_id}.json"
            
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(daily_plan.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Daily plan saved: {plan_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save daily plan: {e}")
            return False
    
    def _create_fallback_plan(self, planning_input: DailyPlanningInput) -> DailyPlan:
        """Create a fallback plan when workflow fails."""
        
        # Create basic action items from input priorities
        action_items = []
        for i, priority in enumerate(planning_input.current_priorities[:5]):
            action_item = ActionItem(
                title=priority,
                description=f"Work on: {priority}",
                priority="medium",
                estimated_time="1 hour",
                category="general"
            )
            action_items.append(action_item)
        
        # Add default items if no priorities provided
        if not action_items:
            default_items = [
                ActionItem(
                    title="Review daily goals",
                    description="Review and prioritize today's objectives",
                    priority="high",
                    estimated_time="15 minutes",
                    category="planning"
                ),
                ActionItem(
                    title="Focus work session",
                    description="Dedicated time for important project work",
                    priority="high",
                    estimated_time="2 hours",
                    category="product"
                ),
                ActionItem(
                    title="Team check-in",
                    description="Connect with team members on progress",
                    priority="medium",
                    estimated_time="30 minutes",
                    category="team"
                )
            ]
            action_items.extend(default_items)
        
        return DailyPlan(
            plan_id=f"fallback_{planning_input.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=planning_input.user_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            action_items=action_items,
            motivational_message="Stay focused on your priorities and make steady progress today!",
            success_metrics=["Complete at least 2 high-priority tasks", "Maintain good energy throughout the day"],
            confidence_score=0.6
        )
    
    async def execute_parallel_planning(
        self,
        planning_inputs: List[DailyPlanningInput],
        max_workers: int = 3
    ) -> List[Tuple[DailyPlan, WorkflowResult]]:
        """Execute parallel daily planning for multiple users.
        
        Supports parallel task processing under 1-minute completion time.
        
        Args:
            planning_inputs: List of planning inputs for different users
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of tuples containing (DailyPlan, WorkflowResult) for each input
        """
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for parallel execution
            tasks = []
            for planning_input in planning_inputs:
                task = asyncio.create_task(
                    self.generate_daily_plan(planning_input)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create fallback plan for failed executions
                fallback_plan = self._create_fallback_plan(planning_inputs[i])
                fallback_result = WorkflowResult(
                    success=False,
                    result_data={"error": str(result)},
                    execution_time=0.0,
                    agent_logs=[],
                    confidence_score=0.0
                )
                processed_results.append((fallback_plan, fallback_result))
            else:
                processed_results.append(result)
        
        execution_time = time.time() - start_time
        
        # Log parallel execution performance
        self.logger.info(
            f"Parallel planning completed: {len(planning_inputs)} plans in {execution_time:.2f}s"
        )
        
        # Validate performance requirement (under 1 minute)
        if execution_time >= 60.0:
            self.logger.warning(f"Parallel planning exceeded 1-minute target: {execution_time:.2f}s")
        
        return processed_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        avg_time = self.total_planning_time / max(1, self.planning_count)
        
        return {
            "total_plans_generated": self.planning_count,
            "average_planning_time": avg_time,
            "total_planning_time": self.total_planning_time,
            "completion_rates": self.completion_rates,
            "performance_target_met": avg_time < 60.0,
            "target_completion_time": 60.0,  # 1 minute target
            "performance_ratio": min(1.0, 60.0 / avg_time) if avg_time > 0 else 0.0
        }

# Example usage and testing functions
async def create_sample_planning_input() -> DailyPlanningInput:
    """Create sample planning input for testing."""
    return DailyPlanningInput(
        user_id="test_user_001",
        current_priorities=[
            "Review product roadmap",
            "Prepare investor presentation",
            "Team standup meeting",
            "Customer feedback analysis"
        ],
        available_time="6 hours",
        energy_level="high",
        upcoming_deadlines=[
            "Investor meeting on Friday",
            "Product demo next week"
        ],
        recent_accomplishments=[
            "Completed user research interviews",
            "Finalized Q1 budget"
        ],
        focus_areas=["product", "fundraising"],
        personal_goals=[
            "Launch MVP by end of quarter",
            "Raise Series A funding"
        ],
        business_context={
            "stage": "early_stage",
            "industry": "fintech",
            "team_size": 8
        }
    )


async def test_daily_planning_workflow():
    """Test the daily planning workflow."""
    from gemini_client import GeminiClient, MockMode
    from context_manager import ContextAssembler
    from confidence_manager import ConfidenceManager
    from agents import AgentOrchestrator
    
    # Initialize components with mock mode for testing
    gemini_client = GeminiClient(mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    orchestrator = AgentOrchestrator(gemini_client, context_manager, confidence_manager)
    
    # Create workflow
    workflow = DailyPlanningWorkflow(orchestrator)
    
    # Create sample input
    planning_input = await create_sample_planning_input()
    
    # Generate daily plan
    daily_plan, workflow_result = await workflow.generate_daily_plan(planning_input)
    
    print(f"Generated daily plan with {len(daily_plan.action_items)} action items")
    print(f"Confidence score: {daily_plan.confidence_score:.2f}")
    print(f"Estimated completion time: {daily_plan.estimated_completion_time}")
    
    return daily_plan, workflow_result


if __name__ == "__main__":
    # Run test
    asyncio.run(test_daily_planning_workflow())