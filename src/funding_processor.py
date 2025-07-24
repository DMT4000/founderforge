"""
Funding form processing workflow using multi-agent system.
Implements specialized validation and processing for funding applications.
"""

import json
import logging
from logging_manager import get_logging_manager, LogLevel, LogCategory
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

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
class FundingFormData:
    """Structured funding form data."""
    company_name: str = ""
    funding_amount: float = 0.0
    business_plan: str = ""
    team_experience: str = ""
    market_size: str = ""
    revenue: Optional[float] = None
    customers: Optional[int] = None
    growth_rate: Optional[float] = None
    competition: Optional[str] = None
    use_of_funds: Optional[str] = None
    business_stage: str = "unknown"
    team_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "company_name": self.company_name,
            "funding_amount": self.funding_amount,
            "business_plan": self.business_plan,
            "team_experience": self.team_experience,
            "market_size": self.market_size,
            "revenue": self.revenue,
            "customers": self.customers,
            "growth_rate": self.growth_rate,
            "competition": self.competition,
            "use_of_funds": self.use_of_funds,
            "business_stage": self.business_stage,
            "team_size": self.team_size
        }


@dataclass
class FundingAssessment:
    """Assessment result for funding application."""
    overall_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    approval_likelihood: str = "unknown"  # low, medium, high
    processing_time: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "approval_likelihood": self.approval_likelihood,
            "processing_time": self.processing_time,
            "confidence": self.confidence
        }


class FundingFormProcessor:
    """Specialized processor for funding form workflows."""
    
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        validation_rules_path: str = "config/funding_validation_rules.json"
    ):
        """Initialize funding form processor.
        
        Args:
            orchestrator: Agent orchestrator instance
            validation_rules_path: Path to validation rules configuration
        """
        self.orchestrator = orchestrator
        self.validation_rules_path = Path(validation_rules_path)
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # Load validation rules
        self.validation_rules = self._load_validation_rules()
        
        # Performance tracking
        self.processing_count = 0
        self.total_processing_time = 0.0
        self.accuracy_scores = []
        
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration file."""
        try:
            if not self.validation_rules_path.exists():
                self.logger.warning(f"Validation rules file not found: {self.validation_rules_path}")
                return self._get_default_validation_rules()
            
            with open(self.validation_rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            
            self.logger.info(f"Loaded validation rules from {self.validation_rules_path}")
            return rules
            
        except Exception as e:
            self.logger.error(f"Failed to load validation rules: {e}")
            return self._get_default_validation_rules()
    
    def _get_default_validation_rules(self) -> Dict[str, Any]:
        """Get default validation rules if config file is not available."""
        return {
            "funding_form_validation": {
                "required_fields": [
                    {"field": "company_name", "type": "string", "min_length": 2},
                    {"field": "funding_amount", "type": "number", "min_value": 1000},
                    {"field": "business_plan", "type": "string", "min_length": 100}
                ],
                "scoring_criteria": [
                    {"category": "team", "weight": 0.25},
                    {"category": "market", "weight": 0.25},
                    {"category": "product", "weight": 0.25},
                    {"category": "financials", "weight": 0.25}
                ]
            },
            "processing_targets": {
                "max_processing_time_seconds": 30,
                "target_accuracy_percentage": 95
            }
        }
    
    async def process_funding_form(
        self,
        form_data: Dict[str, Any],
        user_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[FundingAssessment, WorkflowResult]:
        """Process funding form using multi-agent workflow.
        
        Args:
            form_data: Raw form data dictionary
            user_id: User identifier
            user_context: Optional user context
            
        Returns:
            Tuple of (FundingAssessment, WorkflowResult)
        """
        start_time = time.time()
        
        try:
            # Parse and validate form data
            funding_data = self._parse_form_data(form_data)
            
            # Prepare task data for agents
            task_data = {
                "form_data": funding_data.to_dict(),
                "validation_rules": self.validation_rules["funding_form_validation"],
                "processing_targets": self.validation_rules["processing_targets"],
                "workflow_type": "funding_form"
            }
            
            # Execute multi-agent workflow
            workflow_result = await self.orchestrator.execute_workflow(
                workflow_type="funding_form",
                user_id=user_id,
                task_data=task_data,
                user_context=user_context
            )
            
            # Generate funding assessment from workflow results
            assessment = self._generate_assessment(
                funding_data,
                workflow_result,
                time.time() - start_time
            )
            
            # Update performance metrics
            self._update_performance_metrics(assessment, workflow_result)
            
            # Log processing result
            self._log_processing_result(user_id, funding_data, assessment, workflow_result)
            
            return assessment, workflow_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Funding form processing failed: {str(e)}"
            
            self.logger.error(error_msg)
            
            # Create error assessment
            error_assessment = FundingAssessment(
                overall_score=0.0,
                approval_likelihood="unknown",
                processing_time=processing_time,
                confidence=0.0,
                recommendations=[f"Processing failed: {error_msg}"]
            )
            
            # Create error workflow result
            error_workflow = WorkflowResult(
                success=False,
                result_data={"error": error_msg},
                execution_time=processing_time,
                confidence_score=0.0
            )
            
            return error_assessment, error_workflow
    
    def _parse_form_data(self, form_data: Dict[str, Any]) -> FundingFormData:
        """Parse raw form data into structured format."""
        funding_data = FundingFormData()
        
        # Required fields
        funding_data.company_name = str(form_data.get("company_name", "")).strip()
        funding_data.funding_amount = float(form_data.get("funding_amount", 0))
        funding_data.business_plan = str(form_data.get("business_plan", "")).strip()
        funding_data.team_experience = str(form_data.get("team_experience", "")).strip()
        funding_data.market_size = str(form_data.get("market_size", "")).strip()
        
        # Optional fields
        if "revenue" in form_data and form_data["revenue"] is not None:
            funding_data.revenue = float(form_data["revenue"])
        
        if "customers" in form_data and form_data["customers"] is not None:
            funding_data.customers = int(form_data["customers"])
        
        if "growth_rate" in form_data and form_data["growth_rate"] is not None:
            funding_data.growth_rate = float(form_data["growth_rate"])
        
        funding_data.competition = form_data.get("competition", "")
        funding_data.use_of_funds = form_data.get("use_of_funds", "")
        funding_data.business_stage = form_data.get("business_stage", "unknown")
        funding_data.team_size = int(form_data.get("team_size", 1))
        
        return funding_data
    
    def _generate_assessment(
        self,
        funding_data: FundingFormData,
        workflow_result: WorkflowResult,
        processing_time: float
    ) -> FundingAssessment:
        """Generate funding assessment from workflow results."""
        
        assessment = FundingAssessment(
            processing_time=processing_time,
            confidence=workflow_result.confidence_score
        )
        
        # Extract results from agent outputs
        agent_outputs = workflow_result.result_data
        
        # Get validator results
        validator_output = agent_outputs.get("validator_output", {})
        if validator_output.get("is_valid", False):
            base_score = 0.6  # Base score for valid applications
        else:
            base_score = 0.2  # Low score for invalid applications
            assessment.recommendations.extend(validator_output.get("errors", []))
        
        # Get planner results for scoring
        planner_output = agent_outputs.get("planner_output", {})
        action_items = planner_output.get("action_items", [])
        
        # Calculate category scores based on scoring criteria
        scoring_criteria = self.validation_rules["funding_form_validation"]["scoring_criteria"]
        
        for criterion in scoring_criteria:
            category = criterion["category"]
            weight = criterion["weight"]
            
            # Simple scoring based on data completeness and quality
            category_score = self._calculate_category_score(category, funding_data, action_items)
            assessment.category_scores[category] = category_score
            assessment.overall_score += category_score * weight
        
        # Determine approval likelihood
        if assessment.overall_score >= 0.8:
            assessment.approval_likelihood = "high"
        elif assessment.overall_score >= 0.6:
            assessment.approval_likelihood = "medium"
        else:
            assessment.approval_likelihood = "low"
        
        # Add risk factors
        assessment.risk_factors = self._identify_risk_factors(funding_data)
        
        # Generate recommendations
        if not assessment.recommendations:
            assessment.recommendations = self._generate_recommendations(
                funding_data, assessment, planner_output
            )
        
        return assessment
    
    def _calculate_category_score(
        self,
        category: str,
        funding_data: FundingFormData,
        action_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate score for a specific category."""
        
        if category == "team":
            score = 0.5  # Base score
            if len(funding_data.team_experience) > 100:
                score += 0.2
            if funding_data.team_size >= 2:
                score += 0.2
            if funding_data.team_size >= 5:
                score += 0.1
            return min(score, 1.0)
        
        elif category == "market":
            score = 0.4  # Base score
            if len(funding_data.market_size) > 50:
                score += 0.3
            if funding_data.competition:
                score += 0.2
            if "billion" in funding_data.market_size.lower():
                score += 0.1
            return min(score, 1.0)
        
        elif category == "product":
            score = 0.3  # Base score
            if len(funding_data.business_plan) > 200:
                score += 0.3
            if len(funding_data.business_plan) > 500:
                score += 0.2
            if funding_data.customers and funding_data.customers > 0:
                score += 0.2
            return min(score, 1.0)
        
        elif category == "financials":
            score = 0.4  # Base score
            if funding_data.revenue and funding_data.revenue > 0:
                score += 0.3
            if funding_data.growth_rate and funding_data.growth_rate > 0:
                score += 0.2
            if funding_data.use_of_funds:
                score += 0.1
            return min(score, 1.0)
        
        return 0.5  # Default score
    
    def _identify_risk_factors(self, funding_data: FundingFormData) -> List[Dict[str, Any]]:
        """Identify risk factors in the funding application."""
        risk_factors = []
        
        # High funding with no revenue
        if funding_data.funding_amount > 1000000 and (not funding_data.revenue or funding_data.revenue == 0):
            risk_factors.append({
                "factor": "high_funding_no_revenue",
                "risk_level": "high",
                "message": "High funding request with no current revenue",
                "impact": "May indicate unrealistic expectations or lack of market validation"
            })
        
        # Small team with large funding
        if funding_data.team_size < 3 and funding_data.funding_amount > 2000000:
            risk_factors.append({
                "factor": "small_team_large_funding",
                "risk_level": "medium",
                "message": "Large funding request for small team",
                "impact": "Team may lack capacity to execute on funding plans"
            })
        
        # Vague business plan
        if len(funding_data.business_plan) < 200:
            risk_factors.append({
                "factor": "vague_business_plan",
                "risk_level": "medium",
                "message": "Business plan lacks sufficient detail",
                "impact": "Insufficient information to assess viability"
            })
        
        return risk_factors
    
    def _generate_recommendations(
        self,
        funding_data: FundingFormData,
        assessment: FundingAssessment,
        planner_output: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on assessment."""
        recommendations = []
        
        # Based on overall score
        if assessment.overall_score < 0.5:
            recommendations.append("Consider strengthening your application before resubmitting")
        
        # Based on category scores
        for category, score in assessment.category_scores.items():
            if score < 0.6:
                if category == "team":
                    recommendations.append("Provide more detailed team background and experience")
                elif category == "market":
                    recommendations.append("Include more comprehensive market analysis and sizing")
                elif category == "product":
                    recommendations.append("Expand business plan with more product details and customer validation")
                elif category == "financials":
                    recommendations.append("Provide clearer financial projections and use of funds")
        
        # Based on risk factors
        for risk in assessment.risk_factors:
            if risk["risk_level"] == "high":
                recommendations.append(f"Address high-risk factor: {risk['message']}")
        
        # From planner output
        action_items = planner_output.get("action_items", [])
        for item in action_items[:3]:  # Top 3 recommendations
            if item.get("priority") == "high":
                recommendations.append(item.get("title", ""))
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _update_performance_metrics(
        self,
        assessment: FundingAssessment,
        workflow_result: WorkflowResult
    ) -> None:
        """Update performance tracking metrics."""
        self.processing_count += 1
        self.total_processing_time += assessment.processing_time
        
        # Track accuracy (simplified - in production would compare against known outcomes)
        accuracy_score = assessment.confidence * 100
        self.accuracy_scores.append(accuracy_score)
        
        # Keep only last 100 scores for rolling average
        if len(self.accuracy_scores) > 100:
            self.accuracy_scores = self.accuracy_scores[-100:]
    
    def _log_processing_result(
        self,
        user_id: str,
        funding_data: FundingFormData,
        assessment: FundingAssessment,
        workflow_result: WorkflowResult
    ) -> None:
        """Log processing result to file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "company_name": funding_data.company_name,
                "funding_amount": funding_data.funding_amount,
                "overall_score": assessment.overall_score,
                "approval_likelihood": assessment.approval_likelihood,
                "processing_time": assessment.processing_time,
                "confidence": assessment.confidence,
                "workflow_success": workflow_result.success,
                "agents_executed": len(workflow_result.agent_logs),
                "risk_factors_count": len(assessment.risk_factors)
            }
            
            # Write to funding processing log
            import os
            os.makedirs("data/funding_logs", exist_ok=True)
            
            log_file = f"data/funding_logs/funding_processing_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log processing result: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for funding processing."""
        avg_processing_time = (
            self.total_processing_time / max(1, self.processing_count)
        )
        avg_accuracy = (
            sum(self.accuracy_scores) / max(1, len(self.accuracy_scores))
        )
        
        # Check if meeting targets
        targets = self.validation_rules["processing_targets"]
        meets_time_target = avg_processing_time <= targets["max_processing_time_seconds"]
        meets_accuracy_target = avg_accuracy >= targets["target_accuracy_percentage"]
        
        return {
            "total_processed": self.processing_count,
            "average_processing_time": avg_processing_time,
            "average_accuracy": avg_accuracy,
            "meets_time_target": meets_time_target,
            "meets_accuracy_target": meets_accuracy_target,
            "target_processing_time": targets["max_processing_time_seconds"],
            "target_accuracy": targets["target_accuracy_percentage"]
        }
    
    async def validate_form_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quick validation of form data without full processing."""
        try:
            funding_data = self._parse_form_data(form_data)
            validation_rules = self.validation_rules["funding_form_validation"]
            
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "completeness_score": 0.0
            }
            
            # Check required fields
            required_fields = validation_rules["required_fields"]
            completed_required = 0
            
            for field_rule in required_fields:
                field_name = field_rule["field"]
                field_value = getattr(funding_data, field_name, None)
                
                if not field_value or (isinstance(field_value, str) and not field_value.strip()):
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Required field '{field_name}' is missing")
                else:
                    # Check field-specific validation
                    if field_rule["type"] == "string":
                        if len(str(field_value)) < field_rule.get("min_length", 0):
                            validation_result["errors"].append(
                                f"Field '{field_name}' is too short (minimum {field_rule['min_length']} characters)"
                            )
                        elif len(str(field_value)) > field_rule.get("max_length", 10000):
                            validation_result["warnings"].append(
                                f"Field '{field_name}' is very long (over {field_rule['max_length']} characters)"
                            )
                    elif field_rule["type"] == "number":
                        if float(field_value) < field_rule.get("min_value", 0):
                            validation_result["errors"].append(
                                f"Field '{field_name}' is below minimum value ({field_rule['min_value']})"
                            )
                        elif float(field_value) > field_rule.get("max_value", float('inf')):
                            validation_result["errors"].append(
                                f"Field '{field_name}' exceeds maximum value ({field_rule['max_value']})"
                            )
                    
                    completed_required += 1
            
            # Calculate completeness score
            total_fields = len(required_fields) + len(validation_rules.get("optional_fields", []))
            completed_optional = sum(
                1 for field in validation_rules.get("optional_fields", [])
                if getattr(funding_data, field["field"], None)
            )
            
            validation_result["completeness_score"] = (
                (completed_required + completed_optional) / total_fields
            )
            
            return validation_result
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "completeness_score": 0.0
            }