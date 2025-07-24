"""
Experiment Analysis and Teardown Manager for FounderForge AI Cofounder

This module provides experiment management capabilities including:
- A/B testing framework with local script variants
- Monthly experiment analysis and reporting
- Experiment result documentation and sharing
- Experiment teardown and cleanup tools
"""

import os
import json
import datetime
import shutil
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import logging
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    id: str
    name: str
    description: str
    variant_a_script: str
    variant_b_script: str
    success_metrics: List[str]
    start_date: str
    end_date: Optional[str] = None
    status: str = "active"  # active, completed, failed, cancelled
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExperimentResult:
    """Results from an experiment run"""
    experiment_id: str
    variant: str  # 'A' or 'B'
    run_timestamp: str
    metrics: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    output_log: Optional[str] = None

@dataclass
class ExperimentAnalysis:
    """Analysis results comparing experiment variants"""
    experiment_id: str
    analysis_date: str
    variant_a_results: List[ExperimentResult]
    variant_b_results: List[ExperimentResult]
    statistical_summary: Dict[str, Any]
    recommendation: str
    confidence_level: float

class ExperimentManager:
    """Manages A/B testing experiments and analysis"""
    
    def __init__(self, base_path: str = "data/experiments"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "experiments.db"
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Create necessary directories for experiment management"""
        directories = [
            self.base_path,
            self.base_path / "configs",
            self.base_path / "scripts" / "variant_a",
            self.base_path / "scripts" / "variant_b",
            self.base_path / "results",
            self.base_path / "analysis",
            self.base_path / "reports",
            self.base_path / "archive"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create README files
        self._create_directory_readmes()
    
    def _create_directory_readmes(self):
        """Create README files explaining each experiment directory"""
        readmes = {
            "configs/README.md": """# Experiment Configurations

This directory contains JSON configuration files for experiments.

## Structure
- Each experiment has a unique configuration file
- Configurations define variants, metrics, and parameters
- Version controlled for experiment reproducibility

## Usage
- Create new experiments using the ExperimentManager API
- Modify configurations to adjust experiment parameters
- Archive completed experiment configurations
""",
            "scripts/README.md": """# Experiment Scripts

This directory contains the actual scripts for experiment variants.

## Structure
- variant_a/: Scripts for variant A of experiments
- variant_b/: Scripts for variant B of experiments
- Each variant has its own isolated environment

## Usage
- Scripts are executed automatically during experiments
- Maintain identical interfaces between variants
- Include proper error handling and logging
""",
            "results/README.md": """# Experiment Results

This directory contains raw results from experiment runs.

## Structure
- Results stored as JSON files with timestamps
- Separate files for each experiment run
- Includes metrics, timing, and execution details

## Usage
- Results are automatically generated during runs
- Used for statistical analysis and reporting
- Archived after analysis completion
""",
            "analysis/README.md": """# Experiment Analysis

This directory contains statistical analysis of experiments.

## Structure
- Analysis reports with statistical comparisons
- Confidence intervals and significance tests
- Recommendations based on results

## Usage
- Generated monthly or on-demand
- Used for decision making on experiment outcomes
- Shared with stakeholders for review
""",
            "reports/README.md": """# Experiment Reports

This directory contains formatted reports for sharing.

## Structure
- Monthly experiment summaries
- Individual experiment reports
- Executive summaries and recommendations

## Usage
- Generated automatically from analysis
- Formatted for easy sharing and review
- Include visualizations and key insights
"""
        }
        
        for file_path, content in readmes.items():
            full_path = self.base_path / file_path
            if not full_path.exists():
                full_path.write_text(content)
    
    def _init_database(self):
        """Initialize SQLite database for experiment tracking"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    variant_a_script TEXT,
                    variant_b_script TEXT,
                    success_metrics TEXT,  -- JSON array
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT  -- JSON object
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    variant TEXT,
                    run_timestamp TEXT,
                    metrics TEXT,  -- JSON object
                    execution_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    output_log TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    analysis_date TEXT,
                    statistical_summary TEXT,  -- JSON object
                    recommendation TEXT,
                    confidence_level REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_results ON experiment_results(experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_analysis ON experiment_analysis(experiment_id)")
            conn.commit()
        finally:
            conn.close()
    
    def create_experiment(self, name: str, description: str, 
                         variant_a_script: str, variant_b_script: str,
                         success_metrics: List[str]) -> str:
        """Create a new A/B testing experiment"""
        experiment_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        config = ExperimentConfig(
            id=experiment_id,
            name=name,
            description=description,
            variant_a_script=variant_a_script,
            variant_b_script=variant_b_script,
            success_metrics=success_metrics,
            start_date=datetime.datetime.now().isoformat()
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO experiments 
                (id, name, description, variant_a_script, variant_b_script, 
                 success_metrics, start_date, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.id, config.name, config.description,
                config.variant_a_script, config.variant_b_script,
                json.dumps(config.success_metrics), config.start_date,
                config.status, json.dumps(config.metadata)
            ))
            conn.commit()
        finally:
            conn.close()
        
        # Save configuration file
        config_file = self.base_path / "configs" / f"{experiment_id}.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Create script directories
        variant_a_dir = self.base_path / "scripts" / "variant_a" / experiment_id
        variant_b_dir = self.base_path / "scripts" / "variant_b" / experiment_id
        variant_a_dir.mkdir(exist_ok=True)
        variant_b_dir.mkdir(exist_ok=True)
        
        # Create script files
        (variant_a_dir / "run.py").write_text(variant_a_script)
        (variant_b_dir / "run.py").write_text(variant_b_script)
        
        logger.info(f"Created experiment: {experiment_id} - {name}")
        return experiment_id
    
    def run_experiment(self, experiment_id: str, variant: str = "both", 
                      iterations: int = 1) -> List[ExperimentResult]:
        """Run an experiment variant(s)"""
        config = self.get_experiment_config(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = []
        variants_to_run = ["A", "B"] if variant == "both" else [variant.upper()]
        
        for var in variants_to_run:
            for i in range(iterations):
                result = self._run_single_variant(experiment_id, var, config)
                results.append(result)
                self._save_result(result)
        
        return results
    
    def _run_single_variant(self, experiment_id: str, variant: str, 
                           config: ExperimentConfig) -> ExperimentResult:
        """Run a single variant of an experiment"""
        variant_dir = self.base_path / "scripts" / f"variant_{variant.lower()}" / experiment_id
        script_path = variant_dir / "run.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        start_time = datetime.datetime.now()
        run_timestamp = start_time.isoformat()
        
        try:
            # Run the script and capture output
            result = subprocess.run(
                ["python", str(script_path.resolve())],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Parse metrics from output (expecting JSON on last line)
            metrics = {}
            output_lines = result.stdout.strip().split('\n')
            if output_lines and output_lines[-1].startswith('{'):
                try:
                    metrics = json.loads(output_lines[-1])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse metrics from output: {output_lines[-1]}")
            
            return ExperimentResult(
                experiment_id=experiment_id,
                variant=variant,
                run_timestamp=run_timestamp,
                metrics=metrics,
                execution_time=execution_time,
                success=result.returncode == 0,
                error_message=result.stderr if result.stderr else None,
                output_log=result.stdout
            )
            
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                experiment_id=experiment_id,
                variant=variant,
                run_timestamp=run_timestamp,
                metrics={},
                execution_time=300.0,
                success=False,
                error_message="Script execution timed out",
                output_log=""
            )
        except Exception as e:
            return ExperimentResult(
                experiment_id=experiment_id,
                variant=variant,
                run_timestamp=run_timestamp,
                metrics={},
                execution_time=0.0,
                success=False,
                error_message=str(e),
                output_log=""
            )
    
    def _save_result(self, result: ExperimentResult):
        """Save experiment result to database and file"""
        # Save to database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO experiment_results 
                (experiment_id, variant, run_timestamp, metrics, execution_time, 
                 success, error_message, output_log)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_id, result.variant, result.run_timestamp,
                json.dumps(result.metrics), result.execution_time,
                result.success, result.error_message, result.output_log
            ))
            conn.commit()
        finally:
            conn.close()
        
        # Save to file
        result_file = self.base_path / "results" / f"{result.experiment_id}_{result.variant}_{result.run_timestamp.replace(':', '-')}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT * FROM experiments WHERE id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return ExperimentConfig(
                    id=row[0], name=row[1], description=row[2],
                    variant_a_script=row[3], variant_b_script=row[4],
                    success_metrics=json.loads(row[5]), start_date=row[6],
                    end_date=row[7], status=row[8], metadata=json.loads(row[9])
                )
            return None
        finally:
            conn.close()
    
    def get_experiment_results(self, experiment_id: str) -> Tuple[List[ExperimentResult], List[ExperimentResult]]:
        """Get all results for an experiment, separated by variant"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT * FROM experiment_results 
                WHERE experiment_id = ?
                ORDER BY run_timestamp
            """, (experiment_id,))
            
            variant_a_results = []
            variant_b_results = []
            
            for row in cursor.fetchall():
                result = ExperimentResult(
                    experiment_id=row[1], variant=row[2], run_timestamp=row[3],
                    metrics=json.loads(row[4]), execution_time=row[5],
                    success=bool(row[6]), error_message=row[7], output_log=row[8]
                )
                
                if result.variant == "A":
                    variant_a_results.append(result)
                else:
                    variant_b_results.append(result)
            
            return variant_a_results, variant_b_results
        finally:
            conn.close()
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentAnalysis:
        """Perform statistical analysis of experiment results"""
        config = self.get_experiment_config(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        variant_a_results, variant_b_results = self.get_experiment_results(experiment_id)
        
        if not variant_a_results or not variant_b_results:
            raise ValueError(f"Insufficient results for analysis. A: {len(variant_a_results)}, B: {len(variant_b_results)}")
        
        # Calculate statistical summary
        statistical_summary = self._calculate_statistics(variant_a_results, variant_b_results, config.success_metrics)
        
        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(statistical_summary)
        
        analysis = ExperimentAnalysis(
            experiment_id=experiment_id,
            analysis_date=datetime.datetime.now().isoformat(),
            variant_a_results=variant_a_results,
            variant_b_results=variant_b_results,
            statistical_summary=statistical_summary,
            recommendation=recommendation,
            confidence_level=confidence
        )
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis
    
    def _calculate_statistics(self, variant_a_results: List[ExperimentResult], 
                            variant_b_results: List[ExperimentResult],
                            success_metrics: List[str]) -> Dict[str, Any]:
        """Calculate statistical summary of experiment results"""
        stats = {
            "variant_a": {
                "total_runs": len(variant_a_results),
                "successful_runs": sum(1 for r in variant_a_results if r.success),
                "avg_execution_time": sum(r.execution_time for r in variant_a_results) / len(variant_a_results),
                "metrics": {}
            },
            "variant_b": {
                "total_runs": len(variant_b_results),
                "successful_runs": sum(1 for r in variant_b_results if r.success),
                "avg_execution_time": sum(r.execution_time for r in variant_b_results) / len(variant_b_results),
                "metrics": {}
            },
            "comparison": {}
        }
        
        # Calculate metric statistics
        for metric in success_metrics:
            a_values = [r.metrics.get(metric, 0) for r in variant_a_results if r.success]
            b_values = [r.metrics.get(metric, 0) for r in variant_b_results if r.success]
            
            if a_values and b_values:
                a_avg = sum(a_values) / len(a_values)
                b_avg = sum(b_values) / len(b_values)
                
                stats["variant_a"]["metrics"][metric] = {
                    "average": a_avg,
                    "min": min(a_values),
                    "max": max(a_values),
                    "count": len(a_values)
                }
                
                stats["variant_b"]["metrics"][metric] = {
                    "average": b_avg,
                    "min": min(b_values),
                    "max": max(b_values),
                    "count": len(b_values)
                }
                
                # Simple comparison (in real implementation, would use proper statistical tests)
                improvement = ((b_avg - a_avg) / a_avg * 100) if a_avg != 0 else 0
                stats["comparison"][metric] = {
                    "improvement_percent": improvement,
                    "better_variant": "B" if b_avg > a_avg else "A",
                    "significant": abs(improvement) > 5  # Simple threshold
                }
        
        return stats
    
    def _generate_recommendation(self, stats: Dict[str, Any]) -> Tuple[str, float]:
        """Generate recommendation based on statistical analysis"""
        comparisons = stats.get("comparison", {})
        
        if not comparisons:
            return "Insufficient data for recommendation", 0.0
        
        # Count wins for each variant
        a_wins = sum(1 for comp in comparisons.values() if comp["better_variant"] == "A" and comp["significant"])
        b_wins = sum(1 for comp in comparisons.values() if comp["better_variant"] == "B" and comp["significant"])
        
        total_significant = a_wins + b_wins
        
        if total_significant == 0:
            return "No significant differences found between variants", 0.5
        
        if b_wins > a_wins:
            confidence = b_wins / (a_wins + b_wins)
            return f"Recommend Variant B - shows improvement in {b_wins} out of {len(comparisons)} metrics", confidence
        elif a_wins > b_wins:
            confidence = a_wins / (a_wins + b_wins)
            return f"Recommend Variant A - shows improvement in {a_wins} out of {len(comparisons)} metrics", confidence
        else:
            return "Results are mixed - consider additional testing", 0.5
    
    def _save_analysis(self, analysis: ExperimentAnalysis):
        """Save analysis to database and file"""
        # Save to database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO experiment_analysis 
                (experiment_id, analysis_date, statistical_summary, recommendation, confidence_level)
                VALUES (?, ?, ?, ?, ?)
            """, (
                analysis.experiment_id, analysis.analysis_date,
                json.dumps(analysis.statistical_summary), analysis.recommendation,
                analysis.confidence_level
            ))
            conn.commit()
        finally:
            conn.close()
        
        # Save to file
        analysis_file = self.base_path / "analysis" / f"{analysis.experiment_id}_analysis_{analysis.analysis_date.replace(':', '-')}.json"
        with open(analysis_file, 'w') as f:
            # Create a serializable version
            analysis_dict = asdict(analysis)
            json.dump(analysis_dict, f, indent=2)
    
    def generate_monthly_report(self, year: int = None, month: int = None) -> str:
        """Generate monthly experiment report"""
        if year is None or month is None:
            now = datetime.datetime.now()
            year = now.year
            month = now.month
        
        # Get all experiments from the month
        start_date = datetime.datetime(year, month, 1).isoformat()
        if month == 12:
            end_date = datetime.datetime(year + 1, 1, 1).isoformat()
        else:
            end_date = datetime.datetime(year, month + 1, 1).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT * FROM experiments 
                WHERE start_date >= ? AND start_date < ?
                ORDER BY start_date
            """, (start_date, end_date))
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "start_date": row[6],
                    "status": row[8]
                })
        finally:
            conn.close()
        
        # Generate report
        report_content = f"""# Monthly Experiment Report - {year}-{month:02d}

## Summary
- Total experiments: {len(experiments)}
- Active experiments: {sum(1 for e in experiments if e['status'] == 'active')}
- Completed experiments: {sum(1 for e in experiments if e['status'] == 'completed')}

## Experiments

"""
        
        for exp in experiments:
            report_content += f"""### {exp['name']} ({exp['id']})
- **Description**: {exp['description']}
- **Start Date**: {exp['start_date']}
- **Status**: {exp['status']}

"""
        
        # Save report
        report_file = self.base_path / "reports" / f"monthly_report_{year}_{month:02d}.md"
        report_file.write_text(report_content)
        
        logger.info(f"Generated monthly report: {report_file}")
        return str(report_file)
    
    def teardown_experiment(self, experiment_id: str, archive: bool = True) -> bool:
        """Teardown and optionally archive an experiment"""
        try:
            config = self.get_experiment_config(experiment_id)
            if not config:
                logger.warning(f"Experiment {experiment_id} not found")
                return False
            
            if archive:
                # Create archive directory
                archive_dir = self.base_path / "archive" / experiment_id
                archive_dir.mkdir(exist_ok=True)
                
                # Archive configuration
                config_file = self.base_path / "configs" / f"{experiment_id}.json"
                if config_file.exists():
                    shutil.copy2(config_file, archive_dir / "config.json")
                
                # Archive scripts
                for variant in ["variant_a", "variant_b"]:
                    script_dir = self.base_path / "scripts" / variant / experiment_id
                    if script_dir.exists():
                        shutil.copytree(script_dir, archive_dir / variant, dirs_exist_ok=True)
                
                # Archive results
                results_pattern = f"{experiment_id}_*"
                for result_file in (self.base_path / "results").glob(results_pattern):
                    shutil.copy2(result_file, archive_dir / result_file.name)
                
                # Archive analysis
                for analysis_file in (self.base_path / "analysis").glob(f"{experiment_id}_*"):
                    shutil.copy2(analysis_file, archive_dir / analysis_file.name)
            
            # Update experiment status
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    UPDATE experiments 
                    SET status = 'archived', end_date = ?
                    WHERE id = ?
                """, (datetime.datetime.now().isoformat(), experiment_id))
                conn.commit()
            finally:
                conn.close()
            
            # Clean up active files (if archiving)
            if archive:
                # Remove configuration
                config_file = self.base_path / "configs" / f"{experiment_id}.json"
                if config_file.exists():
                    config_file.unlink()
                
                # Remove scripts
                for variant in ["variant_a", "variant_b"]:
                    script_dir = self.base_path / "scripts" / variant / experiment_id
                    if script_dir.exists():
                        shutil.rmtree(script_dir)
                
                # Remove results (keep in database for analysis)
                for result_file in (self.base_path / "results").glob(f"{experiment_id}_*"):
                    result_file.unlink()
            
            logger.info(f"Experiment {experiment_id} teardown completed (archived: {archive})")
            return True
            
        except Exception as e:
            logger.error(f"Error during experiment teardown: {e}")
            return False
    
    def list_experiments(self, status: str = None) -> List[Dict[str, Any]]:
        """List all experiments with optional status filter"""
        conn = sqlite3.connect(self.db_path)
        try:
            if status:
                cursor = conn.execute("""
                    SELECT id, name, description, start_date, end_date, status 
                    FROM experiments 
                    WHERE status = ?
                    ORDER BY start_date DESC
                """, (status,))
            else:
                cursor = conn.execute("""
                    SELECT id, name, description, start_date, end_date, status 
                    FROM experiments 
                    ORDER BY start_date DESC
                """)
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "start_date": row[3],
                    "end_date": row[4],
                    "status": row[5]
                })
            
            return experiments
        finally:
            conn.close()
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get overall experiment statistics"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Count experiments by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM experiments GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Count total results
            cursor = conn.execute("SELECT COUNT(*) FROM experiment_results")
            total_results = cursor.fetchone()[0]
            
            # Count successful results
            cursor = conn.execute("SELECT COUNT(*) FROM experiment_results WHERE success = 1")
            successful_results = cursor.fetchone()[0]
            
            return {
                "total_experiments": sum(status_counts.values()),
                "by_status": status_counts,
                "total_results": total_results,
                "successful_results": successful_results,
                "success_rate": successful_results / total_results if total_results > 0 else 0
            }
        finally:
            conn.close()