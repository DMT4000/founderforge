#!/usr/bin/env python3
"""
Monthly Experiment Analysis Script for FounderForge AI Cofounder

This script performs monthly analysis and teardown of experiments:
- Analyzes all experiments from the specified month
- Generates comprehensive reports with statistical analysis
- Archives completed experiments
- Creates documentation for sharing

Usage:
    python scripts/monthly_experiment_analysis.py [--year YEAR] [--month MONTH] [--archive]
"""

import argparse
import sys
import os
import datetime
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment_manager import ExperimentManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/monthly_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MonthlyAnalyzer:
    """Handles monthly experiment analysis and teardown"""
    
    def __init__(self):
        self.experiment_manager = ExperimentManager()
        self.analysis_results = []
    
    def analyze_month(self, year: int, month: int, archive_completed: bool = False):
        """Analyze all experiments from a specific month"""
        logger.info(f"Starting monthly analysis for {year}-{month:02d}")
        
        # Get experiments from the month
        experiments = self._get_monthly_experiments(year, month)
        
        if not experiments:
            logger.info(f"No experiments found for {year}-{month:02d}")
            return
        
        logger.info(f"Found {len(experiments)} experiments to analyze")
        
        # Analyze each experiment
        for exp in experiments:
            try:
                logger.info(f"Analyzing experiment: {exp['name']} ({exp['id']})")
                analysis = self.experiment_manager.analyze_experiment(exp['id'])
                self.analysis_results.append(analysis)
                
                # Archive if requested and experiment is completed
                if archive_completed and exp['status'] == 'completed':
                    logger.info(f"Archiving completed experiment: {exp['id']}")
                    self.experiment_manager.teardown_experiment(exp['id'], archive=True)
                    
            except Exception as e:
                logger.error(f"Error analyzing experiment {exp['id']}: {e}")
                continue
        
        # Generate comprehensive report
        self._generate_comprehensive_report(year, month)
        
        # Generate sharing documentation
        self._generate_sharing_docs(year, month)
        
        logger.info(f"Monthly analysis completed for {year}-{month:02d}")
    
    def _get_monthly_experiments(self, year: int, month: int):
        """Get all experiments from a specific month"""
        start_date = datetime.datetime(year, month, 1)
        if month == 12:
            end_date = datetime.datetime(year + 1, 1, 1)
        else:
            end_date = datetime.datetime(year, month + 1, 1)
        
        all_experiments = self.experiment_manager.list_experiments()
        monthly_experiments = []
        
        for exp in all_experiments:
            exp_date = datetime.datetime.fromisoformat(exp['start_date'])
            if start_date <= exp_date < end_date:
                monthly_experiments.append(exp)
        
        return monthly_experiments
    
    def _generate_comprehensive_report(self, year: int, month: int):
        """Generate comprehensive monthly analysis report"""
        report_path = Path(f"data/experiments/reports/comprehensive_analysis_{year}_{month:02d}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        total_experiments = len(self.analysis_results)
        successful_analyses = sum(1 for a in self.analysis_results if a.confidence_level > 0.5)
        
        # Group by recommendation
        recommend_a = sum(1 for a in self.analysis_results if "Variant A" in a.recommendation)
        recommend_b = sum(1 for a in self.analysis_results if "Variant B" in a.recommendation)
        mixed_results = total_experiments - recommend_a - recommend_b
        
        report_content = f"""# Comprehensive Experiment Analysis Report - {year}-{month:02d}

## Executive Summary

- **Total Experiments Analyzed**: {total_experiments}
- **Successful Analyses**: {successful_analyses} ({successful_analyses/total_experiments*100:.1f}% if total_experiments > 0 else 0)
- **Variant A Recommendations**: {recommend_a}
- **Variant B Recommendations**: {recommend_b}
- **Mixed/Inconclusive Results**: {mixed_results}

## Key Insights

"""
        
        # Add insights for each experiment
        for analysis in self.analysis_results:
            config = self.experiment_manager.get_experiment_config(analysis.experiment_id)
            if config:
                report_content += f"""### {config.name} ({analysis.experiment_id})

**Description**: {config.description}

**Recommendation**: {analysis.recommendation}
**Confidence Level**: {analysis.confidence_level:.2f}

**Key Metrics**:
"""
                
                # Add metric comparisons
                for metric, comparison in analysis.statistical_summary.get("comparison", {}).items():
                    improvement = comparison.get("improvement_percent", 0)
                    better_variant = comparison.get("better_variant", "Unknown")
                    significant = comparison.get("significant", False)
                    
                    report_content += f"- **{metric}**: {improvement:+.1f}% (Variant {better_variant} better) {'✓ Significant' if significant else '○ Not significant'}\n"
                
                report_content += "\n"
        
        # Add recommendations section
        report_content += """## Overall Recommendations

"""
        
        if recommend_b > recommend_a:
            report_content += "- **Primary Finding**: Variant B approaches show more promise across experiments\n"
        elif recommend_a > recommend_b:
            report_content += "- **Primary Finding**: Variant A approaches show more promise across experiments\n"
        else:
            report_content += "- **Primary Finding**: Results are mixed, suggesting need for more targeted experiments\n"
        
        if mixed_results > total_experiments * 0.3:
            report_content += "- **Concern**: High number of inconclusive results suggests need for better experiment design\n"
        
        report_content += f"""
## Next Steps

1. Review experiments with low confidence scores
2. Design follow-up experiments for inconclusive results
3. Implement winning variants from high-confidence recommendations
4. Archive completed experiments to maintain clean workspace

---
*Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Comprehensive report generated: {report_path}")
    
    def _generate_sharing_docs(self, year: int, month: int):
        """Generate documentation for sharing experiment results"""
        sharing_dir = Path(f"data/experiments/sharing/{year}_{month:02d}")
        sharing_dir.mkdir(parents=True, exist_ok=True)
        
        # Create executive summary for stakeholders
        exec_summary_path = sharing_dir / "executive_summary.md"
        
        total_experiments = len(self.analysis_results)
        high_confidence = sum(1 for a in self.analysis_results if a.confidence_level > 0.7)
        
        exec_content = f"""# Monthly Experiment Results - {year}-{month:02d}

## Summary for Stakeholders

This month we conducted {total_experiments} experiments to improve system performance and user experience.

### Key Results
- **{high_confidence}** experiments showed clear winners with high confidence
- **{total_experiments - high_confidence}** experiments need additional testing or showed mixed results

### Impact on Product
"""
        
        # Add impact summary for each high-confidence result
        for analysis in self.analysis_results:
            if analysis.confidence_level > 0.7:
                config = self.experiment_manager.get_experiment_config(analysis.experiment_id)
                if config:
                    exec_content += f"- **{config.name}**: {analysis.recommendation}\n"
        
        exec_content += f"""
### Next Month's Focus
Based on this month's results, we will:
1. Implement winning variants from high-confidence experiments
2. Design follow-up tests for inconclusive results
3. Focus on areas showing the most promise for improvement

---
*For detailed technical analysis, see the comprehensive report*
"""
        
        with open(exec_summary_path, 'w', encoding='utf-8') as f:
            f.write(exec_content)
        
        # Create detailed results JSON for technical teams
        technical_results = {
            "month": f"{year}-{month:02d}",
            "generated_at": datetime.datetime.now().isoformat(),
            "summary": {
                "total_experiments": total_experiments,
                "high_confidence_results": high_confidence,
                "recommendations": {
                    "variant_a": sum(1 for a in self.analysis_results if "Variant A" in a.recommendation),
                    "variant_b": sum(1 for a in self.analysis_results if "Variant B" in a.recommendation),
                    "mixed": sum(1 for a in self.analysis_results if "mixed" in a.recommendation.lower())
                }
            },
            "experiments": []
        }
        
        for analysis in self.analysis_results:
            config = self.experiment_manager.get_experiment_config(analysis.experiment_id)
            if config:
                technical_results["experiments"].append({
                    "id": analysis.experiment_id,
                    "name": config.name,
                    "description": config.description,
                    "recommendation": analysis.recommendation,
                    "confidence_level": analysis.confidence_level,
                    "statistical_summary": analysis.statistical_summary
                })
        
        technical_path = sharing_dir / "technical_results.json"
        with open(technical_path, 'w', encoding='utf-8') as f:
            json.dump(technical_results, f, indent=2)
        
        logger.info(f"Sharing documentation generated in: {sharing_dir}")

def main():
    """Main function to run monthly analysis"""
    parser = argparse.ArgumentParser(description="Run monthly experiment analysis")
    parser.add_argument("--year", type=int, default=datetime.datetime.now().year,
                       help="Year to analyze (default: current year)")
    parser.add_argument("--month", type=int, default=datetime.datetime.now().month,
                       help="Month to analyze (default: current month)")
    parser.add_argument("--archive", action="store_true",
                       help="Archive completed experiments after analysis")
    
    args = parser.parse_args()
    
    # Validate month
    if not 1 <= args.month <= 12:
        logger.error("Month must be between 1 and 12")
        sys.exit(1)
    
    try:
        analyzer = MonthlyAnalyzer()
        analyzer.analyze_month(args.year, args.month, args.archive)
        
        logger.info("Monthly analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Monthly analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()