#!/usr/bin/env python3
"""
Experiment Documentation and Sharing Tools for FounderForge AI Cofounder

This script provides tools for documenting and sharing experiment results:
- Generate formatted reports for different audiences
- Create visualizations of experiment data
- Export results in various formats
- Prepare sharing packages for stakeholders

Usage:
    python scripts/experiment_documentation.py report --experiment-id exp_123 --format markdown
    python scripts/experiment_documentation.py export --experiment-id exp_123 --format json
    python scripts/experiment_documentation.py package --month 2024-01 --audience stakeholders
"""

import argparse
import sys
import os
import datetime
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment_manager import ExperimentManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/experiment_docs.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ExperimentDocumentationTool:
    """Tools for documenting and sharing experiment results"""
    
    def __init__(self):
        self.experiment_manager = ExperimentManager()
        self.output_dir = Path("data/experiments/documentation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_experiment_report(self, experiment_id: str, format_type: str = "markdown",
                                 audience: str = "technical") -> str:
        """Generate a formatted report for a specific experiment"""
        
        try:
            # Get experiment data
            config = self.experiment_manager.get_experiment_config(experiment_id)
            if not config:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            analysis = self.experiment_manager.analyze_experiment(experiment_id)
            variant_a_results, variant_b_results = self.experiment_manager.get_experiment_results(experiment_id)
            
            # Generate report based on format and audience
            if format_type == "markdown":
                if audience == "technical":
                    report_content = self._generate_technical_markdown(config, analysis, variant_a_results, variant_b_results)
                else:
                    report_content = self._generate_stakeholder_markdown(config, analysis)
            elif format_type == "html":
                report_content = self._generate_html_report(config, analysis, variant_a_results, variant_b_results)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Save report
            filename = f"{experiment_id}_report_{audience}.{format_type}"
            report_path = self.output_dir / filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Generated {audience} report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None
    
    def _generate_technical_markdown(self, config, analysis, variant_a_results, variant_b_results) -> str:
        """Generate detailed technical report in Markdown format"""
        
        report = f"""# Technical Experiment Report: {config.name}

**Experiment ID**: {config.id}
**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Overview

**Description**: {config.description}
**Start Date**: {config.start_date}
**Status**: {config.status}
**Success Metrics**: {', '.join(config.success_metrics)}

## Hypothesis and Variants

### Variant A (Control)
- **Script**: `{config.variant_a_script[:100]}...`
- **Total Runs**: {len(variant_a_results)}
- **Successful Runs**: {sum(1 for r in variant_a_results if r.success)}

### Variant B (Treatment)
- **Script**: `{config.variant_b_script[:100]}...`
- **Total Runs**: {len(variant_b_results)}
- **Successful Runs**: {sum(1 for r in variant_b_results if r.success)}

## Statistical Analysis

**Analysis Date**: {analysis.analysis_date}
**Confidence Level**: {analysis.confidence_level:.3f}

### Performance Metrics

"""
        
        # Add detailed metrics comparison
        stats = analysis.statistical_summary
        
        for metric in config.success_metrics:
            if metric in stats.get("comparison", {}):
                comparison = stats["comparison"][metric]
                variant_a_stats = stats["variant_a"]["metrics"].get(metric, {})
                variant_b_stats = stats["variant_b"]["metrics"].get(metric, {})
                
                report += f"""#### {metric.title()}

| Metric | Variant A | Variant B | Improvement |
|--------|-----------|-----------|-------------|
| Average | {variant_a_stats.get('average', 0):.3f} | {variant_b_stats.get('average', 0):.3f} | {comparison.get('improvement_percent', 0):+.1f}% |
| Min | {variant_a_stats.get('min', 0):.3f} | {variant_b_stats.get('min', 0):.3f} | - |
| Max | {variant_a_stats.get('max', 0):.3f} | {variant_b_stats.get('max', 0):.3f} | - |
| Count | {variant_a_stats.get('count', 0)} | {variant_b_stats.get('count', 0)} | - |

**Statistical Significance**: {'✓ Significant' if comparison.get('significant', False) else '○ Not significant'}
**Better Variant**: {comparison.get('better_variant', 'Unknown')}

"""
        
        # Add execution details
        report += f"""## Execution Details

### Variant A Results
- **Average Execution Time**: {stats['variant_a'].get('avg_execution_time', 0):.3f}s
- **Success Rate**: {stats['variant_a'].get('successful_runs', 0) / stats['variant_a'].get('total_runs', 1) * 100:.1f}%

### Variant B Results
- **Average Execution Time**: {stats['variant_b'].get('avg_execution_time', 0):.3f}s
- **Success Rate**: {stats['variant_b'].get('successful_runs', 0) / stats['variant_b'].get('total_runs', 1) * 100:.1f}%

## Recommendation

{analysis.recommendation}

## Raw Data

### Failed Runs
"""
        
        # Add failed runs details
        failed_a = [r for r in variant_a_results if not r.success]
        failed_b = [r for r in variant_b_results if not r.success]
        
        if failed_a or failed_b:
            report += f"""
**Variant A Failures**: {len(failed_a)}
**Variant B Failures**: {len(failed_b)}

"""
            for failure in failed_a[:3]:  # Show first 3 failures
                report += f"- Variant A ({failure.run_timestamp}): {failure.error_message}\n"
            
            for failure in failed_b[:3]:  # Show first 3 failures
                report += f"- Variant B ({failure.run_timestamp}): {failure.error_message}\n"
        else:
            report += "No failed runs recorded.\n"
        
        report += f"""
## Next Steps

1. **If high confidence ({analysis.confidence_level:.2f})**: Implement recommended variant
2. **If low confidence**: Design follow-up experiments with larger sample size
3. **Archive experiment**: Use teardown tools to clean up workspace

---
*This report was automatically generated by the FounderForge experiment documentation system.*
"""
        
        return report
    
    def _generate_stakeholder_markdown(self, config, analysis) -> str:
        """Generate stakeholder-friendly report in Markdown format"""
        
        # Determine impact level
        if analysis.confidence_level > 0.8:
            impact_level = "High Impact"
            impact_icon = "🟢"
        elif analysis.confidence_level > 0.6:
            impact_level = "Medium Impact"
            impact_icon = "🟡"
        else:
            impact_level = "Low Impact"
            impact_icon = "🔴"
        
        report = f"""# Experiment Results: {config.name}

{impact_icon} **{impact_level}** | Confidence: {analysis.confidence_level:.0%}

## What We Tested

{config.description}

## Key Finding

{analysis.recommendation}

## Business Impact

"""
        
        # Add business impact based on metrics
        stats = analysis.statistical_summary
        comparisons = stats.get("comparison", {})
        
        significant_improvements = [
            metric for metric, comp in comparisons.items() 
            if comp.get("significant", False) and comp.get("improvement_percent", 0) > 0
        ]
        
        if significant_improvements:
            report += f"This experiment shows measurable improvements in:\n"
            for metric in significant_improvements:
                improvement = comparisons[metric]["improvement_percent"]
                report += f"- **{metric.title()}**: {improvement:+.1f}% improvement\n"
        else:
            report += "No significant improvements were measured in this experiment.\n"
        
        report += f"""

## Next Steps

"""
        
        if analysis.confidence_level > 0.7:
            report += "✅ **Recommended for implementation** - Results show clear benefit\n"
        elif analysis.confidence_level > 0.5:
            report += "⚠️ **Needs more testing** - Results are promising but not conclusive\n"
        else:
            report += "❌ **Not recommended** - No clear benefit demonstrated\n"
        
        report += f"""
## Timeline

- **Experiment Started**: {config.start_date[:10]}
- **Analysis Completed**: {analysis.analysis_date[:10]}
- **Implementation Target**: {(datetime.datetime.now() + datetime.timedelta(days=14)).strftime('%Y-%m-%d')} (if approved)

---
*For technical details, request the technical report from the development team.*
"""
        
        return report
    
    def _generate_html_report(self, config, analysis, variant_a_results, variant_b_results) -> str:
        """Generate HTML report with basic styling"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {config.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Experiment Report: {config.name}</h1>
        <p><strong>ID:</strong> {config.id}</p>
        <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Overview</h2>
    <p><strong>Description:</strong> {config.description}</p>
    <p><strong>Status:</strong> {config.status}</p>
    <p><strong>Confidence Level:</strong> <span class="{'success' if analysis.confidence_level > 0.7 else 'warning' if analysis.confidence_level > 0.5 else 'danger'}">{analysis.confidence_level:.1%}</span></p>
    
    <h2>Recommendation</h2>
    <div class="metric">
        {analysis.recommendation}
    </div>
    
    <h2>Results Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Variant A</th>
            <th>Variant B</th>
            <th>Improvement</th>
            <th>Significant</th>
        </tr>
"""
        
        # Add metrics table
        stats = analysis.statistical_summary
        for metric in config.success_metrics:
            if metric in stats.get("comparison", {}):
                comparison = stats["comparison"][metric]
                variant_a_stats = stats["variant_a"]["metrics"].get(metric, {})
                variant_b_stats = stats["variant_b"]["metrics"].get(metric, {})
                
                html += f"""        <tr>
            <td>{metric.title()}</td>
            <td>{variant_a_stats.get('average', 0):.3f}</td>
            <td>{variant_b_stats.get('average', 0):.3f}</td>
            <td class="{'success' if comparison.get('improvement_percent', 0) > 0 else 'danger'}">{comparison.get('improvement_percent', 0):+.1f}%</td>
            <td>{'✓' if comparison.get('significant', False) else '○'}</td>
        </tr>
"""
        
        html += f"""    </table>
    
    <h2>Execution Details</h2>
    <p><strong>Variant A:</strong> {len(variant_a_results)} runs, {sum(1 for r in variant_a_results if r.success)} successful</p>
    <p><strong>Variant B:</strong> {len(variant_b_results)} runs, {sum(1 for r in variant_b_results if r.success)} successful</p>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        <p>Generated by FounderForge Experiment Documentation System</p>
    </footer>
</body>
</html>"""
        
        return html
    
    def export_experiment_data(self, experiment_id: str, format_type: str = "json") -> str:
        """Export experiment data in various formats"""
        
        try:
            # Get experiment data
            config = self.experiment_manager.get_experiment_config(experiment_id)
            if not config:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            analysis = self.experiment_manager.analyze_experiment(experiment_id)
            variant_a_results, variant_b_results = self.experiment_manager.get_experiment_results(experiment_id)
            
            if format_type == "json":
                return self._export_json(experiment_id, config, analysis, variant_a_results, variant_b_results)
            elif format_type == "csv":
                return self._export_csv(experiment_id, config, variant_a_results, variant_b_results)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return None
    
    def _export_json(self, experiment_id, config, analysis, variant_a_results, variant_b_results) -> str:
        """Export complete experiment data as JSON"""
        
        export_data = {
            "experiment": {
                "id": config.id,
                "name": config.name,
                "description": config.description,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "status": config.status,
                "success_metrics": config.success_metrics,
                "metadata": config.metadata
            },
            "analysis": {
                "analysis_date": analysis.analysis_date,
                "recommendation": analysis.recommendation,
                "confidence_level": analysis.confidence_level,
                "statistical_summary": analysis.statistical_summary
            },
            "results": {
                "variant_a": [
                    {
                        "run_timestamp": r.run_timestamp,
                        "metrics": r.metrics,
                        "execution_time": r.execution_time,
                        "success": r.success,
                        "error_message": r.error_message
                    } for r in variant_a_results
                ],
                "variant_b": [
                    {
                        "run_timestamp": r.run_timestamp,
                        "metrics": r.metrics,
                        "execution_time": r.execution_time,
                        "success": r.success,
                        "error_message": r.error_message
                    } for r in variant_b_results
                ]
            },
            "export_metadata": {
                "exported_at": datetime.datetime.now().isoformat(),
                "exported_by": "experiment_documentation.py",
                "format_version": "1.0"
            }
        }
        
        # Save JSON export
        export_path = self.output_dir / f"{experiment_id}_export.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported JSON data: {export_path}")
        return str(export_path)
    
    def _export_csv(self, experiment_id, config, variant_a_results, variant_b_results) -> str:
        """Export experiment results as CSV"""
        
        export_path = self.output_dir / f"{experiment_id}_results.csv"
        
        with open(export_path, 'w', newline='') as csvfile:
            # Determine all possible metric columns
            all_metrics = set()
            for result in variant_a_results + variant_b_results:
                all_metrics.update(result.metrics.keys())
            
            fieldnames = ['experiment_id', 'variant', 'run_timestamp', 'execution_time', 'success', 'error_message']
            fieldnames.extend(sorted(all_metrics))
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write variant A results
            for result in variant_a_results:
                row = {
                    'experiment_id': experiment_id,
                    'variant': 'A',
                    'run_timestamp': result.run_timestamp,
                    'execution_time': result.execution_time,
                    'success': result.success,
                    'error_message': result.error_message or ''
                }
                row.update(result.metrics)
                writer.writerow(row)
            
            # Write variant B results
            for result in variant_b_results:
                row = {
                    'experiment_id': experiment_id,
                    'variant': 'B',
                    'run_timestamp': result.run_timestamp,
                    'execution_time': result.execution_time,
                    'success': result.success,
                    'error_message': result.error_message or ''
                }
                row.update(result.metrics)
                writer.writerow(row)
        
        logger.info(f"Exported CSV data: {export_path}")
        return str(export_path)
    
    def create_sharing_package(self, month: str, audience: str = "stakeholders") -> str:
        """Create a complete sharing package for a month's experiments"""
        
        try:
            # Parse month (format: YYYY-MM)
            year, month_num = map(int, month.split('-'))
            
            # Create package directory
            package_dir = self.output_dir / f"sharing_package_{month}_{audience}"
            package_dir.mkdir(exist_ok=True)
            
            # Get monthly experiments
            start_date = datetime.datetime(year, month_num, 1)
            if month_num == 12:
                end_date = datetime.datetime(year + 1, 1, 1)
            else:
                end_date = datetime.datetime(year, month_num + 1, 1)
            
            all_experiments = self.experiment_manager.list_experiments()
            monthly_experiments = []
            
            for exp in all_experiments:
                exp_date = datetime.datetime.fromisoformat(exp['start_date'])
                if start_date <= exp_date < end_date:
                    monthly_experiments.append(exp)
            
            if not monthly_experiments:
                logger.warning(f"No experiments found for {month}")
                return None
            
            # Generate individual reports
            for exp in monthly_experiments:
                self.generate_experiment_report(
                    experiment_id=exp['id'],
                    format_type="markdown",
                    audience=audience
                )
                
                # Copy to package directory
                source_file = self.output_dir / f"{exp['id']}_report_{audience}.markdown"
                if source_file.exists():
                    dest_file = package_dir / f"{exp['name'].replace(' ', '_')}_report.md"
                    dest_file.write_text(source_file.read_text(encoding='utf-8'), encoding='utf-8')
            
            # Create package index
            index_content = f"""# Experiment Results Package - {month}

**Audience**: {audience.title()}
**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Experiments**: {len(monthly_experiments)}

## Experiments Included

"""
            
            for exp in monthly_experiments:
                index_content += f"- [{exp['name']}]({exp['name'].replace(' ', '_')}_report.md) - {exp['description']}\n"
            
            index_content += f"""
## Package Contents

- Individual experiment reports (Markdown format)
- This index file
- Summary statistics

## How to Use

1. Review the individual experiment reports for detailed findings
2. Focus on experiments with high confidence levels
3. Implement recommendations from successful experiments
4. Plan follow-up experiments for inconclusive results

---
*Generated by FounderForge Experiment Documentation System*
"""
            
            index_file = package_dir / "README.md"
            index_file.write_text(index_content, encoding='utf-8')
            
            logger.info(f"Created sharing package: {package_dir}")
            return str(package_dir)
            
        except Exception as e:
            logger.error(f"Failed to create sharing package: {e}")
            return None

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Generate experiment documentation and sharing materials")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Report generation command
    report_parser = subparsers.add_parser("report", help="Generate experiment report")
    report_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    report_parser.add_argument("--format", choices=["markdown", "html"], default="markdown",
                              help="Report format")
    report_parser.add_argument("--audience", choices=["technical", "stakeholders"], default="technical",
                              help="Target audience")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export experiment data")
    export_parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json",
                              help="Export format")
    
    # Package command
    package_parser = subparsers.add_parser("package", help="Create sharing package")
    package_parser.add_argument("--month", required=True, help="Month in YYYY-MM format")
    package_parser.add_argument("--audience", choices=["technical", "stakeholders"], default="stakeholders",
                               help="Target audience")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    doc_tool = ExperimentDocumentationTool()
    
    try:
        if args.command == "report":
            result = doc_tool.generate_experiment_report(
                experiment_id=args.experiment_id,
                format_type=args.format,
                audience=args.audience
            )
            if result:
                print(f"✓ Report generated: {result}")
        
        elif args.command == "export":
            result = doc_tool.export_experiment_data(
                experiment_id=args.experiment_id,
                format_type=args.format
            )
            if result:
                print(f"✓ Data exported: {result}")
        
        elif args.command == "package":
            result = doc_tool.create_sharing_package(
                month=args.month,
                audience=args.audience
            )
            if result:
                print(f"✓ Sharing package created: {result}")
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()