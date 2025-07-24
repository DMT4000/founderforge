# FounderForge Prompt Collaboration Notes

## Overview

This document tracks collaborative improvements, experiments, and learnings related to prompt engineering for the FounderForge AI Cofounder system.

## Recent Experiments

### Experiment: test-experiment (T2 - Reversible)
- **Date**: 2025-07-23
- **Type**: T2 (Reversible)
- **Description**: Test experiment for prompt versioning
- **Status**: Completed and merged
- **Metrics**:
  - Accuracy: 95%
  - Response time: 1.2s
  - User satisfaction: 4.5/5
- **Key learnings**: Initial system prompt setup successful
- **Next steps**: Test with more complex prompts

## Prompt Templates

### System Prompt Template
- **Location**: `templates/system_prompt_template.txt`
- **Purpose**: Customizable system prompt for different business types
- **Variables**: business_type, focus_area, business_stage, industry, challenges
- **Usage**: Use for personalizing AI responses based on user context

### Funding Prompt Template
- **Location**: `templates/funding_prompt_template.txt`
- **Purpose**: Specialized prompt for funding-related conversations
- **Variables**: funding_stage, funding_type, target_amount, use_of_funds, timeline, previous_funding
- **Usage**: Activate when user needs funding guidance

## Best Practices

### Experiment Types
- **T1 (One-way door)**: Use for major prompt architecture changes that are hard to reverse
- **T2 (Reversible)**: Use for content updates, parameter tuning, and incremental improvements

### Prompt Versioning
- Always create experiments for prompt changes
- Include clear descriptions and expected outcomes
- Track metrics for each experiment
- Document learnings in this file

### Collaboration Guidelines
- Update this file when completing experiments
- Share insights about what worked and what didn't
- Suggest new experiment ideas
- Document any prompt patterns that emerge

## Ideas for Future Experiments

1. **Context-Aware Prompts**: Experiment with prompts that adapt based on conversation history
2. **Industry-Specific Prompts**: Create specialized prompts for different industries (tech, retail, healthcare)
3. **Confidence-Based Prompts**: Develop prompts that adjust based on AI confidence levels
4. **Multi-Agent Prompts**: Design prompts for different agent roles (planner, validator, coach)

## Metrics to Track

- Response accuracy
- Response time
- User satisfaction scores
- Conversation completion rates
- Fallback activation frequency

## Weekly Review Process

Every week, review:
1. Active experiments and their progress
2. Completed experiments and their outcomes
3. New prompt ideas from user feedback
4. Performance metrics trends
5. Areas for improvement

## Monthly Analysis

Monthly deep-dive should include:
1. Experiment success/failure analysis
2. Prompt performance trends
3. User feedback patterns
4. Technical improvements needed
5. Strategic prompt roadmap updates

---

*Last updated: 2025-07-23*
*Next review: 2025-07-30*