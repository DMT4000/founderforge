#!/usr/bin/env python3
"""
Demonstration of the FounderForge context management system.
Shows how to assemble context from multiple sources and manage token usage.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_manager import ContextAssembler, TokenManager, ContextSummarizer
from models import UserContext, Message, TokenUsage, UserPreferences
from database import initialize_database
import json
from datetime import datetime


def setup_demo_data():
    """Set up demo data for the context system demonstration."""
    print("Setting up demo data...")
    
    # Initialize database
    initialize_database()
    
    # Create demo user data
    user_id = "demo_founder_123"
    
    # Create context assembler
    assembler = ContextAssembler()
    
    # Save some demo business data
    business_info = """
    TechFlow Solutions - AI-powered workflow automation platform
    
    Stage: Series A (raised $2M)
    Industry: SaaS/Enterprise Software
    Team Size: 12 employees
    
    Key Challenges:
    - Scaling customer acquisition
    - Product-market fit validation
    - Building enterprise sales team
    
    Current Focus:
    - Expanding into healthcare vertical
    - Improving user onboarding experience
    - Preparing for Series B fundraising
    """
    
    assembler.save_business_data(user_id, business_info, "company_overview")
    
    # Save some chat history
    messages = [
        Message(content="Hi, I need help with our Series B preparation", role="user"),
        Message(content="I'd be happy to help with Series B preparation. What specific areas would you like to focus on?", role="assistant"),
        Message(content="We're struggling with our pitch deck and financial projections", role="user"),
        Message(content="Let's start with your financial projections. What's your current revenue run rate?", role="assistant"),
        Message(content="We're at $150K MRR with 25% month-over-month growth", role="user"),
        Message(content="That's strong growth! For Series B, investors will want to see a clear path to $1M ARR. Let's work on your growth strategy.", role="assistant")
    ]
    
    for message in messages:
        assembler.save_chat_message(user_id, message)
    
    print(f"Demo data created for user: {user_id}")
    return user_id


def demonstrate_context_assembly(user_id: str):
    """Demonstrate context assembly from multiple sources."""
    print("\n" + "="*60)
    print("CONTEXT ASSEMBLY DEMONSTRATION")
    print("="*60)
    
    assembler = ContextAssembler()
    
    # Assemble context for a query
    query = "What should be my top priorities for the next quarter?"
    context = assembler.assemble_context(user_id, query)
    
    print(f"Query: {query}")
    print(f"Assembled context token count: {context.token_count}")
    print(f"Context sources: {', '.join(context.sources)}")
    print(f"Number of goals: {len(context.goals)}")
    print(f"Business data keys: {list(context.business_data.keys())}")
    print(f"Chat history messages: {len(context.chat_history)}")
    print(f"Guard rails: {len(context.guard_rails)}")
    
    print("\n--- FORMATTED CONTEXT PREVIEW ---")
    prompt_string = context.to_prompt_string()
    # Show first 500 characters
    print(prompt_string[:500] + "..." if len(prompt_string) > 500 else prompt_string)
    
    return context


def demonstrate_token_management(user_id: str):
    """Demonstrate token usage tracking and monitoring."""
    print("\n" + "="*60)
    print("TOKEN MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    token_manager = TokenManager()
    
    # Simulate some token usage
    operations = [
        ("context_assembly", TokenUsage(prompt_tokens=1200, completion_tokens=0, total_tokens=1200)),
        ("ai_response", TokenUsage(prompt_tokens=1200, completion_tokens=300, total_tokens=1500)),
        ("summarization", TokenUsage(prompt_tokens=800, completion_tokens=200, total_tokens=1000)),
        ("context_assembly", TokenUsage(prompt_tokens=1100, completion_tokens=0, total_tokens=1100)),
        ("ai_response", TokenUsage(prompt_tokens=1100, completion_tokens=250, total_tokens=1350))
    ]
    
    print("Logging token usage for various operations...")
    for operation, usage in operations:
        token_manager.log_token_usage(user_id, usage, operation)
        print(f"  {operation}: {usage.total_tokens} tokens")
    
    # Show session usage
    session_usage = token_manager.get_session_usage()
    print(f"\nSession total: {session_usage.total_tokens} tokens")
    
    # Show daily usage
    today = datetime.now().strftime("%Y-%m-%d")
    daily_usage = token_manager.get_daily_usage(user_id, today)
    if daily_usage:
        print(f"Daily usage for {today}: {daily_usage.total_tokens} tokens")
    
    # Show usage summary
    summary = token_manager.get_usage_summary(user_id, days=7)
    print(f"\n7-day usage summary:")
    print(f"  Total tokens: {summary['total_usage'].total_tokens}")
    print(f"  Average daily: {summary['average_daily'].total_tokens}")
    
    # Check usage limits
    limits_check = token_manager.check_usage_limits(user_id, 2000)
    print(f"\nUsage limits check for 2000 tokens:")
    print(f"  Within daily limit: {limits_check['within_daily_limit']}")
    print(f"  Within context limit: {limits_check['within_context_limit']}")
    print(f"  Remaining daily: {limits_check['remaining_daily']}")
    
    if limits_check['recommendations']:
        print("  Recommendations:")
        for rec in limits_check['recommendations']:
            print(f"    - {rec}")
    
    return token_manager


def demonstrate_context_summarization(context, token_manager):
    """Demonstrate context summarization to fit token limits."""
    print("\n" + "="*60)
    print("CONTEXT SUMMARIZATION DEMONSTRATION")
    print("="*60)
    
    summarizer = ContextSummarizer(token_manager)
    
    print(f"Original context: {context.token_count} tokens")
    
    # Test different summarization limits
    limits = [2000, 1000, 500]
    
    for limit in limits:
        print(f"\nSummarizing to {limit} tokens:")
        summarized = summarizer.summarize_context(context, max_tokens=limit)
        
        print(f"  Result: {summarized.token_count} tokens")
        print(f"  Goals: {len(summarized.goals)} (was {len(context.goals)})")
        print(f"  Business data keys: {len(summarized.business_data)} (was {len(context.business_data)})")
        print(f"  Chat history: {len(summarized.chat_history)} (was {len(context.chat_history)})")
        print(f"  Guard rails: {len(summarized.guard_rails)} (was {len(context.guard_rails)})")
        
        # Show token allocation
        if summarized.token_count < context.token_count:
            reduction = ((context.token_count - summarized.token_count) / context.token_count) * 100
            print(f"  Reduction: {reduction:.1f}%")


def demonstrate_integration():
    """Demonstrate the complete context management system integration."""
    print("\n" + "="*60)
    print("INTEGRATION DEMONSTRATION")
    print("="*60)
    
    user_id = "integration_test_user"
    
    # Create all components
    assembler = ContextAssembler()
    token_manager = TokenManager()
    summarizer = ContextSummarizer(token_manager)
    
    # Save some business data
    assembler.save_business_data(user_id, "AI startup in healthcare space, pre-seed stage", "overview")
    
    # Save a conversation
    messages = [
        Message(content="I need help with my pitch deck", role="user"),
        Message(content="I can help with that. What's your main value proposition?", role="assistant"),
        Message(content="We use AI to automate medical record analysis", role="user")
    ]
    
    for msg in messages:
        assembler.save_chat_message(user_id, msg)
    
    # Assemble context
    context = assembler.assemble_context(user_id, "How should I structure my Series A pitch?")
    
    # Log token usage
    token_manager.log_token_usage(user_id, TokenUsage(total_tokens=context.token_count), "context_assembly")
    
    # Check if summarization is needed
    limits_check = token_manager.check_usage_limits(user_id, context.token_count)
    
    if not limits_check['within_context_limit']:
        print("Context exceeds limits, applying summarization...")
        context = summarizer.summarize_context(context)
        token_manager.log_token_usage(user_id, TokenUsage(total_tokens=context.token_count), "summarization")
    
    print(f"Final context ready: {context.token_count} tokens")
    print("Context sources:", ", ".join(context.sources))
    
    # Show formatted context preview
    print("\n--- FINAL CONTEXT PREVIEW ---")
    prompt = context.to_prompt_string()
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)


def main():
    """Run the complete context management system demonstration."""
    print("FounderForge Context Management System Demo")
    print("=" * 50)
    
    try:
        # Setup demo data
        user_id = setup_demo_data()
        
        # Demonstrate context assembly
        context = demonstrate_context_assembly(user_id)
        
        # Demonstrate token management
        token_manager = demonstrate_token_management(user_id)
        
        # Demonstrate context summarization
        demonstrate_context_summarization(context, token_manager)
        
        # Demonstrate integration
        demonstrate_integration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("The context management system successfully:")
        print("✓ Assembled context from multiple local sources")
        print("✓ Tracked and logged token usage")
        print("✓ Applied intelligent summarization when needed")
        print("✓ Maintained data privacy with local-only storage")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()