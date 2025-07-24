#!/usr/bin/env python3
"""
Knowledge Management System Demo

This script demonstrates the capabilities of the FounderForge knowledge
management system including:
- Adding different types of knowledge items
- Searching and retrieving knowledge
- Generating weekly documentation
- Self-feedback mechanisms
- Knowledge statistics and analytics

Usage:
    python examples/knowledge_management_demo.py
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_manager import KnowledgeManager

def demo_basic_operations():
    """Demonstrate basic knowledge management operations"""
    print("="*60)
    print("KNOWLEDGE MANAGEMENT SYSTEM DEMO")
    print("="*60)
    
    # Initialize knowledge manager
    print("\n1. Initializing Knowledge Manager...")
    km = KnowledgeManager(base_path="data/knowledge_demo")
    print("   ✓ Knowledge manager initialized")
    print("   ✓ Directory structure created")
    
    # Add Q&A items
    print("\n2. Adding Q&A Items...")
    qa_items = [
        {
            "question": "How do I optimize database queries in SQLite?",
            "answer": "Use proper indexing, avoid SELECT *, use EXPLAIN QUERY PLAN, and consider connection pooling.",
            "tags": ["database", "sqlite", "optimization"]
        },
        {
            "question": "What's the best way to handle API rate limiting?",
            "answer": "Implement exponential backoff, use retry mechanisms, and cache responses when possible.",
            "tags": ["api", "rate-limiting", "best-practices"]
        },
        {
            "question": "How can I improve LLM response quality?",
            "answer": "Use better prompt engineering, provide more context, implement confidence scoring, and use fallback mechanisms.",
            "tags": ["llm", "ai", "prompt-engineering"]
        }
    ]
    
    for qa in qa_items:
        item_id = km.add_qa_item(qa["question"], qa["answer"], qa["tags"])
        print(f"   ✓ Added Q&A: {qa['question'][:50]}... (ID: {item_id})")
    
    # Add ideas
    print("\n3. Adding Ideas...")
    ideas = [
        {
            "title": "Implement Vector Database",
            "description": "Replace FAISS with a proper vector database like Chroma or Pinecone for better scalability and features.",
            "priority": 4,
            "tags": ["database", "vectors", "scalability"]
        },
        {
            "title": "Add Voice Interface",
            "description": "Integrate speech-to-text and text-to-speech for hands-free interaction with the AI cofounder.",
            "priority": 2,
            "tags": ["ui", "voice", "accessibility"]
        },
        {
            "title": "Multi-language Support",
            "description": "Add support for multiple languages in both input and output to serve international users.",
            "priority": 3,
            "tags": ["i18n", "languages", "global"]
        }
    ]
    
    for idea in ideas:
        item_id = km.add_idea(idea["title"], idea["description"], idea["priority"], idea["tags"])
        print(f"   ✓ Added Idea: {idea['title']} (Priority: {idea['priority']}, ID: {item_id})")
    
    # Add feedback
    print("\n4. Adding Feedback...")
    feedback_items = [
        {
            "title": "Response Time Too Slow",
            "feedback": "Users report that responses take too long, especially for complex queries. Need to optimize context assembly and agent workflows.",
            "type": "performance",
            "tags": ["performance", "user-feedback"]
        },
        {
            "title": "Memory System Confusion",
            "feedback": "Some users are confused about what information is being remembered vs. what needs to be repeated.",
            "type": "usability",
            "tags": ["memory", "ux", "user-feedback"]
        }
    ]
    
    for feedback in feedback_items:
        item_id = km.add_feedback(feedback["title"], feedback["feedback"], feedback["type"], feedback["tags"])
        print(f"   ✓ Added Feedback: {feedback['title']} (Type: {feedback['type']}, ID: {item_id})")
    
    # Add techniques
    print("\n5. Adding Techniques...")
    techniques = [
        {
            "title": "Context Window Management",
            "description": "Technique for managing LLM context windows by prioritizing important information and summarizing less critical content.",
            "example": "def manage_context(context, max_tokens=16000):\n    if count_tokens(context) > max_tokens:\n        return summarize_context(context, max_tokens)\n    return context",
            "tags": ["llm", "context", "optimization"]
        },
        {
            "title": "Confidence-Based Fallbacks",
            "description": "Implementation pattern for providing fallback responses when AI confidence is below threshold.",
            "example": "if confidence < 0.8:\n    return fallback_response(query)\nelse:\n    return ai_response",
            "tags": ["ai", "confidence", "fallbacks"]
        }
    ]
    
    for technique in techniques:
        item_id = km.add_technique(technique["title"], technique["description"], technique["example"], technique["tags"])
        print(f"   ✓ Added Technique: {technique['title']} (ID: {item_id})")
    
    return km

def demo_search_and_retrieval(km):
    """Demonstrate search and retrieval capabilities"""
    print("\n" + "="*60)
    print("SEARCH AND RETRIEVAL DEMO")
    print("="*60)
    
    # Search by keyword
    print("\n1. Searching for 'database' related items...")
    db_items = km.search_knowledge("database")
    print(f"   Found {len(db_items)} items:")
    for item in db_items:
        print(f"   - [{item.type.upper()}] {item.title}")
    
    # Search by type
    print("\n2. Getting all Q&A items...")
    qa_items = km.get_knowledge_items(item_type="qa")
    print(f"   Found {len(qa_items)} Q&A items:")
    for item in qa_items:
        print(f"   - {item.title}")
    
    # Search by priority
    print("\n3. Getting high-priority ideas...")
    all_ideas = km.get_knowledge_items(item_type="idea")
    high_priority = [idea for idea in all_ideas if idea.priority >= 3]
    print(f"   Found {len(high_priority)} high-priority ideas:")
    for idea in high_priority:
        print(f"   - {idea.title} (Priority: {idea.priority})")
    
    # Search for optimization techniques
    print("\n4. Searching for 'optimization' techniques...")
    opt_items = km.search_knowledge("optimization")
    print(f"   Found {len(opt_items)} optimization-related items:")
    for item in opt_items:
        print(f"   - [{item.type.upper()}] {item.title}")

def demo_self_feedback(km):
    """Demonstrate self-feedback generation"""
    print("\n" + "="*60)
    print("SELF-FEEDBACK DEMO")
    print("="*60)
    
    # Simulate system metrics
    print("\n1. Simulating system metrics...")
    system_metrics = {
        "response_time": 7.5,    # Above 5s threshold
        "accuracy": 0.85,        # Below 90% threshold
        "memory_usage": 0.9,     # Above 80% threshold
        "error_rate": 0.03,      # 3% error rate
        "total_requests": 1500,
        "successful_requests": 1455
    }
    
    print("   System Metrics:")
    for metric, value in system_metrics.items():
        if isinstance(value, float) and metric.endswith(('_rate', '_usage', 'accuracy')):
            print(f"   - {metric}: {value:.1%}")
        elif isinstance(value, float):
            print(f"   - {metric}: {value:.2f}")
        else:
            print(f"   - {metric}: {value}")
    
    # Generate self-feedback
    print("\n2. Generating self-feedback...")
    feedback_result = km.generate_self_feedback(system_metrics)
    print(f"   {feedback_result}")
    
    # Show generated feedback items
    print("\n3. Generated feedback items:")
    feedback_items = km.get_knowledge_items(item_type="feedback")
    automated_feedback = [item for item in feedback_items if "automated" in item.tags]
    
    for item in automated_feedback:
        print(f"   - {item.title}")
        print(f"     {item.content[:100]}...")

def demo_weekly_documentation(km):
    """Demonstrate weekly documentation generation"""
    print("\n" + "="*60)
    print("WEEKLY DOCUMENTATION DEMO")
    print("="*60)
    
    print("\n1. Generating weekly documentation...")
    week_start = km.create_weekly_documentation()
    print(f"   ✓ Created documentation for week starting {week_start}")
    
    print("\n2. Retrieving weekly documentation...")
    weekly_doc = km.get_weekly_documentation(week_start)
    
    if weekly_doc:
        print(f"   Week: {weekly_doc.week_start}")
        print(f"   Techniques learned: {len(weekly_doc.techniques_learned)}")
        print(f"   Improvements made: {len(weekly_doc.improvements_made)}")
        print(f"   Feedback collected: {len(weekly_doc.feedback_collected)}")
        print(f"   Next week goals: {len(weekly_doc.next_week_goals)}")
        
        print("\n3. Next week goals:")
        for i, goal in enumerate(weekly_doc.next_week_goals, 1):
            print(f"   {i}. {goal}")
    
    # Show markdown file location
    markdown_file = Path("data/knowledge_demo/weekly_docs") / f"week_{week_start}.md"
    if markdown_file.exists():
        print(f"\n4. Markdown report saved to: {markdown_file}")
        print("   Preview:")
        content = markdown_file.read_text()
        lines = content.split('\n')[:10]  # First 10 lines
        for line in lines:
            print(f"   {line}")
        if len(content.split('\n')) > 10:
            print("   ...")

def demo_statistics(km):
    """Demonstrate knowledge statistics"""
    print("\n" + "="*60)
    print("KNOWLEDGE STATISTICS DEMO")
    print("="*60)
    
    stats = km.get_knowledge_stats()
    
    print(f"\n1. Overall Statistics:")
    print(f"   Total items: {stats.get('total_items', 0)}")
    print(f"   Recent items (last 7 days): {stats.get('recent_items', 0)}")
    print(f"   Last updated: {stats.get('last_updated', 'Unknown')}")
    
    print(f"\n2. Items by Type:")
    for item_type, count in stats.get('by_type', {}).items():
        print(f"   {item_type.capitalize()}: {count}")
    
    print(f"\n3. Items by Priority:")
    for priority, count in sorted(stats.get('by_priority', {}).items(), reverse=True):
        print(f"   Priority {priority}: {count}")

def demo_file_structure(km):
    """Show the created file structure"""
    print("\n" + "="*60)
    print("FILE STRUCTURE DEMO")
    print("="*60)
    
    base_path = Path("data/knowledge_demo")
    
    print(f"\n1. Directory structure created at: {base_path}")
    
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item, next_prefix, max_depth, current_depth + 1)
    
    show_tree(base_path)
    
    print(f"\n2. Sample files created:")
    
    # Show some JSON files
    for subdir in ["qa", "ideas", "feedback", "techniques"]:
        subdir_path = base_path / subdir
        json_files = list(subdir_path.glob("*.json"))
        if json_files:
            print(f"   {subdir}/:")
            for json_file in json_files[:2]:  # Show first 2 files
                print(f"     - {json_file.name}")
            if len(json_files) > 2:
                print(f"     ... and {len(json_files) - 2} more")

def main():
    """Main demo function"""
    try:
        # Run all demos
        km = demo_basic_operations()
        demo_search_and_retrieval(km)
        demo_self_feedback(km)
        demo_weekly_documentation(km)
        demo_statistics(km)
        demo_file_structure(km)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe knowledge management system is now set up with:")
        print("- Sample Q&A items, ideas, feedback, and techniques")
        print("- Generated self-feedback based on system metrics")
        print("- Weekly documentation with goals and summaries")
        print("- Local file storage for easy browsing")
        print("- Search and retrieval capabilities")
        print("\nYou can explore the created files in: data/knowledge_demo/")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()