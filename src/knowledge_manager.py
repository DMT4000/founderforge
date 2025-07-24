"""
Knowledge Management System for FounderForge AI Cofounder

This module provides local knowledge management capabilities including:
- Q&A collection and organization
- Idea tracking and documentation
- Self-feedback mechanisms for continuous improvement
- Weekly documentation automation
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import logging

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeItem:
    """Represents a knowledge item (Q&A, idea, feedback, etc.)"""
    id: str
    type: str  # 'qa', 'idea', 'feedback', 'technique'
    title: str
    content: str
    tags: List[str]
    created_at: str
    updated_at: str
    priority: int = 1  # 1-5 scale
    status: str = 'active'  # active, archived, implemented
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class WeeklyDocumentation:
    """Weekly documentation entry for new tools and techniques"""
    week_start: str
    tools_discovered: List[Dict[str, str]]
    techniques_learned: List[Dict[str, str]]
    improvements_made: List[Dict[str, str]]
    feedback_collected: List[Dict[str, str]]
    next_week_goals: List[str]

class KnowledgeManager:
    """Manages local knowledge base for continuous learning and improvement"""
    
    def __init__(self, base_path: str = "data/knowledge"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "knowledge.db"
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Create necessary directories for knowledge management"""
        directories = [
            self.base_path,
            self.base_path / "qa",
            self.base_path / "ideas", 
            self.base_path / "feedback",
            self.base_path / "techniques",
            self.base_path / "weekly_docs",
            self.base_path / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create README files for each directory
        self._create_directory_readmes()
    
    def _create_directory_readmes(self):
        """Create README files explaining each knowledge directory"""
        readmes = {
            "qa/README.md": """# Q&A Collection

This directory contains questions and answers collected during system usage.

## Structure
- Each Q&A is stored as a JSON file with metadata
- Files are organized by date and topic
- Use tags for easy searching and categorization

## Usage
- Add new Q&A items using the KnowledgeManager API
- Review weekly for patterns and improvements
- Archive outdated or resolved items
""",
            "ideas/README.md": """# Ideas Collection

This directory stores innovative ideas and potential improvements.

## Structure
- Ideas are stored with priority and implementation status
- Include context and potential impact
- Track progress from conception to implementation

## Usage
- Capture ideas immediately when they arise
- Review and prioritize during weekly sessions
- Move implemented ideas to archive with results
""",
            "feedback/README.md": """# Self-Feedback Mechanisms

This directory contains feedback loops and improvement tracking.

## Structure
- Performance feedback from system usage
- User interaction analysis
- Continuous improvement suggestions
- Success/failure pattern analysis

## Usage
- Automated feedback collection from system metrics
- Manual feedback entry for qualitative insights
- Weekly review and action planning
""",
            "techniques/README.md": """# Tools and Techniques Documentation

This directory documents new tools, techniques, and best practices.

## Structure
- Tool documentation with usage examples
- Technique guides with implementation details
- Best practices and lessons learned
- Integration guides for new technologies

## Usage
- Document new discoveries immediately
- Include practical examples and use cases
- Update existing documentation as techniques evolve
""",
            "weekly_docs/README.md": """# Weekly Documentation

This directory contains automated weekly documentation summaries.

## Structure
- Weekly reports generated automatically
- Summary of discoveries, improvements, and learnings
- Goal setting and progress tracking
- Trend analysis and pattern recognition

## Usage
- Review weekly reports for insights
- Use for planning and goal setting
- Track long-term progress and evolution
""",
            "templates/README.md": """# Templates

This directory contains templates for knowledge management activities.

## Structure
- Document templates for consistent formatting
- Workflow templates for common processes
- Report templates for standardized outputs
- Checklist templates for systematic reviews

## Usage
- Use templates to maintain consistency
- Customize templates for specific needs
- Create new templates as patterns emerge
"""
        }
        
        for file_path, content in readmes.items():
            full_path = self.base_path / file_path
            if not full_path.exists():
                full_path.write_text(content)
    
    def _init_database(self):
        """Initialize SQLite database for knowledge management"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,  -- JSON array
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'active',
                    metadata TEXT  -- JSON object
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weekly_docs (
                    week_start TEXT PRIMARY KEY,
                    tools_discovered TEXT,  -- JSON array
                    techniques_learned TEXT,  -- JSON array
                    improvements_made TEXT,  -- JSON array
                    feedback_collected TEXT,  -- JSON array
                    next_week_goals TEXT  -- JSON array
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_items(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_items(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_priority ON knowledge_items(priority)")
            conn.commit()
        finally:
            conn.close()
    
    def add_knowledge_item(self, item: KnowledgeItem) -> bool:
        """Add a new knowledge item to the database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_items 
                (id, type, title, content, tags, created_at, updated_at, priority, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.type, item.title, item.content,
                json.dumps(item.tags), item.created_at, item.updated_at,
                item.priority, item.status, json.dumps(item.metadata)
            ))
            conn.commit()
            
            # Also save as JSON file for easy browsing
            self._save_item_as_file(item)
            logger.info(f"Added knowledge item: {item.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge item: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def _save_item_as_file(self, item: KnowledgeItem):
        """Save knowledge item as JSON file for easy access"""
        # Map item types to directory names (plural)
        type_to_dir = {
            "qa": "qa",
            "idea": "ideas",
            "feedback": "feedback", 
            "technique": "techniques"
        }
        
        dir_name = type_to_dir.get(item.type, item.type)
        type_dir = self.base_path / dir_name
        
        # Ensure the type directory exists
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean filename to avoid invalid characters
        clean_title = "".join(c for c in item.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title.replace(' ', '_')[:50]
        filename = f"{item.id}_{clean_title}.json"
        file_path = type_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(item), f, indent=2)
            logger.info(f"Saved item as file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving item as file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue without file save if there's an issue
    
    def get_knowledge_items(self, item_type: Optional[str] = None, 
                          status: str = 'active', limit: int = 100) -> List[KnowledgeItem]:
        """Retrieve knowledge items from database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            if item_type:
                cursor = conn.execute("""
                    SELECT * FROM knowledge_items 
                    WHERE type = ? AND status = ?
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT ?
                """, (item_type, status, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM knowledge_items 
                    WHERE status = ?
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT ?
                """, (status, limit))
            
            items = []
            for row in cursor.fetchall():
                item = KnowledgeItem(
                    id=row[0], type=row[1], title=row[2], content=row[3],
                    tags=json.loads(row[4]), created_at=row[5], updated_at=row[6],
                    priority=row[7], status=row[8], metadata=json.loads(row[9])
                )
                items.append(item)
            
            return items
                
        except Exception as e:
            logger.error(f"Error retrieving knowledge items: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def add_qa_item(self, question: str, answer: str, tags: List[str] = None) -> str:
        """Add a Q&A item to the knowledge base"""
        if tags is None:
            tags = []
            
        item_id = f"qa_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.datetime.now().isoformat()
        
        item = KnowledgeItem(
            id=item_id,
            type="qa",
            title=question[:100] + "..." if len(question) > 100 else question,
            content=f"Q: {question}\n\nA: {answer}",
            tags=tags,
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"question": question, "answer": answer}
        )
        
        self.add_knowledge_item(item)
        return item_id
    
    def add_idea(self, title: str, description: str, priority: int = 1, 
                tags: List[str] = None) -> str:
        """Add an idea to the knowledge base"""
        if tags is None:
            tags = []
            
        item_id = f"idea_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.datetime.now().isoformat()
        
        item = KnowledgeItem(
            id=item_id,
            type="idea",
            title=title,
            content=description,
            tags=tags,
            created_at=timestamp,
            updated_at=timestamp,
            priority=priority,
            metadata={"implementation_status": "proposed"}
        )
        
        self.add_knowledge_item(item)
        return item_id
    
    def add_feedback(self, title: str, feedback: str, feedback_type: str = "general",
                    tags: List[str] = None) -> str:
        """Add feedback to the knowledge base"""
        if tags is None:
            tags = []
            
        item_id = f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.datetime.now().isoformat()
        
        item = KnowledgeItem(
            id=item_id,
            type="feedback",
            title=title,
            content=feedback,
            tags=tags + [feedback_type],
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"feedback_type": feedback_type}
        )
        
        self.add_knowledge_item(item)
        return item_id
    
    def add_technique(self, title: str, description: str, usage_example: str = "",
                     tags: List[str] = None) -> str:
        """Add a new technique or tool to the knowledge base"""
        if tags is None:
            tags = []
            
        item_id = f"technique_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.datetime.now().isoformat()
        
        content = description
        if usage_example:
            content += f"\n\n## Usage Example\n{usage_example}"
        
        item = KnowledgeItem(
            id=item_id,
            type="technique",
            title=title,
            content=content,
            tags=tags,
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"has_example": bool(usage_example)}
        )
        
        self.add_knowledge_item(item)
        return item_id
    
    def create_weekly_documentation(self) -> str:
        """Create weekly documentation summary"""
        week_start = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Collect items from the past week
        week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get new techniques
            cursor = conn.execute("""
                SELECT title, content FROM knowledge_items 
                WHERE type = 'technique' AND created_at > ?
                ORDER BY created_at DESC
            """, (week_ago,))
            techniques = [{"title": row[0], "content": row[1][:200] + "..."} 
                         for row in cursor.fetchall()]
            
            # Get new ideas
            cursor = conn.execute("""
                SELECT title, content FROM knowledge_items 
                WHERE type = 'idea' AND created_at > ?
                ORDER BY priority DESC, created_at DESC
            """, (week_ago,))
            ideas = [{"title": row[0], "content": row[1][:200] + "..."} 
                    for row in cursor.fetchall()]
            
            # Get feedback
            cursor = conn.execute("""
                SELECT title, content FROM knowledge_items 
                WHERE type = 'feedback' AND created_at > ?
                ORDER BY created_at DESC
            """, (week_ago,))
            feedback = [{"title": row[0], "content": row[1][:200] + "..."} 
                       for row in cursor.fetchall()]
        finally:
            if conn:
                conn.close()
        
        # Create weekly documentation
        weekly_doc = WeeklyDocumentation(
            week_start=week_start,
            tools_discovered=[],  # Will be populated by external integrations
            techniques_learned=techniques,
            improvements_made=ideas,
            feedback_collected=feedback,
            next_week_goals=self._generate_next_week_goals(techniques, ideas, feedback)
        )
        
        # Save to database
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO weekly_docs 
                (week_start, tools_discovered, techniques_learned, improvements_made, 
                 feedback_collected, next_week_goals)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                weekly_doc.week_start,
                json.dumps(weekly_doc.tools_discovered),
                json.dumps(weekly_doc.techniques_learned),
                json.dumps(weekly_doc.improvements_made),
                json.dumps(weekly_doc.feedback_collected),
                json.dumps(weekly_doc.next_week_goals)
            ))
            conn.commit()
        finally:
            if conn:
                conn.close()
        
        # Save as markdown file
        self._save_weekly_doc_as_markdown(weekly_doc)
        
        logger.info(f"Created weekly documentation for {week_start}")
        return week_start
    
    def _generate_next_week_goals(self, techniques: List[Dict], ideas: List[Dict], 
                                 feedback: List[Dict]) -> List[str]:
        """Generate goals for next week based on current knowledge"""
        goals = []
        
        if techniques:
            goals.append(f"Implement and test {len(techniques)} new techniques discovered")
        
        if ideas:
            high_priority_ideas = [i for i in ideas if "priority" in i.get("content", "")]
            if high_priority_ideas:
                goals.append(f"Develop {len(high_priority_ideas)} high-priority ideas")
        
        if feedback:
            goals.append("Address feedback items and implement improvements")
        
        # Default goals
        goals.extend([
            "Continue knowledge collection and documentation",
            "Review and update existing techniques",
            "Identify areas for improvement and innovation"
        ])
        
        return goals[:5]  # Limit to 5 goals
    
    def _save_weekly_doc_as_markdown(self, doc: WeeklyDocumentation):
        """Save weekly documentation as markdown file"""
        filename = f"week_{doc.week_start}.md"
        file_path = self.base_path / "weekly_docs" / filename
        
        content = f"""# Weekly Documentation - {doc.week_start}

## Tools Discovered
{self._format_items_list(doc.tools_discovered)}

## Techniques Learned
{self._format_items_list(doc.techniques_learned)}

## Improvements Made
{self._format_items_list(doc.improvements_made)}

## Feedback Collected
{self._format_items_list(doc.feedback_collected)}

## Next Week Goals
{chr(10).join(f"- {goal}" for goal in doc.next_week_goals)}

---
*Generated automatically by FounderForge Knowledge Manager*
"""
        
        file_path.write_text(content)
    
    def _format_items_list(self, items: List[Dict]) -> str:
        """Format list of items for markdown display"""
        if not items:
            return "- No items this week\n"
        
        formatted = []
        for item in items:
            title = item.get("title", "Untitled")
            content = item.get("content", "")[:100] + "..." if len(item.get("content", "")) > 100 else item.get("content", "")
            formatted.append(f"- **{title}**: {content}")
        
        return "\n".join(formatted) + "\n"
    
    def get_weekly_documentation(self, week_start: str = None) -> Optional[WeeklyDocumentation]:
        """Retrieve weekly documentation"""
        if week_start is None:
            week_start = datetime.datetime.now().strftime('%Y-%m-%d')
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT * FROM weekly_docs WHERE week_start = ?
            """, (week_start,))
            
            row = cursor.fetchone()
            if row:
                return WeeklyDocumentation(
                    week_start=row[0],
                    tools_discovered=json.loads(row[1]),
                    techniques_learned=json.loads(row[2]),
                    improvements_made=json.loads(row[3]),
                    feedback_collected=json.loads(row[4]),
                    next_week_goals=json.loads(row[5])
                )
            
            return None
                
        except Exception as e:
            logger.error(f"Error retrieving weekly documentation: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def search_knowledge(self, query: str, item_type: str = None) -> List[KnowledgeItem]:
        """Search knowledge items by content or title"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            if item_type:
                cursor = conn.execute("""
                    SELECT * FROM knowledge_items 
                    WHERE type = ? AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)
                    ORDER BY priority DESC, updated_at DESC
                """, (item_type, f"%{query}%", f"%{query}%", f"%{query}%"))
            else:
                cursor = conn.execute("""
                    SELECT * FROM knowledge_items 
                    WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                    ORDER BY priority DESC, updated_at DESC
                """, (f"%{query}%", f"%{query}%", f"%{query}%"))
            
            items = []
            for row in cursor.fetchall():
                item = KnowledgeItem(
                    id=row[0], type=row[1], title=row[2], content=row[3],
                    tags=json.loads(row[4]), created_at=row[5], updated_at=row[6],
                    priority=row[7], status=row[8], metadata=json.loads(row[9])
                )
                items.append(item)
            
            return items
                
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def generate_self_feedback(self, system_metrics: Dict[str, Any]) -> str:
        """Generate self-feedback based on system metrics"""
        feedback_items = []
        
        # Analyze performance metrics
        if "response_time" in system_metrics:
            avg_response_time = system_metrics["response_time"]
            if avg_response_time > 5.0:
                feedback_items.append({
                    "type": "performance",
                    "issue": "High response time",
                    "suggestion": "Optimize context assembly and agent workflows"
                })
        
        # Analyze accuracy metrics
        if "accuracy" in system_metrics:
            accuracy = system_metrics["accuracy"]
            if accuracy < 0.9:
                feedback_items.append({
                    "type": "accuracy",
                    "issue": "Below target accuracy",
                    "suggestion": "Review and improve prompt engineering and context quality"
                })
        
        # Analyze memory usage
        if "memory_usage" in system_metrics:
            memory_usage = system_metrics["memory_usage"]
            if memory_usage > 0.8:  # 80% threshold
                feedback_items.append({
                    "type": "resource",
                    "issue": "High memory usage",
                    "suggestion": "Implement memory optimization and garbage collection"
                })
        
        # Create feedback entries
        for item in feedback_items:
            title = f"{item['type'].title()} Issue: {item['issue']}"
            self.add_feedback(
                title=title,
                feedback=f"Issue: {item['issue']}\nSuggestion: {item['suggestion']}",
                feedback_type=item['type'],
                tags=["automated", "self-feedback", item['type']]
            )
        
        return f"Generated {len(feedback_items)} self-feedback items"
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Count by type
            cursor = conn.execute("""
                SELECT type, COUNT(*) FROM knowledge_items 
                WHERE status = 'active'
                GROUP BY type
            """)
            type_counts = dict(cursor.fetchall())
            
            # Count by priority
            cursor = conn.execute("""
                SELECT priority, COUNT(*) FROM knowledge_items 
                WHERE status = 'active'
                GROUP BY priority
            """)
            priority_counts = dict(cursor.fetchall())
            
            # Recent activity
            week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
            cursor = conn.execute("""
                SELECT COUNT(*) FROM knowledge_items 
                WHERE created_at > ?
            """, (week_ago,))
            recent_items = cursor.fetchone()[0]
            
            return {
                "total_items": sum(type_counts.values()),
                "by_type": type_counts,
                "by_priority": priority_counts,
                "recent_items": recent_items,
                "last_updated": datetime.datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {}
        finally:
            if conn:
                conn.close()