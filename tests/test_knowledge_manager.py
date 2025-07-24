"""
Unit tests for Knowledge Manager

Tests the knowledge management system including:
- Knowledge item creation and retrieval
- Q&A, ideas, feedback, and technique management
- Weekly documentation generation
- Self-feedback mechanisms
- Search functionality
"""

import unittest
import tempfile
import shutil
import os
import json
import datetime
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_manager import KnowledgeManager, KnowledgeItem, WeeklyDocumentation

class TestKnowledgeManager(unittest.TestCase):
    """Test cases for KnowledgeManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.knowledge_manager = KnowledgeManager(base_path=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_directory_creation(self):
        """Test that all necessary directories are created"""
        expected_dirs = [
            "qa", "ideas", "feedback", "techniques", "weekly_docs", "templates"
        ]
        
        for dir_name in expected_dirs:
            dir_path = Path(self.test_dir) / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
            
            # Check README files
            readme_path = dir_path / "README.md"
            self.assertTrue(readme_path.exists(), f"README.md should exist in {dir_name}")
    
    def test_add_qa_item(self):
        """Test adding Q&A items"""
        question = "How do I optimize database queries?"
        answer = "Use proper indexing and avoid N+1 queries"
        tags = ["database", "optimization"]
        
        item_id = self.knowledge_manager.add_qa_item(question, answer, tags)
        
        self.assertIsNotNone(item_id)
        self.assertTrue(item_id.startswith("qa_"))
        
        # Retrieve and verify
        items = self.knowledge_manager.get_knowledge_items(item_type="qa")
        self.assertEqual(len(items), 1)
        
        item = items[0]
        self.assertEqual(item.type, "qa")
        self.assertIn(question, item.content)
        self.assertIn(answer, item.content)
        self.assertEqual(item.tags, tags)
    
    def test_add_idea(self):
        """Test adding ideas"""
        title = "Implement caching layer"
        description = "Add Redis caching to improve performance"
        priority = 4
        tags = ["performance", "caching"]
        
        item_id = self.knowledge_manager.add_idea(title, description, priority, tags)
        
        self.assertIsNotNone(item_id)
        self.assertTrue(item_id.startswith("idea_"))
        
        # Retrieve and verify
        items = self.knowledge_manager.get_knowledge_items(item_type="idea")
        self.assertEqual(len(items), 1)
        
        item = items[0]
        self.assertEqual(item.type, "idea")
        self.assertEqual(item.title, title)
        self.assertEqual(item.content, description)
        self.assertEqual(item.priority, priority)
        self.assertEqual(item.tags, tags)
    
    def test_add_feedback(self):
        """Test adding feedback"""
        title = "Response time improvement needed"
        feedback = "Users report slow response times during peak hours"
        feedback_type = "performance"
        tags = ["performance", "user-feedback"]
        
        item_id = self.knowledge_manager.add_feedback(title, feedback, feedback_type, tags)
        
        self.assertIsNotNone(item_id)
        self.assertTrue(item_id.startswith("feedback_"))
        
        # Retrieve and verify
        items = self.knowledge_manager.get_knowledge_items(item_type="feedback")
        self.assertEqual(len(items), 1)
        
        item = items[0]
        self.assertEqual(item.type, "feedback")
        self.assertEqual(item.title, title)
        self.assertEqual(item.content, feedback)
        self.assertIn(feedback_type, item.tags)
    
    def test_add_technique(self):
        """Test adding techniques"""
        title = "Database Connection Pooling"
        description = "Use connection pooling to improve database performance"
        usage_example = "pool = ConnectionPool(max_connections=10)"
        tags = ["database", "performance"]
        
        item_id = self.knowledge_manager.add_technique(title, description, usage_example, tags)
        
        self.assertIsNotNone(item_id)
        self.assertTrue(item_id.startswith("technique_"))
        
        # Retrieve and verify
        items = self.knowledge_manager.get_knowledge_items(item_type="technique")
        self.assertEqual(len(items), 1)
        
        item = items[0]
        self.assertEqual(item.type, "technique")
        self.assertEqual(item.title, title)
        self.assertIn(description, item.content)
        self.assertIn(usage_example, item.content)
        self.assertEqual(item.tags, tags)
    
    def test_search_knowledge(self):
        """Test knowledge search functionality"""
        # Add some test items
        self.knowledge_manager.add_qa_item(
            "How to optimize database?", 
            "Use indexing", 
            ["database"]
        )
        self.knowledge_manager.add_idea(
            "Database caching", 
            "Implement Redis caching", 
            tags=["database", "cache"]
        )
        self.knowledge_manager.add_technique(
            "SQL Optimization", 
            "Optimize SQL queries for better performance", 
            tags=["database", "sql"]
        )
        
        # Search for database-related items
        results = self.knowledge_manager.search_knowledge("database")
        self.assertEqual(len(results), 3)
        
        # Search within specific type
        qa_results = self.knowledge_manager.search_knowledge("database", item_type="qa")
        self.assertEqual(len(qa_results), 1)
        self.assertEqual(qa_results[0].type, "qa")
        
        # Search for non-existent term
        no_results = self.knowledge_manager.search_knowledge("nonexistent")
        self.assertEqual(len(no_results), 0)
    
    def test_generate_self_feedback(self):
        """Test self-feedback generation"""
        system_metrics = {
            "response_time": 6.0,  # Above threshold
            "accuracy": 0.85,      # Below threshold
            "memory_usage": 0.9,   # Above threshold
            "error_rate": 0.02
        }
        
        result = self.knowledge_manager.generate_self_feedback(system_metrics)
        
        self.assertIn("feedback items", result)
        
        # Check that feedback items were created
        feedback_items = self.knowledge_manager.get_knowledge_items(item_type="feedback")
        self.assertGreater(len(feedback_items), 0)
        
        # Verify feedback content
        feedback_titles = [item.title for item in feedback_items]
        self.assertTrue(any("response time" in title.lower() for title in feedback_titles))
        self.assertTrue(any("accuracy" in title.lower() for title in feedback_titles))
        self.assertTrue(any("memory" in title.lower() for title in feedback_titles))
    
    def test_weekly_documentation(self):
        """Test weekly documentation generation"""
        # Add some test items first
        self.knowledge_manager.add_technique(
            "New Testing Framework", 
            "Discovered pytest-asyncio for async testing"
        )
        self.knowledge_manager.add_idea(
            "API Improvement", 
            "Implement GraphQL API"
        )
        self.knowledge_manager.add_feedback(
            "User Interface Feedback", 
            "Users want dark mode option"
        )
        
        # Generate weekly documentation
        week_start = self.knowledge_manager.create_weekly_documentation()
        
        self.assertIsNotNone(week_start)
        
        # Retrieve and verify
        weekly_doc = self.knowledge_manager.get_weekly_documentation(week_start)
        self.assertIsNotNone(weekly_doc)
        self.assertEqual(weekly_doc.week_start, week_start)
        self.assertGreater(len(weekly_doc.techniques_learned), 0)
        self.assertGreater(len(weekly_doc.improvements_made), 0)
        self.assertGreater(len(weekly_doc.feedback_collected), 0)
        self.assertGreater(len(weekly_doc.next_week_goals), 0)
        
        # Check that markdown file was created
        markdown_file = Path(self.test_dir) / "weekly_docs" / f"week_{week_start}.md"
        self.assertTrue(markdown_file.exists())
        
        # Verify markdown content
        content = markdown_file.read_text()
        self.assertIn("Weekly Documentation", content)
        self.assertIn("Tools Discovered", content)
        self.assertIn("Techniques Learned", content)
        self.assertIn("Next Week Goals", content)
    
    def test_knowledge_stats(self):
        """Test knowledge statistics"""
        # Add various items
        self.knowledge_manager.add_qa_item("Test Q", "Test A")
        self.knowledge_manager.add_idea("Test Idea", "Test Description", priority=3)
        self.knowledge_manager.add_feedback("Test Feedback", "Test Content")
        self.knowledge_manager.add_technique("Test Technique", "Test Description")
        
        stats = self.knowledge_manager.get_knowledge_stats()
        
        self.assertEqual(stats["total_items"], 4)
        self.assertEqual(stats["by_type"]["qa"], 1)
        self.assertEqual(stats["by_type"]["idea"], 1)
        self.assertEqual(stats["by_type"]["feedback"], 1)
        self.assertEqual(stats["by_type"]["technique"], 1)
        self.assertEqual(stats["by_priority"][3], 1)  # The idea with priority 3
        self.assertEqual(stats["recent_items"], 4)  # All items are recent
    
    def test_knowledge_item_file_storage(self):
        """Test that knowledge items are also saved as JSON files"""
        title = "Test Item"
        description = "Test description"
        
        item_id = self.knowledge_manager.add_idea(title, description)
        
        # Check that JSON file was created
        ideas_dir = Path(self.test_dir) / "ideas"
        json_files = list(ideas_dir.glob("*.json"))
        self.assertEqual(len(json_files), 1)
        
        # Verify JSON content
        with open(json_files[0]) as f:
            data = json.load(f)
        
        self.assertEqual(data["id"], item_id)
        self.assertEqual(data["type"], "idea")
        self.assertEqual(data["title"], title)
        self.assertEqual(data["content"], description)
    
    def test_knowledge_item_update(self):
        """Test updating knowledge items"""
        # Add initial item
        item_id = self.knowledge_manager.add_idea("Original Title", "Original Description")
        
        # Create updated item
        updated_item = KnowledgeItem(
            id=item_id,
            type="idea",
            title="Updated Title",
            content="Updated Description",
            tags=["updated"],
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
            priority=5,
            status="active"
        )
        
        # Update the item
        success = self.knowledge_manager.add_knowledge_item(updated_item)
        self.assertTrue(success)
        
        # Verify update
        items = self.knowledge_manager.get_knowledge_items(item_type="idea")
        self.assertEqual(len(items), 1)
        
        item = items[0]
        self.assertEqual(item.title, "Updated Title")
        self.assertEqual(item.content, "Updated Description")
        self.assertEqual(item.priority, 5)
        self.assertEqual(item.tags, ["updated"])

if __name__ == '__main__':
    unittest.main()