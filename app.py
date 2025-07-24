"""
FounderForge AI Cofounder - Streamlit Web Interface
Main chat interface with conversation history, user profile management, and memory controls.
"""

import streamlit as st
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import asyncio
import time
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components
try:
    from src.database import initialize_database, get_db_manager
    from src.memory_repository import get_memory_repository
    from src.context_manager import ContextAssembler, TokenManager
    from src.agents import AgentOrchestrator
    from src.gemini_client import GeminiClient
    from src.confidence_manager import ConfidenceManager
    from src.models import (
        UserContext, Message, Memory, MemoryType, BusinessInfo, 
        UserPreferences, Response, TokenUsage
    )
    from config.settings import settings
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


class FounderForgeApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.initialize_components()
        self.setup_session_state()
    
    def initialize_components(self):
        """Initialize core system components."""
        try:
            # Initialize database
            if not initialize_database():
                st.error("Failed to initialize database")
                st.stop()
            
            # Initialize core components
            self.db_manager = get_db_manager()
            self.memory_repository = get_memory_repository()
            self.context_manager = ContextAssembler()
            self.token_manager = TokenManager()
            self.gemini_client = GeminiClient()
            self.confidence_manager = ConfidenceManager()
            
            # Initialize agent orchestrator
            self.agent_orchestrator = AgentOrchestrator(
                gemini_client=self.gemini_client,
                context_manager=self.context_manager,
                confidence_manager=self.confidence_manager
            )
            
            logger.info("Application components initialized successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize application: {e}")
            st.stop()
    
    def setup_session_state(self):
        """Initialize Streamlit session state."""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'user_context' not in st.session_state:
            st.session_state.user_context = None
        
        if 'conversation_started' not in st.session_state:
            st.session_state.conversation_started = False
    
    def run(self):
        """Run the main application."""
        st.set_page_config(
            page_title="FounderForge AI Cofounder",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main header
        st.title("🚀 FounderForge AI Cofounder")
        st.markdown("*Your AI-powered virtual cofounder for strategy, funding, and operations*")
        
        # Sidebar for user management and controls
        self.render_sidebar()
        
        # Main chat interface
        self.render_chat_interface()
        
        # Footer with system status
        self.render_footer()
    
    def render_sidebar(self):
        """Render the sidebar with user controls."""
        with st.sidebar:
            st.header("👤 User Profile")
            
            # User identification
            current_user = st.text_input(
                "User ID",
                value=st.session_state.user_id,
                help="Your unique user identifier"
            )
            
            if current_user != st.session_state.user_id:
                st.session_state.user_id = current_user
                st.session_state.user_context = None
                st.rerun()
            
            # Load or create user context
            if st.session_state.user_context is None:
                st.session_state.user_context = self.load_user_context(st.session_state.user_id)
            
            # User profile management
            self.render_user_profile()
            
            st.divider()
            
            # Memory management
            self.render_memory_controls()
            
            st.divider()
            
            # System controls
            self.render_system_controls()
    
    def render_user_profile(self):
        """Render user profile management section."""
        st.subheader("📋 Profile Settings")
        
        user_context = st.session_state.user_context
        
        with st.expander("Business Information", expanded=False):
            company_name = st.text_input(
                "Company Name",
                value=user_context.business_info.company_name,
                key="company_name"
            )
            
            industry = st.selectbox(
                "Industry",
                options=["", "Technology", "Healthcare", "Finance", "E-commerce", "Education", "Other"],
                index=0 if not user_context.business_info.industry else 
                      ["", "Technology", "Healthcare", "Finance", "E-commerce", "Education", "Other"].index(user_context.business_info.industry),
                key="industry"
            )
            
            stage = st.selectbox(
                "Business Stage",
                options=["", "Idea", "MVP", "Growth", "Scale"],
                index=0 if not user_context.business_info.stage else
                      ["", "Idea", "MVP", "Growth", "Scale"].index(user_context.business_info.stage),
                key="stage"
            )
            
            team_size = st.number_input(
                "Team Size",
                min_value=0,
                max_value=1000,
                value=user_context.business_info.team_size,
                key="team_size"
            )
            
            description = st.text_area(
                "Business Description",
                value=user_context.business_info.description,
                height=100,
                key="description"
            )
        
        with st.expander("Preferences", expanded=False):
            communication_style = st.selectbox(
                "Communication Style",
                options=["professional", "casual", "technical"],
                index=["professional", "casual", "technical"].index(user_context.preferences.communication_style),
                key="communication_style"
            )
            
            response_length = st.selectbox(
                "Response Length",
                options=["short", "medium", "long"],
                index=["short", "medium", "long"].index(user_context.preferences.response_length),
                key="response_length"
            )
            
            focus_areas = st.multiselect(
                "Focus Areas",
                options=["Strategy", "Funding", "Operations", "Marketing", "Product", "Team"],
                default=user_context.preferences.focus_areas,
                key="focus_areas"
            )
        
        # Goals management
        st.subheader("🎯 Goals")
        
        # Display current goals
        if user_context.goals:
            for i, goal in enumerate(user_context.goals):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{i+1}. {goal}")
                with col2:
                    if st.button("❌", key=f"remove_goal_{i}", help="Remove goal"):
                        user_context.goals.pop(i)
                        self.save_user_context(user_context)
                        st.rerun()
        
        # Add new goal
        new_goal = st.text_input("Add New Goal", key="new_goal")
        if st.button("Add Goal") and new_goal.strip():
            user_context.goals.append(new_goal.strip())
            self.save_user_context(user_context)
            st.rerun()
        
        # Save profile changes
        if st.button("💾 Save Profile", type="primary"):
            self.update_user_profile(user_context)
            st.success("Profile updated successfully!")
    
    def render_memory_controls(self):
        """Render memory management controls."""
        st.subheader("🧠 Memory Management")
        
        user_id = st.session_state.user_id
        
        # Memory statistics
        try:
            memory_stats = self.memory_repository.get_memory_stats(user_id)
            
            st.metric("Total Memories", memory_stats.get("total_memories", 0))
            
            if memory_stats.get("by_type"):
                for memory_type, stats in memory_stats["by_type"].items():
                    st.text(f"{memory_type}: {stats['count']} (avg confidence: {stats['avg_confidence']:.2f})")
        
        except Exception as e:
            st.error(f"Failed to load memory stats: {e}")
        
        # Memory actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 View Memories"):
                self.show_memory_viewer()
        
        with col2:
            if st.button("🗑️ Clear Memories"):
                self.show_memory_deletion_interface()
    
    def render_system_controls(self):
        """Render system control buttons."""
        st.subheader("⚙️ System Controls")
        
        # Token usage
        try:
            session_usage = self.token_manager.get_session_usage()
            st.metric("Session Tokens", session_usage.total_tokens)
            
            daily_usage = self.token_manager.get_daily_usage(st.session_state.user_id)
            if daily_usage:
                st.metric("Daily Tokens", daily_usage.total_tokens)
        
        except Exception as e:
            st.error(f"Failed to load token usage: {e}")
        
        # System actions
        if st.button("🔄 Reset Session"):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.success("Session reset!")
            st.rerun()
        
        if st.button("📊 System Status"):
            self.show_system_status()
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display conversation history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show metadata for assistant messages
                    if message["role"] == "assistant" and "metadata" in message:
                        with st.expander("Response Details", expanded=False):
                            metadata = message["metadata"]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Confidence", f"{metadata.get('confidence', 0):.2f}")
                            with col2:
                                st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
                            with col3:
                                st.metric("Tokens Used", metadata.get('token_usage', {}).get('total_tokens', 0))
                            
                            if metadata.get('sources'):
                                st.text("Sources: " + ", ".join(metadata['sources']))
                            
                            if metadata.get('fallback_used'):
                                st.warning("⚠️ Fallback response used due to low confidence")
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your business..."):
            self.handle_user_message(prompt)
    
    def render_footer(self):
        """Render footer with system status."""
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.text(f"User: {st.session_state.user_id[:8]}...")
        
        with col2:
            st.text(f"Messages: {len(st.session_state.messages)}")
        
        with col3:
            st.text(f"Status: {'🟢 Active' if st.session_state.conversation_started else '🟡 Ready'}")
    
    def handle_user_message(self, message: str):
        """Handle user message and generate response."""
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(message)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = self.generate_response(message)
                
                st.markdown(response["content"])
                
                # Add response to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": response.get("metadata", {})
                })
                
                # Save conversation to context manager
                self.save_conversation_message(message, response["content"])
        
        st.session_state.conversation_started = True
    
    def generate_response(self, user_message: str) -> Dict[str, Any]:
        """Generate AI response to user message."""
        start_time = time.time()
        
        try:
            user_id = st.session_state.user_id
            user_context = st.session_state.user_context
            
            # Assemble context
            context = self.context_manager.assemble_context(user_id, user_message)
            
            # Check if this requires agent workflow
            if self.should_use_agent_workflow(user_message):
                # Use agent orchestrator for complex tasks
                workflow_result = asyncio.run(
                    self.agent_orchestrator.execute_workflow(
                        workflow_type="general",
                        user_id=user_id,
                        task_data={"user_query": user_message, "context": context.to_dict()}
                    )
                )
                
                response_content = workflow_result.result_data.get("final_response", "I apologize, but I couldn't process your request.")
                confidence = workflow_result.confidence_score
                sources = ["agent_workflow"]
                fallback_used = not workflow_result.success
                
            else:
                # Direct Gemini API call for simple queries
                prompt = self.build_conversation_prompt(context, user_message)
                
                gemini_response = self.gemini_client.generate_content(
                    prompt,
                    temperature=0.7,
                    max_output_tokens=20000
                )
                
                response_content = gemini_response.content
                confidence = gemini_response.confidence
                sources = context.sources
                fallback_used = False
            
            processing_time = time.time() - start_time
            
            # Create response object
            response = Response(
                content=response_content,
                confidence=confidence,
                sources=sources,
                fallback_used=fallback_used,
                processing_time=processing_time,
                token_usage=TokenUsage(total_tokens=len(user_message + response_content) // 4)
            )
            
            # Log token usage
            self.token_manager.log_token_usage(
                user_id,
                response.token_usage,
                "chat_response"
            )
            
            return {
                "content": response.content,
                "metadata": {
                    "confidence": response.confidence,
                    "processing_time": response.processing_time,
                    "sources": response.sources,
                    "fallback_used": response.fallback_used,
                    "token_usage": response.token_usage.to_dict() if hasattr(response.token_usage, 'to_dict') else {
                        "total_tokens": response.token_usage.total_tokens,
                        "prompt_tokens": response.token_usage.prompt_tokens,
                        "completion_tokens": response.token_usage.completion_tokens
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            
            return {
                "content": "I apologize, but I encountered an error processing your request. Please try again.",
                "metadata": {
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "sources": [],
                    "fallback_used": True,
                    "error": str(e)
                }
            }
    
    def should_use_agent_workflow(self, message: str) -> bool:
        """Determine if message requires agent workflow processing."""
        workflow_keywords = [
            "plan", "strategy", "funding", "analyze", "evaluate",
            "workflow", "process", "validate", "review"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in workflow_keywords)
    
    def build_conversation_prompt(self, context, user_message: str) -> str:
        """Build prompt for conversation with context."""
        context_str = context.to_prompt_string()
        
        prompt = f"""
{context_str}

Current User Message: {user_message}

Please provide a helpful, accurate response based on the context above. 
Maintain a {context.user_context.preferences.communication_style} tone and provide a {context.user_context.preferences.response_length} response.

Focus on actionable insights and practical guidance for the user's business needs.
"""
        
        return prompt
    
    def load_user_context(self, user_id: str) -> UserContext:
        """Load user context from database."""
        try:
            # Query user from database
            user_query = "SELECT * FROM users WHERE id = ?"
            user_result = self.db_manager.execute_query(user_query, (user_id,))
            
            if user_result:
                user_row = user_result[0]
                preferences_data = {}
                
                if user_row['preferences']:
                    try:
                        preferences_data = json.loads(user_row['preferences'])
                    except json.JSONDecodeError:
                        pass
                
                business_info = BusinessInfo()
                preferences = UserPreferences(**preferences_data)
                
                # Load goals from memories
                memory_query = """
                    SELECT content FROM memories 
                    WHERE user_id = ? AND memory_type = 'LONG_TERM'
                    ORDER BY created_at DESC LIMIT 5
                """
                memory_results = self.db_manager.execute_query(memory_query, (user_id,))
                goals = [row['content'] for row in (memory_results or [])]
                
                return UserContext(
                    user_id=user_id,
                    goals=goals,
                    business_info=business_info,
                    preferences=preferences
                )
            else:
                # Create new user
                return self.create_new_user(user_id)
                
        except Exception as e:
            logger.error(f"Failed to load user context: {e}")
            return UserContext(user_id=user_id)
    
    def create_new_user(self, user_id: str) -> UserContext:
        """Create new user in database."""
        try:
            query = """
                INSERT INTO users (id, name, email, preferences)
                VALUES (?, ?, ?, ?)
            """
            
            default_preferences = UserPreferences()
            preferences_json = json.dumps({
                "communication_style": default_preferences.communication_style,
                "response_length": default_preferences.response_length,
                "focus_areas": default_preferences.focus_areas,
                "timezone": default_preferences.timezone,
                "language": default_preferences.language
            })
            
            self.db_manager.execute_update(query, (user_id, "", "", preferences_json))
            
            logger.info(f"Created new user: {user_id}")
            
            return UserContext(
                user_id=user_id,
                business_info=BusinessInfo(),
                preferences=default_preferences
            )
            
        except Exception as e:
            logger.error(f"Failed to create new user: {e}")
            return UserContext(user_id=user_id)
    
    def update_user_profile(self, user_context: UserContext):
        """Update user profile with current form values."""
        try:
            # Update business info
            user_context.business_info.company_name = st.session_state.get("company_name", "")
            user_context.business_info.industry = st.session_state.get("industry", "")
            user_context.business_info.stage = st.session_state.get("stage", "")
            user_context.business_info.team_size = st.session_state.get("team_size", 0)
            user_context.business_info.description = st.session_state.get("description", "")
            
            # Update preferences
            user_context.preferences.communication_style = st.session_state.get("communication_style", "professional")
            user_context.preferences.response_length = st.session_state.get("response_length", "medium")
            user_context.preferences.focus_areas = st.session_state.get("focus_areas", [])
            
            # Save to database
            self.save_user_context(user_context)
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
            st.error("Failed to update profile")
    
    def save_user_context(self, user_context: UserContext):
        """Save user context to database."""
        try:
            preferences_json = json.dumps({
                "communication_style": user_context.preferences.communication_style,
                "response_length": user_context.preferences.response_length,
                "focus_areas": user_context.preferences.focus_areas,
                "timezone": user_context.preferences.timezone,
                "language": user_context.preferences.language
            })
            
            query = """
                UPDATE users 
                SET preferences = ?
                WHERE id = ?
            """
            
            self.db_manager.execute_update(query, (preferences_json, user_context.user_id))
            
            # Save goals as long-term memories
            for goal in user_context.goals:
                memory = Memory(
                    user_id=user_context.user_id,
                    content=goal,
                    memory_type=MemoryType.LONG_TERM,
                    confidence=1.0
                )
                self.memory_repository.create_memory(memory, confirm=False)
            
        except Exception as e:
            logger.error(f"Failed to save user context: {e}")
    
    def save_conversation_message(self, user_message: str, assistant_response: str):
        """Save conversation message to chat history."""
        try:
            user_id = st.session_state.user_id
            
            # Save to database
            query = """
                INSERT INTO conversations (id, user_id, message, response, token_usage)
                VALUES (?, ?, ?, ?, ?)
            """
            
            conversation_id = str(uuid.uuid4())
            token_usage = (len(user_message) + len(assistant_response)) // 4
            
            self.db_manager.execute_update(query, (
                conversation_id, user_id, user_message, assistant_response, token_usage
            ))
            
            # Save to context manager
            user_msg = Message(content=user_message, role="user")
            assistant_msg = Message(content=assistant_response, role="assistant")
            
            self.context_manager.save_chat_message(user_id, user_msg)
            self.context_manager.save_chat_message(user_id, assistant_msg)
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def show_memory_viewer(self):
        """Show memory viewer in a modal."""
        with st.expander("💭 Memory Viewer", expanded=True):
            user_id = st.session_state.user_id
            
            # Memory type filter
            memory_type_filter = st.selectbox(
                "Filter by Type",
                options=["All", "SHORT_TERM", "LONG_TERM"],
                key="memory_type_filter"
            )
            
            try:
                # Get memories
                if memory_type_filter == "All":
                    memories = self.memory_repository.get_memories_by_user(user_id, limit=50)
                else:
                    memories = self.memory_repository.get_memories_by_user(
                        user_id, 
                        memory_type=MemoryType(memory_type_filter),
                        limit=50
                    )
                
                if memories:
                    for memory in memories:
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.text(memory.content[:100] + "..." if len(memory.content) > 100 else memory.content)
                            
                            with col2:
                                st.text(f"Confidence: {memory.confidence:.2f}")
                            
                            with col3:
                                if st.button("🗑️", key=f"delete_memory_{memory.id}", help="Delete memory"):
                                    if self.memory_repository.delete_memory(memory.id, user_id, confirm=False):
                                        st.success("Memory deleted")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete memory")
                            
                            st.text(f"Created: {memory.created_at.strftime('%Y-%m-%d %H:%M')}")
                            st.divider()
                else:
                    st.info("No memories found")
                    
            except Exception as e:
                st.error(f"Failed to load memories: {e}")
    
    def show_memory_deletion_interface(self):
        """Show memory deletion interface with confirmation."""
        with st.expander("🗑️ Memory Deletion", expanded=True):
            st.warning("⚠️ This action cannot be undone!")
            
            user_id = st.session_state.user_id
            
            # Deletion options
            deletion_type = st.radio(
                "What would you like to delete?",
                options=["All Memories", "Short-term Only", "Long-term Only"],
                key="deletion_type"
            )
            
            # Confirmation
            confirm_text = st.text_input(
                f"Type 'DELETE' to confirm deletion of {deletion_type.lower()}:",
                key="deletion_confirm"
            )
            
            if st.button("🗑️ Delete Memories", type="secondary"):
                if confirm_text == "DELETE":
                    try:
                        if deletion_type == "All Memories":
                            count = self.memory_repository.delete_user_memories(user_id, confirm=False)
                        elif deletion_type == "Short-term Only":
                            count = self.memory_repository.delete_user_memories(
                                user_id, 
                                memory_type=MemoryType.SHORT_TERM,
                                confirm=False
                            )
                        else:  # Long-term Only
                            count = self.memory_repository.delete_user_memories(
                                user_id,
                                memory_type=MemoryType.LONG_TERM,
                                confirm=False
                            )
                        
                        st.success(f"Deleted {count} memories")
                        
                        # Clear goals if long-term memories were deleted
                        if deletion_type in ["All Memories", "Long-term Only"]:
                            st.session_state.user_context.goals = []
                        
                    except Exception as e:
                        st.error(f"Failed to delete memories: {e}")
                else:
                    st.error("Please type 'DELETE' to confirm")
    
    def show_system_status(self):
        """Show system status information."""
        with st.expander("📊 System Status", expanded=True):
            try:
                # Database status
                st.subheader("Database")
                users_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM users")
                memories_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM memories")
                conversations_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM conversations")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Users", users_count[0]['count'] if users_count else 0)
                with col2:
                    st.metric("Memories", memories_count[0]['count'] if memories_count else 0)
                with col3:
                    st.metric("Conversations", conversations_count[0]['count'] if conversations_count else 0)
                
                # Memory repository performance
                st.subheader("Performance")
                try:
                    avg_performance = self.memory_repository.get_avg_performance()
                    st.metric("Avg Query Time", f"{avg_performance*1000:.2f}ms")
                except AttributeError:
                    st.metric("Avg Query Time", "N/A")
                
                # Token usage
                st.subheader("Token Usage")
                session_usage = self.token_manager.get_session_usage()
                daily_usage = self.token_manager.get_daily_usage(st.session_state.user_id)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Session Tokens", session_usage.total_tokens)
                with col2:
                    st.metric("Daily Tokens", daily_usage.total_tokens if daily_usage else 0)
                
                # Component status
                st.subheader("Components")
                components = {
                    "Database": "🟢 Connected",
                    "Memory Repository": "🟢 Active", 
                    "Context Manager": "🟢 Active",
                    "Agent Orchestrator": "🟢 Active",
                    "Gemini Client": "🟢 Connected" if self.gemini_client.is_available() else "🔴 Disconnected"
                }
                
                for component, status in components.items():
                    st.text(f"{component}: {status}")
                
            except Exception as e:
                st.error(f"Failed to load system status: {e}")


def main():
    """Main application entry point."""
    try:
        app = FounderForgeApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        logger.error(f"Application startup failed: {e}")


if __name__ == "__main__":
    main()