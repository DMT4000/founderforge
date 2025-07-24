# FounderForge AI Cofounder - Setup Guide

## Overview

FounderForge is a localhost-first AI virtual cofounder system that assists entrepreneurs with strategic decision-making, funding guidance, and operational support. This guide will help you set up and run the system on your local machine.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for Gemini API calls only

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 16GB for optimal performance
- **Storage**: 5GB free space for logs and data
- **CPU**: Multi-core processor for parallel agent processing

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FounderForge
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# Required: Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Custom configurations
LOG_LEVEL=INFO
DATABASE_PATH=data/founderforge.db
VECTOR_INDEX_PATH=data/vector_index
```

### 4. Initialize the System

```bash
python src/init_db.py
```

### 5. Run the Application

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python cli.py --help
```

## Detailed Setup Instructions

### Step 1: Python Environment Setup

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv founderforge-env

# Activate virtual environment
# On Windows:
founderforge-env\Scripts\activate
# On macOS/Linux:
source founderforge-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n founderforge python=3.9
conda activate founderforge

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Gemini API Setup

1. **Get API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in your `.env` file

2. **Configure API Key:**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```

3. **Test API Connection:**
   ```bash
   python test_gemini_connection.py
   ```

### Step 3: Database Initialization

The system uses SQLite for local data storage:

```bash
# Initialize database with schema
python src/init_db.py

# Verify database creation
ls -la data/founderforge.db
```

### Step 4: Directory Structure Setup

The system will automatically create necessary directories, but you can set them up manually:

```bash
mkdir -p data/{business_data,chat_history,experiments,logs,prompts,vector_index}
mkdir -p data/{agent_logs,funding_logs,workflow_logs,security_events}
```

### Step 5: Configuration Files

#### Feature Flags Configuration
Edit `config/feature_flags.json`:
```json
{
  "enable_advanced_analytics": true,
  "enable_experimental_features": false,
  "enable_debug_logging": false,
  "enable_performance_monitoring": true
}
```

#### Security Configuration
Edit `config/security_config.json`:
```json
{
  "enable_content_filtering": true,
  "enable_pii_detection": true,
  "max_token_limit": 16000,
  "confidence_threshold": 0.8
}
```

## Running the System

### Web Interface (Streamlit)

The web interface provides a user-friendly chat interface with memory management:

```bash
streamlit run app.py
```

**Features:**
- Interactive chat interface
- User profile management
- Memory controls and deletion
- Token usage tracking
- System status monitoring

**Access:** Open http://localhost:8501 in your browser

### Command Line Interface

The CLI provides power-user features and batch processing:

```bash
# Interactive chat
python cli.py chat --user-id your_user_id

# Memory management
python cli.py memory list --user-id your_user_id
python cli.py memory delete --user-id your_user_id --type SHORT_TERM

# System management
python cli.py system status
python cli.py system database --backup backup.db

# Batch processing
python cli.py batch process --file queries.json
```

## Testing the Installation

### 1. Run System Health Check
```bash
python cli.py system status
```

### 2. Run Integration Tests
```bash
# Quick integration test
python test_integration_simple.py

# Agent orchestration test
python test_agent_orchestration.py

# Funding workflow test
python test_funding_workflow.py

# Logging and monitoring test
python test_logging_monitoring_integration.py
```

### 3. Test Web Interface
1. Start the web interface: `streamlit run app.py`
2. Open http://localhost:8501
3. Try a simple query: "Help me with my business strategy"
4. Check that responses are generated and memory is stored

### 4. Test CLI Interface
```bash
# Test basic chat
python cli.py chat --user-id test_user "What should I focus on for my startup?"

# Test memory operations
python cli.py memory list --user-id test_user
```

## Performance Optimization

### Memory and Storage
- **Database**: SQLite performs well for single-user scenarios
- **Vector Store**: FAISS index is optimized for local semantic search
- **Logs**: Automatic cleanup keeps log files manageable

### Performance Targets
- Memory retrieval: < 10ms
- Context assembly: < 2 seconds
- Agent workflows: < 30 seconds (funding), < 60 seconds (planning)
- API responses: < 5 seconds
- Token management: < 16k tokens per context

### Optimization Tips
1. **Regular Cleanup:**
   ```bash
   python cli.py memory delete --user-id your_user_id --type SHORT_TERM
   ```

2. **Monitor Performance:**
   ```bash
   python cli.py system tokens --days 7
   ```

3. **Adjust Configuration:**
   - Reduce `max_token_limit` in security config for faster processing
   - Enable `enable_performance_monitoring` for detailed metrics

## Data Management

### User Data
- **Location**: `data/founderforge.db` (SQLite database)
- **Contents**: User profiles, conversations, memories
- **Backup**: Use `python cli.py system database --backup filename.db`

### Business Data
- **Location**: `data/business_data/`
- **Format**: JSON files with user-specific business information
- **Management**: Accessible through web interface or CLI

### Logs and Monitoring
- **Location**: `data/logs/`
- **Types**: System logs, performance metrics, audit trails
- **Retention**: Configurable, default 30 days

### Vector Index
- **Location**: `data/vector_index/`
- **Purpose**: Semantic search for context retrieval
- **Maintenance**: Automatically managed by the system

## Security Considerations

### Data Privacy
- **Local Storage**: All data remains on your local machine
- **API Calls**: Only prompts are sent to Gemini API, no personal data
- **Encryption**: Consider encrypting the `data/` directory for sensitive information

### Access Control
- **Single User**: System designed for single-user local deployment
- **API Key**: Keep your Gemini API key secure in the `.env` file
- **Logs**: Audit logs track all data access for compliance

### Content Filtering
- **PII Detection**: Basic regex-based detection for common PII patterns
- **Content Safety**: Configurable filters for inappropriate content
- **Confidence Thresholds**: Fallback mechanisms for low-confidence responses

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: attempted relative import with no known parent package
```
**Solution:** Ensure you're running from the project root directory and Python path is set correctly.

#### 2. Database Connection Issues
```
sqlite3.OperationalError: database is locked
```
**Solution:** Close all applications using the database and restart the system.

#### 3. Gemini API Errors
```
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```
**Solution:** Check your API key in the `.env` file and ensure it's valid.

#### 4. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution:** Reduce batch sizes or increase system RAM.

#### 5. Port Already in Use (Streamlit)
```
OSError: [Errno 98] Address already in use
```
**Solution:** Use a different port: `streamlit run app.py --server.port 8502`

### Performance Issues

#### Slow Response Times
1. Check system resources: `python cli.py system status`
2. Clear old memories: `python cli.py memory delete --user-id your_user_id --type SHORT_TERM`
3. Reduce token limits in configuration
4. Monitor API response times

#### High Memory Usage
1. Restart the application periodically
2. Clear vector index: Delete `data/vector_index/` and restart
3. Reduce conversation history retention
4. Monitor with: `python cli.py system tokens`

### Log Analysis

#### System Logs
```bash
# View recent system logs
tail -f data/logs/system/system.log

# Check error logs
grep ERROR data/logs/system/*.log

# Monitor performance
grep PERFORMANCE data/logs/system/*.log
```

#### Agent Logs
```bash
# View agent execution logs
ls data/agent_logs/
cat data/agent_logs/orchestrator_$(date +%Y%m%d).jsonl
```

## Advanced Configuration

### Custom Prompts
Edit prompts in `data/prompts/` directory:
- `system_prompts.json`: Core system prompts
- `agent_prompts.json`: Agent-specific prompts
- `coaching_prompts.json`: Motivational coaching prompts

### Experiment Management
```bash
# Create new experiment
python scripts/manage_ab_experiments.py create --name "new_feature_test"

# Run experiment analysis
python scripts/monthly_experiment_analysis.py
```

### Knowledge Management
```bash
# Update knowledge base
python scripts/weekly_knowledge_update.py

# Manage prompts with Git versioning
python scripts/manage_prompts.py commit --message "Updated coaching prompts"
```

## Backup and Recovery

### Full System Backup
```bash
# Create backup directory
mkdir backup_$(date +%Y%m%d)

# Backup database
python cli.py system database --backup backup_$(date +%Y%m%d)/founderforge.db

# Backup configuration and data
cp -r config/ backup_$(date +%Y%m%d)/
cp -r data/business_data/ backup_$(date +%Y%m%d)/
cp .env backup_$(date +%Y%m%d)/
```

### Recovery
```bash
# Restore database
python cli.py system database --restore backup_20240124/founderforge.db

# Restore configuration
cp -r backup_20240124/config/ .
cp backup_20240124/.env .
```

## Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Clear short-term memories, check logs
2. **Monthly**: Backup database, analyze experiments
3. **Quarterly**: Update dependencies, review performance

### Getting Help
1. Check this documentation first
2. Run system diagnostics: `python cli.py system status`
3. Review log files in `data/logs/`
4. Test with minimal configuration

### Updates and Upgrades
1. Backup your data before updating
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Run database migrations if needed
4. Test system functionality after updates

## Conclusion

FounderForge is designed to be a powerful yet simple-to-deploy AI cofounder system. With proper setup and configuration, it provides comprehensive business intelligence and strategic guidance while maintaining complete data privacy through local deployment.

For additional support or advanced configuration needs, refer to the troubleshooting section or check the system logs for detailed error information.