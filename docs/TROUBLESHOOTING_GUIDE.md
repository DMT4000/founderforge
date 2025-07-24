# FounderForge AI Cofounder - Troubleshooting Guide

## Overview

This guide provides solutions to common issues you may encounter while setting up, running, or maintaining FounderForge AI Cofounder. Issues are organized by category with step-by-step solutions.

## Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system check
python cli.py system status

# Check component health
python test_integration_simple.py

# Verify database integrity
sqlite3 data/founderforge.db "PRAGMA integrity_check;"
```

### Log Analysis
```bash
# View recent system logs
tail -f data/logs/system/system.log

# Check for errors
grep -i error data/logs/system/*.log

# Monitor performance
grep -i performance data/logs/system/*.log
```

## Installation and Setup Issues

### 1. Python Environment Issues

#### Problem: Python Version Compatibility
```
ERROR: Python 3.7 is not supported
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.8+ (Ubuntu/Debian)
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip

# Create virtual environment with correct Python
python3.9 -m venv founderforge-env
source founderforge-env/bin/activate
```

#### Problem: Virtual Environment Issues
```
ERROR: Could not find a version that satisfies the requirement
```

**Solution:**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependency Installation Issues

#### Problem: Package Installation Failures
```
ERROR: Failed building wheel for package-name
```

**Solution:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt install build-essential python3-dev

# Install system dependencies (CentOS/RHEL)
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Clear pip cache and retry
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

#### Problem: FAISS Installation Issues
```
ERROR: Could not find a version that satisfies the requirement faiss-cpu
```

**Solution:**
```bash
# Install FAISS CPU version
pip install faiss-cpu --no-cache-dir

# Alternative: Install from conda-forge
conda install -c conda-forge faiss-cpu

# For Apple Silicon Macs
pip install faiss-cpu --no-deps
pip install numpy
```

### 3. Environment Configuration Issues

#### Problem: Missing .env File
```
ERROR: GEMINI_API_KEY not found in environment
```

**Solution:**
```bash
# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_actual_api_key_here
LOG_LEVEL=INFO
DATABASE_PATH=data/founderforge.db
EOF

# Verify file exists and has correct permissions
ls -la .env
chmod 600 .env
```

#### Problem: Invalid API Key
```
ERROR: 401 Unauthorized - Invalid API key
```

**Solution:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Update your `.env` file:
   ```bash
   echo "GEMINI_API_KEY=your_new_api_key" > .env
   ```
4. Test the connection:
   ```bash
   python test_gemini_connection.py
   ```

## Database Issues

### 1. Database Connection Problems

#### Problem: Database Locked
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Check for processes using the database
lsof data/founderforge.db

# Kill processes if necessary
pkill -f "python.*app.py"
pkill -f "python.*cli.py"

# Remove lock file if it exists
rm -f data/founderforge.db-wal
rm -f data/founderforge.db-shm

# Restart application
streamlit run app.py
```

#### Problem: Database Corruption
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solution:**
```bash
# Backup current database
cp data/founderforge.db data/founderforge.db.backup

# Try to repair
sqlite3 data/founderforge.db ".backup repair.db"
mv repair.db data/founderforge.db

# If repair fails, reinitialize
rm data/founderforge.db
python src/init_db.py

# Restore data from backup if possible
python scripts/restore_from_backup.py data/founderforge.db.backup
```

### 2. Database Performance Issues

#### Problem: Slow Query Performance
```
WARNING: Database query took 5.2 seconds
```

**Solution:**
```bash
# Analyze database
sqlite3 data/founderforge.db "ANALYZE;"

# Add missing indexes
sqlite3 data/founderforge.db << EOF
CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_conversations_user_date ON conversations(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
EOF

# Vacuum database to optimize
sqlite3 data/founderforge.db "VACUUM;"
```

## Application Runtime Issues

### 1. Streamlit Web Interface Issues

#### Problem: Port Already in Use
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port 8501
sudo netstat -tlnp | grep :8501
sudo lsof -i :8501

# Kill the process
sudo kill -9 $(sudo lsof -t -i:8501)

# Or use a different port
streamlit run app.py --server.port 8502
```

#### Problem: Streamlit Won't Start
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install streamlit
pip install streamlit

# Verify installation
streamlit --version

# Run with full path if needed
python -m streamlit run app.py
```

### 2. Memory and Performance Issues

#### Problem: High Memory Usage
```
WARNING: Memory usage at 95%
```

**Solution:**
```bash
# Monitor memory usage
python cli.py system status

# Clear short-term memories
python cli.py memory delete --user-id your_user_id --type SHORT_TERM

# Restart application to free memory
pkill -f streamlit
streamlit run app.py

# Increase system swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Problem: Slow Response Times
```
WARNING: Response time exceeded 10 seconds
```

**Solution:**
```bash
# Check system resources
htop

# Monitor database performance
python cli.py system tokens --days 1

# Optimize context assembly
# Edit config/settings.py to reduce max_token_limit

# Clear vector index to rebuild
rm -rf data/vector_index/*
# Restart application to rebuild index
```

### 3. API Integration Issues

#### Problem: Gemini API Timeout
```
ERROR: Request timeout after 30 seconds
```

**Solution:**
```bash
# Test API connectivity
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"

# Check network connectivity
ping generativelanguage.googleapis.com

# Increase timeout in configuration
# Edit src/gemini_client.py to increase timeout value
```

#### Problem: API Rate Limiting
```
ERROR: 429 Too Many Requests
```

**Solution:**
```bash
# Implement exponential backoff (already in code)
# Monitor API usage
python cli.py system tokens --days 7

# Reduce API calls by:
# 1. Enabling caching
# 2. Using mock mode for testing
# 3. Optimizing prompt length
```

## Agent and Workflow Issues

### 1. Agent Execution Failures

#### Problem: Agent Workflow Timeout
```
ERROR: Workflow execution timeout after 60 seconds
```

**Solution:**
```bash
# Check agent logs
ls data/agent_logs/
tail -f data/agent_logs/orchestrator_$(date +%Y%m%d).jsonl

# Test individual agents
python test_agent_orchestration.py

# Increase timeout in configuration
# Edit src/agents.py to increase workflow timeout
```

#### Problem: LangGraph State Issues
```
ERROR: Invalid state transition in workflow
```

**Solution:**
```bash
# Clear workflow state
rm -rf data/workflow_logs/*

# Test workflow execution
python test_funding_workflow.py
python test_daily_planning_workflow.py

# Reset agent state if needed
python cli.py system database --backup backup.db
# Restart application
```

### 2. Context Assembly Issues

#### Problem: Token Limit Exceeded
```
ERROR: Context exceeds 16000 token limit
```

**Solution:**
```bash
# Check current token usage
python cli.py system tokens --user-id your_user_id

# Clear old memories to reduce context
python cli.py memory delete --user-id your_user_id --type SHORT_TERM

# Adjust token limits in config/security_config.json
{
  "max_token_limit": 12000,
  "enable_context_summarization": true
}
```

## Security and Access Issues

### 1. File Permission Issues

#### Problem: Permission Denied Errors
```
PermissionError: [Errno 13] Permission denied: 'data/founderforge.db'
```

**Solution:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod -R 755 .
chmod 600 .env
chmod -R 700 data/

# For system-wide installation
sudo chown -R founderforge:founderforge /opt/founderforge/
```

### 2. Data Access Issues

#### Problem: User Data Not Found
```
ERROR: No user data found for user_id
```

**Solution:**
```bash
# Check user exists in database
sqlite3 data/founderforge.db "SELECT * FROM users WHERE id = 'your_user_id';"

# Create user if missing
python cli.py user create --user-id your_user_id --name "Your Name"

# Verify user data directories
ls -la data/business_data/
```

## Network and Connectivity Issues

### 1. Local Network Access Issues

#### Problem: Cannot Access from Other Devices
```
Connection refused when accessing from network
```

**Solution:**
```bash
# Configure Streamlit for network access
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
EOF

# Check firewall settings
sudo ufw status
sudo ufw allow 8501/tcp

# Restart application
streamlit run app.py
```

### 2. SSL/TLS Issues

#### Problem: SSL Certificate Errors
```
ERROR: SSL certificate verification failed
```

**Solution:**
```bash
# Update certificates
sudo apt update && sudo apt install ca-certificates

# For development, disable SSL verification (not recommended for production)
export PYTHONHTTPSVERIFY=0

# Or configure proper certificates for production deployment
```

## Logging and Monitoring Issues

### 1. Log File Issues

#### Problem: Log Files Not Created
```
WARNING: Unable to write to log directory
```

**Solution:**
```bash
# Create log directories
mkdir -p data/logs/{system,security,performance}

# Fix permissions
chmod -R 755 data/logs/

# Check disk space
df -h

# Verify logging configuration
python -c "from src.logging_manager import get_logging_manager; print(get_logging_manager())"
```

### 2. Performance Monitoring Issues

#### Problem: Performance Metrics Not Collected
```
WARNING: Performance monitor not initialized
```

**Solution:**
```bash
# Test performance monitoring
python test_logging_monitoring_integration.py

# Check configuration
cat config/feature_flags.json | grep performance

# Enable performance monitoring
python -c "
from src.feature_flag_manager import FeatureFlagManager
fm = FeatureFlagManager()
fm.set_flag('enable_performance_monitoring', True)
"
```

## Data Migration and Backup Issues

### 1. Backup Failures

#### Problem: Backup Script Fails
```
ERROR: Backup failed - insufficient disk space
```

**Solution:**
```bash
# Check disk space
df -h

# Clean up old logs
find data/logs -name "*.log" -mtime +30 -delete

# Compress old backups
gzip backup/*.db

# Run backup manually
python cli.py system database --backup backup_$(date +%Y%m%d).db
```

### 2. Data Recovery Issues

#### Problem: Cannot Restore from Backup
```
ERROR: Backup file corrupted or incompatible
```

**Solution:**
```bash
# Verify backup integrity
sqlite3 backup.db "PRAGMA integrity_check;"

# Try partial recovery
sqlite3 backup.db ".dump" | sqlite3 new_database.db

# Manual data extraction
sqlite3 backup.db << EOF
.mode csv
.output users_backup.csv
SELECT * FROM users;
.output memories_backup.csv
SELECT * FROM memories;
EOF
```

## Development and Testing Issues

### 1. Test Failures

#### Problem: Integration Tests Fail
```
ERROR: Test failed - component not initialized
```

**Solution:**
```bash
# Run tests individually
python test_agent_orchestration.py
python test_funding_workflow.py
python test_logging_monitoring_integration.py

# Check test environment
python -c "import sys; print(sys.path)"

# Reset test environment
rm -rf data/test_*
python src/init_db.py
```

### 2. Development Environment Issues

#### Problem: Hot Reload Not Working
```
WARNING: File changes not detected
```

**Solution:**
```bash
# Enable Streamlit auto-reload
streamlit run app.py --server.runOnSave true

# Check file permissions
ls -la app.py

# Use development configuration
export ENVIRONMENT=development
streamlit run app.py
```

## Advanced Troubleshooting

### 1. Memory Profiling

#### Identify Memory Leaks
```python
# Add to your code for debugging
import tracemalloc
import psutil
import os

def profile_memory():
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    print(f"RSS memory: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

### 2. Performance Profiling

#### Profile Slow Functions
```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions
    
    return result
```

### 3. Database Debugging

#### Analyze Query Performance
```sql
-- Enable query logging in SQLite
PRAGMA query_only = ON;

-- Analyze query plans
EXPLAIN QUERY PLAN SELECT * FROM memories WHERE user_id = 'test';

-- Check table statistics
SELECT name, sql FROM sqlite_master WHERE type='table';
```

## Getting Additional Help

### 1. Diagnostic Information Collection

When reporting issues, collect this information:

```bash
# System information
uname -a
python --version
pip list | grep -E "(streamlit|faiss|sqlite)"

# Application status
python cli.py system status > system_status.txt

# Recent logs
tail -100 data/logs/system/system.log > recent_logs.txt

# Configuration
cat config/feature_flags.json
cat config/security_config.json
```

### 2. Debug Mode

Enable debug mode for detailed logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or modify .env file
echo "LOG_LEVEL=DEBUG" >> .env

# Run with debug output
python -u app.py 2>&1 | tee debug_output.log
```

### 3. Safe Mode

Run in safe mode to isolate issues:

```bash
# Disable all optional features
export SAFE_MODE=true

# Use mock API calls
export MOCK_API_CALLS=true

# Minimal configuration
python cli.py --config minimal_config.json
```

## Prevention and Best Practices

### 1. Regular Maintenance

```bash
# Weekly maintenance script
#!/bin/bash
# maintenance.sh

# Clean old logs
find data/logs -name "*.log" -mtime +7 -delete

# Vacuum database
sqlite3 data/founderforge.db "VACUUM;"

# Clear temporary files
rm -rf /tmp/founderforge_*

# Update dependencies
pip install -r requirements.txt --upgrade

# Run health check
python cli.py system status
```

### 2. Monitoring Setup

```bash
# Set up monitoring cron job
crontab -e

# Add these lines:
# Check system health every hour
0 * * * * /path/to/founderforge/health_check.sh

# Daily backup
0 2 * * * /path/to/founderforge/backup.sh

# Weekly maintenance
0 3 * * 0 /path/to/founderforge/maintenance.sh
```

### 3. Configuration Management

Keep configuration files in version control:

```bash
# Track configuration changes
git add config/
git commit -m "Update configuration"

# Use environment-specific configs
cp config/production.json config/settings.json
```

## Conclusion

This troubleshooting guide covers the most common issues encountered with FounderForge AI Cofounder. For issues not covered here:

1. Check the system logs for detailed error messages
2. Run the diagnostic commands provided
3. Try the safe mode options
4. Collect diagnostic information for further analysis

Remember to always backup your data before attempting major fixes, and test solutions in a development environment when possible.