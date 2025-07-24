# FounderForge AI Cofounder - Deployment Guide

## Overview

This guide covers deployment options for FounderForge AI Cofounder, from local development to production-ready setups. The system is designed as a localhost-first application but can be adapted for various deployment scenarios.

## Deployment Options

### 1. Local Development (Recommended)
- **Use Case**: Personal use, development, testing
- **Requirements**: Local machine with Python 3.8+
- **Data**: Stored locally, complete privacy
- **Scalability**: Single user

### 2. Local Network Deployment
- **Use Case**: Small team, office network
- **Requirements**: Local server or powerful workstation
- **Data**: Shared local storage
- **Scalability**: Multiple users on same network

### 3. Cloud VM Deployment
- **Use Case**: Remote access, team collaboration
- **Requirements**: Cloud VM (AWS, GCP, Azure)
- **Data**: Cloud storage with encryption
- **Scalability**: Configurable based on VM size

## Local Development Deployment

### Prerequisites
```bash
# System requirements
python --version  # 3.8+
pip --version
git --version

# Available disk space
df -h  # Ensure 2GB+ free space
```

### Quick Deployment
```bash
# 1. Clone repository
git clone <https://github.com/DMT4000/founderforge>
cd FounderForge

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 5. Initialize system
python src/init_db.py

# 6. Start application
streamlit run app.py
```

### Development Configuration
Create `config/development.json`:
```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "enable_hot_reload": true,
  "mock_api_calls": false,
  "performance_monitoring": true
}
```

## Local Network Deployment

### Server Setup

#### Hardware Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB+ SSD storage
- **Network**: Stable network connection

#### Installation Steps
```bash
# 1. Set up server environment
sudo apt update && sudo apt upgrade -y
sudo apt install python3.9 python3.9-venv python3-pip git -y

# 2. Create application user
sudo useradd -m -s /bin/bash founderforge
sudo su - founderforge

# 3. Deploy application
git clone <repository-url>
cd FounderForge
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure for network access
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "HOST=0.0.0.0" >> .env
echo "PORT=8501" >> .env
```

#### Network Configuration
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

[browser]
gatherUsageStats = false
EOF
```

#### Firewall Setup
```bash
# Ubuntu/Debian
sudo ufw allow 8501/tcp
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

#### Service Configuration
Create `/etc/systemd/system/founderforge.service`:
```ini
[Unit]
Description=FounderForge AI Cofounder
After=network.target

[Service]
Type=simple
User=founderforge
WorkingDirectory=/home/founderforge/FounderForge
Environment=PATH=/home/founderforge/FounderForge/venv/bin
ExecStart=/home/founderforge/FounderForge/venv/bin/streamlit run app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable founderforge
sudo systemctl start founderforge
sudo systemctl status founderforge
```

### Multi-User Configuration

#### User Management
```bash
# Create user-specific data directories
mkdir -p data/users/{user1,user2,user3}

# Set up user isolation in app.py
# Modify session state to include user isolation
```

#### Database Isolation
```python
# In src/database.py, add user-specific database paths
def get_user_database_path(user_id):
    return f"data/users/{user_id}/founderforge.db"
```

## Cloud VM Deployment

### AWS EC2 Deployment

#### Instance Setup
```bash
# 1. Launch EC2 instance
# - AMI: Ubuntu 20.04 LTS
# - Instance Type: t3.medium (minimum)
# - Security Group: Allow port 8501
# - Storage: 20GB GP2

# 2. Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Update system
sudo apt update && sudo apt upgrade -y
sudo apt install python3.9 python3.9-venv python3-pip git nginx -y
```

#### Application Deployment
```bash
# 1. Clone and setup
git clone <repository-url>
cd FounderForge
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cat > .env << EOF
GEMINI_API_KEY=your_api_key_here
HOST=127.0.0.1
PORT=8501
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF

# 3. Initialize database
python src/init_db.py

# 4. Test deployment
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

#### Nginx Reverse Proxy
```nginx
# /etc/nginx/sites-available/founderforge
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/founderforge /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### SSL Configuration (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Google Cloud Platform Deployment

#### Compute Engine Setup
```bash
# 1. Create VM instance
gcloud compute instances create founderforge-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --tags=http-server,https-server

# 2. Configure firewall
gcloud compute firewall-rules create allow-founderforge \
    --allow tcp:8501 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow FounderForge access"

# 3. Connect and deploy
gcloud compute ssh founderforge-vm
# Follow AWS deployment steps
```

### Azure VM Deployment

#### Virtual Machine Setup
```bash
# 1. Create resource group
az group create --name FounderForgeRG --location eastus

# 2. Create VM
az vm create \
    --resource-group FounderForgeRG \
    --name FounderForgeVM \
    --image UbuntuLTS \
    --admin-username azureuser \
    --generate-ssh-keys \
    --size Standard_B2s

# 3. Open port
az vm open-port --port 8501 --resource-group FounderForgeRG --name FounderForgeVM

# 4. Connect and deploy
az vm show --resource-group FounderForgeRG --name FounderForgeVM --show-details --query publicIps
ssh azureuser@your-vm-ip
# Follow AWS deployment steps
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/{business_data,chat_history,experiments,logs,prompts,vector_index}

# Initialize database
RUN python src/init_db.py

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  founderforge:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - founderforge
    restart: unless-stopped
```

### Docker Deployment Commands
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f founderforge

# Scale (if needed)
docker-compose up -d --scale founderforge=2

# Update deployment
docker-compose pull
docker-compose up -d
```

## Production Considerations

### Security

#### Environment Variables
```bash
# Use secure environment variable management
# Never commit .env files to version control

# Production .env example
GEMINI_API_KEY=your_secure_api_key
LOG_LEVEL=INFO
ENVIRONMENT=production
ENABLE_DEBUG=false
SECURE_COOKIES=true
SESSION_TIMEOUT=3600
```

#### Data Encryption
```bash
# Encrypt data directory
sudo apt install ecryptfs-utils -y
sudo ecryptfs-setup-private --username founderforge

# Mount encrypted directory
sudo mount -t ecryptfs /home/founderforge/FounderForge/data /home/founderforge/FounderForge/data
```

#### Access Control
```bash
# Set proper file permissions
chmod 600 .env
chmod -R 700 data/
chown -R founderforge:founderforge /home/founderforge/FounderForge/
```

### Monitoring and Logging

#### Log Management
```bash
# Configure log rotation
sudo cat > /etc/logrotate.d/founderforge << EOF
/home/founderforge/FounderForge/data/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 founderforge founderforge
}
EOF
```

#### System Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Monitor application
htop -p $(pgrep -f streamlit)
```

#### Application Monitoring
```python
# Add to app.py for production monitoring
import psutil
import time

def monitor_system():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
```

### Backup and Recovery

#### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/founderforge"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/founderforge/FounderForge"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup database
cp $APP_DIR/data/founderforge.db $BACKUP_DIR/$DATE/

# Backup configuration
cp -r $APP_DIR/config $BACKUP_DIR/$DATE/
cp $APP_DIR/.env $BACKUP_DIR/$DATE/

# Backup business data
cp -r $APP_DIR/data/business_data $BACKUP_DIR/$DATE/

# Compress backup
tar -czf $BACKUP_DIR/founderforge_backup_$DATE.tar.gz -C $BACKUP_DIR $DATE
rm -rf $BACKUP_DIR/$DATE

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: founderforge_backup_$DATE.tar.gz"
```

```bash
# Schedule daily backups
crontab -e
# Add: 0 2 * * * /home/founderforge/backup.sh
```

### Performance Optimization

#### Database Optimization
```sql
-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_conversations_user_date ON conversations(user_id, timestamp);
```

#### Caching
```python
# Add Redis caching for production
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_context(user_id, context):
    redis_client.setex(f"context:{user_id}", 300, json.dumps(context))

def get_cached_context(user_id):
    cached = redis_client.get(f"context:{user_id}")
    return json.loads(cached) if cached else None
```

## Troubleshooting Deployment Issues

### Common Deployment Problems

#### Port Conflicts
```bash
# Check port usage
sudo netstat -tlnp | grep :8501
sudo lsof -i :8501

# Kill conflicting processes
sudo kill -9 $(sudo lsof -t -i:8501)
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R founderforge:founderforge /home/founderforge/FounderForge/
chmod +x /home/founderforge/FounderForge/venv/bin/streamlit
```

#### Memory Issues
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Database Corruption
```bash
# Check database integrity
sqlite3 data/founderforge.db "PRAGMA integrity_check;"

# Repair if needed
sqlite3 data/founderforge.db ".backup backup.db"
mv backup.db data/founderforge.db
```

### Performance Troubleshooting

#### Slow Response Times
1. Check system resources: `htop`
2. Monitor database queries: Enable SQL logging
3. Check API response times: Monitor Gemini API calls
4. Optimize context assembly: Reduce token limits

#### High Memory Usage
1. Monitor Python processes: `ps aux | grep python`
2. Check for memory leaks: Use memory profiling
3. Restart application periodically: Set up cron job
4. Optimize vector store: Clear old embeddings

## Maintenance and Updates

### Regular Maintenance Tasks

#### Daily
- Monitor system resources
- Check application logs
- Verify backup completion

#### Weekly
- Update system packages
- Clean temporary files
- Review performance metrics

#### Monthly
- Update Python dependencies
- Analyze usage patterns
- Review security logs

### Update Procedures

#### Application Updates
```bash
# 1. Backup current version
./backup.sh

# 2. Stop application
sudo systemctl stop founderforge

# 3. Update code
git pull origin main

# 4. Update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 5. Run migrations (if any)
python src/migrate_db.py

# 6. Test deployment
python test_integration_simple.py

# 7. Start application
sudo systemctl start founderforge
```

#### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python
sudo apt install python3.10 python3.10-venv

# Recreate virtual environment if needed
python3.10 -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

## Conclusion

FounderForge AI Cofounder can be deployed in various configurations, from simple local development to production-ready cloud deployments. Choose the deployment option that best fits your needs:

- **Local Development**: Quick setup for personal use
- **Local Network**: Team collaboration within organization
- **Cloud VM**: Remote access and scalability
- **Docker**: Containerized deployment for consistency

Each deployment option maintains the core principle of data privacy while providing the flexibility needed for different use cases. Follow the security and monitoring guidelines to ensure a robust production deployment.