# FounderForge AI Cofounder - Sample Configurations

## Overview

This document provides sample configurations for different deployment scenarios and use cases. Use these as starting points and customize them according to your specific needs.

## Environment Configuration Examples

### Development Environment (.env)
```bash
# Development configuration
GEMINI_API_KEY=your_development_api_key_here
LOG_LEVEL=DEBUG
ENVIRONMENT=development
DATABASE_PATH=data/founderforge_dev.db
VECTOR_INDEX_PATH=data/vector_index_dev
ENABLE_DEBUG=true
ENABLE_HOT_RELOAD=true
MOCK_API_CALLS=false
PERFORMANCE_MONITORING=true
```

### Production Environment (.env)
```bash
# Production configuration
GEMINI_API_KEY=your_production_api_key_here
LOG_LEVEL=INFO
ENVIRONMENT=production
DATABASE_PATH=data/founderforge.db
VECTOR_INDEX_PATH=data/vector_index
ENABLE_DEBUG=false
ENABLE_HOT_RELOAD=false
MOCK_API_CALLS=false
PERFORMANCE_MONITORING=true
SESSION_TIMEOUT=3600
SECURE_COOKIES=true
```

### Testing Environment (.env)
```bash
# Testing configuration
GEMINI_API_KEY=test_api_key
LOG_LEVEL=WARNING
ENVIRONMENT=testing
DATABASE_PATH=data/founderforge_test.db
VECTOR_INDEX_PATH=data/vector_index_test
ENABLE_DEBUG=false
MOCK_API_CALLS=true
PERFORMANCE_MONITORING=false
```

## Feature Flags Configuration

### Development Feature Flags (config/feature_flags.json)
```json
{
  "enable_advanced_analytics": true,
  "enable_experimental_features": true,
  "enable_debug_logging": true,
  "enable_performance_monitoring": true,
  "enable_a_b_testing": true,
  "enable_mock_responses": false,
  "enable_context_caching": true,
  "enable_parallel_processing": true,
  "enable_auto_backup": false,
  "enable_user_feedback": true
}
```

### Production Feature Flags (config/feature_flags.json)
```json
{
  "enable_advanced_analytics": true,
  "enable_experimental_features": false,
  "enable_debug_logging": false,
  "enable_performance_monitoring": true,
  "enable_a_b_testing": false,
  "enable_mock_responses": false,
  "enable_context_caching": true,
  "enable_parallel_processing": true,
  "enable_auto_backup": true,
  "enable_user_feedback": false
}
```

### Testing Feature Flags (config/feature_flags.json)
```json
{
  "enable_advanced_analytics": false,
  "enable_experimental_features": false,
  "enable_debug_logging": false,
  "enable_performance_monitoring": false,
  "enable_a_b_testing": false,
  "enable_mock_responses": true,
  "enable_context_caching": false,
  "enable_parallel_processing": false,
  "enable_auto_backup": false,
  "enable_user_feedback": false
}
```

## Security Configuration

### Standard Security (config/security_config.json)
```json
{
  "enable_content_filtering": true,
  "enable_pii_detection": true,
  "enable_rate_limiting": true,
  "enable_input_validation": true,
  "max_token_limit": 16000,
  "confidence_threshold": 0.8,
  "max_memory_retention_days": 90,
  "enable_audit_logging": true,
  "allowed_file_types": [".json", ".txt", ".csv"],
  "max_file_size_mb": 10,
  "session_timeout_minutes": 60,
  "max_concurrent_users": 10,
  "rate_limit_requests_per_minute": 60,
  "enable_data_encryption": false,
  "pii_patterns": [
    "\\b\\d{3}-\\d{2}-\\d{4}\\b",
    "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
    "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"
  ]
}
```

### High Security (config/security_config.json)
```json
{
  "enable_content_filtering": true,
  "enable_pii_detection": true,
  "enable_rate_limiting": true,
  "enable_input_validation": true,
  "max_token_limit": 12000,
  "confidence_threshold": 0.9,
  "max_memory_retention_days": 30,
  "enable_audit_logging": true,
  "allowed_file_types": [".json", ".txt"],
  "max_file_size_mb": 5,
  "session_timeout_minutes": 30,
  "max_concurrent_users": 5,
  "rate_limit_requests_per_minute": 30,
  "enable_data_encryption": true,
  "pii_patterns": [
    "\\b\\d{3}-\\d{2}-\\d{4}\\b",
    "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
    "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b",
    "\\b\\d{3}[\\s-]?\\d{3}[\\s-]?\\d{4}\\b",
    "\\b[A-Z]{2}\\d{6,8}\\b"
  ]
}
```

## Funding Validation Rules

### Standard Validation (config/funding_validation_rules.json)
```json
{
  "required_fields": [
    "company_name",
    "funding_amount",
    "business_plan",
    "team_experience",
    "market_size"
  ],
  "field_validations": {
    "company_name": {
      "min_length": 2,
      "max_length": 100,
      "pattern": "^[A-Za-z0-9\\s\\-\\.]+$"
    },
    "funding_amount": {
      "min_value": 10000,
      "max_value": 100000000,
      "type": "number"
    },
    "business_plan": {
      "min_length": 100,
      "max_length": 5000
    },
    "team_experience": {
      "min_length": 50,
      "max_length": 2000
    },
    "market_size": {
      "min_length": 20,
      "max_length": 1000
    }
  },
  "scoring_weights": {
    "team": 0.3,
    "market": 0.25,
    "product": 0.25,
    "financials": 0.2
  },
  "risk_factors": {
    "high_funding_amount": 5000000,
    "early_stage_keywords": ["idea", "concept", "planning"],
    "red_flag_keywords": ["guaranteed", "no risk", "revolutionary"]
  },
  "processing_targets": {
    "max_processing_time_seconds": 30,
    "min_accuracy_percentage": 95,
    "min_confidence_score": 0.8
  }
}
```

### Strict Validation (config/funding_validation_rules.json)
```json
{
  "required_fields": [
    "company_name",
    "funding_amount",
    "business_plan",
    "team_experience",
    "market_size",
    "revenue",
    "customers",
    "competition",
    "use_of_funds"
  ],
  "field_validations": {
    "company_name": {
      "min_length": 3,
      "max_length": 50,
      "pattern": "^[A-Za-z0-9\\s\\-\\.]+$"
    },
    "funding_amount": {
      "min_value": 50000,
      "max_value": 50000000,
      "type": "number"
    },
    "business_plan": {
      "min_length": 500,
      "max_length": 3000
    },
    "team_experience": {
      "min_length": 200,
      "max_length": 1500
    },
    "market_size": {
      "min_length": 100,
      "max_length": 800
    },
    "revenue": {
      "min_value": 0,
      "type": "number"
    },
    "customers": {
      "min_value": 0,
      "type": "number"
    }
  },
  "scoring_weights": {
    "team": 0.35,
    "market": 0.25,
    "product": 0.25,
    "financials": 0.15
  },
  "risk_factors": {
    "high_funding_amount": 2000000,
    "early_stage_keywords": ["idea", "concept", "planning", "prototype"],
    "red_flag_keywords": ["guaranteed", "no risk", "revolutionary", "disrupting everything"]
  },
  "processing_targets": {
    "max_processing_time_seconds": 20,
    "min_accuracy_percentage": 98,
    "min_confidence_score": 0.9
  }
}
```

## Streamlit Configuration

### Development Streamlit Config (~/.streamlit/config.toml)
```toml
[global]
developmentMode = true

[server]
headless = false
port = 8501
address = "localhost"
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
showErrorDetails = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[logger]
level = "debug"
```

### Production Streamlit Config (~/.streamlit/config.toml)
```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 50

[browser]
gatherUsageStats = false
showErrorDetails = false

[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[logger]
level = "info"
```

## Docker Configuration

### Development Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/{business_data,chat_history,experiments,logs,prompts,vector_index}

# Set environment variables
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG

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

### Production Docker Compose
```yaml
version: '3.8'

services:
  founderforge:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - founderforge
    restart: unless-stopped

  backup:
    image: alpine:latest
    volumes:
      - ./data:/data:ro
      - ./backups:/backups
    command: |
      sh -c "
        while true; do
          tar -czf /backups/backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
          find /backups -name '*.tar.gz' -mtime +7 -delete
          sleep 86400
        done
      "
    restart: unless-stopped
```

## Nginx Configuration

### Basic Nginx Config (nginx.conf)
```nginx
events {
    worker_connections 1024;
}

http {
    upstream founderforge {
        server founderforge:8501;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://founderforge;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 86400;
        }

        location /_stcore/health {
            proxy_pass http://founderforge;
            access_log off;
        }
    }
}
```

### SSL Nginx Config (nginx.conf)
```nginx
events {
    worker_connections 1024;
}

http {
    upstream founderforge {
        server founderforge:8501;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        location / {
            proxy_pass http://founderforge;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 86400;
        }
    }
}
```

## Systemd Service Configuration

### Development Service (/etc/systemd/system/founderforge-dev.service)
```ini
[Unit]
Description=FounderForge AI Cofounder (Development)
After=network.target

[Service]
Type=simple
User=developer
WorkingDirectory=/home/developer/FounderForge
Environment=PATH=/home/developer/FounderForge/venv/bin
Environment=ENVIRONMENT=development
Environment=LOG_LEVEL=DEBUG
ExecStart=/home/developer/FounderForge/venv/bin/streamlit run app.py --server.port=8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Production Service (/etc/systemd/system/founderforge.service)
```ini
[Unit]
Description=FounderForge AI Cofounder
After=network.target

[Service]
Type=simple
User=founderforge
Group=founderforge
WorkingDirectory=/opt/founderforge
Environment=PATH=/opt/founderforge/venv/bin
Environment=ENVIRONMENT=production
Environment=LOG_LEVEL=INFO
ExecStart=/opt/founderforge/venv/bin/streamlit run app.py --server.port=8501 --server.address=127.0.0.1
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=founderforge

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/founderforge/data /opt/founderforge/logs

[Install]
WantedBy=multi-user.target
```

## Monitoring Configuration

### Prometheus Configuration (prometheus.yml)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'founderforge'
    static_configs:
      - targets: ['localhost:8501']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

rule_files:
  - "founderforge_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "FounderForge Monitoring",
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "founderforge_response_time_seconds",
            "legendFormat": "Response Time"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "founderforge_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "founderforge_active_users",
            "legendFormat": "Active Users"
          }
        ]
      }
    ]
  }
}
```

## Backup Configuration

### Automated Backup Script (backup.sh)
```bash
#!/bin/bash

# Configuration
BACKUP_DIR="/backup/founderforge"
APP_DIR="/opt/founderforge"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Stop application for consistent backup
systemctl stop founderforge

# Backup database
cp $APP_DIR/data/founderforge.db $BACKUP_DIR/$DATE/

# Backup configuration
cp -r $APP_DIR/config $BACKUP_DIR/$DATE/
cp $APP_DIR/.env $BACKUP_DIR/$DATE/

# Backup business data
cp -r $APP_DIR/data/business_data $BACKUP_DIR/$DATE/

# Backup logs (last 7 days)
find $APP_DIR/data/logs -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/$DATE/ \;

# Start application
systemctl start founderforge

# Compress backup
tar -czf $BACKUP_DIR/founderforge_backup_$DATE.tar.gz -C $BACKUP_DIR $DATE
rm -rf $BACKUP_DIR/$DATE

# Clean old backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

# Log backup completion
echo "$(date): Backup completed - founderforge_backup_$DATE.tar.gz" >> $BACKUP_DIR/backup.log

# Send notification (optional)
# curl -X POST -H 'Content-type: application/json' \
#   --data '{"text":"FounderForge backup completed: '$DATE'"}' \
#   YOUR_SLACK_WEBHOOK_URL
```

## Logging Configuration

### Logrotate Configuration (/etc/logrotate.d/founderforge)
```
/opt/founderforge/data/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 founderforge founderforge
    postrotate
        systemctl reload founderforge
    endscript
}
```

### Rsyslog Configuration (/etc/rsyslog.d/founderforge.conf)
```
# FounderForge logging
if $programname == 'founderforge' then /var/log/founderforge.log
& stop
```

## Usage Examples

### CLI Usage Examples
```bash
# User management
python cli.py user create --name "John Doe" --email "john@example.com"
python cli.py user list --format json

# Memory management
python cli.py memory list --user-id user123 --type LONG_TERM
python cli.py memory search --user-id user123 "funding strategy"
python cli.py memory delete --user-id user123 --confirm

# System management
python cli.py system status
python cli.py system database --backup backup_$(date +%Y%m%d).db
python cli.py system tokens --user-id user123 --days 7

# Batch processing
python cli.py batch process --file batch_queries.json --output results.json
```

### API Usage Examples (if implementing REST API)
```bash
# Health check
curl http://localhost:8501/api/health

# User creation
curl -X POST http://localhost:8501/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com"}'

# Chat interaction
curl -X POST http://localhost:8501/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "message": "Help me with my business strategy"}'

# Memory retrieval
curl http://localhost:8501/api/users/user123/memories?type=LONG_TERM
```

## Conclusion

These sample configurations provide a solid foundation for deploying FounderForge AI Cofounder in various environments. Customize them according to your specific requirements:

1. **Development**: Use debug settings and relaxed security
2. **Production**: Enable security features and monitoring
3. **Testing**: Use mock responses and minimal logging

Remember to:
- Keep sensitive information (API keys) secure
- Regularly update configurations based on usage patterns
- Monitor system performance and adjust settings accordingly
- Backup configurations along with data