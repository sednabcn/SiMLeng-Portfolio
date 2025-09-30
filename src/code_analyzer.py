#!/usr/bin/env python3
"""
Code Analysis Module
Analyzes code structure, complexity, and AI/ML patterns
"""

import ast
import re
import json
import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
import hashlib

class CodeAnalyzer:
    """Analyzes code files for AI/ML patterns, complexity, and structure."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # AI/ML code patterns
        self.ml_patterns = {
            'model_training': [
                r'\.fit\(',
                r'\.train\(',
                r'model\.compile\(',
                r'optimizer\s*=',
                r'loss\s*=',
                r'epochs\s*=',
                r'batch_size\s*='
            ],
            'data_preprocessing': [
                r'train_test_split',
                r'StandardScaler',
                r'MinMaxScaler',
                r'LabelEncoder',
                r'OneHotEncoder',
                r'\.fillna\(',
                r'\.dropna\(',
                r'\.transform\('
            ],
            'model_evaluation': [
                r'\.evaluate\(',
                r'\.predict\(',
                r'accuracy_score',
                r'classification_report',
                r'confusion_matrix',
                r'cross_val_score',
                r'GridSearchCV',
                r'RandomizedSearchCV'
            ],
            'neural_networks': [
                r'Dense\(',
                r'Conv2D\(',
                r'LSTM\(',
                r'Dropout\(',
                r'BatchNormalization',
                r'nn\.Module',
                r'nn\.Linear',
                r'nn\.Conv2d'
            ],
            'deep_learning': [
                r'torch\.nn',
                r'tf\.keras',
                r'layers\.',
                r'Sequential\(',
                r'Model\(',
                r'forward\(',
                r'backward\(',
                r'autograd'
            ],
            'data_loading': [
                r'DataLoader',
                r'Dataset',
                r'ImageFolder',
                r'pd\.read_csv',
                r'load_dataset',
                r'torchvision\.datasets'
            ],
            'model_persistence': [
                r'\.save\(',
                r'\.load\(',
                r'pickle\.dump',
                r'pickle\.load',
                r'joblib\.dump',
                r'joblib\.load',
                r'torch\.save',
                r'torch\.load'
            ]
        }
        
        # Code quality patterns
        self.quality_patterns = {
            'documentation': [
                r'""".*?"""',
                r"'''.*?'''",
                r'#.*$'
            ],
            'type_hints': [
                r':\s*\w+',
                r'->\s*\w+',
                r'typing\.',
                r'List\[',
                r'Dict\[',
                r'Optional\['
            ],
            'error_handling': [
                r'try:',
                r'except\s+\w+:',
                r'raise\s+\w+',
                r'assert\s+'
            ],
            'testing': [
                r'def\s+test_',
                r'unittest\.',
                r'pytest\.',
                r'@pytest\.',
                r'TestCase'
            ]
        }
    
    async def analyze_repository_code(self, repository_contents: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all code files in the repository."""
        self.logger.info("Starting code analysis")
        
        analysis = {
            'file_statistics': {},
            'code_patterns': {},
            'quality_metrics': {},
            'complexity_metrics': {},
            'ai_ml_indicators': {},
            'notebooks': [],
            'has_notebooks': False,
            'has_models': False,
            'has_tests': False,
            'code_samples': []
        }
        
        files = repository_contents.get('files', [])
        
        # Analyze different file types
        python_files = [f for f in files if f['extension'] == 'py']
        notebook_files = [f for f in files if f['extension'] == 'ipynb']
        config_files = [f for f in files if f['extension'] in ['yaml', 'yml', 'json', 'toml']]
        
        analysis['file_statistics'] = {
            'total_files': len(files),
            'python_files': len(python_files),
            'notebook_files': len(notebook_files),
            'config_files': len(config_files),
            'file_extensions': self._get_file_extensions(files)
        }
        
        analysis['has_notebooks'] = len(notebook_files) > 0
        analysis['has_models'] = self._has_model_files(files)
        analysis['has_tests'] = self._has_test_files(python_files)
        
        # Analyze Python files
        if python_files:
            python_analysis = await self._analyze_python_files(python_files)
            analysis.update(python_analysis)
        
        # Analyze notebooks
        if notebook_files:
            notebook_analysis = await self._analyze_notebooks(notebook_files)
            analysis['notebooks'] = notebook_analysis
        
        return analysis
    
    def _get_file_extensions(self, files: List[Dict]) -> Dict[str, int]:
        """Get count of files by extension."""
        extensions = {}
        for file in files:
            ext = file.get('extension', 'no_extension')
            extensions[ext] = extensions.get(ext, 0) + 1
        return extensions
    
    def _has_model_files(self, files: List[Dict]) -> bool:
        """Check if repository contains model files."""
        model_extensions = {'pkl', 'joblib', 'h5', 'pt', 'pth', 'onnx', 'pb'}
        return any(f['extension'] in model_extensions for f in files)
    
    def _has_test_files(self, python_files: List[Dict]) -> bool:
        """Check if repository contains test files."""
        test_patterns = ['test_', '_test.py', 'tests.py']
        return any(
            any(pattern in f['name'].lower() for pattern in test_patterns)
            for f in python_files
        )
    
    async def _analyze_python_files(self, python_files: List[Dict]) -> Dict[str, Any]:
        """Analyze Python source files."""
        analysis = {
            'code_patterns': {pattern: 0 for pattern in self.ml_patterns.keys()},
            'quality_metrics': {pattern: 0 for pattern in self.quality_patterns.keys()},
            'complexity_metrics': {
                'total_lines': 0,
                'total_functions': 0,
                'total_classes': 0,
                'average_function_length': 0,
                'cyclomatic_complexity': 0
            },
            'ai_ml_indicators': {},
            'code_samples': []
        }
        
        total_files_analyzed = 0
        
        # Analyze up to 20 Python files for performance
        for file_info in python_files[:20]:
            file_analysis = await self._analyze_python_file(file_info)
            if file_analysis:
                total_files_analyzed += 1
                
                # Aggregate patterns
                for pattern, count in file_analysis.get('patterns', {}).items():
                    if pattern in analysis['code_patterns']:
                        analysis['code_patterns'][pattern] += count
                
                # Aggregate quality metrics
                for metric, count in file_analysis.get('quality', {}).items():
                    if metric in analysis['quality_metrics']:
                        analysis['quality_metrics'][metric] += count
                
                # Aggregate complexity
                complexity = file_analysis.get('complexity', {})
                analysis['complexity_metrics']['total_lines'] += complexity.get('lines', 0)
                analysis['complexity_metrics']['total_functions'] += complexity.get('functions', 0)
                analysis['complexity_metrics']['total_classes'] += complexity.get('classes', 0)
                
                # Add interesting code samples
                if file_analysis.get('code_sample'):
                    analysis['code_samples'].append({
                        'file': file_info['path'],
                        'sample': file_analysis['code_sample'],
                        'type': file_analysis.get('sample_type', 'general')
                    })
        
        # Calculate averages
        if analysis['complexity_metrics']['total_functions'] > 0:
            analysis['complexity_metrics']['average_function_length'] = (
                analysis['complexity_metrics']['total_lines'] / 
                analysis['complexity_metrics']['total_functions']
            )
        
        # Calculate AI/ML relevance indicators
        analysis['ai_ml_indicators'] = self._calculate_ai_ml_indicators(analysis['code_patterns'])
        
        analysis['files_analyzed'] = total_files_analyzed
        
        return analysis
    
    async def _analyze_python_file(self, file_info: Dict) -> Optional[Dict[str, Any]]:
        """Analyze a single Python file."""
        # In a real implementation, you would fetch the actual file content
        # For this example, we'll simulate the analysis
        
        file_analysis = {
            'patterns': {pattern: 0 for pattern in self.ml_patterns.keys()},
            'quality': {pattern: 0 for pattern in self.quality_patterns.keys()},
            'complexity': {
                'lines': 0,
                'functions': 0,
                'classes': 0,
                'imports': 0
            },
            'code_sample': None,
            'sample_type': None
        }
        
        # Simulate file content analysis
        # In practice, you'd use the RepoAnalyzer to get actual file content
        file_size = file_info.get('size', 0)
        file_name = file_info.get('name', '')
        
        # Estimate complexity based on file size (rough approximation)
        estimated_lines = max(file_size // 50, 10)  # Rough estimate
        file_analysis['complexity']['lines'] = estimated_lines
        file_analysis['complexity']['functions'] = estimated_lines // 20
        file_analysis['complexity']['classes'] = max(estimated_lines // 100, 0)
        
        # Check for AI/ML indicators in filename
        if self._has_ml_filename_indicators(file_name):
            # Simulate finding ML patterns
            file_analysis['patterns']['model_training'] = 1
            file_analysis['patterns']['data_preprocessing'] = 1
            file_analysis['sample_type'] = 'ml_training'
        
        return file_analysis
    
    def _has_ml_filename_indicators(self, filename: str) -> bool:
        """Check if filename suggests ML/AI content."""
        ml_keywords = [
            'model', 'train', 'predict', 'neural', 'network', 'deep', 'learning',
            'classifier', 'regressor', 'features', 'data', 'preprocess'
        ]
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in ml_keywords)
    
    async def _analyze_notebooks(self, notebook_files: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze Jupyter notebooks."""
        notebook_analyses = []
        
        for notebook_info in notebook_files[:10]:  # Limit to 10 notebooks
            analysis = await self._analyze_notebook(notebook_info)
            if analysis:
                notebook_analyses.append(analysis)
        
        return notebook_analyses
    
    async def _analyze_notebook(self, notebook_info: Dict) -> Optional[Dict[str, Any]]:
        """Analyze a single Jupyter notebook."""
        # Simulate notebook analysis
        analysis = {
            'file_path': notebook_info['path'],
            'file_name': notebook_info['name'],
            'estimated_cells': max(notebook_info.get('size', 0) // 1000, 5),
            'has_markdown': True,
            'has_plots': False,
            'ml_patterns': [],
            'topics': []
        }
        
        # Check filename for topics
        filename = notebook_info['name'].lower()
        if any(word in filename for word in ['eda', 'exploration', 'analysis']):
            analysis['topics'].append('exploratory_data_analysis')
        if any(word in filename for word in ['model', 'train', 'ml']):
            analysis['topics'].append('machine_learning')
        if any(word in filename for word in ['viz', 'plot', 'graph']):
            analysis['topics'].append('visualization')
            analysis['has_plots'] = True
        
        return analysis
    
    def _calculate_ai_ml_indicators(self, code_patterns: Dict[str, int]) -> Dict[str, Any]:
        """Calculate AI/ML relevance indicators."""
        indicators = {
            'ml_score': 0.0,
            'dl_score': 0.0,
            'data_science_score': 0.0,
            'production_readiness': 0.0,
            'dominant_pattern': None
        }
        
        # Calculate scores based on pattern counts
        pattern_weights = {
            'model_training': 3.0,
            'neural_networks': 2.5,
            'deep_learning': 2.5,
            'data_preprocessing': 2.0,
            'model_evaluation': 2.0,
            'data_loading': 1.5,
            'model_persistence': 1.0
        }
        
        total_weighted_score = 0.0
        max_pattern_score = 0.0
        dominant_pattern = None
        
        for pattern, count in code_patterns.items():
            if count > 0 and pattern in pattern_weights:
                weight = pattern_weights[pattern]
                score = count * weight
                total_weighted_score += score
                
                if score > max_pattern_score:
                    max_pattern_score = score
                    dominant_pattern = pattern
        
        # Normalize scores
        if total_weighted_score > 0:
            indicators['ml_score'] = min(total_weighted_score / 10.0, 10.0)
            
            # Specific scores
            dl_patterns = ['neural_networks', 'deep_learning']
            dl_score = sum(code_patterns.get(p, 0) * pattern_weights.get(p, 0) for p in dl_patterns)
            indicators['dl_score'] = min(dl_score / 5.0, 10.0)
            
            ds_patterns = ['data_preprocessing', 'data_loading', 'model_evaluation']
            ds_score = sum(code_patterns.get(p, 0) * pattern_weights.get(p, 0) for p in ds_patterns)
            indicators['data_science_score'] = min(ds_score / 5.0, 10.0)
        
        indicators['dominant_pattern'] = dominant_pattern
        
        return indicators
    
    async def analyze_code_quality(self, file_content: str, file_type: str = 'python') -> Dict[str, Any]:
        """Analyze code quality metrics for a specific file."""
        quality = {
            'readability_score': 0.0,
            'documentation_ratio': 0.0,
            'complexity_score': 0.0,
            'maintainability_score': 0.0,
            'issues': []
        }
        
        if file_type == 'python':
            quality = await self._analyze_python_quality(file_content)
        
        return quality
    
    async def _analyze_python_quality(self, code_content: str) -> Dict[str, Any]:
        """Analyze Python code quality."""
        quality = {
            'readability_score': 0.0,
            'documentation_ratio': 0.0,
            'complexity_score': 0.0,
            'maintainability_score': 0.0,
            'issues': []
        }
        
        try:
            # Parse the code using AST
            tree = ast.parse(code_content)
            
            # Count various elements
            lines = code_content.split('\n')
            total_lines = len([line for line in lines if line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            blank_lines = len([line for line in lines if not line.strip()])
            
            # Calculate documentation ratio
            docstring_lines = self._count_docstring_lines(tree)
            total_doc_lines = comment_lines + docstring_lines
            quality['documentation_ratio'] = total_doc_lines / max(total_lines, 1)
            
            # Calculate complexity
            complexity = self._calculate_cyclomatic_complexity(tree)
            quality['complexity_score'] = min(complexity / 10.0, 10.0)
            
            # Calculate readability (simplified)
            avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
            readability = 10.0 - min(avg_line_length / 100.0 * 10, 9.0)
            quality['readability_score'] = max(readability, 1.0)
            
            # Overall maintainability
            quality['maintainability_score'] = (
                quality['readability_score'] * 0.3 +
                quality['documentation_ratio'] * 10 * 0.4 +
                (10 - quality['complexity_score']) * 0.3
            )
            
        except SyntaxError as e:
            quality['issues'].append(f"Syntax error: {e}")
        except Exception as e:
            quality['issues'].append(f"Analysis error: {e}")
        
        return quality
    
    def _count_docstring_lines(self, tree: ast.AST) -> int:
        """Count lines in docstrings."""
        docstring_lines = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value
                    docstring_lines += len(docstring.split('\n'))
        
        return docstring_lines
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    async def extract_code_samples(self, repository_contents: Dict[str, Any], 
                                 max_samples: int = 5) -> List[Dict[str, Any]]:
        """Extract interesting code samples from the repository."""
        samples = []
        
        python_files = [f for f in repository_contents.get('files', []) 
                       if f['extension'] == 'py']
        
        # Look for interesting files
        priority_patterns = [
            ('model', 'model_definition'),
            ('train', 'training_script'),
            ('predict', 'prediction_script'),
            ('main', 'main_script'),
            ('utils', 'utility_functions'),
            ('data', 'data_processing')
        ]
        
        for pattern, sample_type in priority_patterns:
            matching_files = [f for f in python_files 
                            if pattern in f['name'].lower()]
            
            for file_info in matching_files[:2]:  # Max 2 per pattern
                # Simulate extracting code sample
                sample = {
                    'file_path': file_info['path'],
                    'sample_type': sample_type,
                    'description': f"Code sample from {file_info['name']}",
                    'code': f"# Sample code from {file_info['name']}\n# (Content would be extracted from actual file)",
                    'lines': min(file_info.get('size', 0) // 50, 50)
                }
                samples.append(sample)
                
                if len(samples) >= max_samples:
                    return samples
        
        return samples
    
    async def detect_ml_workflows(self, repository_contents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect ML workflows and pipelines in the repository."""
        workflows = []
        
        files = repository_contents.get('files', [])
        
        # Look for workflow indicators
        workflow_files = [
            f for f in files 
            if any(keyword in f['name'].lower() 
                  for keyword in ['pipeline', 'workflow', 'train', 'experiment'])
        ]
        
        for file_info in workflow_files:
            workflow = {
                'file_path': file_info['path'],
                'workflow_type': self._classify_workflow_type(file_info['name']),
                'estimated_complexity': 'medium',
                'components': []
            }
            
            # Estimate workflow components based on filename
            filename = file_info['name'].lower()
            if 'data' in filename:
                workflow['components'].append('data_processing')
            if any(word in filename for word in ['train', 'fit']):
                workflow['components'].append('model_training')
            if any(word in filename for word in ['eval', 'test', 'valid']):
                workflow['components'].append('evaluation')
            if 'predict' in filename:
                workflow['components'].append('prediction')
            
            workflows.append(workflow)
        
        return workflows
    
    def _classify_workflow_type(self, filename: str) -> str:
        """Classify the type of ML workflow."""
        filename = filename.lower()
        
        if 'pipeline' in filename:
            return 'pipeline'
        elif 'experiment' in filename:
            return 'experiment'
        elif 'train' in filename:
            return 'training'
        elif 'predict' in filename:
            return 'inference'
        elif 'eval' in filename:
            return 'evaluation'
        else:
            return 'general'
    
    def generate_code_summary(self, code_analysis: Dict[str, Any], 
                            repository_name: str) -> str:
        """Generate a human-readable summary of code analysis."""
        summary_parts = [f"Code Analysis Summary for {repository_name}:"]
        
        # File statistics
        file_stats = code_analysis.get('file_statistics', {})
        total_files = file_stats.get('total_files', 0)
        python_files = file_stats.get('python_files', 0)
        notebook_files = file_stats.get('notebook_files', 0)
        
        summary_parts.append(f"Files: {total_files} total, {python_files} Python, {notebook_files} notebooks")
        
        # AI/ML indicators
        ai_ml = code_analysis.get('ai_ml_indicators', {})
        ml_score = ai_ml.get('ml_score', 0)
        dominant_pattern = ai_ml.get('dominant_pattern', 'None')
        
        summary_parts.append(f"ML Relevance Score: {ml_score:.1f}/10")
        summary_parts.append(f"Primary Pattern: {dominant_pattern}")
        
        # Code patterns
        patterns = code_analysis.get('code_patterns', {})
        active_patterns = [k for k, v in patterns.items() if v > 0]
        
        if active_patterns:
            summary_parts.append(f"Detected Patterns: {', '.join(active_patterns)}")
        
        # Complexity
        complexity = code_analysis.get('complexity_metrics', {})
        total_lines = complexity.get('total_lines', 0)
        total_functions = complexity.get('total_functions', 0)
        
        if total_lines > 0:
            summary_parts.append(f"Code Size: {total_lines} lines, {total_functions} functions")
        
        # Special features
        features = []
        if code_analysis.get('has_notebooks'):
            features.append('Jupyter notebooks')
        if code_analysis.get('has_models'):
            features.append('saved models')
        if code_analysis.get('has_tests'):
            features.append('test files')
        
        if features:
            summary_parts.append(f"Features: {', '.join(features)}")
        
        return "\n".join(summary_parts)
