#!/usr/bin/env python3
"""
Émile-Mini Code Analyzer
========================

Analyzes Python modules and generates plain-language summaries of functions,
classes, and their purposes. Perfect for documentation and code understanding.

Usage:
    python code_analyzer.py
    python code_analyzer.py --module agent.py
    python code_analyzer.py --all
"""

import ast
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict

class CodeAnalyzer:
    """Analyzes Python code and generates human-readable summaries"""

    def __init__(self):
        self.modules = {}
        self.summaries = defaultdict(dict)

    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analyze a single Python file and extract information"""

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)
            module_info = {
                'filepath': filepath,
                'docstring': ast.get_docstring(tree),
                'functions': [],
                'classes': [],
                'imports': [],
                'summary': self._generate_module_summary(filepath.name)
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node)
                    module_info['functions'].append(func_info)

                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    module_info['classes'].append(class_info)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    module_info['imports'].extend(import_info)

            return module_info

        except Exception as e:
            return {
                'filepath': filepath,
                'error': str(e),
                'summary': f"Could not analyze {filepath.name}: {e}"
            }

    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition"""

        # Extract parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Generate plain language summary
        purpose = self._infer_function_purpose(node.name, params, docstring)

        return {
            'name': node.name,
            'parameters': params,
            'docstring': docstring,
            'purpose': purpose,
            'is_private': node.name.startswith('_'),
            'is_property': any(isinstance(d, ast.Name) and d.id == 'property'
                             for d in node.decorator_list)
        }

    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition"""

        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                methods.append(method_info)

        docstring = ast.get_docstring(node)
        purpose = self._infer_class_purpose(node.name, methods, docstring if docstring is not None else "") # Pass empty string if docstring is None

        return {
            'name': node.name,
            'docstring': docstring,
            'methods': methods,
            'purpose': purpose,
            'base_classes': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }

    def _analyze_import(self, node) -> List[str]:
        """Analyze import statements"""
        imports = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

        return imports

    def _infer_function_purpose(self, name: str, params: List[str], docstring: str | None) -> str: # Allow docstring to be None
        """Generate plain language description of function purpose"""

        if docstring:
            # Extract first line of docstring as primary purpose
            first_line = docstring.split('\n')[0].strip()
            if first_line:
                return first_line

        # Infer from name patterns
        if name.startswith('get_'):
            return f"Retrieves {name[4:].replace('_', ' ')}"
        elif name.startswith('set_'):
            return f"Sets or updates {name[4:].replace('_', ' ')}"
        elif name.startswith('is_') or name.startswith('has_'):
            return f"Checks if {name[3:].replace('_', ' ')}"
        elif name.startswith('create_'):
            return f"Creates {name[7:].replace('_', ' ')}"
        elif name.startswith('update_'):
            return f"Updates {name[7:].replace('_', ' ')}"
        elif name.startswith('calculate_'):
            return f"Calculates {name[10:].replace('_', ' ')}"
        elif name.startswith('process_'):
            return f"Processes {name[8:].replace('_', ' ')}"
        elif name.startswith('generate_'):
            return f"Generates {name[9:].replace('_', ' ')}"
        elif name.startswith('analyze_'):
            return f"Analyzes {name[8:].replace('_', ' ')}"
        elif name == '__init__':
            return "Initializes the object with starting values"
        elif name == '__str__':
            return "Returns string representation of the object"
        elif name == '__repr__':
            return "Returns detailed string representation for debugging"
        elif name == 'step':
            return "Performs one iteration or cycle of the process"
        elif 'execute' in name:
            return f"Executes {name.replace('execute_', '').replace('_', ' ')}"
        elif 'run' in name:
            return f"Runs {name.replace('run_', '').replace('_', ' ')}"
        else:
            # Generic inference based on parameters
            if not params:
                return f"Performs {name.replace('_', ' ')} operation"
            elif len(params) == 1 and params[0] == 'self':
                return f"Performs {name.replace('_', ' ')} on this object"
            else:
                return f"Performs {name.replace('_', ' ')} using {', '.join(params[1:] if params and params[0] == 'self' else params)}"

    def _infer_class_purpose(self, name: str, methods: List[Dict], docstring: str) -> str:
        """Generate plain language description of class purpose"""

        if docstring:
            first_line = docstring.split('\n')[0].strip()
            if first_line:
                return first_line

        # Infer from name patterns
        if name.endswith('Agent'):
            return f"An intelligent agent that can {self._infer_agent_capabilities(methods)}"
        elif name.endswith('Module'):
            return f"A module that handles {name[:-6].lower().replace('_', ' ')} functionality"
        elif name.endswith('Manager'):
            return f"Manages {name[:-7].lower().replace('_', ' ')} operations and state"
        elif name.endswith('Engine'):
            return f"Core engine for {name[:-6].lower().replace('_', ' ')} processing"
        elif name.endswith('Environment'):
            return f"Environment where agents can interact and {self._infer_environment_features(methods)}"
        elif 'Memory' in name:
            return f"Memory system for storing and retrieving {name.lower().replace('memory', '').replace('_', ' ')} information"
        elif 'Context' in name:
            return f"Manages context and {name.lower().replace('context', '').replace('_', ' ')} state transitions"
        else:
            # Generic inference
            method_purposes = [m['purpose'] for m in methods if not m['name'].startswith('_')]
            if method_purposes:
                return f"Handles {name.replace('_', ' ').lower()} with capabilities like {', '.join(method_purposes[:3])}"
            else:
                return f"A {name.replace('_', ' ').lower()} class"

    def _infer_agent_capabilities(self, methods: List[Dict]) -> str:
        """Infer what an agent can do based on its methods"""
        capabilities = []
        for method in methods:
            if 'step' in method['name']:
                capabilities.append("take actions")
            elif 'learn' in method['name']:
                capabilities.append("learn from experience")
            elif 'goal' in method['name']:
                capabilities.append("pursue goals")
            elif 'memory' in method['name']:
                capabilities.append("remember experiences")
            elif 'social' in method['name']:
                capabilities.append("interact socially")

        return ', '.join(capabilities) if capabilities else "perform cognitive operations"

    def _infer_environment_features(self, methods: List[Dict]) -> str:
        """Infer what an environment provides based on its methods"""
        features = []
        for method in methods:
            if 'step' in method['name']:
                features.append("take actions")
            elif 'reset' in method['name']:
                features.append("restart scenarios")
            elif 'reward' in method['name']:
                features.append("receive rewards")
            elif 'observe' in method['name'] or 'visual' in method['name']:
                features.append("make observations")

        return ', '.join(features) if features else "exist and interact"

    def _generate_module_summary(self, filename: str) -> str:
        """Generate overall module purpose summary"""

        summaries = {
            'agent.py': "Main cognitive agent that orchestrates all mental processes",
            'qse_core.py': "Quantum surplus emergence engine - the core dynamics",
            'symbolic.py': "Converts quantum states into symbolic reasoning",
            'context.py': "Manages how the agent reframes situations",
            'goal.py': "Handles goal formation and pursuit using Q-learning",
            'memory.py': "Hierarchical memory system (working, episodic, semantic)",
            'config.py': "Configuration settings and parameters",
            'embodied_qse_emile.py': "Embodied version with spatial navigation",
            'social_qse_agent_v2.py': "Social agents that teach and learn from each other",
            'main.py': "Entry point for running basic experiments",
            'simulator.py': "Simulation runner and experiment orchestration",
            'viz.py': "Visualization and plotting utilities",
            'experiment_runner.py': "Research-grade experimental framework"
        }

        return summaries.get(filename, f"Module handling {filename.replace('.py', '').replace('_', ' ')} functionality")

    def analyze_project(self, project_path: Path = None) -> Dict[str, Any]:
        """Analyze all Python files in the project"""

        if project_path is None:
            project_path = Path('.')

        python_files = list(project_path.glob('*.py'))
        project_analysis = {
            'project_path': project_path,
            'modules': {},
            'total_functions': 0,
            'total_classes': 0,
            'module_purposes': {}
        }

        for py_file in python_files:
            if py_file.name.startswith('.') or py_file.name.startswith('__'):
                continue

            module_info = self.analyze_file(py_file)
            project_analysis['modules'][py_file.name] = module_info

            if 'functions' in module_info:
                project_analysis['total_functions'] += len(module_info['functions'])
            if 'classes' in module_info:
                project_analysis['total_classes'] += len(module_info['classes'])

            project_analysis['module_purposes'][py_file.name] = module_info.get('summary', 'Unknown purpose')

        return project_analysis

    def generate_report(self, analysis: Dict[str, Any], detailed: bool = True) -> str:
        """Generate a human-readable report"""

        report = []
        report.append("🧠 ÉMILE-MINI CODE ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"📁 Project: {analysis['project_path']}")
        report.append(f"📊 Total modules: {len(analysis['modules'])}")
        report.append(f"🔧 Total functions: {analysis['total_functions']}")
        report.append(f"🏗️ Total classes: {analysis['total_classes']}")
        report.append("")

        # Module overview
        report.append("📋 MODULE OVERVIEW")
        report.append("-" * 30)
        for module_name, purpose in analysis['module_purposes'].items():
            report.append(f"• {module_name:<25} → {purpose}")
        report.append("")

        if detailed:
            # Detailed analysis per module
            for module_name, module_info in analysis['modules'].items():
                if 'error' in module_info:
                    report.append(f"❌ {module_name}: {module_info['error']}")
                    continue

                report.append(f"📦 {module_name.upper()}")
                report.append("-" * len(module_name))
                report.append(f"Purpose: {module_info['summary']}")

                if module_info.get('docstring'):
                    report.append(f"Description: {module_info['docstring'].split(chr(10))[0]}")

                # Classes
                if module_info.get('classes'):
                    report.append(f"\n🏗️ Classes ({len(module_info['classes'])}):")
                    for cls in module_info['classes']:
                        report.append(f"  • {cls['name']}: {cls['purpose']}")
                        if cls['methods']:
                            key_methods = [m for m in cls['methods'] if not m['name'].startswith('_')][:3]
                            if key_methods:
                                method_list = ', '.join([m['name'] for m in key_methods])
                                report.append(f"    Key methods: {method_list}")

                # Functions
                if module_info.get('functions'):
                    public_functions = [f for f in module_info['functions'] if not f['is_private']]
                    if public_functions:
                        report.append(f"\n🔧 Public Functions ({len(public_functions)}):")
                        for func in public_functions:
                            report.append(f"  • {func['name']}: {func['purpose']}")

                report.append("")

        return "\n".join(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Analyze Émile-Mini code and generate summaries')
    parser.add_argument('--module', '-m', help='Analyze specific module')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all modules')
    parser.add_argument('--brief', '-b', action='store_true', help='Brief summary only')
    parser.add_argument('--output', '-o', help='Save report to file')

    args = parser.parse_args()

    analyzer = CodeAnalyzer()

    if args.module:
        # Analyze single module
        module_path = Path(args.module)
        if not module_path.exists():
            print(f"❌ Module {args.module} not found")
            return

        module_info = analyzer.analyze_file(module_path)
        analysis = {
            'project_path': Path('.'),
            'modules': {module_path.name: module_info},
            'total_functions': len(module_info.get('functions', [])),
            'total_classes': len(module_info.get('classes', [])),
            'module_purposes': {module_path.name: module_info.get('summary', 'Unknown')}
        }
    else:
        # Analyze whole project
        analysis = analyzer.analyze_project()

    # Generate report
    detailed = not args.brief
    report = analyzer.generate_report(analysis, detailed=detailed)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {args.output}")
    else:
        print(report)

if __name__ == "__main__":
    main()
