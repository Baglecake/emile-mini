
#!/usr/bin/env python3
"""
Émile-Mini Project Analyzer
===========================
Analyzes the /content/emile_mini directory and generates plain-speech summaries.

Usage:
    python analyze_emile.py
    python analyze_emile.py --path /content/emile_mini
    python analyze_emile.py --detailed
"""

import ast
import os
from pathlib import Path
import argparse
from collections import defaultdict

class EmileAnalyzer:
    """Analyzes Émile-mini project structure and generates summaries"""
    
    def __init__(self, project_path="/content/emile_mini"):
        self.project_path = Path(project_path)
        self.results = {}
        
    def analyze_file(self, filepath):
        """Analyze a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree) or ""
            
            classes = []
            functions = []
            imports = []
            
            # Walk through AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    classes.append(class_info)
                elif isinstance(node, ast.FunctionDef):
                    # Check if it's a top-level function (not inside a class)
                    if self._is_top_level_function(node, tree):
                        func_info = self._analyze_function(node)
                        if not func_info['is_private']:
                            functions.append(func_info)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_imports(node)
                    imports.extend(import_info)
            
            return {
                'filename': filepath.name,
                'filepath': str(filepath),
                'module_docstring': module_docstring,
                'classes': classes,
                'functions': functions,
                'imports': imports,
                'summary': self._get_module_summary(filepath.name),
                'lines_of_code': len(source.splitlines())
            }
            
        except Exception as e:
            return {
                'filename': filepath.name,
                'filepath': str(filepath),
                'error': str(e),
                'summary': f"Could not analyze {filepath.name}: {e}"
            }
    
    def _analyze_class(self, node):
        """Analyze a class definition"""
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                methods.append(method_info)
        
        public_methods = [m for m in methods if not m['is_private']]
        key_methods = public_methods[:5]  # Top 5 public methods
        
        return {
            'name': node.name,
            'purpose': self._infer_class_purpose(node.name, methods),
            'methods': methods,
            'public_methods': public_methods,
            'key_methods': key_methods,
            'method_count': len(methods)
        }
    
    def _analyze_function(self, node):
        """Analyze a function definition"""
        params = [arg.arg for arg in node.args.args]
        docstring = ast.get_docstring(node) or ""
        
        return {
            'name': node.name,
            'parameters': params,
            'docstring': docstring,
            'purpose': self._infer_function_purpose(node.name, params, docstring),
            'is_private': node.name.startswith('_'),
            'param_count': len(params)
        }
    
    def _analyze_imports(self, node):
        """Analyze import statements"""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    
    def _is_top_level_function(self, func_node, tree):
        """Check if function is at module level (not inside a class)"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return False
        return True
    
    def _infer_function_purpose(self, name, params, docstring=""):
        """Infer function purpose from name, parameters, and docstring"""
        
        # Use docstring if available and meaningful
        if docstring and len(docstring.strip()) > 10:
            first_line = docstring.split('\n')[0].strip()
            if first_line and not first_line.startswith('"""'):
                return first_line
        
        # Infer from name patterns
        patterns = {
            'get_': f"Retrieves {name[4:].replace('_', ' ')}",
            'set_': f"Sets or updates {name[4:].replace('_', ' ')}",
            'is_': f"Checks if {name[3:].replace('_', ' ')}",
            'has_': f"Checks if {name[4:].replace('_', ' ')}",
            'create_': f"Creates {name[7:].replace('_', ' ')}",
            'update_': f"Updates {name[7:].replace('_', ' ')}",
            'calculate_': f"Calculates {name[10:].replace('_', ' ')}",
            'process_': f"Processes {name[8:].replace('_', ' ')}",
            'generate_': f"Generates {name[9:].replace('_', ' ')}",
            'analyze_': f"Analyzes {name[8:].replace('_', ' ')}",
            'run_': f"Runs {name[4:].replace('_', ' ')}",
            'execute_': f"Executes {name[8:].replace('_', ' ')}"
        }
        
        for prefix, description in patterns.items():
            if name.startswith(prefix):
                return description
        
        # Special cases
        if name == '__init__':
            return "Initializes the object with starting values"
        elif name == 'step':
            return "Performs one iteration or cycle of the process"
        elif name == 'main':
            return "Main entry point for the module"
        elif 'plot' in name:
            return f"Creates visualization for {name.replace('plot_', '').replace('_', ' ')}"
        else:
            return f"Performs {name.replace('_', ' ')} operation"
    
    def _infer_class_purpose(self, name, methods):
        """Infer class purpose from name and methods"""
        
        method_names = [m['name'] for m in methods]
        
        if name.endswith('Agent'):
            capabilities = []
            if any('step' in m for m in method_names):
                capabilities.append("take actions")
            if any('learn' in m for m in method_names):
                capabilities.append("learn from experience")
            if any('goal' in m for m in method_names):
                capabilities.append("pursue goals")
            if any('memory' in m or 'remember' in m for m in method_names):
                capabilities.append("remember experiences")
            if any('social' in m for m in method_names):
                capabilities.append("interact socially")
            
            caps_str = ', '.join(capabilities) if capabilities else "perform cognitive operations"
            return f"An intelligent agent that can {caps_str}"
        
        elif name.endswith('Module'):
            return f"A module that handles {name[:-6].lower().replace('_', ' ')} functionality"
        elif name.endswith('Manager'):
            return f"Manages {name[:-7].lower().replace('_', ' ')} operations and state"
        elif name.endswith('Engine'):
            return f"Core engine for {name[:-6].lower().replace('_', ' ')} processing"
        elif name.endswith('Environment'):
            return f"Environment where agents can interact and take actions"
        elif 'Memory' in name:
            return f"Memory system for storing and retrieving information"
        elif 'Context' in name:
            return f"Manages context and state transitions"
        elif 'Config' in name:
            return f"Configuration settings and parameters"
        else:
            return f"Handles {name.replace('_', ' ').lower()} operations"
    
    def _get_module_summary(self, filename):
        """Get specific module purpose"""
        summaries = {
            'agent.py': 'Main cognitive agent that orchestrates all mental processes',
            'qse_core.py': 'Quantum surplus emergence engine - the core dynamics',
            'symbolic.py': 'Converts quantum states into symbolic reasoning',
            'context.py': 'Manages how the agent reframes situations',
            'goal.py': 'Handles goal formation and pursuit using Q-learning',
            'memory.py': 'Hierarchical memory system (working, episodic, semantic)',
            'config.py': 'Configuration settings and parameters',
            'embodied_qse_emile.py': 'Embodied version with spatial navigation',
            'social_qse_agent_v2.py': 'Social agents that teach and learn from each other',
            'main.py': 'Entry point for running basic experiments',
            'simulator.py': 'Simulation runner and experiment orchestration',
            'viz.py': 'Visualization and plotting utilities',
            'cli.py': 'Command-line interface for easy usage',
            '__init__.py': 'Package initialization and exports'
        }
        
        return summaries.get(filename, f"Module handling {filename.replace('.py', '').replace('_', ' ')} functionality")
    
    def analyze_project(self):
        """Analyze the entire project"""
        
        if not self.project_path.exists():
            return {'error': f"Path {self.project_path} does not exist"}
        
        # Find all Python files
        python_files = list(self.project_path.glob('*.py'))
        
        project_analysis = {
            'project_path': str(self.project_path),
            'total_files': len(python_files),
            'modules': {},
            'total_classes': 0,
            'total_functions': 0,
            'total_lines': 0
        }
        
        for py_file in python_files:
            if py_file.name.startswith('.'):
                continue
            
            file_analysis = self.analyze_file(py_file)
            project_analysis['modules'][py_file.name] = file_analysis
            
            if 'classes' in file_analysis:
                project_analysis['total_classes'] += len(file_analysis['classes'])
            if 'functions' in file_analysis:
                project_analysis['total_functions'] += len(file_analysis['functions'])
            if 'lines_of_code' in file_analysis:
                project_analysis['total_lines'] += file_analysis['lines_of_code']
        
        self.results = project_analysis
        return project_analysis
    
    def generate_report(self, detailed=True):
        """Generate human-readable report"""
        
        if not self.results:
            return "No analysis results found. Run analyze_project() first."
        
        report = []
        
        # Header
        report.append("🧠 ÉMILE-MINI PROJECT ANALYSIS")
        report.append("=" * 50)
        report.append(f"📁 Project Path: {self.results['project_path']}")
        report.append(f"📊 Total Files: {self.results['total_files']}")
        report.append(f"🏗️ Total Classes: {self.results['total_classes']}")
        report.append(f"🔧 Total Functions: {self.results['total_functions']}")
        report.append(f"📝 Total Lines: {self.results['total_lines']}")
        report.append("")
        
        # Module overview
        report.append("📋 MODULE OVERVIEW")
        report.append("-" * 30)
        
        for filename, module_info in self.results['modules'].items():
            if 'error' in module_info:
                report.append(f"❌ {filename:<25} → {module_info['error']}")
            else:
                report.append(f"• {filename:<25} → {module_info['summary']}")
        report.append("")
        
        if detailed:
            # Detailed analysis
            for filename, module_info in self.results['modules'].items():
                if 'error' in module_info:
                    continue
                
                report.append(f"📦 {filename.upper()}")
                report.append("-" * len(filename))
                report.append(f"Purpose: {module_info['summary']}")
                
                if module_info.get('module_docstring'):
                    first_line = module_info['module_docstring'].split('\n')[0].strip()
                    if first_line:
                        report.append(f"Description: {first_line}")
                
                # Classes
                if module_info.get('classes'):
                    report.append(f"\n🏗️ Classes ({len(module_info['classes'])}):")
                    for cls in module_info['classes']:
                        report.append(f"  • {cls['name']}: {cls['purpose']}")
                        if cls['key_methods']:
                            method_names = [m['name'] for m in cls['key_methods'][:3]]
                            report.append(f"    Key methods: {', '.join(method_names)}")
                
                # Functions
                if module_info.get('functions'):
                    report.append(f"\n🔧 Functions ({len(module_info['functions'])}):")
                    for func in module_info['functions']:
                        report.append(f"  • {func['name']}: {func['purpose']}")
                
                report.append(f"\n📏 Lines of code: {module_info.get('lines_of_code', 'Unknown')}")
                report.append("")
        
        # Summary insights
        report.append("🎯 ARCHITECTURE INSIGHTS")
        report.append("-" * 30)
        
        # Categorize modules
        core_modules = []
        extension_modules = []
        utility_modules = []
        
        for filename, module_info in self.results['modules'].items():
            if filename in ['agent.py', 'qse_core.py', 'symbolic.py', 'context.py', 'goal.py', 'memory.py']:
                core_modules.append(filename)
            elif 'embodied' in filename or 'social' in filename:
                extension_modules.append(filename)
            else:
                utility_modules.append(filename)
        
        report.append(f"🧠 Core Cognitive ({len(core_modules)} modules):")
        report.append(f"   {', '.join(core_modules)}")
        report.append("")
        
        if extension_modules:
            report.append(f"🌍 Extensions ({len(extension_modules)} modules):")
            report.append(f"   {', '.join(extension_modules)}")
            report.append("")
        
        if utility_modules:
            report.append(f"🛠️ Utilities ({len(utility_modules)} modules):")
            report.append(f"   {', '.join(utility_modules)}")
            report.append("")
        
        report.append("✨ Key Innovation:")
        report.append("   Bidirectional quantum-symbolic causation enables enactive cognition")
        
        return "\n".join(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Analyze Émile-Mini project structure')
    parser.add_argument('--path', '-p', default='/content/emile_mini', 
                       help='Path to emile-mini project (default: /content/emile_mini)')
    parser.add_argument('--brief', '-b', action='store_true', 
                       help='Generate brief summary only')
    parser.add_argument('--output', '-o', help='Save report to file')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EmileAnalyzer(args.path)
    
    # Analyze project
    print("🔍 Analyzing Émile-mini project...")
    results = analyzer.analyze_project()
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Generate report
    detailed = not args.brief
    report = analyzer.generate_report(detailed=detailed)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {args.output}")
    else:
        print(report)

if __name__ == "__main__":
    main()
