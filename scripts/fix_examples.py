#!/usr/bin/env python3
"""
Fix imports in example scripts to work with installed emile-mini package
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single file"""
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # List of modules that should be prefixed with emile_mini
    emile_modules = [
        'agent', 'config', 'qse_core', 'symbolic', 'context', 'goal', 
        'memory', 'viz', 'simulator', 'embodied_qse_emile', 
        'social_qse_agent_v2', 'maze_environment', 'maze_comparison',
        'visual_maze_demo', 'extinction_experiment'
    ]
    
    # Fix "from module import ..." patterns
    for module in emile_modules:
        # Pattern: from module import ...
        pattern = f'from {module} import'
        replacement = f'from emile_mini.{module} import'
        content = content.replace(pattern, replacement)
        
        # Pattern: import module
        pattern = f'^import {module}$'
        replacement = f'import emile_mini.{module} as {module}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Special case for relative imports like "from . import"
    content = re.sub(r'from \. import', 'from emile_mini import', content)
    
    # Check if we made changes
    if content != original_content:
        # Backup original
        backup_path = str(filepath) + '.backup'
        with open(backup_path, 'w') as f:
            f.write(original_content)
        
        # Write fixed version
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {filepath}")
        print(f"   (backup saved as {backup_path})")
        return True
    else:
        print(f"⏭️  No changes needed for {filepath}")
        return False

def fix_all_examples(examples_dir="examples"):
    """Fix imports in all Python files in the examples directory"""
    
    examples_path = Path(examples_dir)
    if not examples_path.exists():
        print(f"❌ Examples directory '{examples_dir}' not found")
        return
    
    print(f"🔧 Fixing imports in {examples_dir}/")
    print("=" * 50)
    
    fixed_count = 0
    
    # Find all Python files
    python_files = list(examples_path.glob("*.py"))
    
    if not python_files:
        print(f"❌ No Python files found in {examples_dir}/")
        return
    
    for py_file in python_files:
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print("=" * 50)
    print(f"🎉 Fixed {fixed_count}/{len(python_files)} files")
    
    if fixed_count > 0:
        print("\n📝 Now you can run the examples like:")
        for py_file in python_files:
            print(f"   python {py_file}")

def restore_backups(examples_dir="examples"):
    """Restore all backup files"""
    
    examples_path = Path(examples_dir)
    backup_files = list(examples_path.glob("*.backup"))
    
    if not backup_files:
        print("📁 No backup files found")
        return
    
    print(f"🔄 Restoring {len(backup_files)} backup files...")
    
    for backup_file in backup_files:
        original_file = backup_file.with_suffix('')
        
        # Read backup content
        with open(backup_file, 'r') as f:
            content = f.read()
        
        # Write to original file
        with open(original_file, 'w') as f:
            f.write(content)
        
        # Remove backup
        backup_file.unlink()
        
        print(f"✅ Restored {original_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix imports in emile-mini example scripts")
    parser.add_argument("--examples-dir", default="examples", 
                       help="Path to examples directory (default: examples)")
    parser.add_argument("--restore", action="store_true",
                       help="Restore backup files instead of fixing")
    
    args = parser.parse_args()
    
    if args.restore:
        restore_backups(args.examples_dir)
    else:
        fix_all_examples(args.examples_dir)

if __name__ == "__main__":
    main()
