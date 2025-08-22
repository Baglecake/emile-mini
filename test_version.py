#!/usr/bin/env python3
"""
Simple test script to verify CLI version output for v0.5.0 release.
"""

import subprocess
import sys


def test_cli_version():
    """Test that CLI outputs correct version."""
    try:
        # Test the installed command first
        result = subprocess.run(
            ["emile-mini", "-V"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        expected = "emile-mini 0.5.0"
        actual = (result.stdout + result.stderr).strip()
        
        if expected in actual and result.returncode == 0:
            print(f"✅ CLI installed command test PASSED: '{actual}'")
            return True
        else:
            print(f"❌ CLI installed command test FAILED: expected '{expected}', got '{actual}' (returncode: {result.returncode})")
            
            # Fallback to testing the module import directly
            try:
                import sys
                import io
                from contextlib import redirect_stdout, redirect_stderr
                from emile_mini.cli import main

                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        main(['-V'])
                except SystemExit:
                    pass

                captured = stdout_capture.getvalue().strip()
                if expected in captured:
                    print(f"✅ CLI module test PASSED: '{captured}'")
                    return True
                else:
                    print(f"❌ CLI module test FAILED: expected '{expected}', got '{captured}'")
                    return False
                    
            except Exception as e:
                print(f"❌ CLI module test FAILED: {e}")
                return False
            
    except FileNotFoundError:
        print("⚠️ emile-mini command not found, trying module import...")
        # Try module import directly as fallback
        try:
            import sys
            import io
            from contextlib import redirect_stdout, redirect_stderr
            from emile_mini.cli import main

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    main(['-V'])
            except SystemExit:
                pass

            expected = "emile-mini 0.5.0"
            captured = stdout_capture.getvalue().strip()
            if expected in captured:
                print(f"✅ CLI module test PASSED: '{captured}'")
                return True
            else:
                print(f"❌ CLI module test FAILED: expected '{expected}', got '{captured}'")
                return False
                
        except Exception as e:
            print(f"❌ CLI module test FAILED: {e}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI version test FAILED: command timed out")
        return False
    except Exception as e:
        print(f"❌ CLI version test FAILED: {e}")
        return False


def test_package_version():
    """Test that package __version__ is correct."""
    try:
        import emile_mini
        expected = "0.5.0"
        actual = emile_mini.__version__
        
        if actual == expected:
            print(f"✅ Package version test PASSED: '{actual}'")
            return True
        else:
            print(f"❌ Package version test FAILED: expected '{expected}', got '{actual}'")
            return False
            
    except Exception as e:
        print(f"❌ Package version test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("Testing v0.5.0 release version consistency...")
    print("=" * 50)
    
    success = True
    success &= test_package_version()
    success &= test_cli_version()
    
    print("=" * 50)
    if success:
        print("🎉 All version tests PASSED!")
        sys.exit(0)
    else:
        print("💥 Some version tests FAILED!")
        sys.exit(1)