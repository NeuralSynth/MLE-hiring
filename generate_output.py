import sys
import os
import subprocess

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(root_dir, "code")
    main_script = os.path.join(code_dir, "main.py")
    validate_script = os.path.join(code_dir, "validate_output.py")
    
    print("Starting Multi-Domain Support Triage Agent pipeline...")
    
    # Run main.py
    env = os.environ.copy()
    env["PYTHONPATH"] = code_dir + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run([sys.executable, main_script], cwd=root_dir, env=env)
    
    if result.returncode != 0:
        print("Error: Pipeline execution failed.")
        sys.exit(result.returncode)
        
    print("\nPipeline execution complete. Starting output format validation...")
    
    # Run validate_output.py
    result_val = subprocess.run([sys.executable, validate_script], cwd=code_dir, env=env)
    
    if result_val.returncode == 0:
        print("\nOutput generated and validated successfully!")
    else:
        print("\nOutput generated but format validation failed. Please check the errors above.")
        sys.exit(result_val.returncode)

if __name__ == "__main__":
    main()
