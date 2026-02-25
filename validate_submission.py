"""
Validate your model before DIAMBRA submission
"""
import os
import zipfile


def validate_submission():
    print("=" * 70)
    print("DIAMBRA Submission Validation")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    # Check 1: Model file exists
    checks_total += 1
    model_path = "./models_phase3/sfiii_phase3_4078200_steps.zip"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file exists: {model_path} ({size_mb:.1f} MB)")
        checks_passed += 1
    else:
        print(f"✗ Model file NOT found: {model_path}")

    # Check 2: Model is valid ZIP
    checks_total += 1
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            print(f"✓ Model ZIP is valid ({len(files)} files inside)")
            checks_passed += 1
    except Exception as e:
        print(f"✗ Model ZIP validation failed: {e}")

    # Check 3: Agent wrapper exists
    checks_total += 1
    agent_path = "./submit_agent.py"
    if os.path.exists(agent_path):
        print(f"✓ Agent wrapper script exists: {agent_path}")
        checks_passed += 1
    else:
        print(f"✗ Agent wrapper NOT found: {agent_path}")

    # Check 4: Verify model contains required files
    checks_total += 1
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            required_files = ['data', 'pytorch_variables.pth']
            has_required = all(any(req in f for f in files) for req in required_files)
            if has_required:
                print(f"✓ Model contains required PPO files")
                checks_passed += 1
            else:
                print(f"✗ Model missing required files. Found: {files}")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")

    # Skip live loading test (can hang due to multiprocessing issues)
    print("⚠ Skipping live model loading test (test with submit_agent.py instead)")

    # Summary
    print("\n" + "=" * 70)
    print(f"Validation: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print("✓ Ready for submission!")
        print("\nSubmission file: ./models_phase3/sfiii_phase3_4078200_steps.zip")
        return True
    else:
        print("✗ Fix issues before submission")
        return False


if __name__ == "__main__":
    validate_submission()