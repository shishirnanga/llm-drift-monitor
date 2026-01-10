import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "" if exists else ""
    print(f"  {status} {filepath}")
    return exists


def check_directory_exists(dirpath: str) -> bool:
    """Check if a directory exists and print status."""
    exists = os.path.isdir(dirpath)
    status = "" if exists else ""
    print(f"  {status} {dirpath}/")
    return exists


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           LLM DRIFT MONITOR - SETUP VERIFICATION              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    all_good = True
    
    # Step 1: Check file structure
    print("1⃣  Checking file structure...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        ".env",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/base.py",
        "src/models/openai_model.py",
        "src/models/anthropic_model.py",
        "src/tests/__init__.py",
        "src/tests/test_cases.py",
        "src/tests/scoring.py",
        "src/storage/__init__.py",
        "src/storage/results.py",
        "src/runner.py",
    ]
    
    for f in required_files:
        if not check_file_exists(f):
            all_good = False
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "dashboard",
        "notebooks",
        "scripts",
    ]
    
    for d in required_dirs:
        if not check_directory_exists(d):
            all_good = False
    
    if not all_good:
        print("\n Some files/directories are missing!")
        print("   Please create them before continuing.")
        return 1
    
    print("\n   All files present! ")
    
    # Step 2: Check environment variables
    print("\n2⃣  Checking API keys...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    has_openai = openai_key and not openai_key.startswith("sk-your")
    has_anthropic = anthropic_key and not anthropic_key.startswith("sk-ant-your")
    
    print(f"  {'' if has_openai else ''} OPENAI_API_KEY")
    print(f"  {'' if has_anthropic else ''} ANTHROPIC_API_KEY")
    
    if not has_openai and not has_anthropic:
        print("\n No API keys found!")
        print("   Please add your API keys to the .env file.")
        return 1
    
    print("\n   API keys configured! ")
    
    # Step 3: Test imports
    print("\n3⃣  Testing imports...")
    
    try:
        from src.models import OpenAIModel, AnthropicModel, ModelResponse
        print("   src.models")
    except Exception as e:
        print(f"   src.models: {e}")
        all_good = False
    
    try:
        from src.tests import get_all_tests, score_response, TestCase
        print("   src.tests")
    except Exception as e:
        print(f"   src.tests: {e}")
        all_good = False
    
    try:
        from src.storage import ResultsStorage, TestResult, TestRun
        print("   src.storage")
    except Exception as e:
        print(f"   src.storage: {e}")
        all_good = False
    
    try:
        from src.runner import DriftMonitorRunner
        print("   src.runner")
    except Exception as e:
        print(f"   src.runner: {e}")
        all_good = False
    
    if not all_good:
        print("\n Import errors! Check the error messages above.")
        return 1
    
    print("\n   All imports working! ")
    
    # Step 4: Test API connections
    print("\n4⃣  Testing API connections...")
    
    models_working = []
    
    if has_openai:
        try:
            from src.models import OpenAIModel
            model = OpenAIModel()
            response = model.query("Say 'hello' and nothing else.")
            if response.success:
                print(f"   OpenAI API working (latency: {response.latency_ms}ms)")
                models_working.append(model)
            else:
                print(f"   OpenAI API error: {response.error}")
        except Exception as e:
            print(f"   OpenAI API error: {e}")
    
    if has_anthropic:
        try:
            from src.models import AnthropicModel
            model = AnthropicModel()
            response = model.query("Say 'hello' and nothing else.")
            if response.success:
                print(f"   Anthropic API working (latency: {response.latency_ms}ms)")
                models_working.append(model)
            else:
                print(f"   Anthropic API error: {response.error}")
        except Exception as e:
            print(f"   Anthropic API error: {e}")
    
    if not models_working:
        print("\n No working API connections!")
        return 1
    
    print(f"\n   {len(models_working)} API(s) working! ")
    
    # Step 5: Run mini test suite
    print("\n5⃣  Running mini test suite (3 tests)...")
    
    from src.tests import get_all_tests
    from src.runner import DriftMonitorRunner
    
    # Get just 3 tests
    tests = get_all_tests()[:3]
    
    runner = DriftMonitorRunner(models=models_working, verbose=False)
    run = runner.run_all_tests(tests=tests, save=True)
    
    print(f"   Ran {len(run.results)} tests")
    
    # Show results
    for result in run.results:
        status = "" if result.score == 1.0 else ""
        print(f"     {status} {result.test_id} ({result.model_name}): {result.score:.0%}")
    
    avg_score = sum(r.score for r in run.results) / len(run.results)
    print(f"\n   Average score: {avg_score:.1%}")
    
    # Step 6: Verify storage
    print("\n6⃣  Verifying storage...")
    
    from src.storage import ResultsStorage
    storage = ResultsStorage()
    
    loaded_run = storage.load_run(run.run_id)
    if loaded_run:
        print(f"   Results saved and loaded successfully")
        print(f"     File: data/raw/run_{run.run_id}.json")
    else:
        print("   Could not load saved results")
        all_good = False
    
    # Final summary
    print("\n" + "="*60)
    if all_good:
        print(" ALL CHECKS PASSED!")
        print("="*60)
        print("""
Your LLM Drift Monitor is ready to use!

Next steps:
  1. Run the full test suite:
     python main.py

  2. Set up daily automated runs (see scripts/schedule.sh)

  3. After a few days of data, check the dashboard:
     streamlit run dashboard/app.py

Happy monitoring! 🚀
        """)
        return 0
    else:
        print(" SOME CHECKS FAILED")
        print("="*60)
        print("Please fix the issues above before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())