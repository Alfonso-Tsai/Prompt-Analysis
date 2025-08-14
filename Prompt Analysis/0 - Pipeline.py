import subprocess, sys, shlex
from pathlib import Path

def run_script(script_path, *args, timeout=None):
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path), *args]
    print("🚀 Running:", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        print(f"✅ Finished: {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_path} (returncode={e.returncode})")
        sys.exit(e.returncode)

if __name__ == "__main__":

    normalization_script = "1 - Normalization.py"
    normalization_script_predict = "1-1 - Normalization Predict.py"
    vectorization_script = "2 - Vectorization.py"
    model_build_script = "3 - Build Model.py"
    prediction_script = "4 - Prediction.py"

    test_input_file = "RA_analysis.xlsx"
    vectorized_test_input_file = "RA_analysis - Normalized.xlsx"
    model_build_input_file = "RA_analysis - Normalized - Vectorized.csv"

    prediction_input_file = "feedback-merge-categorization.xlsx"
    vectorized_prediction_input_file = "feedback-merge-categorization - Normalized.xlsx"
    model_prediction_input_file = "feedback-merge-categorization - Normalized - Vectorized.csv"

    # run_script(normalization_script, "--input", test_input_file)
    # run_script(normalization_script_predict, "--input", prediction_input_file)
    # run_script(vectorization_script, "--input", vectorized_test_input_file)
    # run_script(vectorization_script, "--input", vectorized_prediction_input_file)
    # run_script(model_build_script, "--input", model_build_input_file)
    run_script(prediction_script, "--input", model_prediction_input_file)

    print("🎯 Pipeline completed successfully.")
