# backend/routes/evaluate.py

import json
import time
from pathlib import Path
from datetime import datetime
from flask import Blueprint, request, jsonify
from typing import Dict, Tuple, Union
from jupyter_client.manager import KernelManager, KernelClient
from queue import Empty
import pandas as pd
import numpy as np
import re

evaluation_bp = Blueprint('evaluation_api', __name__)

QUESTIONS_BASE_PATH = Path(__file__).parent.parent / "data" / "questions"
SUBMISSIONS_PATH = Path(__file__).parent.parent / "data" / "submissions"
USERS_FILE_PATH = Path(__file__).parent.parent / "data" / "users.json"
USER_GENERATED_PATH = Path(__file__).parent.parent / "data" / "user_generated"
USER_KERNELS: Dict[str, Tuple[KernelManager, KernelClient]] = {}

# --- HELPER FUNCTIONS ---
def extract_and_compare_value(student_output: str, label: str, expected_value: float, tolerance: float) -> Tuple[bool, str]:
    try:
        pattern = re.compile(re.escape(label) + r'\s*(-?[\d\.]+)')
        match = pattern.search(student_output)
        if not match: return False, f"Failed. The required label '{label}' was not found in the output."
        extracted_string = match.group(1)
        extracted_value = float(extracted_string)
        if abs(extracted_value - expected_value) <= tolerance: return True, f"Passed. Found value {extracted_value:.4f} is within tolerance."
        else: return False, f"Failed. Found value {extracted_value:.4f}, expected around {expected_value}."
    except ValueError: return False, f"Failed. Found label '{label}', but could not parse '{extracted_string}' as a number."
    except Exception as e: return False, f"An unexpected error occurred during numerical parsing: {e}"

def check_keywords_in_text(student_output: str, keywords_str: str, threshold: float = 0.8) -> Tuple[bool, str]:
    student_output_lower = student_output.lower()
    keywords = [kw.strip().lower() for kw in keywords_str.split() if kw.strip()]
    if not keywords: return True, "No keywords specified."
    matched_count = sum(1 for kw in keywords if kw in student_output_lower)
    match_ratio = matched_count / len(keywords)
    if match_ratio >= threshold: return True, f"Passed ({match_ratio:.0%})"
    else:
        missing = [kw for kw in keywords if kw not in student_output_lower]
        return False, f"Failed. Missing keywords: {missing}"

def compare_csvs(student_path: Union[Path, str], solution_path: Union[Path, str], key_columns=None, threshold: float = 0.9, tolerance: float = 1e-5) -> Tuple[bool, float]:
    try:
        student_path, solution_path = Path(student_path), Path(solution_path)
        if not student_path.exists():
            print(f"DEBUG: Student file does not exist at {student_path}")
            return False, 0.0
        if not solution_path.exists():
            print(f"DEBUG: Solution file does not exist at {solution_path}")
            return False, 0.0
        
        df_student = pd.read_csv(student_path)
        df_solution = pd.read_csv(solution_path)
        
        similarity_score = 0.0

        if key_columns and len(key_columns) == 2:
            merge_key, compare_col = key_columns
            if merge_key not in df_student.columns or merge_key not in df_solution.columns: return False, 0.0
            if compare_col not in df_student.columns or compare_col not in df_solution.columns: return False, 0.0
            df_student[merge_key] = df_student[merge_key].astype(df_solution[merge_key].dtype)
            merged = pd.merge(df_student, df_solution, on=merge_key, suffixes=('_student', '_solution'))
            col_student, col_solution = merged[f'{compare_col}_student'], merged[f'{compare_col}_solution']
            matches = np.isclose(col_student, col_solution, atol=tolerance).sum()
            similarity_score = (matches / len(col_solution)) if len(col_solution) > 0 else 1.0
        else:
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # START OF MODIFIED SECTION
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            if df_student.shape != df_solution.shape:
                print(f"DEBUG: Shape mismatch. Student: {df_student.shape}, Solution: {df_solution.shape}")
                print("DEBUG: Performing partial comparison on the top-left corner as a fallback.")

                # Determine the size of the comparison grid (up to 5x5)
                min_rows = min(df_student.shape[0], df_solution.shape[0], 5)
                min_cols = min(df_student.shape[1], df_solution.shape[1], 5)

                if min_rows == 0 or min_cols == 0:
                    print("DEBUG: Cannot perform partial comparison on empty or single-dimension data.")
                    return False, 0.0

                # Slice the dataframes to the smaller intersection
                student_subset = df_student.iloc[:min_rows, :min_cols]
                solution_subset = df_solution.iloc[:min_rows, :min_cols]
                
                # Compare only the numeric columns within this subset
                numeric_cols = solution_subset.select_dtypes(include=np.number).columns
                valid_cols = [col for col in numeric_cols if col in student_subset.columns]
                
                if not valid_cols:
                    print("DEBUG: No common numeric columns in the top-left corner to compare.")
                    return False, 0.0

                student_numeric_subset = student_subset[valid_cols]
                solution_numeric_subset = solution_subset[valid_cols]

                # Perform the tolerant comparison
                matches = np.isclose(student_numeric_subset.values, solution_numeric_subset.values, atol=tolerance).sum()
                total_cells = student_numeric_subset.size
                
                similarity_score = matches / total_cells if total_cells > 0 else 0.0
                print(f"DEBUG: Partial comparison score: {similarity_score:.2f} (Threshold: {threshold})")

            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            # END OF MODIFIED SECTION
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            else:
                # This is the original logic for when shapes match perfectly. It remains unchanged.
                numeric_cols = df_solution.select_dtypes(include=np.number).columns
                if len(numeric_cols) == 0:
                    is_equal = df_student.equals(df_solution)
                    similarity_score = 1.0 if is_equal else 0.0
                else:
                    student_numeric = df_student[numeric_cols]
                    solution_numeric = df_solution[numeric_cols]
                    matches = np.isclose(student_numeric, solution_numeric, atol=tolerance).sum()
                    total_numeric_cells = len(numeric_cols) * df_solution.shape[0]
                    similarity_score = matches / total_numeric_cells if total_numeric_cells > 0 else 1.0

        final_pass_status = similarity_score >= threshold
        
        # This debug info block is helpful and remains unchanged
        if not final_pass_status:
            print("\n--- CSV COMPARISON FAILED: DEBUG INFO ---")
            # ... (rest of the debug printing logic) ...

        return final_pass_status, similarity_score

    except Exception as e:
        print(f"ERROR during CSV comparison: {e}"); return False, 0.0   
# ------------------------------------------------

def run_code_on_kernel(kc: KernelClient, code: str, user_input: str = "", working_dir: str = None, timeout: int = 45) -> Tuple[str, str]:
    prep_script = ""
    if working_dir:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        py_working_dir = str(Path(working_dir).resolve()).replace("\\", "/")
        prep_script = f"import os\nos.chdir(r'{py_working_dir}')\n"

    full_script = f"""
{prep_script}
import builtins
_input_lines = {json.dumps(user_input)}.splitlines()
_input_lines.reverse()
def _mock_input(prompt=''):
    try: return _input_lines.pop()
    except IndexError: return ''
builtins.input = _mock_input
{code}
"""
    msg_id = kc.execute(full_script); stdout, stderr = [], []; start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        try:
            msg = kc.get_iopub_msg(timeout=1)
            if msg.get('parent_header', {}).get('msg_id') != msg_id: continue
            msg_type = msg['header']['msg_type']
            content = msg.get('content', {})
            if msg_type == 'stream':
                if content['name'] == 'stdout': stdout.append(content['text'])
                else: stderr.append(content['text'])
            elif msg_type == 'error': stderr.append('\\n'.join(content.get('traceback', [])))
            elif msg_type == 'status' and content.get('execution_state') == 'idle': break
        except Empty: pass
    else: stderr.append(f"\\n[Kernel Timeout] Execution exceeded {timeout} seconds.")
    return "".join(stdout).strip(), "".join(stderr).strip()

@evaluation_bp.route('/session/start', methods=['POST'])
def start_session():
    data = request.get_json(); session_id = data.get('sessionId')
    if not session_id: return jsonify({'error': 'sessionId is required.'}), 400
    if session_id in USER_KERNELS: return jsonify({'message': f'Session {session_id} already exists.'})
    try:
        km = KernelManager(); km.start_kernel()
        kc = km.client(); kc.start_channels(); kc.wait_for_ready(timeout=60)
        USER_KERNELS[session_id] = (km, kc)
        return jsonify({'message': f'Session {session_id} started successfully.'})
    except Exception as e:
        if 'km' in locals() and km.is_alive(): km.shutdown_kernel()
        return jsonify({'error': 'The code execution engine failed to start.', 'details': str(e)}), 500

@evaluation_bp.route('/validate', methods=['POST'])
def validate_cell():
    data = request.get_json()
    session_id, subject, level, q_id, p_id, code, username = data.get('sessionId'), data.get('subject'), data.get('level'), data.get('questionId'), data.get('partId'), data.get('cellCode'), data.get('username')

    if not all([session_id, subject, level, q_id, code, username]): return jsonify({'error': 'Missing required fields'}), 400
    if not code.strip(): return jsonify({'error': 'Code cannot be empty.'}), 400
    if session_id not in USER_KERNELS: return jsonify({'error': 'User session not found.'}), 404
    
    _km, kc = USER_KERNELS[session_id]
    student_dir = USER_GENERATED_PATH / username

    try:
        q_path = QUESTIONS_BASE_PATH / subject / f"level{level}" / "questions.json"
        with open(q_path, 'r', encoding='utf-8') as f: all_q = json.load(f)
        q_data = next((q for q in all_q if q['id'] == q_id), None)
        if not q_data: return jsonify({'error': f'Question with ID {q_id} not found.'}), 404
        part_data = next((p for p in q_data.get('parts', []) if p['part_id'] == p_id), q_data) if p_id else q_data
    except FileNotFoundError: return jsonify({'error': f"Question file not found at path: {q_path}"}), 500
    except Exception as e: return jsonify({'error': f'Could not load question data: {str(e)}'}), 500

    test_results = []
    
    if subject == 'ds':
        test_cases = q_data.get("test_cases", [])
        if not test_cases: return jsonify({'error': f'No test cases found for question {q_id}.'}), 500
        for i, case in enumerate(test_cases):
            user_input = case.get("input", "")
            stdout, stderr = run_code_on_kernel(kc, code, user_input=user_input)
            if stderr:
                test_results.append(False)
                continue
            passed = stdout.strip() == case.get("output", "").strip()
            test_results.append(passed)
            
    elif subject == 'ml':
        stdout, stderr = run_code_on_kernel(kc, code, working_dir=student_dir)
        if stderr:
            print(f"  - ERROR: Student code failed to execute.\n{stderr}")
            test_results.append(False)
        else:
            # ML-specific logic, which may use key_columns
            key_cols = part_data.get("key_columns") # Get key_columns if it exists
            solution_file = part_data.get("solution_file")
            
            if isinstance(solution_file, str):
                sol_path = Path(solution_file)
                student_file_path = student_dir / sol_path.name
                passed, score = compare_csvs(student_file_path, sol_path, key_columns=key_cols, tolerance=float(part_data.get('tolerance', 1e-5)))
                print(f"  - Comparing '{student_file_path.name}': {'PASSED' if passed else 'FAILED'} (Score: {score:.2f})")
                test_results.append(bool(passed))
            else:
                test_results.append(False)

    # ... (inside the validate_cell function)

    elif subject == 'Speech Recognition':
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # START OF ADDED VALIDATION LOGIC
        # This logic checks if the student is using the correct input file,
        # determined dynamically from the question's prompt text.
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

        # 1. Get the prompt text from the loaded question data.
        prompt_text = part_data.get("prompt", "")
        
        # 2. Extract the expected .wav filename from the prompt text.
        expected_match = re.search(r'([\w.-]+\.wav)', prompt_text)
        
        # 3. If a .wav file is mentioned in the prompt, validate the student's code.
        if expected_match:
            expected_filename = expected_match.group(1)
            
            # 4. Extract the .wav file path from the student's code.
            student_match = re.search(r'["\']([^"\']+\.wav)["\']', code)
            
            if not student_match:
                print("  - FAILED: Could not find a .wav file path in the student's code.")
                # Return immediately with a clear error for the student.
                return jsonify({
                    "test_results": [False], 
                    "stdout": "", 
                    "stderr": "Validation Error: Your code must contain the full path to the input .wav file as a string (e.g., \"/path/to/Audio36.wav\")."
                })

            student_path_str = student_match.group(1)
            student_filename = Path(student_path_str).name

            # 5. Compare the expected filename with the one the student used.
            if student_filename != expected_filename:
                print(f"  - FAILED: Input file mismatch. Expected '{expected_filename}', but code uses '{student_filename}'.")
                # Return immediately with a clear error for the student.
                return jsonify({
                    "test_results": [False], 
                    "stdout": "", 
                    "stderr": f"Validation Error: Incorrect input file. The prompt requires you to use '{expected_filename}', but your code uses '{student_filename}'."
                })
        
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # END OF ADDED VALIDATION LOGIC
        # If the check above passes, the original logic below is executed.
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # --- Original Logic Starts Here ---
        stdout, stderr = run_code_on_kernel(kc, code, working_dir=student_dir)
        if stderr:
            print(f"  - ERROR: Student code failed to execute.\n{stderr}")
            # It's better to pass stderr to the frontend for debugging.
            return jsonify({"test_results": [False], "stdout": stdout, "stderr": stderr})
        else:
            # Speech Rec logic, which does NOT use key_columns
            solution_files = part_data.get("solution_file")
            tolerance = float(part_data.get('tolerance', 1e-5))

            if isinstance(solution_files, list):
                all_files_passed = True
                for sol_path_str in solution_files:
                    sol_path = Path(sol_path_str)
                    student_file_path = student_dir / sol_path.name
                    passed, score = compare_csvs(student_file_path, sol_path, tolerance=tolerance)
                    print(f"  - Comparing '{student_file_path.name}': {'PASSED' if passed else 'FAILED'} (Score: {score:.2f})")
                    if not passed: all_files_passed = False
                test_results.append(bool(all_files_passed))
            elif isinstance(solution_files, str):
                sol_path = Path(solution_files)
                student_file_path = student_dir / sol_path.name
                passed, score = compare_csvs(student_file_path, sol_path, tolerance=tolerance)
                print(f"  - Comparing '{student_file_path.name}': {'PASSED' if passed else 'FAILED'} (Score: {score:.2f})")
                test_results.append(bool(passed))
            else:
                test_results.append(False)
    else:
        return jsonify({'error': f"No validation logic defined for subject: '{subject}'"}), 400

    return jsonify({"test_results": test_results})

@evaluation_bp.route('/run', methods=['POST'])
def run_cell():
    data = request.get_json()
    session_id, student_code, user_input, username = data.get('sessionId'), data.get('cellCode', 'pass'), data.get('userInput', ''), data.get('username')
    if not all([session_id, student_code, username]):
        return jsonify({'error': 'Session ID, code, and username are required.'}), 400
    if not student_code.strip():
        return jsonify({'stdout': '', 'stderr': 'Cannot run empty code.'})
    if session_id not in USER_KERNELS:
        return jsonify({'error': 'User session not found or invalid.'}), 404
    _km, kc = USER_KERNELS[session_id]
    student_dir = USER_GENERATED_PATH / username
    try:
        stdout, stderr = run_code_on_kernel(kc, student_code, user_input=user_input, working_dir=student_dir)
        return jsonify({'stdout': stdout, 'stderr': stderr})
    except Exception as e: 
        return jsonify({'stdout': '', 'stderr': str(e)}), 500

@evaluation_bp.route('/submit', methods=['POST'])
def submit_answers():
    data = request.get_json()
    session_id, username, subject, level = data.get('sessionId'), data.get('username'), data.get('subject'), data.get('level')
    answers, all_passed = data.get('answers', []), all(ans.get('passed', False) for ans in data.get('answers', []))
    status = 'passed' if all_passed else 'failed'
    submission = { 'subject': subject, 'level': f"level{level}", 'status': status, 'timestamp': datetime.now().isoformat(), 'answers': answers }
    user_submission_file = SUBMISSIONS_PATH / f"{username}.json"
    try:
        with open(user_submission_file, 'r+', encoding='utf-8') as f:
            user_submissions = json.load(f); user_submissions.append(submission); f.seek(0); json.dump(user_submissions, f, indent=2)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(user_submission_file, 'w', encoding='utf-8') as f: json.dump([submission], f, indent=2)
    updated_user = None
    if all_passed:
        with open(USERS_FILE_PATH, 'r+', encoding='utf-8') as f:
            users_json = json.load(f)
            user = next((u for u in users_json['users'] if u['username'] == username), None)
            if user:
                if subject not in user['progress']: user['progress'][subject] = {}
                user['progress'][subject][f"level{level}"] = 'completed'
                next_level = f"level{int(level) + 1}"
                if user['progress'][subject].get(next_level) == 'locked': user['progress'][subject][next_level] = 'unlocked'
                updated_user = {k: v for k, v in user.items() if k != 'password'}
            f.seek(0); json.dump(users_json, f, indent=2); f.truncate()
    if session_id in USER_KERNELS:
        km, kc = USER_KERNELS.pop(session_id)
        if kc.is_alive(): kc.stop_channels()
        if km.is_alive(): km.shutdown_kernel()
    return jsonify({'success': True, 'message': "Submission received.", 'updatedUser': updated_user})