import io
from unittest.mock import patch
import traceback
import sys
import contextlib
import signal

def check_code_structure(extracted_code):
    # Check if the phrase "__name__ == '__main__'" is present in the code
    if "__name__ == '__main__'" not in extracted_code:
        return False, "Missing `if __name__ == '__main__':` block."
    return True, None

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException(
        "The previous code execution timed out. This may indicate a performance issue, such as an infinite loop or inefficient logic. "
        "The input provided was sufficiently small and valid, so the problem is likely due to a flaw in the code logic rather than the input itself. "
        "Please review the code for potential errors or inefficiencies that could cause it to run indefinitely or take an excessive amount of time."
    )

def run_extracted_code_with_timeout(extracted_code, test_input, timeout=5):
    # Check code structure first
    is_valid, error_message = check_code_structure(extracted_code)
    if not is_valid:
        return None, error_message

    # Set up mock input and output redirection
    output = io.StringIO()
    error = None
    test_input_lines = [line.strip() for line in test_input.strip().split('\n') if line.strip()]

    def mock_input():
        if not test_input_lines:
            raise ValueError("No input data provided")
        return test_input_lines.pop(0)

    # Set the timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set a timeout of `timeout` seconds

    try:
        with patch('builtins.input', mock_input), contextlib.redirect_stdout(output):
            code_obj = compile(extracted_code, '<string>', 'exec')
            local_scope = {'__name__': '__main__'}
            exec(code_obj, local_scope)
        signal.alarm(0)  # Disable the alarm
    except TimeoutException:
        return None, "Error: Code execution timed out."
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        line_no = tb[-1].lineno
        code_lines = extracted_code.split('\n')
        error_line = code_lines[line_no - 1] if line_no <= len(code_lines) else "Unknown"
        error = f"Error on line {line_no}: {error_line.strip()}\nException: {exc_type.__name__}: {str(exc_value)}"
        return None, error

    result = output.getvalue()
    if not result.strip():
        return None, "Error: Code did not produce any output."

    return result, None

# Compare the output generated by the code with the expected output
def compare_with_expected_output(generated_output, expected_output):
    # Check if either output is None
    if generated_output is None:
        return 0, ["Generated output is None"]

    generated_output_lines = generated_output.strip().splitlines()
    expected_output_lines = expected_output.strip().splitlines()

    total_cases = len(expected_output_lines)
    matching_cases = 0
    failed_cases = []

    for i, (generated_line, expected_line) in enumerate(zip(generated_output_lines, expected_output_lines), start=1):
        if generated_line.strip() == expected_line.strip():
            matching_cases += 1
        else:
            failed_cases.append(f"Test Case #{i}: Expected '{expected_line}' but got '{generated_line}'")

    score = (matching_cases / total_cases) * 100 if total_cases > 0 else 0
    return score, failed_cases

# Evaluate generated code on test cases
def evaluate_generated_code_on_test_cases(extracted_code, test_input, test_output):
    # Run the code and get the output
    generated_output, error = run_extracted_code(extracted_code, test_input)
    
    if generated_output is None or generated_output.strip() == "":
        return 0, error or "Error: The code ran without any problem. There's no error in parsing the input. But it produces no output. Please check the entire code again.", generated_output, []
    
    # If there's an error, return it
    if error:
        return 0, error, generated_output, []

    # Compare the generated output with expected output
    score, failed_cases = compare_with_expected_output(generated_output, test_output)
    
    if failed_cases:
        error_msg = f"Test cases failed: {failed_cases}"
        return score, error_msg, generated_output, failed_cases

    return score, error, generated_output, failed_cases