# Message templates for each step

def get_problem_understanding_template(problem_description):
    return f"""
    Task: I want you to present your understanding of the programming problem that I will send right now. Your goal is to capture all the important ideas and constraints accurately as stated in the original problem.
    Pay attention to small details, nuances, notes, and examples in the problem description.

    This is the problem: 
    '{problem_description}'
    
    Provide your understanding in the following JSON structure:
    {{
      "understanding": {{
        "goal": "State the main objective of the problem in your own words.",
        "constraints": "List all constraints and limitations of the problem as you understand them.",
        "test_cases": {{
          "input_format": "Describe how input is structured, including edge cases if applicable.",
          "output_format": "Describe how output is structured."
        }},
        "important_ideas": [
          "List key idea 1 using your own interpretation.",
          "List key idea 2."
          "List additional ideas as needed."
        ],
        "difficulty_assessment": {{
          "estimated_difficulty": "Assess the difficulty level of this problem (easy, medium, hard, super hard) based on the complexity of logic, constraints.,
          "justification": "Provide reasoning for your difficulty assessment."
        }}
      }}
    }}
    """


def analyze_original_test_cases_template(problem_description):
    return f"""
    Task: Based on the problem description: '{problem_description}', your job is to analyze the original test case input and output, map each component to its corresponding variable from the problem description, and explain how these inputs lead to the specified output based on the logic and constraints of the problem.
    
    You should start by identifying the format of the test cases and specifying the structure of the input and output.
    
    **Clarification about Input Structure**:
    - The input may consist of multiple test cases, and for each test case, variables can appear on the same line or different lines.
    - If multiple values are provided across multiple lines, clearly specify how the input is structured line by line.
    - Ensure the separation of input components is maintained based on their appearance on separate lines.
    
    **General Example**:
    - Input:
      3 (number of test cases)
      Test Case 1:
        Line 1: N, K (e.g., 4 17)
        Line 2: traveler_1 time (e.g., 1)
        Line 3: traveler_2 time (e.g., 2)
        ...
      Test Case 2: values or variables as per the problem
      ...

    - Output:
      Expected output format specified by the problem (e.g., Case #1: YES, result values, etc.).
    
    **Provide the analysis in the following generalized JSON structure**:
    {{
      "format_description": "Describe the format of the test cases based on the problem (number of test cases, how input values are structured).",
      "original_test_case_analysis": [
        {{
          "total_number_of_test_cases": "Extract the total number of test cases from the input.",
          "test_case_X": {{
            "input": {{
              "line_1": {{
                "variable_1": "Extract the first variable or input component (e.g., N).",
                "variable_2": "Extract the second variable or input component (e.g., K)."
              }},
              "line_2_to_N": {{
                "components": [
                  {{
                    "component_name": "Describe the first element of the list.",
                    "value": "What's the value of that component?"
                  }},
                  ...
                ]
              }}
            }},
            "output": {{
              "target_output": "The expected output as per the problem statement (e.g., `Case #X: YES/NO`, integer result, etc.).",
              "output_explanation": "Explain why these inputs lead to this specific output, considering the problem's constraints and logic."
            }}
          }},
          "and so on for all test cases..."
        }}
      ]
    }}
    """

def generate_ai_test_cases_prompt(problem_description,test_case_analysis):
    return f"""
    Task: Based on #Sample Input and #Sample Output of {problem_description} provided and the following analysis: '{test_case_analysis}', generate 5 new AI-generated test cases. The goal is to observe patterns from the existing test cases and create new cases that are diverse and challenge different edge cases of the problem.
    
    Provide the new test cases in the following JSON structure:
    {{
      "ai_generated_test_cases": [
        {{
          "test_case_1": {{
            "input": "Input for test case 1",
            "expected_output": "Expected output for test case 1"
          }},
          "test_case_2": {{
            "input": "Input for test case 2",
            "expected_output": "Expected output for test case 2"
          }},
          and so on...
        }}
      ]
    }}
    """



def get_solution_ideas_template(problem_description, test_case_analysis, num_solutions):
    return f"""
    Task: Based on your analysis of {problem_description} and {test_case_analysis}, come up with {num_solutions} ideas that can pass all test cases (original and AI-generated). 

    Provide the ideas in the following JSON structure:
    {{
      "solutions": [
        {{
          "solution_1": {{
            "name": "Give the name or category of the first (basic) approach.",
            "strategy": "Explain the general strategy for this approach."
          }},
          "and so on... if I require more ideas"
        }}
      ]
    }}
    """

def evaluate_solutions_template(solution_ideas, problem_understanding, problem_difficulty):
    return f"""
    Task: You are given multiple solutions based on the analysis of the solution ideas: '{solution_ideas}'. Your goal is to choose the best solution based on the description below.

    Problem understanding:
    Goal: '{problem_understanding.get('understanding', {}).get('goal', 'No goal specified')}'
    
    Guidelines:
    - The main consideration should be that the solution can fully solve the problem in a simple and robust manner, especially given the difficulty level ('{problem_difficulty}').
    - Ensure the solution has a reasonable runtime - less than three seconds on a modern computer, based on the problem's constraints, including large inputs.
    - Consider trade-offs between simplicity, robustness, and efficiency depending on the problem's difficulty.

    Provide your evaluation in the following JSON format:
    {{
        "selected_solution": {{
            "solution_name": "The name of the chosen solution",
            "justification": {{
                "goal_alignment": "Explain how the solution addresses the main goal of the problem: '{problem_understanding.get('understanding', {}).get('goal', 'No goal provided')}'.",
                "constraint_handling": "Evaluate how well the solution meets the problem's constraints: '{problem_understanding.get('understanding', {}).get('constraints', 'No constraints provided')}'.",
                "input_output_handling": {{
                    "input_format": "Evaluate whether the solution correctly handles the input format: '{problem_understanding.get('understanding', {}).get('test_cases', {}).get('input_format', 'No input format provided')}'.",
                    "output_format": "Evaluate whether the solution produces the correct output format: '{problem_understanding.get('understanding', {}).get('test_cases', {}).get('output_format', 'No output format provided')}'."
                }},
                "important_ideas": "Explain how the solution incorporates key ideas from the problem understanding: '{problem_understanding.get('understanding', {}).get('important_ideas', 'No key ideas provided')}'.",
                "edge_case_handling": "Evaluate how the solution handles edge cases (if applicable).",
                "time_efficiency": "Provide the estimated time complexity and evaluate if it's suitable given the constraints.",
                "space_efficiency": "Provide the estimated space complexity and evaluate if it's efficient."
            }},
            "tradeoffs": {{
                "simplicity_vs_efficiency": "Explain any trade-offs between simplicity and efficiency, particularly considering the difficulty level ('{problem_difficulty}')."
            }},
            "improvements": "Suggest any future improvements or optimizations to further enhance the solution."
        }}
    }}
    """




def get_code_generation_template(selected_solution, test_case_analysis):
    return f"""
    Task: Now that you’ve identified the solution: '{selected_solution}' that passed all test cases, write Python code for that solution.
    Use your understanding of the input-output structure from: '{test_case_analysis}' to write valid Python code.

    **Guidelines**:
    1. Write clear comments: Add detailed comments throughout the code to explain each part of the logic.
    2. The input format must strictly follow the structure described in the test cases. For example, if the test input is provided in multiple lines, make sure your code can handle it line by line.
    3. Never use `input = sys.stdin.read` or `sys.stdin` to read input.
    4. Always use the `input()` Python built-in function to handle input directly. Ensure the code can handle multiple test cases as described in the problem and parse the input accordingly.
    5. Ensure that the output format is as described in the problem (e.g., "Case #1: YES/NO").
    6. Your code should be able to read from an input file and parse inputs in the correct order and structure (e.g., integers, arrays, etc.), based on the given test cases.
    
    Provide the Python code in this structured JSON format:
    {{
      "solution_code": {{
        "language": "Python",
        "code": "Your Python code as a string here, following the correct input-output format",
        "solution_name": "the name of the chosen solution",
        "description": "A brief description of the code based on the solution. Explain how the code fully implements the idea of the solution, strictly adhering to the input-output structure of the test cases."
      }}
    }}
    """

def iterate_public_tests(generated_code, error_message):
    return f"""
    Task: The generated code has encountered the following issue: {error_message}. Based on the latest code: '{generated_code}', modify and improve the Python code to fix the specific error at the indicated line.

    Guidelines:
    1. Fix the issue on the specific line mentioned in the error message.
    2. Add clear comments to explain the changes and ensure the logic is sound.
    3. Never use input = sys.stdin.read to read input.
    4. Ensure that the input() function is used to read input directly and in the correct format.
    5. **Only return the updated Python code in JSON format. Do not include any other text.**

    Also, include an explanation of your improvements.

    Provide the Python code in the following JSON format:
    {{
      "solution_code": {{
        "language": "Python",
        "code": "Your improved Python code here",
        "improvement": "Explain what was fixed, including specific references to line numbers or logic changes that address the issue raised in the error message."
      }}
    }}
    """
