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
    Task: Based on the problem description: 
    
    '{problem_description}', 
    
    your job is to analyze the original test case input and output, map each component to its corresponding variable from the problem description, and explain how these inputs lead to the specified output based on the logic and constraints of the problem.
    
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
                "component_name": "Name of the element/variable.",
                "value": "What's the value of that component?" 
                if there are more than 1 component just write their names and values in the same line  
              }},
              create more if there are more than 1 line in a test case
            }},
            "output": {{
              "target_output": "The expected output as per the problem statement (e.g., `Case #X: YES/NO`, integer result, etc.).",
              "output_explanation": "Explain why these inputs lead to this specific output, considering the problem's constraints and logic."
            }}
          }},
          "and so on for all test cases..."
        }}
      ],
      "test_case_reflection": {{
        "key_observations": [
          "List important observations from analyzing the test cases. These could be patterns, edge cases, or critical insights for solving the problem.
          For example: after analysing the test case, it appears that both single and double cheese burgers have 2 buns.
          "
        ],
        "variable_roles": {{
          "variable_name": "Explain the role or significance of each variable in the problem, based on how it's used across test cases."
        }},
        "problem_solving_hints": [
          "Provide hints or guidelines for approaching the problem, based on insights from the test cases."
        ],
        "general_formula": "If applicable, provide a general formula or approach for solving the problem, derived from analyzing the test cases.",
      }}
    }}
    
    Ensure that your analysis in the 'test_case_reflection' section captures general insights about the problem that go beyond individual test cases. This should include patterns observed across all test cases, important considerations for solving the problem efficiently, and any key relationships between variables that become apparent from analyzing multiple examples.
    """
def refine_problem_understanding_template(problem_understanding, test_case_analysis):
    return f"""
    Task: Now that you have analyzed the test cases and re-evaluated your initial understanding, refine the problem understanding. Focus on any new insights, corrections, or additional ideas that emerged from examining the test cases.

    Take into consideration:
    - Any constraints or nuances that were missed in the original understanding.
    - The input-output structures observed in the test cases, which might differ from the original understanding.
    - Assume most of your original important ideas were incorrect, and base your updates on the output explanations from the test case analysis.

    Your goal is to provide a refined understanding of the problem. Incorporate details from both the problem statement and the test cases to make the understanding more precise.

    Here is the original understanding: 
    '{problem_understanding}'

    Here is the test case analysis: 
    '{test_case_analysis}'

    Provide the refined problem understanding in the following JSON structure:
    {{
      "refined_problem_understanding": {{
        "goal": "State the refined objective of the problem.",
        "updated_constraints": "List updated constraints and any new limitations you discovered.",

        "test_cases_update": {{
          "input_format": "Update the input format based on the test case analysis if it has changed.",
          "output_format": "Update the output format based on the test case analysis."
        }},
        "important_ideas_update": [
          "Based on the output explanation in the test case analysis, update the important ideas assuming the initial understanding was mostly wrong."
        ],
        "difficulty_assessment_update": {{
          "updated_difficulty": "Reassess the difficulty of this problem (easy, medium, hard, super hard) based on new insights from the test case analysis.",
          "justification": "Provide reasoning for the updated difficulty assessment."
        }}
      }}
    }}
    """


def generate_ai_test_cases_prompt(refine_problem_understanding,test_case_analysis):
    return f"""
    Task: Based on the understanding of the problem 

    {refine_problem_understanding} 

    provided and the following analysis: 
    
    '{test_case_analysis}', 

    generate 5 new AI-generated test cases. The goal is to observe patterns from the existing test cases and create new cases that are diverse and challenge different edge cases of the problem.
    
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



def get_solution_ideas_template(refine_problem_understanding, test_case_analysis, num_solutions):
    return f"""
    Task: Based on your understanding of the problem:

    {refine_problem_understanding} 

    and analysis of the test cases:

    {test_case_analysis}, 

    come up with {num_solutions} ideas that can pass all test cases (original and AI-generated). 

    Provide the ideas in the following JSON structure:
    {{
      "solutions": [
        {{
          "name": "Give the name or category of the first approach.",
          "strategy": "Explain the general strategy for this approach."
        }}
        // Additional solution objects will be added here up to the requested number
      ]
    }}
    """

def evaluate_solutions_template(solution_ideas, refine_problem_understanding, test_case_analysis, problem_difficulty):
    return f"""
    Task: You are given multiple solutions based on the analysis of the solution ideas: 

    '{solution_ideas}'. 

    Your goal is to choose the best solution based on the description below.

    Problem understanding:
    Goal: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('goal', 'No goal specified')}'
    
    Guidelines:
    - The main consideration should be that the solution can fully solve the problem in a simple and robust manner, especially given the difficulty level ('{problem_difficulty}').
    - Ensure the solution has a reasonable runtime - less than three seconds on a modern computer, based on the problem's constraints, including large inputs.
    - Consider trade-offs between simplicity, robustness, and efficiency depending on the problem's difficulty.

    Provide your evaluation in the following JSON format:
    {{
        "selected_solution": {{
            "solution_name": "The name of the chosen solution",
            "justification": {{
                "goal_alignment": "Explain how the solution addresses the main goal of the problem: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('goal', 'No goal provided')}'.",
                "constraint_handling": "Evaluate how well the solution meets the problem's constraints: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('constraints', 'No constraints provided')}'.",
                "important_ideas": "Explain how the solution incorporates key ideas from the problem understanding: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('important_ideas', 'No key ideas provided')}'.",
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
You are tasked with generating Python code for the solution: 
{selected_solution}

based on the provided test case analysis: 

{test_case_analysis}

Your code should solve the problem and pass all test cases, using the specified input-output structure. 

Guidelines:
1. Divide the code into small, well-named sub-functions, each no longer than 10 lines.
2. Use Python's built-in `input()` function to handle input directly. Do not use `sys.stdin` or `input = sys.stdin.read`.
3. Ensure the code can correctly process the provided `sample_input` and produce the expected `sample_output`.
4. Use the `if __name__ == "__main__":` block to run the solution.
5. Do not include any error handling (`try...except`), and do not raise any exceptions as errors will be captured separately.

Provide the Python code in this JSON format:
{{
  "solution_code": {{
    "sample_input": "Extract the correct first test case input",
    "sample_output": "Expected output for the first test case",
    "language": "Python",
    "code": "Your Python code as a string here, ensuring it can process the input and output correctly",
    "solution_name": "Name of the chosen solution",
    "description": "Brief explanation of how the code implements the solution."
  }}
}}
"""

def iterate_public_tests(generated_code, error_message):
    return f"""
    Task: The generated code has encountered the following issue: 
    
    {error_message}. 

    Based on the latest code: 

    '{generated_code}',

    If the problem is: there are many failed test cases. Think of another solution that may improve the passed rate.
    If the issue is Execution/Runtime error. Focus specificly on the line causes the error and try to fix it!

    Guidelines:
    1. The fixed code should be robust and general, and work for other input examples as well. The fixed should be different from the original code, and not just a copy-paste of the original code.
    2. You must divide the fixed code into small sub-functions, with meaningful names and functionality.
    3. Never use `input = sys.stdin.read` or `sys.stdin` to read input. Always use the `input()` Python built-in function to handle input directly.
    4. In order to have valid Python code. Your code must correctly process the sample_input and sample_output
    5. The 'code' field must USE if __name__ == "__main__": to execute the final answer!

    Provide the Python code in the following JSON format:
    {{
      "solution_code": {{
        "language": "Python",
        "error_line": "The line that may cause the error from the latest code"
        "code": "Your improved Python code here.",
        "improvement": "Explain what was fixed, including specific references to line or logic changes that address the issue raised in the error message."
      }}
    }}
    """
