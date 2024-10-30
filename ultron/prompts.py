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
      "input_format": "Describe how input is structured.",
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
    Line 1: N, K 
    Line 2: traveler_1 time 
    Line 3: traveler_2 time 
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
      "test_cases": [
        {{
          "input": {{
            "line_1": {{
              "component_name": "Name of the element/variable.",
              "value": "What's the value of that component? If there are more than 1 component just write their names and values in the same line" 
            }},
            "line_2": {{
              "component_name": "Name of the second element/variable if present.",
              "value": "What's the value of that second component?"
            }}
          }},
          "output": {{
            "target_output": "The expected output as per the problem statement (e.g., `Case #X: YES/NO`, integer result, etc.).",
            "output_explanation": "Explain why these inputs lead to this specific output, considering the problem's constraints and logic."
          }}
        }},
        {{
          "input": {{
            "line_1": {{
              "component_name": "Name of the element/variable.",
              "value": "What's the value of that component? If there are more than 1 component just write their names and values in the same line" 
            }},
            "line_2": {{
              "component_name": "Name of the second element/variable if present.",
              "value": "What's the value of that second component?"
            }}
          }},
          "output": {{
            "target_output": "The expected output as per the problem statement (e.g., `Case #X: YES/NO`, integer result, etc.).",
            "output_explanation": "Explain why these inputs lead to this specific output, considering the problem's constraints and logic."
          }}
        }},
        ...
      ]
    }}
  ],
  "test_case_reflection": {{
    "key_observations": [
      "List important observations from analyzing the test cases. These could be patterns, edge cases, or critical insights for solving the problem. Be as specific as possible. Your observation could be a formula, an algorithm or a chain of steps that lead to the result."
    ],j
    "variable_roles": {{
      "variable_name": "Explain the role or significance of each variable in the problem, based on how it's used across test cases."
    }},
    "problem_solving_hints": [
      "Provide hints or guidelines for approaching the problem, based on insights from the test cases."
    ],
    "general_formula": "If applicable, provide a general formula or approach for solving the problem, derived from analyzing the test cases."
  }}
}}

Ensure that your analysis in the 'test_case_reflection' section captures general insights about the problem that go beyond individual test cases. This should include patterns observed across all test cases, important considerations for solving the problem efficiently, and any key relationships between variables that become apparent from analyzing multiple examples.
"""

def refine_problem_understanding_template(problem_understanding, test_case_analysis, reflection = ""):
    if reflection == "":
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
    else:
        return f"""
Task: The problem understanding has evolved based on the improvements and insights gained after the previous iteration. 
Refine the problem understanding further, considering the new insights from the reflection process along with the test case analysis.

Take the reflection carefully into consideration to update your understanding.

Your goal is to incorporate insights from the problem statement, test case analysis, and reflection to provide a precise and updated understanding.

Here is the original understanding: 
'{problem_understanding}'

Here is the test case analysis: 
'{test_case_analysis}'

Here is the reflection from previous iterations: 
'{reflection}'

Provide the refined problem understanding in the following JSON structure:
{{
  "refined_problem_understanding": {{
    "goal": "State the refined objective of the problem based on the reflection and test case analysis.",
    "updated_constraints": "List updated constraints and any new limitations discovered from the reflection and test case analysis.",

    "test_cases_update": {{
      "input_format": "Update the input format based on the test case analysis if it has changed.",
      "output_format": "Update the output format based on the test case analysis."
    }},
    "important_ideas_update": [
      "Based on the output explanation in the test case analysis and reflection, update the important ideas to reflect the latest insights."
    ],
    "difficulty_assessment_update": {{
      "updated_difficulty": "Reassess the difficulty of this problem (easy, medium, hard, super hard) based on insights from the reflection and test case analysis.",
      "justification": "Provide reasoning for the updated difficulty assessment."
    }},
    "changes_based_on_reflection": [
      "List specific updates made to the problem understanding based on insights gained from the reflection."
    ]
  }}
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
  ]
}}
"""

def evaluate_solutions_template(solution_ideas, refine_problem_understanding, test_case_analysis):
    return f"""
Task: You are given multiple solutions based on the analysis of the solution ideas: 

'{solution_ideas}'. 

Your goal is to choose the best solution based on the description below.

Problem goal:
Goal: "{refine_problem_understanding.get('refined_problem_understanding', {}).get('goal', 'No goal specified')}"

Test case analysis:
{test_case_analysis}
Guidelines:
- The main consideration should be that the solution can fully solve the problem in a simple and robust manner, especially given the difficulty level ("{refine_problem_understanding.get('refined_problem_understanding', {}).get('difficulty_assessment_update', 'No assessment')}").
- Ensure the solution has a reasonable runtime - less than three seconds on a modern computer, based on the problem's constraints, including large inputs.
- Consider trade-offs between simplicity, robustness, and efficiency depending on the problem's difficulty.

Provide your evaluation in the following JSON format:
{{
    "selected_solution": {{
        "solution_name": "The name of the chosen solution",
        "justification": {{
            "goal_alignment": "Explain how the solution addresses the main goal of the problem: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('goal', 'No goal provided')}'.",
            "constraint_handling": "Evaluate how well the solution meets the problem's constraints: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('updated_constraints', 'No constraints provided')}'.",
            "important_ideas": "Explain how the solution incorporates key ideas from the problem understanding: '{refine_problem_understanding.get('refined_problem_understanding', {}).get('important_ideas_update', 'No key ideas provided')}'.",
            "edge_case_handling": "Evaluate how the solution handles edge cases (if applicable).",
            "time_efficiency": "Provide the estimated time complexity and evaluate if it's suitable given the constraints.",
            "space_efficiency": "Provide the estimated space complexity and evaluate if it's efficient."
        }},
        "tradeoffs": {{
            "simplicity_vs_efficiency": "Explain any trade-offs between simplicity and efficiency, particularly considering the difficulty level ('{refine_problem_understanding.get('refined_problem_understanding', {}).get('difficulty_assessment_update', 'No assessment')}')."
        }},
        "improvements": "Suggest any future improvements or optimizations to further enhance the solution."
    }}
}}
"""

def get_code_generation_template(selected_solution, test_case_analysis, refine_problem_understanding):
    return f"""
You are tasked with generating Python code for the solution: 
{selected_solution}

based on the provided test case analysis: 

{test_case_analysis}

And your own understanding of the problem:
{refine_problem_understanding}

Follow the instructions below:

Code generation guidelines:
1. Your code should solve the problem and pass all test cases, using the specified input-output structure. 
2. Divide the code into small, well-named sub-functions.
3. Use Python's built-in `input()` function to handle input directly. Do not use `sys.stdin` or `input = sys.stdin.read`.
4. Ensure the code can correctly process the provided `sample_input` and produce the expected `sample_output`.
5. Do not include any error handling (`try...except`), and do not raise any exceptions as errors will be captured separately.
6. Always include an `if __name__ == '__main__':` block, ensuring the code is executable as a standalone script.

The output must always follow this example structure:
Case #1: YES
Case #2: NO
Case #3: YES
Case #4: NO
Case #5: NO

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

def reflect_execution_error(generated_code, error_message, test_case_analysis, error_history):
    return f"""
Task: The generated code has encountered an execution or runtime issue:

{error_message}

Current code:
'{generated_code}'

The test cases being evaluated:
{test_case_analysis}

Error History:
- The errors and their number of occurrence so far: 
{error_history}

Instructions:
- Analyze the current error and identify the specific line or section of code responsible.
- Consider any recurring patterns in `error_history` and avoid repeating similar mistakes, especially those that have occurred frequently.
- Design a robust solution that not only addresses this error but also reduces the likelihood of similar issues in future iterations.

Provide your analysis in the JSON format below, focusing on clearly articulating the error’s root cause, proposed modifications, and next steps for improvement:

{{
  "error_reflection": {{
    "root_cause": "Explain the fundamental reason for this error, identifying deeper issues if patterns exist in error history.",
    "trigger_condition": "Describe the specific input/state that triggered this error.",
    "potential_patterns": "Identify any patterns in recurring errors that need to be addressed."
  }},
  "changes_needed": {{
    "fix_strategy": "Detail a targeted approach to fix this specific error, incorporating lessons from past errors if applicable.",
    "code_modifications": {{
      "problematic_line": "Pinpoint the exact line or section causing the issue.",
      "proposed_fix": "Provide the corrected version of the code, focusing on robustness.",
      "safety_checks": "Describe additional checks or constraints to prevent similar errors in the future."
    }},
    "expected_outcome": "Explain why these changes should resolve the error and improve overall stability."
  }},
  "next_step": "Specify what to do next based on this reflection, such as adjusting a function or exploring a new solution approach."
}}
"""


def reflect_failed_test(generated_code, failed_tests, test_case_analysis, failure_history):
    return f"""
Task: The generated code has failed these test cases:

{failed_tests}

Based on the latest code:

'{generated_code}'

The test cases are:
{test_case_analysis}

Failure History:
- You have a history of making these mistakes and their corresponding number of times that it has occured.
{failure_history}

You must follow the instructions below:
  A. Improvement guidelines:
    1. Identify and explore new strategies to address recurring patterns in past failures, aiming to avoid similar issues, especially the ones that occur many times!
    2. Exploit parts of the code that are working well and consistently yield correct results.
    3. Adjust or restructure problematic sections while preserving effective logic from previous iterations.
    4. And more importantly, for the failure that has occured many times don't be afraid to try new things even though you might ruin the program because I will give you more information to fix things!

Provide your analysis in the following JSON format:
{{
  "failure_reflection": {{
    "root_cause": "Identify the underlying reason for the failed tests",
    "trigger_condition": "Specify the input/state that caused the failure",
    "potential_patterns": "Describe any recurring failure patterns that need attention"
  }},
  "improvements_needed": {{
    "explore_strategy": "Propose a new approach or strategy to address the failed test cases",
    "exploit_strategy": "Describe what aspects of the code are effective and should be retained",
    "code_modifications": {{
      "problematic_section": "Describe the problematic code section",
      "proposed_changes": "The adjustments to improve functionality",
      "safety_checks": "Additional checks to prevent similar failures"
    }},
  }},
  "next_step": "What to do next based on the reflection (e.g., modify a specific function or strategy)."
}}
"""

#final step
def improve_final_code_efficiency(final_code, refine_problem_understanding):
    return f"""
Here is the final code that needs optimization:
{final_code}

Based on your correct understanding of the problem:
{refine_problem_understanding}

Task: The current solution has passed all sample test cases but needs to be optimized to handle larger inputs effectively. 
Your goal is to improve its runtime efficiency so that it can process full test cases within the constraints in your understanding.

Instructions:
1. Analyze Bottlenecks: Identify any inefficient parts of the code, such as nested loops or redundant calculations.
2. Optimize Data Structures: Where possible, replace costly operations with efficient data structures (e.g., dictionaries, heaps, binary search trees).
3. Reduce Redundancies: Remove or simplify repeated calculations, and consider precomputing values where feasible.
4. Simplify Logic: Rewrite the code to reduce the number of operations in critical sections.

Performance Goals: minimize time complexity to handle larger inputs

You must provide your optimized code in the following JSON format:
{{
  "optimized_code": {{
    "language": "Python",
    "code": "Your optimized Python code here, ensuring it achieves the performance goals",
    "optimization_explanation": "Describe the changes made to improve efficiency and why they should achieve better runtime performance on large inputs."
  }}
}}
"""
