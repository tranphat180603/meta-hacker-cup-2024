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
      "List key idea 2.",
      "List additional ideas as needed."
    ],
    "difficulty_assessment": {{
      "estimated_difficulty": "Assess the difficulty level of this problem (easy, medium, hard, super hard) based on the complexity of logic, constraints.",
      "justification": "Provide reasoning for your difficulty assessment."
    }}
  }}
}}
"""
    
def get_image_understanding_prompt(problem_description):
    return f"""
Task: Given the programming problem description below, analyze the image and capture essential details that could assist in solving the problem. Focus on extracting information that directly relates to understanding the problem constraints, input/output requirements, and solution approach.

Problem Description:
"{problem_description}"

Please provide your analysis in the following JSON format:

{{
  "image_understanding": {{
    "core_components": [
      {{
        "name": "Name of the component",
        "properties": [
          {{
            "property_name": "Specific attribute or characteristic",
            "value": "Quantitative or qualitative value",
            "relevance": "How this property affects problem constraints or solution"
          }}
        ],
        "constraints": [
          {{
            "rule": "Any rule or limitation related to this component",
            "impact": "How this constraint affects the solution"
          }}
        ]
      }}
    ],
    "relationships": [
      {{
        "type": "Type of relationship (e.g., 'composition', 'transformation', 'dependency')",
        "description": "Description of how elements relate to each other",
        "formula": "Mathematical or logical relationship if applicable",
        "constraints": "Any constraints on this relationship"
      }}
    ],
    "patterns": [
      {{
        "name": "Pattern name or type",
        "sequence": "Description of how the pattern progresses",
        "requirements": "What's needed to form this pattern",
        "mathematical_properties": "Any relevant mathematical relationships"
      }}
    ],
    "derived_variables": [
      {{
        "name": "Variable name",
        "description": "What this variable represents",
        "calculation": "How to calculate or derive this variable",
        "constraints": "Valid range or conditions"
      }}
    ],
    "solution_insights": [
      {{
        "observation": "Key insight from the image",
        "implication": "How this affects the solution approach",
        "validation_criteria": "How to verify if this insight is correctly applied"
      }}
    ],
    "edge_cases": [
      {{
        "scenario": "Description of edge case visible in image",
        "considerations": "What needs to be handled for this case",
        "example": "Visual example from image if available"
      }}
    ]
  }}
}}

Focus on capturing:
1. Quantitative relationships and formulas visible in the image
2. Pattern rules and their exceptions
3. Component relationships that affect validity
4. Minimum/maximum values or constraints shown
5. Edge cases or special scenarios illustrated
6. Any mathematical or logical patterns that emerge from the visual representation
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
  "format_description": "Describe the input format of the test cases, specifying the structure of values in each line for the first test case.",
  "first_test_case_analysis": {{
    "input": {{
      "line_1": {{
        "component_name": "Name of the first element or variable.",
        "value": "The value or values in line 1, including multiple components if applicable."
      }},
      "line_2": {{
        "component_name": "Name of the second element or variable, if present.",
        "value": "The value or values in line 2."
      }}
    }},
    "output": {{
      "target_output": "Expected output for this test case (e.g., `Case #1: YES/NO`).",
      "output_explanation": "Explain why these inputs produce this specific output based on the problem's description."
    }}
  }},
  "test_case_reflection": {{
    "key_observations": [
      "Summarize important patterns or insights noticed in the first test case that could apply generally."
    ],
    "variable_roles": {{
      "variable_name": "Describe each variable's role in the problem based on its usage in the first test case."
    }},
    "problem_solving_hints": [
      "List hints or strategies derived from the first test case for approaching this problem."
    ],
    "general_formula": "If applicable, provide a general formula or rule observed from the first test case."
  }}
}}

Ensure that your analysis in the 'test_case_reflection' section captures general insights that go beyond the specific example, providing patterns and hints applicable to other cases.

Ensure that your analysis in the 'test_case_reflection' section captures general insights about the problem that go beyond individual test cases. This should include patterns observed across all test cases, important considerations for solving the problem efficiently, and any key relationships between variables that become apparent from analyzing multiple examples.
"""


def refine_problem_understanding_template(problem_understanding, test_case_analysis, reflection="", img_understanding=""):
    # Basic structure when no reflection or image understanding is provided
    if not reflection and not img_understanding:
        return f"""
Task: Refine the problem understanding based on test case analysis. Identify any new insights, corrections, or adjustments needed.

Consider:
- Missed constraints or nuances.
- Observed input-output structures in the test cases.
- Assume initial ideas were mostly incorrect; base updates on the test case analysis.

Here is the original understanding:
'{problem_understanding}'

Here is the test case analysis:
'{test_case_analysis}'

Provide the refined problem understanding in JSON format:
{{
  "refined_problem_understanding": {{
    "goal": "State the refined objective of the problem.",
    "updated_constraints": "List updated constraints and any new limitations discovered.",
    "test_cases_update": {{
      "input_format": "Update the input format if changed.",
      "output_format": "Update the output format if changed."
    }},
    "important_ideas_update": [
      "List new or corrected important ideas based on test case analysis."
    ],
    "general_formula_update": "Update the general formula if applicable based on test case analysis.",
    "difficulty_assessment_update": {{
      "updated_difficulty": "Reassess the problem difficulty (easy, medium, hard, super hard).",
      "justification": "Explain the reasoning for the updated difficulty."
    }}
  }}
}}

"""
    # Structure when only reflection is provided
    elif reflection and not img_understanding:
        return f"""
Task: Refine the problem understanding by integrating insights from test case analysis and reflection.

Consider:
- Key takeaways or patterns identified in the reflection.
- Constraints or nuances observed from test cases.
- Make any necessary adjustments based on insights from both reflection and test case analysis.

Here is the original understanding:
'{problem_understanding}'

Here is the test case analysis:
'{test_case_analysis}'

Here is the reflection from previous iterations:
'{reflection}'

Provide the refined problem understanding in JSON format:
{{
  "refined_problem_understanding": {{
    "changes_based_on_reflection": [
      "Summarize updates made based on reflection insights."
    ],
    "goal": "State the refined objective of the problem based on the combined insights.",
    "updated_constraints": "List updated constraints identified from reflection and test case analysis.",
    "test_cases_update": {{
      "input_format": "Describe any updates to input format.",
      "output_format": "Describe any updates to output format."
    }},
    "important_ideas_update": [
      "List new or corrected important ideas based on test case analysis and reflection."
    ],
    "general_formula_update": "Update general formula if applicable based on test case and reflection insights.",
    "difficulty_assessment_update": {{
      "updated_difficulty": "Reassess the difficulty level based on insights.",
      "justification": "Provide reasoning for the updated difficulty."
    }}
  }}
}}
"""

    # Structure when only image understanding is provided
    elif img_understanding and not reflection:
        return f"""
Task: Refine the problem understanding by integrating insights from test case analysis and relevant image details.

Consider:
- Visual insights related to the problem’s structure, patterns, or components.
- Observed constraints or nuances in the test cases.
- Assume initial ideas were mostly incorrect; base updates on test cases and image details.

Here is the original understanding:
'{problem_understanding}'

Here is the test case analysis:
'{test_case_analysis}'

Here is the image understanding:
'{img_understanding}'

Provide the refined problem understanding in JSON format:
{{
  "refined_problem_understanding": {{
    "image_insights": [
      "Summarize specific updates based on image insights."
    ],
    "goal": "State the refined objective, incorporating visual and test case insights.",
    "updated_constraints": "List any revised constraints observed.",
    "test_cases_update": {{
      "input_format": "Describe any updates to input format.",
      "output_format": "Describe any updates to output format."
    }},
    "important_ideas_update": [
      "List important ideas based on test case and image insights."
    ],
    "general_formula_update": "Update general formula if applicable, using new insights.",
    "difficulty_assessment_update": {{
      "updated_difficulty": "Reassess difficulty based on insights.",
      "justification": "Explain the reasoning for this assessment."
    }}
  }}
}}
"""

    # Structure when both reflection and image understanding are provided
    else:
        return f"""
Task: Refine the problem understanding by integrating insights from test case analysis, reflection, and image understanding.

Consider:
- Reflection insights that highlight recurring issues or patterns from past attempts.
- Visual insights related to the problem’s structure or relevant components.
- Observed constraints or nuances in the test cases.

Here is the original understanding:
'{problem_understanding}'

Here is the test case analysis:
'{test_case_analysis}'

Here is the reflection:
'{reflection}'

Here is the image understanding:
'{img_understanding}'

Provide the refined problem understanding in JSON format:
{{
  "refined_problem_understanding": {{
    "changes_based_on_reflection": [
      "Summarize updates based on reflection insights."
    ],
    "image_insights": [
      "Summarize updates based on image understanding."
    ],
    "goal": "State the refined objective of the problem.",
    "updated_constraints": "List any updated constraints from combined insights.",
    "test_cases_update": {{
      "input_format": "Update the input format based on insights.",
      "output_format": "Update the output format based on insights."
    }},
    "important_ideas_update": [
      "List important ideas based on test case, reflection, and image insights."
    ],
    "general_formula_update": "Update the general formula if applicable, based on combined insights.",
    "difficulty_assessment_update": {{
      "updated_difficulty": "Reassess the difficulty based on all insights.",
      "justification": "Provide reasoning for the updated difficulty assessment."
    }}
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

Provide the the ideas in valid JSON format following the structure below.
Note that: there must not be any text outside of the JSON format for validity!
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
            "goal_alignment": "Explain how the solution addresses the main goal of the problem: \"{refine_problem_understanding.get('refined_problem_understanding', {}).get('goal', 'No goal provided')}\".",
            "constraint_handling": "Evaluate how well the solution meets the problem's constraints: \"{refine_problem_understanding.get('refined_problem_understanding', {}).get('updated_constraints', 'No constraints provided')}\".",
            "important_ideas": "Explain how the solution incorporates key ideas from the problem understanding: \"{refine_problem_understanding.get('refined_problem_understanding', {}).get('important_ideas_update', 'No key ideas provided')}\".",
            "edge_case_handling": "Evaluate how the solution handles edge cases (if applicable).",
            "time_efficiency": "Provide the estimated time complexity and evaluate if it's suitable given the constraints.",
            "space_efficiency": "Provide the estimated space complexity and evaluate if it's efficient."
        }},
        "tradeoffs": {{
            "simplicity_vs_efficiency": "Explain any trade-offs between simplicity and efficiency, particularly considering the difficulty level (\"{refine_problem_understanding.get('refined_problem_understanding', {}).get('difficulty_assessment_update', 'No assessment')}\")."
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
3. Use Python's built-in `input()` function to handle input directly.
4. Ensure the code can correctly process the provided `sample_input` and produce the expected `sample_output`.
5. Do not include any error handling (`try...except`), and do not raise any exceptions as errors will be captured separately.
6. Always include an `if __name__ == '__main__':` block, ensuring the code is executable as a standalone script.

##IMPORTANT:***
IN ANY GIVEN CIRCUSTANCES, MUST NEVER use `sys.stdin` or `input = sys.stdin.read` since it will definitely affect the performance of the process!

The output must always follow this example structure:
Case #1: YES
Case #2: NO
Case #3: YES
Case #4: NO
Case #5: NO

Provide the Python code in the following JSON format. Note that newlines within the `"code"` field should be represented by `\\n` to ensure JSON compatibility:

{{
  "solution_code": {{
    "sample_input": "Extract the correct first test case input",
    "sample_output": "Expected output for the first test case",
    "language": "Python",
    "code": "Your Python code here, with each line separated by \\n for JSON compatibility.",
    "solution_name": "Name of the chosen solution",
    "description": "Brief explanation of how the code implements the solution."
  }}
}}

"""


def reflect_execution_error(generated_code, error_message, test_case_analysis, error_history):
    return f"""
Task: The generated code has encountered an execution or runtime issue:

{error_message}

Current code and some information:
'{generated_code}'

The test case analysis
{test_case_analysis}

Error History:
- The errors and their number of occurrence so far: 
{error_history}

Instructions:
- Analyze the current error and identify the specific line or section of code responsible.
- Consider any recurring patterns in `error_history` and avoid repeating similar mistakes, especially those that have occurred frequently.
- Design a robust solution that not only addresses this error but also reduces the likelihood of similar issues in future iterations.

Provide your analysis in the JSON format below, focusing on clearly articulating the error’s root cause and changes needed for improvement:

{{
  "error_reflection": {{
    "root_cause": "Brief description of the specific error source.",
    "correction_needed": "Precise modification required to resolve the issue."
  }},
  "changes_needed": {{
    "fix_strategy": "Direct solution to the identified problem.",
    "code_modifications": {{
      "problematic_line": "Specific line(s) causing the error",
      "proposed_fix": "Exact correction for the line(s)"
    }}
  }}
}}
"""



def reflect_failed_test(generated_code, failed_tests, refine_problem_understanding, failure_history):
    return f"""
Task: Analyze the test failures and provide comprehensive insights for improvement.

Failed Test Cases:
{failed_tests}

Current Code:
'{generated_code}'

Problem Understanding:
{refine_problem_understanding}

Failure History:
{failure_history}

Provide your analysis in the JSON format below:

{{
  "test_case_analysis": {{
    "failed_cases": [
      {{
        "input": "The input that caused failure",
        "expected": "Expected output",
        "actual": "Actual output",
        "pattern": "Pattern this failure represents (e.g., edge case, boundary condition)"
      }}
    ],
    "failure_categorization": {{
      "type": "Type of failure (logic error, constraint handling)",
      "scope": "Local to specific cases or global issue",
      "frequency": "New or recurring pattern"
    }}
  }},
  "solution_revision": {{
    "core_logic_issues": [
      {{
        "component": "Component causing issue",
        "current_approach": "Current code logic",
        "proposed_fix": "Updated code snippet with correction"
      }}
    ]
  }},
  "implementation_plan": {{
    "priority_fixes": [
      {{
        "component": "Component to fix",
        "approach": "How to fix",
        "validation_steps": "Specific steps to confirm the fix works (test cases to run, expected output)"
      }}
    ]
  }},
  "learning_points": {{
    "insights": [
      {{
        "insight": "Detailed insight learned from failure",
        "application": "How to apply this insight in future tasks"
      }}
    ]
  }}
}}

Note: Provide detailed, specific information rather than generic observations. Focus on actionable insights that directly address the failed test cases.
"""


def improve_final_code_efficiency(final_code, refine_problem_understanding, timeout_msg=None):
    if timeout_msg:
        return f"""
Here is the final code that needs optimization due to slow performance on larger inputs:
{final_code}

Issue: The solution is unable to complete execution within acceptable time limits and needs significant optimization to handle full input cases. Here’s the feedback received:
"{timeout_msg}"

Task: Based on your correct understanding of the problem and the identified performance bottleneck, revise the code to improve its runtime efficiency so that it can handle larger input sizes effectively.

Instructions:
1. **Identify Inefficiencies**: Examine the code for any sections with nested loops, redundant calculations, or repetitive operations. Focus on areas contributing to high computational cost.
2. **Optimize Data Structures**: Use more efficient data structures like dictionaries, heaps, or binary search trees where applicable, replacing any costly operations with optimal alternatives.
3. **Reduce Redundant Calculations**: Consolidate repeated logic, avoid recalculating values, and consider precomputing reusable results where feasible.
4. **Simplify Logic**: Rewrite sections of the code to minimize operations in critical paths, focusing on reducing overall time complexity.

Provide the Python code in the following JSON format. Note that newlines within the `"optimized_code"` field should be represented by `\\n` to ensure JSON compatibility:

{{
  "optimization": {{
    "language": "Python",
    "previous_code": "{final_code}",
    "optimized_code": "Your improved Python code here, adjusted to meet the performance requirements",
    "improvement_explanation": {{
      "summary": "Summarize the main improvements and any breakthroughs made.",
      "details": "Explain the specific changes made in the optimized code that improve performance. Describe why these changes are effective and how they address the performance bottleneck noted in the previous code."
    }}
  }}
}}
"""
    else:
        return f"""
Here is the final code that needs optimization:
{final_code}

Based on your correct understanding of the problem:
{refine_problem_understanding}

Task: The current solution has passed all sample test cases but needs to be optimized to handle larger inputs effectively. 
Your goal is to improve its runtime efficiency so that it can process full test cases within the constraints in your understanding.

Instructions:
1. **Analyze Bottlenecks**: Identify any inefficient parts of the code, such as nested loops or redundant calculations.
2. **Optimize Data Structures**: Where possible, replace costly operations with efficient data structures (e.g., dictionaries, heaps, binary search trees).
3. **Reduce Redundancies**: Remove or simplify repeated calculations, and consider precomputing values where feasible.
4. **Simplify Logic**: Rewrite the code to reduce the number of operations in critical sections.


Provide the Python code in the following JSON format. Note that newlines within the `"optimized_code"` field should be represented by `\\n` to ensure JSON compatibility:
{{
  "optimization": {{
    "language": "Python",
    "previous_code": "{final_code}",
    "optimized_code": "Your improved Python code here, adjusted to meet the performance requirements",
    "improvement_explanation": {{
      "summary": "Summarize the main improvements and any breakthroughs made.",
      "details": "Explain the specific changes made in the optimized code that improve performance. Describe why these changes are effective and how they address the performance bottleneck noted in the previous code."
    }}
  }}
}}
"""


