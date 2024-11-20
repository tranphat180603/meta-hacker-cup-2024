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

def analyze_original_test_cases_template(problem_description, reflection=""):
    # Define `general_formula_update` by default to prevent referencing issues
    general_formula_update = ""
    
    # Extract `general_formula_update` from reflection if provided
    if isinstance(reflection, dict):
      general_formula_update = reflection.get("reflection", {}).get("solution_revision", {}).get("general_formula_update", "")

    reflection_section = f"""
Here is the reflection from previous iterations:
'{reflection}'

Instructions:
- Use the insights from the reflection to identify any previous issues, patterns, or constraints that were missed.
- Incorporate these insights into your analysis to avoid repeating the same mistakes.
- Make adjustments to your interpretation of the test case based on what was learned from previous attempts.
""" if reflection else ""

    changes_based_on_reflection_field = """
  "changes_based_on_reflection": [
    "Summarize updates made based on reflection insights."
  ],
  """ if reflection else ""

    # Conditionally include the "general_formula" field if reflection exists
    general_formula_field = (
        f'"general_formula": "Rewrite exactly what you see here: {general_formula_update}"' if general_formula_update else ""
    )

    return f"""
Task: Based on the problem description: 

'{problem_description}', 

your job is to analyze the original test case input and output, map each component to its corresponding variable from the problem description, and explain how these inputs lead to the specified output based on the logic and constraints of the problem.

Choose only the first test case to analyze.

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

- Output:
  Expected output format specified by the problem (e.g., Case #1: YES, result values, etc.). 

{reflection_section}

**Provide the analysis in the following generalized JSON structure**:
{{
  {changes_based_on_reflection_field}
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
    }},
    "solution_approach": {{
      "core_logic": "Summarize the main logic or approach derived from the output explanation. This should capture the essential steps needed to reach the solution.",
      "variables_used": "List the key variables that play a role in determining the output based on the logic."
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
      "List hints or strategies derived from key_observations for approaching this problem."
    ],
    {general_formula_field}
  }}
}}

Ensure that your analysis in the 'test_case_reflection' section captures general insights that go beyond the specific example, providing patterns and hints applicable to other cases.
"""


def refine_problem_understanding_template(problem_understanding, test_case_analysis, reflection="", img_understanding=""):
    # Optional sections based on inputs
    reflection_section = f"""
Here is the reflection from previous iterations:
'{reflection}'
""" if reflection else ""

    img_understanding_section = f"""
Here is the image understanding:
'{img_understanding}'
""" if img_understanding else ""

    general_formula_update = ""
    
    # Extract `general_formula_update` from reflection if it exists
    if reflection:
      general_formula_update = reflection.get("reflection", {}).get("solution_revision", {}).get("general_formula_update", "")


    # Conditionally include "general_formula_update" field; add instruction if not available
    general_formula_update_field = (
        f'"general_formula_update": "Rewrite exactly what you see here: {general_formula_update}",' if general_formula_update else 
        '"general_formula_update": "Write specific, condensed, short and precise formula or algorithmic expression that captures the essential logic required to solve this problem based on your understanding of the problem and test case analysis. Focus on using clear mathematical or logical expressions that are both efficient and easy to follow. The formula should highlight key variables, dependencies, and any conditional checks needed to meet the problem’s requirements, minimizing any unnecessary complexity.",'     )
    return f"""
Task: Refine the problem understanding based on test case analysis{', reflection insights' if reflection else ''}{', and image understanding' if img_understanding else ''}.

Consider:
- Missed constraints or nuances.
- Observed input-output structures in the test cases.
{'- Key takeaways or patterns identified in the reflection.' if reflection else ''}
{'- Visual insights related to the problem’s structure, patterns, or components.' if img_understanding else ''}
- Assume initial ideas were mostly incorrect; base updates on the combined insights.

Here is the original understanding:
'{problem_understanding}'

Here is the test case analysis:
'{test_case_analysis}'

{reflection_section}

{img_understanding_section}

Provide the refined problem understanding in JSON format:
{{
  "refined_problem_understanding": {{
    "changes_based_on_reflection": {["Summarize updates made based on reflection insights."] if reflection else []},
    "image_insights": {["Summarize specific updates based on image insights."] if img_understanding else []},
    "goal": "State the refined objective of the problem.",
    "updated_constraints": "List updated constraints and any new limitations discovered.",
    "test_cases_update": {{
      "input_format": "Update the input format if changed.",
      "output_format": "Update the output format if changed."
    }},
    "important_ideas_update": [
      "List new or corrected important ideas based on test case analysis{', reflection,' if reflection else ''}{' and image insights' if img_understanding else ''}."
    ],
    "difficulty_assessment_update": {{
      "updated_difficulty": "Reassess the problem difficulty (easy, medium, hard, super hard).",
      "justification": "Explain the reasoning for the updated difficulty."
    }},
    {general_formula_update_field}
  }}
}}
"""


def get_solution_ideas_template(refine_problem_understanding, test_case_analysis, num_solutions):
    # Extract `general_formula_update` from `refine_problem_understanding` using `get`
    general_formula_update = refine_problem_understanding.get("refined_problem_understanding", {}).get("general_formula_update", "") if isinstance(refine_problem_understanding, dict) else ""

    # Conditionally include `general_formula_update` field if it exists
    general_formula_update_field = (
        f'"general_formula_update": "{general_formula_update}",' if general_formula_update else ""
    )

    return f"""
Task: Based on your understanding of the problem:

{refine_problem_understanding} 

and analysis of the test cases:

{test_case_analysis}, 

come up with {num_solutions} ideas that can pass all test cases (original and AI-generated). 

Use the existing general formula from `refine_problem_understanding` to maintain consistency in the solution approach.

Provide the ideas in valid JSON format following the structure below.
Note that: there must not be any text outside of the JSON format for validity!
{{
  {general_formula_update_field}
  "solutions": [
    {{
      "name": "Give the name or category of the first approach.",
      "strategy": "Explain the general strategy for this approach."
    }}
  ]
}}
"""

def evaluate_solutions_template(solution_ideas, refine_problem_understanding, test_case_analysis):
    # Safely access `general_formula_update` from `refine_problem_understanding`
    general_formula_update = refine_problem_understanding.get("refined_problem_understanding", {}).get("general_formula_update", "") if isinstance(refine_problem_understanding, dict) else ""

    # Conditionally include the `general_formula_update` field if it exists
    general_formula_update_field = (
        f'"general_formula_update": "{general_formula_update}",' if general_formula_update else ""
    )

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
        {general_formula_update_field}
        "solution_name": "The name of the chosen solution",
        "justification": {{
            "goal_alignment": "Explain how the solution addresses the main goal of the problem: \"{refine_problem_understanding.get('refined_problem_understanding', {}).get('goal', 'No goal provided')}\".",
            "constraint_handling": "Evaluate how well the solution meets the problem's constraints: \"{refine_problem_understanding.get('refined_problem_understanding', {}).get('updated_constraints', 'No constraints provided')}\".",
            "important_ideas": "Explain how the solution incorporates key ideas from the problem understanding: \"{refine_problem_understanding.get('refined_problem_understanding', {}).get('important_ideas_update', 'No key ideas provided')}\".",
            "edge_case_handling": "Evaluate how the solution handles edge cases (if applicable).",
            "time_efficiency": "Provide the estimated time complexity and evaluate if it's suitable given the constraints.",
            "space_efficiency": "Provide the estimated space complexity and evaluate if it's efficient."
        }},
    }}
}}
"""


def get_code_generation_template(selected_solution, test_case_analysis, refine_problem_understanding):
    # Safely access `general_formula_update` from `refine_problem_understanding`
    general_formula_update = (
        refine_problem_understanding.get("refined_problem_understanding", {}).get("general_formula_update", "")
    )

    # Conditionally include `general_formula_update` field if it exists
    general_formula_update_field = (
        f'"general_formula_update": "{general_formula_update}",' if general_formula_update else ""
    )

    return f"""
You are tasked with generating Python code for the solution: 
{selected_solution}

based on the provided test case analysis: 

{test_case_analysis}

And your own understanding of the problem:
{refine_problem_understanding}

Follow the instructions below:

Code generation guidelines:
1. Your code should be valid Python code.
2. Divide the code into small, well-named sub-functions.
3. Always use Python's built-in `input()` function to handle input directly.
4. Always include an `if __name__ == '__main__':` block, ensuring the code is executable as a standalone script.

##SUPER IMPORTANT:***
IN ANY GIVEN CIRCUMSTANCES, DO NOT USE "input = sys.stdin.read" and/or "sys.stdin.readline()" since it will definitely affect the performance of the process!

Provide the Python code in the following JSON format. Note that newlines within the `"code"` field should be represented by "\\n" to ensure JSON compatibility:

{{
  "solution_code": {{
    {general_formula_update_field}
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
    # Extract `general_formula_update` from generated_code if it exists
    general_formula_update = (
        generated_code.get("solution_code", {}).get("general_formula_update", "")
        if isinstance(generated_code, dict) else ""
    )

    # Conditionally include `general_formula_update` field if it exists
    general_formula_update_field = (
        f'"general_formula_update": "{general_formula_update}",'
        if general_formula_update
        else '"general_formula_update": "Update or refine the formula if needed based on the error."'
    )

    return f"""
Task: Review the following error and code, then generate updated Python code to fix the issue.

Error Message:
{error_message}

Current Code:
'{generated_code}'

Error History:
{error_history}

Test Case Analysis:
{test_case_analysis}

Instructions:
1. Identify the cause of the error and propose a code modification to fix it.
2. If necessary, refine the general formula to address any underlying logic issues.
3. Output the updated code in JSON format, as shown below.

Return valid JSON format exactly like you are provided without text outside of the JSON format:
{{
  "solution_code": {{
    {general_formula_update_field}
    "code": "Your updated Python code here, including any changes to address the error."
  }}
}}
"""




def reflect_failed_test(generated_code, failed_tests, refine_problem_understanding, failure_history):
    # Include failure history if provided
    failure_history_section = ""
    if failure_history:
        failure_history_section = f"""
Failure History:
Each entry below includes a unique combination of previous failed cases and the solution formula that led to failure, along with a count of how often this specific failure occurred across different iterations.
Use this information to identify patterns, avoid reusing approaches that frequently fail, and prioritize developing a fundamentally different strategy.

{failure_history}
"""

    general_formula_update = (
        refine_problem_understanding.get("refined_problem_understanding", {}).get("general_formula_update", "")
    )

    # Conditionally include `general_formula_update` field with a specific instruction if it exists
    general_formula_update_field = (
        f'General formula update": {general_formula_update}. This approach has failed and there are very likely that the core logic/formula is totally wrong for this problem!' 
        if general_formula_update else ""
    )

    return f"""
Task: Carefully analyze the failed test cases and failure history. Identify recurring patterns, underlying logic issues, and constraints that may be causing these failures. Your goal is to provide a comprehensive reflection that offers insights for significant improvement and adaptation to avoid repeating similar mistakes.

This iteration failed on these tests:
{failed_tests}

Current Code:
'{generated_code}'

Problem Understanding:
{refine_problem_understanding}

{failure_history_section}

{general_formula_update_field}

Instructions:
- Examine each failed case in `failure_history` for patterns, such as recurring logic issues or missed constraints.
- Use the occurrence counts to identify the most frequent failures and prioritize redesigning these parts of the approach.
- Develop a new strategy that differs fundamentally from previously failed approaches recorded in `failure_history`.

Reflect on the issues in JSON format, capturing the core issues, any new or alternative approaches suggested, specific changes for future solutions, and updates to the general formula.

{{
  "reflection": {{
    "patterns_in_failures": [
      {{
        "pattern_description": "Describe any repeating patterns or issues observed across failed cases."
      }}
    ],
    "revised_strategy": [
      {{
        "approach": "Summarize a new or adapted approach to address the failures identified, avoiding previously used failing solutions.",
        "reasoning": "Explain why this approach may be more effective than previous attempts.",
        "specific_steps": "List specific steps or adjustments needed to implement the approach."
      }}
    ],
    "solution_revision": {{
      "core_logic_issues": [
        {{
          "component": "Component or function causing issue",
          "current_approach": "Current code logic that led to failure. View from the general_formula_update",
          "proposed_fix": "Updated or new code snippet that addresses the failure, try to think of the approach that is fudamentally different from the core formula in the current approach."
        }}
      ],
      "general_formula_update": "Write specific, condensed, short and precise formula or algorithmic expression that captures the essential logic required to solve this problem. Image writing ideas to solve the problem, not the full code to solve the problem. Note that, the core formula/logic of this new approach must be completely different from the previous approach."
    }},
    "learning_points": {{
      "insights": [
        {{
          "insight": "Detailed explanation of what was learned from analyzing the patterns and issues.",
          "application": "How this insight will be applied in future solutions to avoid repeating similar mistakes."
        }}
      ]
    }}
  }}
}}

Note: Provide specific information rather than general observations. Each component should be unique to the analysis and focused on breaking recurring failure patterns. Aim for actionable, explorative insights that directly address the failed test cases and encourage the generation of varied approaches.
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


