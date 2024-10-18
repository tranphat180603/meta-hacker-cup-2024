import os

# Extract problem cases and include sample_input and sample_output in the problem_description
def extract_problem_cases_from_hf(dataset):
    problem_cases = []
    for example in dataset['full']:
        sample_input = example["sample_input"]
        sample_output = example["sample_output"]
        
        # Format the problem description
        problem_description = f"""
{example['statement']}
        
### Sample Input
{sample_input}

### Sample Output
{sample_output}
"""

        # Append the formatted problem description to the list of problems
        problem_cases.append({
            "name": example["name"],
            "year": example["year"],
            "round": example["round"],
            "problem_description": problem_description,
            "sample_input": sample_input,
            "sample_output": sample_output
        })
    return problem_cases


def extract_problem_cases_from_folder(dataset_path):
    problem_cases = []
    
    # Traverse the directory structure
    for root, dirs, files in os.walk(dataset_path):
        # Get the directory name (problem name)
        problem_name = os.path.basename(root)
        
        # Check if the required files are in the current directory
        if 'statement.txt' in files and 'sample_in.txt' in files and 'sample_out.txt' in files:
            # Read content from the necessary files
            with open(os.path.join(root, 'statement.txt'), 'r') as statement_file:
                statement = statement_file.read().strip()
            
            with open(os.path.join(root, 'sample_in.txt'), 'r') as sample_in_file:
                sample_input = sample_in_file.read().strip()
                
            with open(os.path.join(root, 'sample_out.txt'), 'r') as sample_out_file:
                sample_output = sample_out_file.read().strip()
                
            # Concatenate the information into a problem description
            problem_description = f"""
{statement}
            
### Sample Input
{sample_input}

### Sample Output
{sample_output}
"""
            
            # Add the problem description to the list
            problem_cases.append({
                "name": problem_name,  # The folder name is used as the problem name
                "problem_description": problem_description,
                "sample_input": sample_input,
                "sample_output": sample_output
            })
    
    return problem_cases