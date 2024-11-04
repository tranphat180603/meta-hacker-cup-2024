import os

# Extract problem cases and include sample_input and sample_output in the problem_description
def extract_problem_cases_from_hf(dataset, start_index=0, end_index=None, problem_name=None):
    problem_cases = []
    data_subset = dataset['full']
    end_index = end_index or len(data_subset['name'])  # Set end_index if None

    for idx in range(start_index, end_index):
        # Check if problem_name is specified and if it matches the current problem
        if problem_name and data_subset["name"][idx].lower() != problem_name.lower():
            continue

        sample_input = data_subset['sample_input'][idx]
        sample_output = data_subset['sample_output'][idx]
        full_input = data_subset['input'][idx]
        img_raw = data_subset['images'][idx][0] if data_subset['images'][idx] else ""  # Extract first image if available

        # Format the problem description
        problem_description = f"""
{data_subset['statement'][idx]}

### Sample Input
{sample_input}

### Sample Output
{sample_output}
"""

        # Append the formatted problem case
        problem_cases.append({
            "name": data_subset["name"][idx],
            "year": data_subset["year"][idx],
            "round": data_subset["round"][idx],
            "problem_description": problem_description,
            "image": img_raw,  # Pass the raw image string here
            "sample_input": sample_input,
            "sample_output": sample_output,
            "full_input": full_input
        })

        # If problem_name is specified, exit after finding the first match to save time
        if problem_name:
            break

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

            with open(os.path.join(root, 'full_in.txt'), 'r') as full_input_file:
                full_input = full_input_file.read().strip()
                
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
                "sample_output": sample_output,
                "full_input": full_input
            })
    
    return problem_cases