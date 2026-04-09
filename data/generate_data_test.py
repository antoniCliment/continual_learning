import sys
import os
import time
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful assistant." # Modify this if you want specific system behavior
GEN_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_text_file(file_path):
    """Loads text content from a file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        sys.exit(1)

def generate_response(client, model_id, system_msg, user_msg):
    """Wraps the OpenAI API call with basic error handling."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            **GEN_PARAMS
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None
    
# Transform df to have only input, ouput columns
def transform_df(input_df):
    records = []
    for _, row in input_df.iterrows():
        characteristic = row["characteristic"]
        for i, text_type in enumerate(text_types):
            records.append({
                "input": f"Text type: {text_type} and characteristic: {characteristic}.",
                "output": row[f"text_type_{i+1}"]
            })
    return pd.DataFrame(records)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Setup and Input Validation
    if len(sys.argv) < 3:
        print("Usage: python generate_data.py [model_name] [output_folder]")
        print("Example: python generate_data.py gpt-4o-mini gen_v1")
        print("Example: python generate_data.py gpt-4 gen_v1")
        sys.exit(1)

    model_id, folder_name = sys.argv[1], sys.argv[2]
    
    # Ensure API key is present
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key)

    # 2. File Paths and Safety Checks
    os.makedirs(folder_name, exist_ok=True)
    output_file = os.path.join(folder_name, "all_data.csv")

    # Check to avoid accidental overwrite of expensive API generated data
    if os.path.exists(output_file):
        confirm = input(f"Warning: '{output_file}' already exists. Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    print(f"Using OpenAI Model: {model_id}")

    # 3. Load Data
    prompt_template = load_text_file("./data/toy/prompt.txt")
    characteristics_raw = load_text_file("./data/toy/characteristics.txt")
    text_type_raw = load_text_file("./data/toy/text_type.txt")
    text_types = [line.strip() for line in text_type_raw.splitlines() if line.strip()]
    characteristics = [line.strip() for line in characteristics_raw.splitlines() if line.strip()]

    print(f"Loaded {len(characteristics)} characteristics to process.")
    print(f"Text types available: {text_types}")

    # 4. Generation Loop
    results = []
    
    print("Starting generation...")
    try:
        for n, characteristic in enumerate(characteristics):
            for text_type in text_types:

                # Format the user prompt using the template
                formatted_prompt = prompt_template.format(text_type=text_type, characteristic=characteristic)
                generated_text = generate_response(
                    client, 
                    model_id, 
                    SYSTEM_PROMPT, 
                    formatted_prompt
                )

                if generated_text:
                    results.append({
                        "input": f"Text type: {text_type}. Characteristic: {characteristic}",
                        "output": generated_text
                    })
                    
                    # Save incrementally
                    df = pd.DataFrame(results)
                    df.to_csv(output_file, index=False)
            
            print(f"Completed {n+1}/{len(characteristics)}")


    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial progress...")

    # 5. Final Save and Metadata
    if results:
        df = pd.DataFrame(results)
        
        # Save master file (redundant but safe)
        df.to_csv(output_file, index=False)
        print(f"Saved all data to {output_file}")

        # 6. Save Metadata
        with open(os.path.join(folder_name, "generation_metadata.txt"), 'w') as meta_file:
            meta_file.write(f"Model used: {model_id}\n")
            meta_file.write(f"System Prompt: {SYSTEM_PROMPT}\n")
            meta_file.write(f"Number of examples: {len(df)}\n")
            meta_file.write(f"Hyperparameters: {GEN_PARAMS}\n")
            meta_file.write(f"Prompt Template: {prompt_template}\n")
    else:
        print("No results were generated.")