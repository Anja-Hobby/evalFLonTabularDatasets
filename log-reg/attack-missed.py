import os
import subprocess
import json
import yaml

def extract_setup_values_from_txt(file_path):
    """
    Extract setup values from the .txt file.

    Parameters:
    - file_path: Path to the .txt file containing model parameters.

    Returns:
    - setup_values: The extracted setup values from the .txt file.
    """
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Model parameters:"):
                # Extract the part after "Model parameters:"
                setup_values_str = line.replace("Model parameters:", "").strip()

                # Replace single quotes with double quotes to make it valid JSON
                setup_values_str = setup_values_str.replace("'", "\"")

                # Replace Python's None with JSON's null
                setup_values_str = setup_values_str.replace("None", "null")

                # Replace Python's True/False with JSON's true/false
                setup_values_str = setup_values_str.replace("True", "true").replace("False", "false")

                # Debugging: print the string after replacement
                print(f"Modified setup values string: {setup_values_str}")

                # Load the JSON object from the modified string
                try:
                    setup_values = json.loads(setup_values_str)
                    return setup_values
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {file_path}: {e}")
                    return {}

    return {}

def getfolderpaths():
    """
    Get the folder paths from the 'conf.yaml' file.

    Returns:
    - folder_paths: A list of folder paths.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return list(config['missed-folders'].values())

def run_mia_on_files_in_folder():
    """
    Run the mia.py script on all .txt and .npz files in the given folder.

    Parameters:
    - folder_path: The path to the folder containing the files.
    - num_classes: The number of classes for the model (default is 2).
    """
    num_classes = 2
    files = []
    folder_paths = getfolderpaths()
    for folder_path in folder_paths:
        files_in_path = os.listdir(folder_path)
        # Prepend the folder path to the filenames so we can access them later
        files.extend([os.path.join(folder_path, f) for f in files_in_path])

    # Now 'files' contains the full paths to all files across all folders
    txt_files = [f for f in files if f.endswith(".txt")]
    npz_files = [f for f in files if f.endswith(".npz")]

    for txt_file in txt_files:
        npz_file = txt_file.replace(".txt", "")
        print(f"Processing {txt_file} and {npz_file}")

        # Extract setup values from .txt file
        setup_values = extract_setup_values_from_txt(txt_file)

        # Prepare the command to run mia.py with arguments
        command = [
            'python3', 'mia.py',
            '--location', os.path.dirname(txt_file) + '/',  # Folder where the model file is located
            '--modeltoattack', os.path.basename(npz_file),      # The model file to attack (npz file)
            '--extralines', f"'{txt_file}'",  # Extra information from the txt file
            '--num_classes', str(num_classes),  # Number of classes
            '--setupvalues', json.dumps(setup_values)  # Model parameters from the txt file
        ]

        # Run the mia.py script with the prepared arguments
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            print(f"Output for {npz_file}:\n{result.stdout}")
            if result.stderr:
                print(f"Error for {npz_file}:\n{result.stderr}")
        except Exception as e:
            print(f"Failed to run MIA for {npz_file}: {e}")

if __name__ == "__main__":
    """
    Main entry point of the script. Input variables retrieved through the yaml, conf.yaml.
    """
    run_mia_on_files_in_folder()