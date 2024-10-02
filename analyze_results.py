import os
import re
import json
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import yaml
import numpy as np
    

def clean_json_string(json_str):
    """ 
    Convert single quotes to double quotes and handle trailing commas.
    Also replace Python None, True, and False with their JSON equivalents.
    """
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'None', 'null', json_str)
    json_str = re.sub(r'False', 'false', json_str)
    json_str = re.sub(r'True', 'true', json_str)
    return json_str

def extract_data_from_file(filepath):
    """
    Extract model parameters, strategy parameters, last results, and traceability percentages from a file.
    """
    # retrieve file
    with open(filepath, 'r') as file:
        lines = file.readlines()
    print("analyzing file name: ", filepath)

    # Extract model parameters and strategy parameters
    model_params = None
    strategy_params = None
    for line in lines:
        if line.startswith("Model parameters"):
            model_params_str = line.split(": ", 1)[1]
            model_params = json.loads(clean_json_string(model_params_str))
        elif line.startswith("Strategy parameters"):
            strategy_params_str = line.split(": ", 1)[1]
            strategy_params = json.loads(clean_json_string(strategy_params_str))

    # Extract last accuracy, f1_score, precision, and recall
    results = []
    for line in lines:
        if re.match(r"{'accuracy': .*", line) or re.match(r"{'f1_score': .*", line):
            results.append(json.loads(clean_json_string(line.strip())))
    last_result = results[-1] if results else {}
    
    # Extract percentages of traceable records
    percentage_traceable = None
    for line in lines:
        if line.startswith("Percentage of traceable records"):
            percentage_traceable = float(line.split(": ", 1)[1].strip().replace("%", ""))
    
    for line in lines:
        if line.startswith("Percentage of maybe-traceable records"):
            percentage_maybe_traceable = float(line.split(": ", 1)[1].strip().replace("%", ""))
    
    return last_result, percentage_traceable, percentage_maybe_traceable, model_params, strategy_params

def normalize_values(values):
    """
    Normalize a the values to the range [0, 1] using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(values.reshape(-1, 1)).flatten()

def process_files(directory):
    """
    Process all output files in the specified directory to extract and normalize data.
    """
    files = glob(os.path.join(directory, "*.txt"))
    print("Found", len(files), "files.")

    # Retrieve data
    data = []
    for file in files:
        last_result, percentage_traceable, percentage_maybe_traceable, model_params, strategy_params = extract_data_from_file(file)
        
        if not last_result or percentage_traceable is None:
            continue
        
        data.append({
            "filename": os.path.basename(file),
            "results": last_result,
            "percentage_traceable": percentage_traceable,
            "percentage_maybe_traceable": percentage_maybe_traceable,
            "traceability": percentage_traceable + 0.5*(percentage_maybe_traceable-percentage_traceable),
            "model_parameters": model_params,
            "strategy_parameters": strategy_params
        })
    
    print("normalize_values")
    # Normalize values
    if data:
        normalized_f1_scores = normalize_values(np.array([d["results"]["f1_score"] for d in data]))
        normalized_traceability = normalize_values(np.array([d["traceability"] for d in data]))
        
        for i, d in enumerate(data):
            d["normalized_results"] = {
                "traceability": normalized_traceability[i],
                "f1_score": normalized_f1_scores[i],
                "accuracy": d["results"]["accuracy"],
                "precision": d["results"]["precision"],
                "recall": d["results"]["recall"]
            }
            d["score"] = ((1 - normalized_traceability[i]) + normalized_f1_scores[i])/2
    
    print("sort")
    # Sort by score
    data = sorted(data, key=lambda x: x["score"], reverse=True)
    
    return data

def write_results_to_file(data, output_filepath):
    """
    Write processed results to an output file
    """
    with open(output_filepath, 'w') as file:
        file.write("Results file, sorted by score\n")
        file.write("name is model_[datasetname]_[rounds]_[strategyname]_[amount of clients]_[timestamp of start]\n")
        file.write("=====================================\n\n")
        for d in data:
            file.write(f"filename: {d['filename']}\n")
            file.write(f"score: {d['score']}\n")
            file.write(f"privacy score: {{'traceability: '{d['traceability']}, 'percentage_traceable': {d['percentage_traceable']*100}, 'percentage_maybe_traceable' {d['percentage_maybe_traceable']*100} }}\n")
            file.write(f"normalized results: {d['normalized_results']}\n")
            file.write(f"results: {d['results']}\n")
            file.write(f"model parameters: {d['model_parameters']}\n")
            file.write(f"strategy parameters: {d['strategy_parameters']}\n")
            file.write("\n")

def read_yaml():
    with open('analyze.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    
    directory = read_yaml()['directory']
    print("Analyzing files in directory:", directory)
    output_filepath = read_yaml()['resulting_file_name']
    print("Writing results to:", output_filepath)
    data = process_files(directory)
    write_results_to_file(data, output_filepath)
    print("Done analyzing files.")