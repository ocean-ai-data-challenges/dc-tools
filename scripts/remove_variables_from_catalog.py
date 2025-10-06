import os
import json
import glob
import sys

def remove_variables_from_features(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        if "variables" in properties:
            del properties["variables"]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_json_directory(directory, ext=".novars.json"):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for json_file in json_files:
        base, _ = os.path.splitext(json_file)
        output_file = base + ext
        remove_variables_from_features(json_file, output_file)
        print(f"Processed: {json_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory> [extension]")
        sys.exit(1)
    directory = sys.argv[1]
    ext = sys.argv[2] if len(sys.argv) > 2 else ".novars.json"
    process_json_directory(directory, ext)
