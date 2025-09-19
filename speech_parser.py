import pandas as pd
import json

def excel_to_json(input_excel, output_json):
    df = pd.read_excel(input_excel)

    tasks = []

    for _, row in df.iterrows():
        # Handle input file (single file only)
        input_file = str(row.get("Input File", "")).strip()
        input_path = "/home/student/Desktop/PS_SOFTWARE/PS/backend/data/datasets/Speech-Recognition/input/" + input_file if input_file else ""

        # Handle output files (comma-separated in Excel)
        output_files = str(row.get("Output File", "")).strip()
        if output_files:
            output_files = ["/home/student/Desktop/PS_SOFTWARE/PS/backend/data/datasets/Speech-Recognition/solution/" + f.strip()
                            for f in output_files.split(",")]
        else:
            output_files = []

        task = {
            "id": str(row.get("S.No", "")).strip(),
            "title": str(row.get("Scenario", "")).strip(),
            "description": str(row.get("Task", "")).strip(),
            "datasets": {
                "input_file": input_path
            },
            "parts": [
                {
                    "part_id": str(row.get("S.No", "")).strip(),
                    "type": "csv_similarity",
                    "description": str(row.get("Task", "")).strip(),
                    "solution_file": output_files if len(output_files) > 1 else (output_files[0] if output_files else "")
                }
            ]
        }
        tasks.append(task)

    # Save JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON saved to {output_json}")


if __name__ == "__main__":
    excel_to_json("Speech_Recognition.xlsx", "tasks.json")
