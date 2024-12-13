import json

def get_criteria()->str:
    with open('resource/band_assignments.json', 'r') as file:
        evaluation_criteria = json.load(file)

    criteria_string = ""
    for band in evaluation_criteria['bands']:
        criteria_string += f"{band['band']}:\n"
        for key, value in band['criteria'].items():
            criteria_string += f"  {key}: {value}\n"
    
    return criteria_string