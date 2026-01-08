# simple_collect.py
import json
from pathlib import Path

# Create directories
Path("data/collected/raw").mkdir(parents=True, exist_ok=True)
Path("data/collected/processed").mkdir(parents=True, exist_ok=True)

# Create a simple config
config = {
    "web_sources": [
        {
            "name": "CDC Chronic Disease",
            "url": "https://www.cdc.gov/chronicdisease/about/index.htm",
            "type": "html"
        }
    ],
    "synthetic_prompts": [
        "What are the best foods for diabetes management?",
        "How to naturally lower blood pressure?",
        "Exercise recommendations for arthritis patients"
    ]
}

# Save config
config_path = Path("data_collection/sources.json")
config_path.parent.mkdir(exist_ok=True)
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f" Created config file: {config_path}")

# Create some initial synthetic data
synthetic_data = [
    {
        "doc_id": "synth_001",
        "title": "Diabetes Management Guide",
        "content": "Managing diabetes involves regular monitoring of blood sugar levels, following a balanced diet, engaging in physical activity, and taking medications as prescribed. Key strategies include carbohydrate counting, portion control, and stress management.",
        "source": "synthetic",
        "topic": "diabetes"
    },
    {
        "doc_id": "synth_002",
        "title": "Hypertension Lifestyle Tips",
        "content": "To lower blood pressure naturally: reduce sodium intake, increase potassium-rich foods, exercise regularly, maintain healthy weight, limit alcohol, and manage stress through techniques like meditation and deep breathing.",
        "source": "synthetic",
        "topic": "hypertension"
    },
    {
        "doc_id": "synth_003",
        "title": "Chronic Pain Management",
        "content": "Managing chronic pain involves a multi-faceted approach: physical therapy, gentle exercise, stress reduction techniques, proper sleep hygiene, and pain medications as prescribed. Mind-body techniques like mindfulness can be particularly helpful.",
        "source": "synthetic",
        "topic": "pain management"
    }
]

# Save synthetic data
data_file = Path("data/collected/raw/synthetic_data.json")
with open(data_file, 'w') as f:
    json.dump(synthetic_data, f, indent=2)

print(f" Created initial data file: {data_file}")
print(f" Created {len(synthetic_data)} synthetic documents")
print("\n Next steps:")
print("1. Test the config: python -c \"import json; print(json.load(open('data_collection/sources.json')))\"")
print("2. Run data collection: python run_data_collection.py")