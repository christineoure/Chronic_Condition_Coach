# test_config.py
import json
from pathlib import Path

def test_config_file():
    config_path = Path("data_collection/sources.json")
    
    print(f"Checking config file at: {config_path}")
    print(f"File exists: {config_path.exists()}")
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                print(f"File size: {len(content)} characters")
                print(f"First 100 chars: {content[:100]}...")
                
                # Try to parse JSON
                config = json.loads(content)
                print(" JSON is valid!")
                print(f"Web sources: {len(config.get('web_sources', []))}")
                print(f"Synthetic prompts: {len(config.get('synthetic_prompts', []))}")
                
        except json.JSONDecodeError as e:
            print(f" Invalid JSON: {e}")
            print("Creating valid config file...")
            create_valid_config(config_path)
        except Exception as e:
            print(f" Error: {e}")
    else:
        print(" Config file doesn't exist")
        create_valid_config(config_path)

def create_valid_config(path):
    """Create a valid config file"""
    valid_config = {
        "web_sources": [
            {
                "name": "Test Source",
                "url": "https://www.cdc.gov/chronicdisease/about/index.htm",
                "type": "html"
            }
        ],
        "synthetic_prompts": [
            "Test prompt 1",
            "Test prompt 2"
        ]
    }
    
    try:
        with open(path, 'w') as f:
            json.dump(valid_config, f, indent=2)
        print(f" Created valid config at {path}")
    except Exception as e:
        print(f" Failed to create config: {e}")

if __name__ == "__main__":
    test_config_file()