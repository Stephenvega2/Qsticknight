import hashlib
import base64
import json

def generate_hash(data: str) -> str:
    """Generate SHA-256 hash of data"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string"""
    return base64.b64encode(data).decode('utf-8')

def decode_base64(data: str) -> bytes:
    """Decode base64 string to bytes"""
    return base64.b64decode(data.encode('utf-8'))

def load_json(filepath: str) -> dict:
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load JSON: {str(e)}")

def save_json(data: dict, filepath: str):
    """Save data to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise Exception(f"Failed to save JSON: {str(e)}")
