import os

ENDPOINTS = {}

def parse_endpoints_from_environ():
    global ENDPOINTS
    for name, value in os.environ.items():
        if name.startswith('HF_INFERENCE_ENDPOINT_'):
            ENDPOINTS[name[len('HF_INFERENCE_ENDPOINT_'):].lower()] = value
            
