import requests
import json
from datetime import datetime, timedelta, UTC

# Assume the FastAPI application is running at this base URL
BASE_URL = "http://localhost:8000/api/v1"  # The CLI server (geo-infer-health serve) mounts the API here
ENVIRONMENT_URL = f"{BASE_URL}/environment"


def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
    except json.JSONDecodeError:
        print(f"Response Text: {response.text}")
    print("---")


def submit_sample_env_readings():
    print("Submitting sample environmental readings...")
    readings_data = [
        {
            "data_id": "reading001",
            "parameter_name": "PM2.5",
            "value": 12.5,
            "unit": "µg/m³",
            "location": {"latitude": 34.0522, "longitude": -118.2437},
            "timestamp": datetime.now(UTC).isoformat(),
        },
        {
            "data_id": "reading002",
            "parameter_name": "PM2.5",
            "value": 15.2,
            "unit": "µg/m³",
            "location": {"latitude": 34.0530, "longitude": -118.2445},
            "timestamp": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        },
        {
            "data_id": "reading003",
            "parameter_name": "NO2",
            "value": 40.0,
            "unit": "ppb",
            "location": {"latitude": 34.0522, "longitude": -118.2437},
            "timestamp": datetime.now(UTC).isoformat(),
        },
        {
            "data_id": "reading004",
            "parameter_name": "PM2.5",
            "value": 10.1,
            "unit": "µg/m³",
            "location": {"latitude": 34.0500, "longitude": -118.2400},
            "timestamp": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        },
    ]
    for reading in readings_data:
        response = requests.post(f"{ENVIRONMENT_URL}/readings/", json=reading)
        print_response(response)


def get_env_readings():
    print("Getting environmental readings (PM2.5 only)...")
    params = {"limit": 10, "parameter_name": "PM2.5"}
    response = requests.get(f"{ENVIRONMENT_URL}/readings/", params=params)
    print_response(response)


def get_readings_near_loc_example():
    print("Getting readings near a location...")
    params = {
        "latitude": 34.0525,
        "longitude": -118.2440,
        "radius_km": 0.5,
        "parameter_name": "PM2.5",
        "start_time_iso": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        "end_time_iso": datetime.now(UTC).isoformat(),
    }
    response = requests.post(f"{ENVIRONMENT_URL}/readings/near_location", params=params)
    print_response(response)


def get_average_exposure_example():
    print("Calculating average exposure for target locations...")
    payload = {
        "target_locations_query": [
            {"latitude": 34.0522, "longitude": -118.2437},
            {"latitude": 34.0550, "longitude": -118.2300},
        ],
        "radius_km": 1.0,
        "parameter_name": "PM2.5",
        "time_window_days": 2,
    }
    response = requests.post(
        f"{ENVIRONMENT_URL}/exposure/average", json=payload
    )  # Note: Changed to json=payload
    print_response(response)


if __name__ == "__main__":
    submit_sample_env_readings()
    get_env_readings()
    get_readings_near_loc_example()
    get_average_exposure_example()

    print("\nEnvironmental Health API examples completed.")
    print(f"Ensure your FastAPI server is running and accessible at {BASE_URL}")
    print("And that the GEO-INFER-HEALTH API routers are correctly mounted.")
