import requests
import json
from datetime import datetime, UTC

# Assume the FastAPI application is running at this base URL
BASE_URL = "http://localhost:8000/api/v1"  # The CLI server (geo-infer-health serve) mounts the API here

SURVEILLANCE_URL = f"{BASE_URL}/surveillance"


def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
    except json.JSONDecodeError:
        print(f"Response Text: {response.text}")
    print("---")


def submit_sample_disease_reports():
    print("Submitting sample disease reports...")
    reports_data = [
        {
            "report_id": "case001",
            "disease_code": "COVID-19",
            "location": {
                "latitude": 34.0522,
                "longitude": -118.2437,
                "crs": "EPSG:4326",
            },
            "report_date": datetime.now(UTC).isoformat(),
            "case_count": 5,
            "source": "Hospital A",
        },
        {
            "report_id": "case002",
            "disease_code": "COVID-19",
            "location": {
                "latitude": 34.0530,
                "longitude": -118.2440,
                "crs": "EPSG:4326",
            },
            "report_date": datetime.now(UTC).isoformat(),
            "case_count": 3,
            "source": "Clinic B",
        },
        {
            "report_id": "case003",
            "disease_code": "Flu",
            "location": {
                "latitude": 34.0500,
                "longitude": -118.2500,
                "crs": "EPSG:4326",
            },
            "report_date": datetime.now(UTC).isoformat(),
            "case_count": 10,
            "source": "Hospital A",
        },
        {
            "report_id": "case004",
            "disease_code": "COVID-19",
            "location": {
                "latitude": 34.0525,
                "longitude": -118.2430,
                "crs": "EPSG:4326",
            },
            "report_date": datetime.now(UTC).isoformat(),
            "case_count": 2,
            "source": "Community Testing",
        },
    ]
    for report in reports_data:
        response = requests.post(f"{SURVEILLANCE_URL}/reports/", json=report)
        print_response(response)


def get_disease_reports():
    print("Getting all disease reports...")
    response = requests.get(f"{SURVEILLANCE_URL}/reports/?limit=10")
    print_response(response)


def add_sample_population_data():
    print("Adding sample population data...")
    pop_data = [
        {
            "area_id": "district_1",
            "population_count": 50000,
            "age_distribution": {"0-18": 15000, "19-65": 30000, "65+": 5000},
        },
        {"area_id": "district_2", "population_count": 75000},
    ]
    for data in pop_data:
        response = requests.post(f"{SURVEILLANCE_URL}/population_data/", json=data)
        print_response(response)


def identify_hotspots():
    print("Identifying hotspots...")
    # Parameters can be adjusted
    params = {
        "threshold_case_count": 3,
        "scan_radius_km": 0.5,
        # "min_density_cases_per_sq_km": 5.0
    }
    response = requests.post(f"{SURVEILLANCE_URL}/hotspots/identify", params=params)
    print_response(response)


def get_local_incidence():
    print("Calculating local incidence rate...")
    params = {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "radius_km": 1.0,
        "time_window_days": 30,
    }
    response = requests.post(f"{SURVEILLANCE_URL}/incidence_rate/local", params=params)
    print_response(response)


if __name__ == "__main__":
    # Run example functions
    submit_sample_disease_reports()
    get_disease_reports()
    add_sample_population_data()
    # Note: For identify_hotspots and get_local_incidence to work meaningfully,
    # the FastAPI app needs to have its in-memory DBs populated by the above calls first.
    # In a real scenario, data persistence would handle this.
    identify_hotspots()
    get_local_incidence()

    print("\nDisease Surveillance API examples completed.")
    print(f"Ensure your FastAPI server is running and accessible at {BASE_URL}")
    print("And that the GEO-INFER-HEALTH API routers are correctly mounted.")
