import requests
import json

# Assume the FastAPI application is running at this base URL
BASE_URL = "http://localhost:8000/api/v1"  # The CLI server (geo-infer-health serve) mounts the API here
ACCESSIBILITY_URL = f"{BASE_URL}/accessibility"


def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
    except json.JSONDecodeError:
        print(f"Response Text: {response.text}")
    print("---")


def add_sample_health_facilities():
    print("Adding sample health facilities...")
    facilities_data = [
        {
            "facility_id": "hosp001",
            "name": "General Hospital Downtown",
            "facility_type": "Hospital",
            "location": {"latitude": 34.0500, "longitude": -118.2400},
            "capacity": 500,
            "services_offered": ["Emergency", "Surgery", "Pediatrics"],
        },
        {
            "facility_id": "clinic001",
            "name": "Westside Health Clinic",
            "facility_type": "Clinic",
            "location": {"latitude": 34.0600, "longitude": -118.4000},
            "services_offered": ["General Checkup", "Vaccinations"],
        },
        {
            "facility_id": "hosp002",
            "name": "North Valley Medical Center",
            "facility_type": "Hospital",
            "location": {"latitude": 34.2500, "longitude": -118.4500},
            "capacity": 250,
            "services_offered": ["Emergency", "Cardiology"],
        },
    ]
    for facility in facilities_data:
        response = requests.post(f"{ACCESSIBILITY_URL}/facilities/", json=facility)
        print_response(response)


def get_health_facilities():
    print("Getting all health facilities...")
    response = requests.get(f"{ACCESSIBILITY_URL}/facilities/?limit=5")
    print_response(response)


def add_accessibility_population_sample():
    print("Adding sample population data for accessibility context...")
    pop_data = {
        "area_id": "city_main",
        "population_count": 1200000,
        "age_distribution": {"0-18": 300000, "19-65": 700000, "65+": 200000},
    }
    response = requests.post(f"{ACCESSIBILITY_URL}/population_data/", json=pop_data)
    print_response(response)


def find_facilities_nearby_example():
    print("Finding nearby facilities...")
    params = {
        "latitude": 34.0550,
        "longitude": -118.2450,
        "radius_km": 5.0,
        "facility_type": "Hospital",
    }
    response = requests.post(f"{ACCESSIBILITY_URL}/facilities/nearby", params=params)
    print_response(response)


def get_nearest_facility_example():
    print("Getting nearest facility...")
    # FastAPI parses repeated query parameters into the required_services list.
    response = requests.post(
        f"{ACCESSIBILITY_URL}/facilities/nearest",
        params={
            "latitude": 34.0580,
            "longitude": -118.3000,
            "required_services": [
                "Emergency",
                "Surgery",
            ],  # FastAPI Query can often handle this
        },
    )
    print_response(response)


def get_facility_ratio_example():
    print("Getting facility to population ratio...")
    area_id = "city_main"
    params = {"facility_type": "Hospital"}
    response = requests.get(
        f"{ACCESSIBILITY_URL}/facility_population_ratio/{area_id}", params=params
    )
    print_response(response)


if __name__ == "__main__":
    add_sample_health_facilities()
    get_health_facilities()
    add_accessibility_population_sample()
    find_facilities_nearby_example()
    get_nearest_facility_example()
    get_facility_ratio_example()

    print("\nHealthcare Accessibility API examples completed.")
    print(f"Ensure your FastAPI server is running and accessible at {BASE_URL}")
    print("And that the GEO-INFER-HEALTH API routers are correctly mounted.")
