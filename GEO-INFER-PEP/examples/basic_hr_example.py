"""
Basic HR Data Processing Example

This example demonstrates how to:
1. Import HR data from a CSV file
2. Clean and enrich the data
3. Generate reports and dashboards
4. Display key metrics

Prerequisites:
- A CSV file with HR data (see sample format below)
- The geo_infer_pep module installed

Sample CSV format for hr_data.csv:
employee_id,first_name,last_name,email,hire_date,status,job_title,department,gender
emp001,John,Doe,john.doe@company.com,2022-01-15,active,Software Engineer,Technology,male
emp002,Jane,Smith,jane.smith@company.com,2021-03-20,active,Product Manager,Product,female
emp003,Bob,Johnson,bob.johnson@company.com,2023-06-10,active,Data Analyst,Analytics,male
emp004,Alice,Brown,alice.brown@company.com,2020-11-05,terminated,Senior Developer,Technology,female
"""

import os
import tempfile
import csv
from geo_infer_pep.methods import (
    import_hr_data_from_csv,
    generate_comprehensive_hr_dashboard,
    clear_all_data,
)


def create_sample_hr_data():
    """Create a sample HR CSV file for demonstration."""
    sample_data = [
        [
            "employee_id",
            "first_name",
            "last_name",
            "email",
            "hire_date",
            "status",
            "job_title",
            "department",
            "gender",
        ],
        [
            "emp001",
            "John",
            "Doe",
            "john.doe@company.com",
            "2022-01-15",
            "active",
            "Software Engineer",
            "Technology",
            "male",
        ],
        [
            "emp002",
            "Jane",
            "Smith",
            "jane.smith@company.com",
            "2021-03-20",
            "active",
            "Product Manager",
            "Product",
            "female",
        ],
        [
            "emp003",
            "Bob",
            "Johnson",
            "bob.johnson@company.com",
            "2023-06-10",
            "active",
            "Data Analyst",
            "Analytics",
            "male",
        ],
        [
            "emp004",
            "Alice",
            "Brown",
            "alice.brown@company.com",
            "2020-11-05",
            "terminated",
            "Senior Developer",
            "Technology",
            "female",
        ],
        [
            "emp005",
            "Carol",
            "Davis",
            "carol.davis@company.com",
            "2022-08-12",
            "active",
            "Marketing Manager",
            "Marketing",
            "female",
        ],
        [
            "emp006",
            "David",
            "Wilson",
            "david.wilson@company.com",
            "2021-11-18",
            "active",
            "Sales Representative",
            "Sales",
            "male",
        ],
        [
            "emp007",
            "Eva",
            "Garcia",
            "eva.garcia@company.com",
            "2023-02-28",
            "active",
            "HR Specialist",
            "Human Resources",
            "female",
        ],
        [
            "emp008",
            "Frank",
            "Miller",
            "frank.miller@company.com",
            "2020-09-14",
            "on_leave",
            "Operations Manager",
            "Operations",
            "male",
        ],
    ]

    # Create temporary CSV file
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "sample_hr_data.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample_data)

    return csv_path


def main():
    """Main example function demonstrating HR data processing."""
    print("🚀 GEO-INFER-PEP Basic HR Example")
    print("=" * 50)

    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample HR data...")
    csv_path = create_sample_hr_data()
    print(f"Created sample data at: {csv_path}")

    # Step 2: Import and process HR data
    print("\n📥 Step 2: Importing and processing HR data...")
    try:
        employees = import_hr_data_from_csv(csv_path)
        print(
            f"✅ Successfully imported and processed {len(employees)} employee records"
        )

        # Display some processed data
        print("\n👥 Sample processed employees:")
        for emp in employees[:3]:  # Show first 3 employees
            print(f"  - {emp.first_name} {emp.last_name} ({emp.employee_id})")
            print(f"    Department: {emp.department}, Title: {emp.job_title}")
            print(f"    Status: {emp.employment_status.value}")
            if "tenure_years" in emp.custom_fields:
                print(f"    Tenure: {emp.custom_fields['tenure_years']:.1f} years")
            print()

    except Exception as e:
        print(f"❌ Error importing HR data: {str(e)}")
        return

    # Step 3: Generate HR dashboard
    print("\n📈 Step 3: Generating HR dashboard...")
    try:
        dashboard = generate_comprehensive_hr_dashboard()

        if "message" in dashboard and "No employee data" in dashboard["message"]:
            print("❌ No data available for dashboard generation")
            return

        print("✅ HR Dashboard generated successfully!")
        print("\n📊 Key Metrics:")
        print(f"  - Total Employees: {dashboard.get('total_employees', 0)}")
        print(f"  - Active Employees: {dashboard.get('active_employees', 0)}")

        # Display department breakdown
        dept_breakdown = dashboard.get("headcount_by_department", {})
        if dept_breakdown:
            print("\n🏢 Department Breakdown:")
            for dept, count in dept_breakdown.items():
                print(f"  - {dept}: {count} employees")

        # Display diversity metrics
        diversity = dashboard.get("diversity_report", {})
        if diversity:
            print("\n🌈 Diversity Metrics:")
            gender_diversity = diversity.get("diversity_by_gender", {})
            if "counts" in gender_diversity:
                for gender, count in gender_diversity["counts"].items():
                    percentage = gender_diversity["percentages"].get(gender, 0)
                    print(f"  - {gender}: {count} ({percentage:.1f}%)")

    except Exception as e:
        print(f"❌ Error generating dashboard: {str(e)}")

    # Step 4: Clean up
    print("\n🧹 Step 4: Cleaning up...")
    try:
        clear_all_data()
        os.remove(csv_path)
        print("✅ Cleanup completed successfully!")
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {str(e)}")

    print("\n🎉 Example completed successfully!")
    print("\n💡 Next steps you can try:")
    print("  - Modify the sample data to test different scenarios")
    print("  - Import your own HR data using import_hr_data_from_csv()")
    print("  - Explore other methods like generate_comprehensive_crm_dashboard()")
    print("  - Check the documentation for more advanced features")


if __name__ == "__main__":
    main()
