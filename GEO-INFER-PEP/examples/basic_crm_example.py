"""
Basic CRM Data Processing Example

This example demonstrates how to:
1. Import CRM data from a CSV file
2. Clean and enrich customer data with engagement analysis
3. Generate customer segmentation and reports
4. Display customer insights and metrics

Prerequisites:
- A CSV file with CRM data (see sample format below)
- The geo_infer_pep module installed

Sample CSV format for crm_data.csv:
id,first_name,last_name,email,phone,company_name,title,address_street,address_city,address_state,address_country,created_at,updated_at,lead_source,status,tags,notes
1,John,Doe,john.doe@techcorp.com,555-0101,Tech Corp,CTO,123 Tech St,San Francisco,CA,USA,2023-01-15T10:30:00,2023-12-15T14:20:00,website,active,vip,enterprise,Initial contact at conference
2,Jane,Smith,jane.smith@startup.io,555-0102,Startup.io,CEO,456 Startup Ave,Austin,TX,USA,2023-03-20T09:15:00,2023-12-18T11:45:00,referral,lead,startup,Interested in premium features
3,Bob,Johnson,bob.johnson@enterprise.com,555-0103,Enterprise Ltd,VP Sales,789 Corp Blvd,New York,NY,USA,2022-11-10T16:20:00,2023-12-10T08:30:00,linkedin,customer,enterprise,Long-term client
"""

import os
import tempfile
import csv
from geo_infer_pep.methods import (
    import_crm_data_from_csv,
    generate_comprehensive_crm_dashboard,
    clear_all_data,
)


def create_sample_crm_data():
    """Create a sample CRM CSV file for demonstration."""
    sample_data = [
        [
            "id",
            "first_name",
            "last_name",
            "email",
            "phone",
            "company_name",
            "title",
            "address_street",
            "address_city",
            "address_state",
            "address_country",
            "created_at",
            "updated_at",
            "lead_source",
            "status",
            "tags",
            "notes",
        ],
        [
            "1",
            "John",
            "Doe",
            "john.doe@techcorp.com",
            "555-0101",
            "Tech Corp",
            "CTO",
            "123 Tech St",
            "San Francisco",
            "CA",
            "USA",
            "2023-01-15T10:30:00",
            "2023-12-15T14:20:00",
            "website",
            "active",
            "vip,enterprise",
            "Initial contact at conference",
        ],
        [
            "2",
            "Jane",
            "Smith",
            "jane.smith@startup.io",
            "555-0102",
            "Startup.io",
            "CEO",
            "456 Startup Ave",
            "Austin",
            "TX",
            "USA",
            "2023-03-20T09:15:00",
            "2023-12-18T11:45:00",
            "referral",
            "lead",
            "startup",
            "Interested in premium features",
        ],
        [
            "3",
            "Bob",
            "Johnson",
            "bob.johnson@enterprise.com",
            "555-0103",
            "Enterprise Ltd",
            "VP Sales",
            "789 Corp Blvd",
            "New York",
            "NY",
            "USA",
            "2022-11-10T16:20:00",
            "2023-12-10T08:30:00",
            "linkedin",
            "customer",
            "enterprise",
            "Long-term client",
        ],
        [
            "4",
            "Alice",
            "Brown",
            "alice.brown@smallbiz.com",
            "555-0104",
            "Small Business Inc",
            "Owner",
            "321 Main St",
            "Chicago",
            "IL",
            "USA",
            "2023-06-05T12:00:00",
            "2023-12-12T10:15:00",
            "cold_outreach",
            "lead",
            "",
            "Small business owner",
        ],
        [
            "5",
            "Charlie",
            "Wilson",
            "charlie.wilson@consulting.com",
            "555-0105",
            "Wilson Consulting",
            "Principal",
            "654 Consulting Ln",
            "Seattle",
            "WA",
            "USA",
            "2023-02-28T15:45:00",
            "2023-12-20T16:30:00",
            "referral",
            "active",
            "consulting",
            "Active consulting client",
        ],
        [
            "6",
            "Diana",
            "Garcia",
            "diana.garcia@retail.com",
            "555-0106",
            "Retail Chain",
            "Store Manager",
            "987 Retail Rd",
            "Miami",
            "FL",
            "USA",
            "2022-08-14T11:20:00",
            "2023-11-25T09:10:00",
            "website",
            "churned",
            "",
            "Customer churned last month",
        ],
    ]

    # Create temporary CSV file
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "sample_crm_data.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample_data)

    return csv_path


def main():
    """Main example function demonstrating CRM data processing."""
    print("🚀 GEO-INFER-PEP Basic CRM Example")
    print("=" * 50)

    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample CRM data...")
    csv_path = create_sample_crm_data()
    print(f"Created sample data at: {csv_path}")

    # Step 2: Import and process CRM data
    print("\n📥 Step 2: Importing and processing CRM data...")
    try:
        customers = import_crm_data_from_csv(csv_path)
        print(
            f"✅ Successfully imported and processed {len(customers)} customer records"
        )

        # Display some processed data with enrichment
        print("\n👥 Sample processed customers with enrichment:")
        for cust in customers[:3]:  # Show first 3 customers
            print(f"  - {cust.first_name} {cust.last_name} ({cust.customer_id})")
            print(f"    Company: {cust.company}, Title: {cust.job_title}")
            print(
                f"    Status: {cust.status}, Tags: {', '.join(cust.tags) if cust.tags else 'None'}"
            )

            # Show enrichment data from interaction history
            if cust.interaction_history:
                recent_interactions = [
                    i
                    for i in cust.interaction_history
                    if i.channel == "engagement_analysis"
                ]
                if recent_interactions:
                    print(f"    Engagement: {recent_interactions[0].summary}")

                segment_interactions = [
                    i
                    for i in cust.interaction_history
                    if i.channel == "segmentation_analysis"
                ]
                if segment_interactions:
                    print(f"    Segment: {segment_interactions[0].summary}")

            print()

    except Exception as e:
        print(f"❌ Error importing CRM data: {str(e)}")
        return

    # Step 3: Generate CRM dashboard
    print("\n📈 Step 3: Generating CRM dashboard...")
    try:
        dashboard = generate_comprehensive_crm_dashboard()

        if "message" in dashboard and "No customer data" in dashboard["message"]:
            print("❌ No data available for dashboard generation")
            return

        print("✅ CRM Dashboard generated successfully!")
        print("\n📊 Key Metrics:")
        print(f"  - Total Customers: {dashboard.get('total_customers', 0)}")
        print(f"  - Active Customers: {dashboard.get('active_customers', 0)}")

        # Display status breakdown
        status_breakdown = dashboard.get("status_breakdown", {})
        if status_breakdown:
            print("\n📈 Customer Status Breakdown:")
            for status, count in status_breakdown.items():
                print(f"  - {status.title()}: {count} customers")

        # Display conversion report insights
        conversion_report = dashboard.get("conversion_report", {})
        if conversion_report:
            print("\n🎯 Conversion Insights:")
            print(
                f"  - Total Identified Leads: {conversion_report.get('total_identified_leads', 0)}"
            )
            print(
                f"  - Total Converted Customers: {conversion_report.get('total_converted_customers', 0)}"
            )
            conversion_rate = conversion_report.get(
                "lead_to_customer_conversion_rate", 0
            )
            print(f"  - Conversion Rate: {conversion_rate:.1f}%")

        print(f"\n📅 Data Freshness: {dashboard.get('data_freshness', 'Unknown')}")

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
    print("  - Modify the sample data to test different customer scenarios")
    print("  - Import your own CRM data using import_crm_data_from_csv()")
    print("  - Explore other methods like generate_comprehensive_talent_dashboard()")
    print("  - Check the documentation for advanced customer segmentation features")


if __name__ == "__main__":
    main()
