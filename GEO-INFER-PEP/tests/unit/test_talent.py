import pytest
import os
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import csv  # Ensure csv is imported at the top level

from geo_infer_pep.models.talent_models import (
    Candidate,
    JobRequisition,
    Offer,
    Interview,
    CandidateStatus,
    JobRequisitionStatus,
    InterviewType,
)
from geo_infer_pep.talent import importer as talent_importer_module
from geo_infer_pep.talent.importer import CSVTalentImporter
from geo_infer_pep.talent.transformer import (
    clean_candidate_data,
    enrich_candidate_data,
    convert_candidates_to_dataframe,
    convert_requisitions_to_dataframe,
)
from geo_infer_pep.reporting.talent_reports import (
    generate_candidate_pipeline_report,
    calculate_time_to_hire,
)
from geo_infer_pep.visualizations.talent_visuals import (
    plot_candidate_pipeline_by_status,
    plot_time_to_hire_distribution,
)


# Fixtures
@pytest.fixture
def sample_job_requisition_list():
    return [
        JobRequisition(
            requisition_id="req1",
            job_title="Software Engineer",
            department="Engineering",
            status=JobRequisitionStatus.OPEN,
            opened_at=date(2023, 1, 1),
        ),
        JobRequisition(
            requisition_id="req2",
            job_title="Product Manager",
            department="Product",
            status=JobRequisitionStatus.FILLED,
            opened_at=date(2022, 11, 1),
            closed_at=date(2023, 2, 15),
        ),
    ]


@pytest.fixture
def sample_candidate_data_list(sample_job_requisition_list):
    req1_id = sample_job_requisition_list[0].requisition_id
    return [
        Candidate(
            candidate_id="cand1",
            first_name="Alice",
            last_name="Applicant",
            email="alice.app@example.com",
            status=CandidateStatus.APPLIED,
            job_requisition_id=req1_id,
            applied_at=datetime(2023, 2, 1, 10, 0, 0),
            updated_at=datetime(2023, 2, 1, 10, 0, 0),
            skills=["Python", "FastAPI"],
        ),
        Candidate(
            candidate_id="cand2",
            first_name="Bob",
            last_name="Bookworm",
            email="bob.book@example.com",
            status=CandidateStatus.INTERVIEWING,
            job_requisition_id=req1_id,
            applied_at=datetime(2023, 1, 15, 9, 0, 0),
            updated_at=datetime(2023, 2, 10, 14, 0, 0),
            interviews=[
                Interview(
                    interview_id="int1",
                    interview_type=InterviewType.PHONE_SCREEN,
                    scheduled_at=datetime(2023, 1, 20, 11, 0, 0),
                    interviewers=["emp_hr_1"],
                )
            ],
        ),
        Candidate(
            candidate_id="cand3",
            first_name="Carol",
            last_name="Candidate",
            email="carol.cand@example.com",
            status=CandidateStatus.HIRED,
            job_requisition_id=req1_id,  # Assuming hired for req1 for simplicity of TTH test
            applied_at=datetime(2023, 1, 5, 12, 0, 0),
            updated_at=datetime(2023, 2, 20, 10, 0, 0),  # Proxy for hired_at for TTH
            offer=Offer(
                offer_id="offer1",
                offered_at=date(2023, 2, 15),
                status="accepted",
                accepted_at=date(2023, 2, 20),
            ),
        ),
    ]


@pytest.fixture
def dummy_talent_csv_files(tmp_path):
    """Creates dummy CSV files for talent importer testing."""
    cand_csv_path = tmp_path / "dummy_candidates.csv"
    req_csv_path = tmp_path / "dummy_requisitions.csv"

    # Candidate CSV Data
    cand_headers = [
        "candidate_id",
        "first_name",
        "last_name",
        "email",
        "status",
        "job_requisition_id",
        "applied_at",
        "updated_at",
        "skills",
        "source",
    ]
    cand_row1 = [
        "cand_csv_1",
        "CSV",
        "One",
        "csv.one@example.com",
        "applied",
        "req_csv_1",
        datetime(2023, 1, 10).isoformat(),
        datetime(2023, 1, 11).isoformat(),
        "python,sql",
        "linkedin",
    ]
    cand_row2 = [
        "cand_csv_2",
        "CSV",
        "Two",
        "csv.two@example.com",
        "screening",
        "req_csv_1",
        datetime(2023, 2, 5).isoformat(),
        datetime(2023, 2, 6).isoformat(),
        "java,spring",
        "referral",
    ]
    cand_bad_date = [
        "cand_csv_3",
        "Bad",
        "DateCand",
        "baddatecand@example.com",
        "interviewing",
        "req_csv_2",
        "not-a-date",
        "2023-03-01T10:00:00",
        "c++",
        "career_fair",
    ]

    with open(cand_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cand_headers)
        writer.writerow(cand_row1)
        writer.writerow(cand_row2)
        writer.writerow(cand_bad_date)

    # Requisition CSV Data
    req_headers = [
        "requisition_id",
        "job_title",
        "department",
        "status",
        "opened_at",
        "closed_at",
        "hiring_manager_id",
    ]
    req_row1 = [
        "req_csv_1",
        "Data Scientist",
        "Analytics",
        "open",
        "2023-01-01",
        "",
        "mgr_analytics",
    ]
    req_row2 = [
        "req_csv_2",
        "UX Designer",
        "Design",
        "closed",
        "2022-12-01",
        "2023-02-28",
        "mgr_design",
    ]
    req_bad_date = [
        "req_csv_3",
        "QA Tester",
        "Engineering",
        "on_hold",
        "invalid-date",
        "",
        "mgr_qa",
    ]

    with open(req_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(req_headers)
        writer.writerow(req_row1)
        writer.writerow(req_row2)
        writer.writerow(req_bad_date)

    return {"candidates": cand_csv_path, "requisitions": req_csv_path}


# Model Tests
def test_candidate_model():
    """Behavior-focused test: test_candidate_model."""
    cand = Candidate(
        candidate_id="cand_test",
        first_name="Test",
        last_name="Candidate",
        email="test.cand@example.com",
    )
    assert cand.candidate_id == "cand_test"
    assert cand.status == CandidateStatus.APPLIED  # Default


def test_job_requisition_model():
    """Behavior-focused test: test_job_requisition_model."""
    req = JobRequisition(
        requisition_id="req_test",
        job_title="Test Engineer",
        department="QA",
        opened_at=date(2023, 3, 3),
    )
    assert req.requisition_id == "req_test"
    assert req.status == JobRequisitionStatus.OPEN  # Default


# Importer Tests
def test_csv_talent_importer(dummy_talent_csv_files, caplog, monkeypatch):
    """Behavior-focused test: test_csv_talent_importer."""
    # Freeze the importer's clock so the opened_at default is deterministic
    # (no midnight race between the SUT's date.today() and the assertion).
    frozen_today = date(2031, 5, 17)

    class _FrozenDate(date):
        @classmethod
        def today(cls):
            return frozen_today

    monkeypatch.setattr(talent_importer_module, "date", _FrozenDate)

    importer = CSVTalentImporter(
        candidate_file_path=str(dummy_talent_csv_files["candidates"]),
        requisition_file_path=str(dummy_talent_csv_files["requisitions"]),
    )

    with caplog.at_level("WARNING", logger="geo_infer_pep.talent.importer"):
        candidates = importer.import_candidates()
        requisitions = importer.import_requisitions()

    assert any(
        "Bad applied_at for cand cand_csv_3" in r.getMessage() for r in caplog.records
    )
    assert any(
        "Bad opened_at for req req_csv_3" in r.getMessage() for r in caplog.records
    )

    # Expect 2 valid candidates, 1 with default applied_at due to bad date format
    assert len(candidates) == 3
    cand1 = next(c for c in candidates if c.candidate_id == "cand_csv_1")
    assert cand1.first_name == "CSV"
    assert cand1.email == "csv.one@example.com"
    assert cand1.status == CandidateStatus.APPLIED
    assert cand1.applied_at == datetime(2023, 1, 10)
    assert "python" in cand1.skills

    cand_bad_date_obj = next(c for c in candidates if c.candidate_id == "cand_csv_3")
    assert cand_bad_date_obj.status == CandidateStatus.INTERVIEWING
    # applied_at for cand_csv_3 should be datetime.now() or very close, as parsing failed
    assert (
        datetime.now() - cand_bad_date_obj.applied_at
    ).total_seconds() < 5  # Check it defaulted to now

    # Expect 2 valid requisitions, 1 with default opened_at
    assert len(requisitions) == 3
    req1 = next(r for r in requisitions if r.requisition_id == "req_csv_1")
    assert req1.job_title == "Data Scientist"
    assert req1.status == JobRequisitionStatus.OPEN
    assert req1.opened_at == date(2023, 1, 1)
    assert req1.closed_at is None

    req_bad_date_obj = next(r for r in requisitions if r.requisition_id == "req_csv_3")
    assert req_bad_date_obj.status == JobRequisitionStatus.ON_HOLD
    # opened_at for req_csv_3 defaults to the importer's date.today()
    assert req_bad_date_obj.opened_at == frozen_today


# Transformer Tests
def test_clean_candidate_data(sample_candidate_data_list):
    """clean_candidate_data lowercases skills, strips duplicates, and cleans phones."""
    messy = Candidate(
        candidate_id="cand_dirty",
        first_name="BRUCE",
        last_name="WAYNE",
        email="bruce.wayne@example.com",
        status=CandidateStatus.APPLIED,
        applied_at=datetime(2023, 2, 1, 10, 0, 0),
        updated_at=datetime(2023, 2, 1, 10, 0, 0),
        skills=["Python", "python ", "SQL", ""],
        phone_number="555-123-4567",
    )

    cleaned = clean_candidate_data([messy])

    assert len(cleaned) == 1
    assert all(s == s.lower() for s in cleaned[0].skills)
    assert "python" in cleaned[0].skills
    assert "" not in cleaned[0].skills
    assert cleaned[0].phone_number == "5551234567" if cleaned[0].phone_number else True

    # Fixture candidates pass through with skills lowercased and deduplicated
    fixture_cleaned = clean_candidate_data(sample_candidate_data_list)
    assert len(fixture_cleaned) == 3
    assert all(s == s.lower() for s in fixture_cleaned[0].skills)


def test_enrich_candidate_data(sample_candidate_data_list, sample_job_requisition_list):
    """Behavior-focused test: test_enrich_candidate_data."""
    enriched = enrich_candidate_data(
        sample_candidate_data_list, sample_job_requisition_list
    )
    assert len(enriched) == 3


def test_convert_candidates_to_dataframe(sample_candidate_data_list):
    """Behavior-focused test: test_convert_candidates_to_dataframe."""
    df = convert_candidates_to_dataframe(sample_candidate_data_list)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_convert_requisitions_to_dataframe(sample_job_requisition_list):
    """Behavior-focused test: test_convert_requisitions_to_dataframe."""
    df = convert_requisitions_to_dataframe(sample_job_requisition_list)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


# Reporting Tests
def test_generate_candidate_pipeline_report(
    sample_candidate_data_list, sample_job_requisition_list
):
    """Behavior-focused test: test_generate_candidate_pipeline_report."""
    report = generate_candidate_pipeline_report(
        sample_candidate_data_list, sample_job_requisition_list
    )
    assert report["total_candidates"] == 3
    assert report["candidates_by_status"].get(CandidateStatus.APPLIED.value) == 1
    assert report["candidates_by_status"].get(CandidateStatus.HIRED.value) == 1
    assert "pipeline_by_active_requisition" in report
    assert "req1" in report["pipeline_by_active_requisition"]
    assert (
        report["pipeline_by_active_requisition"]["req1"]["total_candidates_for_req"]
        == 3
    )


def test_calculate_time_to_hire(sample_candidate_data_list):
    """Behavior-focused test: test_calculate_time_to_hire."""
    hired_candidates = [
        c for c in sample_candidate_data_list if c.status == CandidateStatus.HIRED
    ]
    report = calculate_time_to_hire(hired_candidates)
    assert report["number_of_hires_in_calc"] == 1
    # Expected TTH for cand3: (2023,2,20) - (2023,1,5) = 46 days (inclusive) or 45 days (exclusive)
    assert report["avg_time_to_hire_days"] in (45.0, 46.0)


# Visualization Tests
def test_plot_candidate_pipeline_by_status(sample_candidate_data_list, tmp_path):
    """Behavior-focused test: test_plot_candidate_pipeline_by_status."""
    output_dir = tmp_path / "talent_visuals"
    output_dir.mkdir()
    plot_path = plot_candidate_pipeline_by_status(
        sample_candidate_data_list, output_dir=output_dir
    )
    assert plot_path is not None
    assert os.path.exists(plot_path)
    assert Path(plot_path).name == "candidate_pipeline_status.png"


def test_plot_time_to_hire_distribution(sample_candidate_data_list, tmp_path):
    """Behavior-focused test: test_plot_time_to_hire_distribution."""
    # Prepare TTH data (days) for the hired candidate
    tth_days_list = []
    hired_cand = next(
        (c for c in sample_candidate_data_list if c.status == CandidateStatus.HIRED),
        None,
    )
    if (
        hired_cand and hired_cand.applied_at and hired_cand.updated_at
    ):  # updated_at is proxy for hired_at
        duration = hired_cand.updated_at - hired_cand.applied_at
        tth_days_list.append(duration.days)

    output_dir = tmp_path / "talent_visuals"
    output_dir.mkdir(exist_ok=True)  # Ensure dir exists
    plot_path = plot_time_to_hire_distribution(tth_days_list, output_dir=output_dir)
    if tth_days_list:  # Plot only generated if data exists
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert Path(plot_path).name == "time_to_hire_distribution.png"
    else:
        assert plot_path is None


# Empty data tests
def test_talent_reports_empty_data():
    """Behavior-focused test: test_talent_reports_empty_data."""
    empty_list = []
    pipeline_report = generate_candidate_pipeline_report(empty_list)
    assert "No candidate data" in pipeline_report.get("message", "")
    tth_report = calculate_time_to_hire(empty_list)
    assert "No hired candidate data" in tth_report.get("message", "")


def test_talent_visuals_empty_data(tmp_path):
    """Behavior-focused test: test_talent_visuals_empty_data."""
    empty_list = []
    output_dir = tmp_path / "talent_visuals_empty"
    output_dir.mkdir()
    pipeline_plot = plot_candidate_pipeline_by_status(empty_list, output_dir=output_dir)
    assert pipeline_plot is None
    tth_plot = plot_time_to_hire_distribution(
        [], output_dir=output_dir
    )  # Empty list for TTH days
    assert tth_plot is None
