"""Tests for policy validation and regulatory framework operations."""

import datetime

from geo_infer_norms.models.regulation import Regulation, RegulatoryFramework


class TestRegulationLifecycle:
    def test_regulation_equality(self):
        reg1 = Regulation(
            id="reg-1",
            name="Test",
            description="Test",
            regulation_type="environmental",
            issuing_authority="EPA",
            effective_date=datetime.date(2020, 1, 1),
        )
        reg2 = Regulation(
            id="reg-1",
            name="Different Name",
            description="Different",
            regulation_type="environmental",
            issuing_authority="EPA",
            effective_date=datetime.date(2020, 1, 1),
        )
        assert reg1 == reg2  # Same ID

    def test_regulation_hash(self):
        reg1 = Regulation(
            id="reg-1",
            name="Test",
            description="Test",
            regulation_type="environmental",
            issuing_authority="EPA",
            effective_date=datetime.date(2020, 1, 1),
        )
        reg_set = {reg1}
        assert reg1 in reg_set

    def test_regulation_not_yet_effective(self):
        future_date = datetime.date.today() + datetime.timedelta(days=365)
        reg = Regulation.create(
            name="Future Reg",
            description="Not yet effective",
            regulation_type="safety",
            issuing_authority="OSHA",
            effective_date=future_date,
        )
        assert reg.is_active() is False

    def test_regulation_attribute_management(self):
        reg = Regulation.create(
            name="Test",
            description="Test",
            regulation_type="environmental",
            issuing_authority="EPA",
            effective_date=datetime.date(2020, 1, 1),
        )
        reg.update_attribute("max_penalty", 50000)
        assert reg.attributes["max_penalty"] == 50000


class TestRegulatoryFrameworkOperations:
    def test_framework_active_no_dates(self):
        fw = RegulatoryFramework.create(
            name="Test",
            description="Test",
            domain="env",
            issuing_authority="EPA",
        )
        assert fw.is_active() is True

    def test_framework_expired(self):
        fw = RegulatoryFramework.create(
            name="Test",
            description="Test",
            domain="env",
            issuing_authority="EPA",
            effective_date=datetime.date(2019, 1, 1),
            expiration_date=datetime.date(2020, 12, 31),
        )
        assert fw.is_active() is False

    def test_framework_version_update(self):
        fw = RegulatoryFramework.create(
            name="Test",
            description="Test",
            domain="env",
            issuing_authority="EPA",
        )
        fw.update_version("2.0")
        assert fw.version == "2.0"

    def test_framework_regulation_management(self):
        fw = RegulatoryFramework.create(
            name="Test",
            description="Test",
            domain="env",
            issuing_authority="EPA",
        )
        fw.add_regulation("reg-1")
        fw.add_regulation("reg-2")
        fw.add_regulation("reg-1")  # Duplicate
        assert len(fw.regulations) == 2

    def test_framework_attribute_management(self):
        fw = RegulatoryFramework.create(
            name="Test",
            description="Test",
            domain="env",
            issuing_authority="EPA",
        )
        fw.update_attribute("scope", "national")
        assert fw.attributes["scope"] == "national"
