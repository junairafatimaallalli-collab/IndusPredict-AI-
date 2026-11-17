"""
Tests for IndusPredict AI Machine Management Service.
"""

import pytest
from induspredict.core.config import CRITICALITY_CRITICAL, CRITICALITY_HIGH
from induspredict.services.auth_service import AuthService
from induspredict.services.machine_service import MachineService


def test_get_all_machines(test_engine):
    """Verify machine listing and filtering."""
    machines = MachineService.get_all_machines()
    assert len(machines) >= 5

    # Filter by type
    compressors = MachineService.get_all_machines(type_filter="Screw Compressor")
    assert len(compressors) >= 1
    assert all(m["type"] == "Screw Compressor" for m in compressors)

    # Search query
    results = MachineService.get_all_machines(search_query="Siemens")
    assert len(results) >= 1
    assert any("Siemens" in m["name"] or "Siemens" in m["manufacturer"] for m in results)


def test_machine_crud_lifecycle(test_engine):
    """Test full Create, Read, Update, Delete lifecycle for an industrial asset."""
    admin_session = AuthService.authenticate("admin", "Admin@123")
    assert admin_session is not None

    # 1. Create Machine
    create_data = {
        "code": "TEST-PMP-99",
        "name": "Experimental High-Pressure Cryo Pump",
        "type": "Centrifugal Pump",
        "manufacturer": "CryoTech Systems",
        "model": "CT-500",
        "serial": "SN-998877",
        "location": "Cryogenic Testing Bay",
        "criticality": CRITICALITY_CRITICAL,
        "operating_hours": 250.5,
        "description": "High pressure pump under prototype validation.",
    }

    res_create = MachineService.create_machine(create_data, user_session=admin_session)
    assert res_create["success"] is True
    m_id = res_create["machine_id"]

    # 2. Read Machine
    details = MachineService.get_machine_details(m_id)
    assert details is not None
    assert details["code"] == "TEST-PMP-99"
    assert details["operating_hours"] == 250.5
    assert len(details["sensors"]) == 4  # Default 4 sensors attached

    # 3. Update Machine
    update_data = {
        "name": "Updated Cryo Pump Name",
        "operating_hours": 320.0,
        "status": "warning",
    }
    res_update = MachineService.update_machine(m_id, update_data, user_session=admin_session)
    assert res_update["success"] is True

    updated_details = MachineService.get_machine_details(m_id)
    assert updated_details["name"] == "Updated Cryo Pump Name"
    assert updated_details["operating_hours"] == 320.0
    assert updated_details["status"] == "warning"

    # 4. Add Sub-Component
    comp_res = MachineService.add_component(
        m_id,
        {
            "name": "Cryogenic Ceramic Ball Bearing",
            "type": "Rolling Bearing",
            "part_number": "CR-7721",
            "lifespan": 15000.0,
            "hours": 320.0,
        },
    )
    assert comp_res["success"] is True

    details_with_comp = MachineService.get_machine_details(m_id)
    assert len(details_with_comp["components"]) == 1

    # 5. Delete Machine
    res_delete = MachineService.delete_machine(m_id, user_session=admin_session)
    assert res_delete["success"] is True
    assert res_delete["deleted_code"] == "TEST-PMP-99"

    # Verify deleted
    assert MachineService.get_machine_details(m_id) is None
    AuthService.logout()


def test_machine_validation(test_engine):
    """Verify input validation rules when registering machines."""
    admin_session = AuthService.authenticate("admin", "Admin@123")

    # Missing code
    res1 = MachineService.create_machine({"code": "", "name": "Valid Name", "type": "Pump", "manufacturer": "ACME", "location": "Bay 1"})
    assert res1["success"] is False

    # Negative operating hours
    res2 = MachineService.create_machine({
        "code": "NEG-01",
        "name": "Negative Machine",
        "type": "Motor",
        "manufacturer": "ACME",
        "location": "Bay 1",
        "operating_hours": -10.0,
    })
    assert res2["success"] is False
    assert "non-negative" in res2["error"]

    AuthService.logout()
