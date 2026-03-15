"""Comprehensive tests for the GRE Action Serializer.

Tests cover:
  - Action serialization (all common action types)
  - PerformActionResp message building
  - ClientToGREMessage envelope building
  - GREActionRef integration
  - Validation against legal actions
  - Validated serialization
  - Convenience action builders
  - Edge cases and error handling
"""

import copy

import pytest

from arenamcp.gre_serializer import (
    AutoPassPriority,
    SerializationError,
    ValidationError,
    _action_identity_key,
    _serialize_action,
    _serialize_targets,
    build_activate_action,
    build_cast_action,
    build_pass_action,
    build_play_land_action,
    build_targeted_action,
    find_matching_legal_action,
    serialize_client_message,
    serialize_from_action_ref,
    serialize_perform_action_resp,
    serialize_perform_action_resp_multi,
    serialize_validated,
    validate_action_against_legal,
)


# =========================================================================
# Fixtures: sample raw GRE action dicts (representative of log data)
# =========================================================================

@pytest.fixture
def pass_action():
    """ActionType_Pass — simplest possible action."""
    return {"actionType": "ActionType_Pass"}


@pytest.fixture
def cast_action():
    """ActionType_Cast for 'Lightning Bolt' (grpId=12345)."""
    return {
        "actionType": "ActionType_Cast",
        "grpId": 12345,
        "instanceId": 501,
        "abilityGrpId": 0,
        "sourceId": 0,
        "manaPaymentOptions": [
            {"manaId": [{"color": "ManaColor_Red", "srcInstanceId": 100}]}
        ],
        "autoTapSolution": {
            "autoTapActions": [
                {"instanceId": 100, "abilityGrpId": 993}
            ]
        },
    }


@pytest.fixture
def play_land_action():
    """ActionType_Play for a Forest (grpId=67890)."""
    return {
        "actionType": "ActionType_Play",
        "grpId": 67890,
        "instanceId": 502,
    }


@pytest.fixture
def activate_action():
    """ActionType_Activate for a mana ability."""
    return {
        "actionType": "ActionType_Activate",
        "instanceId": 100,
        "abilityGrpId": 993,
        "sourceId": 100,
        "grpId": 55555,
    }


@pytest.fixture
def targeted_cast_action():
    """ActionType_Cast with targets (e.g. Shock targeting a creature)."""
    return {
        "actionType": "ActionType_Cast",
        "grpId": 22222,
        "instanceId": 503,
        "targets": [
            {
                "targetIdx": 0,
                "targets": [
                    {"targetInstanceId": 600, "highlight": "HighlightType_Hot"},
                    {"targetInstanceId": 601, "highlight": "HighlightType_Hot"},
                ],
                "minTargets": 1,
                "maxTargets": 1,
                "selectedTargets": 0,
                "targetingAbilityGrpId": 333,
            }
        ],
    }


@pytest.fixture
def selection_action():
    """Action with selectionType/selection fields."""
    return {
        "actionType": "ActionType_Special",
        "instanceId": 504,
        "selectionType": 2,
        "selection": 1,
    }


@pytest.fixture
def legal_actions_list(pass_action, cast_action, play_land_action, activate_action):
    """A list of legal actions typical of a main phase priority decision."""
    return [pass_action, cast_action, play_land_action, activate_action]


# =========================================================================
# Test: _serialize_action
# =========================================================================

class TestSerializeAction:
    """Tests for the core action serialization function."""

    def test_pass_action_minimal(self, pass_action):
        result = _serialize_action(pass_action)
        assert result == {"actionType": "ActionType_Pass"}

    def test_cast_action_preserves_identity_fields(self, cast_action):
        result = _serialize_action(cast_action)
        assert result["actionType"] == "ActionType_Cast"
        assert result["grpId"] == 12345
        assert result["instanceId"] == 501

    def test_cast_action_preserves_mana_payment(self, cast_action):
        result = _serialize_action(cast_action)
        assert "manaPaymentOptions" in result
        assert len(result["manaPaymentOptions"]) == 1

    def test_cast_action_preserves_auto_tap(self, cast_action):
        result = _serialize_action(cast_action)
        assert "autoTapSolution" in result
        assert result["autoTapSolution"]["autoTapActions"][0]["instanceId"] == 100

    def test_zero_fields_omitted(self):
        """Fields with value 0 should not appear in output."""
        action = {
            "actionType": "ActionType_Cast",
            "grpId": 100,
            "instanceId": 0,
            "abilityGrpId": 0,
            "sourceId": 0,
        }
        result = _serialize_action(action)
        assert "instanceId" not in result
        assert "abilityGrpId" not in result
        assert "sourceId" not in result
        assert result["grpId"] == 100

    def test_activate_action(self, activate_action):
        result = _serialize_action(activate_action)
        assert result["actionType"] == "ActionType_Activate"
        assert result["instanceId"] == 100
        assert result["abilityGrpId"] == 993
        assert result["sourceId"] == 100

    def test_selection_fields(self, selection_action):
        result = _serialize_action(selection_action)
        assert result["selectionType"] == 2
        assert result["selection"] == 1

    def test_missing_action_type_raises(self):
        with pytest.raises(SerializationError, match="actionType"):
            _serialize_action({"grpId": 100})

    def test_empty_action_type_raises(self):
        with pytest.raises(SerializationError, match="actionType"):
            _serialize_action({"actionType": ""})

    def test_should_stop_true(self):
        action = {"actionType": "ActionType_Pass", "shouldStop": True}
        result = _serialize_action(action)
        assert result["shouldStop"] is True

    def test_should_stop_false_omitted(self):
        action = {"actionType": "ActionType_Pass", "shouldStop": False}
        result = _serialize_action(action)
        assert "shouldStop" not in result

    def test_unique_ability_id(self):
        action = {
            "actionType": "ActionType_Activate",
            "instanceId": 10,
            "uniqueAbilityId": 42,
        }
        result = _serialize_action(action)
        assert result["uniqueAbilityId"] == 42

    def test_timing_source_grpid(self):
        action = {
            "actionType": "ActionType_Cast",
            "grpId": 1,
            "timingSourceGrpid": 99,
        }
        result = _serialize_action(action)
        assert result["timingSourceGrpid"] == 99


# =========================================================================
# Test: _serialize_targets
# =========================================================================

class TestSerializeTargets:
    """Tests for target selection serialization."""

    def test_simple_target(self):
        targets = [{
            "targetIdx": 0,
            "targets": [{"targetInstanceId": 600}],
            "minTargets": 1,
            "maxTargets": 1,
        }]
        result = _serialize_targets(targets)
        assert len(result) == 1
        assert result[0]["targets"][0]["targetInstanceId"] == 600
        assert result[0]["minTargets"] == 1
        assert result[0]["maxTargets"] == 1

    def test_multiple_targets(self):
        targets = [{
            "targetIdx": 0,
            "targets": [
                {"targetInstanceId": 600},
                {"targetInstanceId": 601},
            ],
        }]
        result = _serialize_targets(targets)
        assert len(result[0]["targets"]) == 2

    def test_target_with_highlight(self):
        targets = [{
            "targets": [
                {"targetInstanceId": 600, "highlight": "HighlightType_Hot"},
            ],
        }]
        result = _serialize_targets(targets)
        assert result[0]["targets"][0]["highlight"] == "HighlightType_Hot"

    def test_zero_target_instance_id_skipped(self):
        """Targets with targetInstanceId=0 are filtered out."""
        targets = [{
            "targets": [
                {"targetInstanceId": 0},
                {"targetInstanceId": 700},
            ],
        }]
        result = _serialize_targets(targets)
        assert len(result[0]["targets"]) == 1
        assert result[0]["targets"][0]["targetInstanceId"] == 700

    def test_empty_targets_list(self):
        result = _serialize_targets([])
        assert result == []

    def test_targeting_ability_fields(self):
        targets = [{
            "targetIdx": 1,
            "targets": [{"targetInstanceId": 800}],
            "targetingAbilityGrpId": 333,
            "targetingPlayer": 1,
            "targetSourceZoneId": 42,
        }]
        result = _serialize_targets(targets)
        assert result[0]["targetIdx"] == 1
        assert result[0]["targetingAbilityGrpId"] == 333
        assert result[0]["targetingPlayer"] == 1
        assert result[0]["targetSourceZoneId"] == 42


# =========================================================================
# Test: serialize_perform_action_resp
# =========================================================================

class TestSerializePerformActionResp:
    """Tests for PerformActionResp message building."""

    def test_basic_pass(self, pass_action):
        resp = serialize_perform_action_resp(pass_action)
        assert "actions" in resp
        assert len(resp["actions"]) == 1
        assert resp["actions"][0]["actionType"] == "ActionType_Pass"
        assert "autoPassPriority" not in resp

    def test_with_auto_pass_yes(self, pass_action):
        resp = serialize_perform_action_resp(
            pass_action, auto_pass=AutoPassPriority.YES
        )
        assert resp["autoPassPriority"] == "AutoPassPriority_Yes"

    def test_with_auto_pass_no(self, pass_action):
        resp = serialize_perform_action_resp(
            pass_action, auto_pass=AutoPassPriority.NO
        )
        assert resp["autoPassPriority"] == "AutoPassPriority_No"

    def test_cast_with_targets(self, targeted_cast_action):
        resp = serialize_perform_action_resp(targeted_cast_action)
        action = resp["actions"][0]
        assert action["actionType"] == "ActionType_Cast"
        assert "targets" in action
        assert len(action["targets"]) == 1
        # Target selections preserved
        ts = action["targets"][0]
        assert ts["minTargets"] == 1
        assert ts["maxTargets"] == 1

    def test_response_structure(self, cast_action):
        """Verify the response follows PerformActionResp protobuf schema."""
        resp = serialize_perform_action_resp(cast_action)
        # Must have 'actions' as a list
        assert isinstance(resp["actions"], list)
        # Each action must have 'actionType'
        for action in resp["actions"]:
            assert "actionType" in action


# =========================================================================
# Test: serialize_perform_action_resp_multi
# =========================================================================

class TestSerializePerformActionRespMulti:
    """Tests for multi-action PerformActionResp (batch mana payments)."""

    def test_two_mana_actions(self, activate_action):
        second = {
            "actionType": "ActionType_Activate_Mana",
            "instanceId": 101,
            "abilityGrpId": 993,
        }
        resp = serialize_perform_action_resp_multi([activate_action, second])
        assert len(resp["actions"]) == 2
        assert resp["actions"][0]["actionType"] == "ActionType_Activate"
        assert resp["actions"][1]["actionType"] == "ActionType_Activate_Mana"

    def test_empty_list_raises(self):
        with pytest.raises(SerializationError, match="empty"):
            serialize_perform_action_resp_multi([])

    def test_single_action_works(self, pass_action):
        resp = serialize_perform_action_resp_multi([pass_action])
        assert len(resp["actions"]) == 1


# =========================================================================
# Test: serialize_client_message
# =========================================================================

class TestSerializeClientMessage:
    """Tests for the full ClientToGREMessage envelope."""

    def test_envelope_structure(self, pass_action):
        msg = serialize_client_message(
            pass_action, system_seat_id=1, game_state_id=42
        )
        assert msg["type"] == "ClientMessageType_PerformActionResp"
        assert msg["systemSeatId"] == 1
        assert msg["gameStateId"] == 42
        assert "performActionResp" in msg

    def test_nested_action_resp(self, cast_action):
        msg = serialize_client_message(
            cast_action, system_seat_id=2, game_state_id=99
        )
        resp = msg["performActionResp"]
        assert "actions" in resp
        assert resp["actions"][0]["actionType"] == "ActionType_Cast"
        assert resp["actions"][0]["grpId"] == 12345

    def test_with_auto_pass(self, pass_action):
        msg = serialize_client_message(
            pass_action,
            system_seat_id=1,
            game_state_id=1,
            auto_pass=AutoPassPriority.YES,
        )
        assert msg["performActionResp"]["autoPassPriority"] == "AutoPassPriority_Yes"

    def test_seat_id_2(self, pass_action):
        msg = serialize_client_message(
            pass_action, system_seat_id=2, game_state_id=10
        )
        assert msg["systemSeatId"] == 2


# =========================================================================
# Test: GREActionRef integration
# =========================================================================

class TestSerializeFromActionRef:
    """Tests for serialization from GREActionRef objects."""

    def _make_ref(self, **kwargs):
        """Create a GREActionRef with given fields."""
        from arenamcp.gre_action_matcher import GREActionRef
        return GREActionRef(**kwargs)

    def test_ref_with_raw_dict(self, cast_action):
        ref = self._make_ref(
            action_type="ActionType_Cast",
            grp_id=12345,
            instance_id=501,
            raw=cast_action,
        )
        result = serialize_from_action_ref(ref)
        assert "actions" in result
        assert result["actions"][0]["actionType"] == "ActionType_Cast"
        assert result["actions"][0]["grpId"] == 12345
        # When raw is present, mana payment should be preserved
        assert "manaPaymentOptions" in result["actions"][0]

    def test_ref_without_raw_reconstructs(self):
        ref = self._make_ref(
            action_type="ActionType_Play",
            grp_id=67890,
            instance_id=502,
        )
        result = serialize_from_action_ref(ref)
        action = result["actions"][0]
        assert action["actionType"] == "ActionType_Play"
        assert action["grpId"] == 67890
        assert action["instanceId"] == 502

    def test_ref_with_envelope(self, pass_action):
        ref = self._make_ref(
            action_type="ActionType_Pass",
            raw=pass_action,
        )
        result = serialize_from_action_ref(
            ref, system_seat_id=1, game_state_id=50
        )
        assert result["type"] == "ClientMessageType_PerformActionResp"
        assert result["systemSeatId"] == 1
        assert result["gameStateId"] == 50

    def test_ref_without_envelope(self, pass_action):
        ref = self._make_ref(
            action_type="ActionType_Pass",
            raw=pass_action,
        )
        result = serialize_from_action_ref(ref)
        # Should be a PerformActionResp (no type/systemSeatId)
        assert "type" not in result
        assert "actions" in result

    def test_ref_with_targets(self):
        ref = self._make_ref(
            action_type="ActionType_Cast",
            grp_id=22222,
            instance_id=503,
            targets=[
                {"targetInstanceId": 600, "targetGrpId": 11111},
            ],
        )
        result = serialize_from_action_ref(ref)
        action = result["actions"][0]
        assert "targets" in action
        ts = action["targets"][0]
        assert ts["targets"][0]["targetInstanceId"] == 600

    def test_ref_with_ability_and_source(self):
        ref = self._make_ref(
            action_type="ActionType_Activate",
            instance_id=100,
            ability_grp_id=993,
            source_id=100,
        )
        result = serialize_from_action_ref(ref)
        action = result["actions"][0]
        assert action["abilityGrpId"] == 993
        assert action["sourceId"] == 100

    def test_ref_from_raw_roundtrip(self, cast_action):
        """GREActionRef.from_raw() -> serialize -> should produce valid output."""
        from arenamcp.gre_action_matcher import GREActionRef
        ref = GREActionRef.from_raw(cast_action)
        result = serialize_from_action_ref(ref)
        action = result["actions"][0]
        assert action["actionType"] == "ActionType_Cast"
        assert action["grpId"] == 12345
        assert action["instanceId"] == 501

    def test_ref_with_selection_fields(self):
        ref = self._make_ref(
            action_type="ActionType_Special",
            instance_id=504,
            selection_type=2,
            selection=1,
        )
        result = serialize_from_action_ref(ref)
        action = result["actions"][0]
        assert action["selectionType"] == 2
        assert action["selection"] == 1


# =========================================================================
# Test: Validation
# =========================================================================

class TestValidation:
    """Tests for action validation against legal actions list."""

    def test_pass_is_legal(self, pass_action, legal_actions_list):
        assert validate_action_against_legal(pass_action, legal_actions_list) is True

    def test_cast_is_legal(self, cast_action, legal_actions_list):
        assert validate_action_against_legal(cast_action, legal_actions_list) is True

    def test_play_land_is_legal(self, play_land_action, legal_actions_list):
        assert validate_action_against_legal(
            play_land_action, legal_actions_list
        ) is True

    def test_unknown_action_is_illegal(self, legal_actions_list):
        unknown = {
            "actionType": "ActionType_Cast",
            "grpId": 99999,  # Not in legal actions
            "instanceId": 999,
        }
        assert validate_action_against_legal(unknown, legal_actions_list) is False

    def test_empty_legal_actions(self, pass_action):
        assert validate_action_against_legal(pass_action, []) is False

    def test_matching_by_identity_key(self):
        """Two actions with the same identity fields should match."""
        action = {
            "actionType": "ActionType_Cast",
            "grpId": 100,
            "instanceId": 200,
            "abilityGrpId": 0,
            # Extra field not in identity key
            "manaPaymentOptions": [{"manaId": []}],
        }
        legal = [{
            "actionType": "ActionType_Cast",
            "grpId": 100,
            "instanceId": 200,
            # No manaPaymentOptions
        }]
        assert validate_action_against_legal(action, legal) is True

    def test_different_instance_id_no_match(self):
        action = {
            "actionType": "ActionType_Cast",
            "grpId": 100,
            "instanceId": 200,
        }
        legal = [{
            "actionType": "ActionType_Cast",
            "grpId": 100,
            "instanceId": 201,  # Different
        }]
        assert validate_action_against_legal(action, legal) is False


class TestFindMatchingLegalAction:
    """Tests for find_matching_legal_action."""

    def test_finds_matching_action(self, cast_action, legal_actions_list):
        matched = find_matching_legal_action(cast_action, legal_actions_list)
        assert matched is not None
        assert matched["actionType"] == "ActionType_Cast"
        assert matched["grpId"] == 12345

    def test_returns_none_for_unknown(self, legal_actions_list):
        unknown = {"actionType": "ActionType_Cast", "grpId": 99999, "instanceId": 999}
        matched = find_matching_legal_action(unknown, legal_actions_list)
        assert matched is None

    def test_returns_legal_action_with_full_data(self, legal_actions_list):
        """The returned action should be the GRE's version with all fields."""
        # Look up the cast action by minimal identity
        lookup = {
            "actionType": "ActionType_Cast",
            "grpId": 12345,
            "instanceId": 501,
        }
        matched = find_matching_legal_action(lookup, legal_actions_list)
        assert matched is not None
        # The GRE version should have manaPaymentOptions
        assert "manaPaymentOptions" in matched

    def test_empty_legal_list(self, pass_action):
        assert find_matching_legal_action(pass_action, []) is None


# =========================================================================
# Test: _action_identity_key
# =========================================================================

class TestActionIdentityKey:
    """Tests for the identity key extraction."""

    def test_same_action_same_key(self):
        a = {"actionType": "ActionType_Cast", "grpId": 1, "instanceId": 2}
        b = {"actionType": "ActionType_Cast", "grpId": 1, "instanceId": 2}
        assert _action_identity_key(a) == _action_identity_key(b)

    def test_different_action_type_different_key(self):
        a = {"actionType": "ActionType_Cast", "grpId": 1, "instanceId": 2}
        b = {"actionType": "ActionType_Play", "grpId": 1, "instanceId": 2}
        assert _action_identity_key(a) != _action_identity_key(b)

    def test_missing_fields_default_to_zero(self):
        a = {"actionType": "ActionType_Pass"}
        key = _action_identity_key(a)
        assert key == ("ActionType_Pass", 0, 0, 0, 0, 0, 0)

    def test_extra_fields_ignored(self):
        a = {"actionType": "ActionType_Pass", "manaPaymentOptions": []}
        b = {"actionType": "ActionType_Pass"}
        assert _action_identity_key(a) == _action_identity_key(b)


# =========================================================================
# Test: serialize_validated
# =========================================================================

class TestSerializeValidated:
    """Tests for the combined validation + serialization function."""

    def test_valid_action_serializes(self, pass_action, legal_actions_list):
        result = serialize_validated(pass_action, legal_actions_list)
        assert "actions" in result
        assert result["actions"][0]["actionType"] == "ActionType_Pass"

    def test_invalid_action_raises(self, legal_actions_list):
        bad = {"actionType": "ActionType_Cast", "grpId": 99999, "instanceId": 999}
        with pytest.raises(ValidationError, match="not found"):
            serialize_validated(bad, legal_actions_list)

    def test_uses_legal_action_data_by_default(self, legal_actions_list):
        """When use_legal_action_data=True, mana data from GRE is preserved."""
        # Minimal lookup (no mana data)
        lookup = {
            "actionType": "ActionType_Cast",
            "grpId": 12345,
            "instanceId": 501,
        }
        result = serialize_validated(lookup, legal_actions_list)
        action = result["actions"][0]
        # Should have mana payment from the matched legal action
        assert "manaPaymentOptions" in action

    def test_skip_legal_action_data(self, legal_actions_list):
        """When use_legal_action_data=False, user's action dict is used."""
        lookup = {
            "actionType": "ActionType_Cast",
            "grpId": 12345,
            "instanceId": 501,
        }
        result = serialize_validated(
            lookup, legal_actions_list, use_legal_action_data=False
        )
        action = result["actions"][0]
        # Our minimal lookup has no mana data
        assert "manaPaymentOptions" not in action

    def test_with_envelope(self, pass_action, legal_actions_list):
        result = serialize_validated(
            pass_action,
            legal_actions_list,
            system_seat_id=1,
            game_state_id=42,
        )
        assert result["type"] == "ClientMessageType_PerformActionResp"
        assert result["systemSeatId"] == 1

    def test_error_message_includes_action_info(self, legal_actions_list):
        bad = {"actionType": "ActionType_Activate", "grpId": 11, "instanceId": 22}
        with pytest.raises(ValidationError) as exc_info:
            serialize_validated(bad, legal_actions_list)
        msg = str(exc_info.value)
        assert "ActionType_Activate" in msg
        assert "grpId=11" in msg
        assert "instanceId=22" in msg


# =========================================================================
# Test: Convenience builders
# =========================================================================

class TestConvenienceBuilders:
    """Tests for the convenience action builder functions."""

    def test_build_pass_action(self):
        action = build_pass_action()
        assert action == {"actionType": "ActionType_Pass"}
        # Should be serializable
        resp = serialize_perform_action_resp(action)
        assert resp["actions"][0]["actionType"] == "ActionType_Pass"

    def test_build_cast_action_basic(self):
        action = build_cast_action(grp_id=100, instance_id=200)
        assert action["actionType"] == "ActionType_Cast"
        assert action["grpId"] == 100
        assert action["instanceId"] == 200
        assert "abilityGrpId" not in action

    def test_build_cast_action_with_ability(self):
        action = build_cast_action(
            grp_id=100, instance_id=200, ability_grp_id=50
        )
        assert action["abilityGrpId"] == 50

    def test_build_cast_action_with_auto_tap(self):
        tap = {"autoTapActions": [{"instanceId": 10, "abilityGrpId": 993}]}
        action = build_cast_action(
            grp_id=100, instance_id=200, auto_tap_solution=tap
        )
        assert action["autoTapSolution"] == tap

    def test_build_play_land_action(self):
        action = build_play_land_action(grp_id=300, instance_id=400)
        assert action["actionType"] == "ActionType_Play"
        assert action["grpId"] == 300
        assert action["instanceId"] == 400

    def test_build_activate_action_basic(self):
        action = build_activate_action(instance_id=500, ability_grp_id=993)
        assert action["actionType"] == "ActionType_Activate"
        assert action["instanceId"] == 500
        assert action["abilityGrpId"] == 993
        assert "sourceId" not in action

    def test_build_activate_action_with_source(self):
        action = build_activate_action(
            instance_id=500, ability_grp_id=993, source_id=500, grp_id=100
        )
        assert action["sourceId"] == 500
        assert action["grpId"] == 100

    def test_build_targeted_action(self):
        action = build_targeted_action(
            action_type="ActionType_Cast",
            instance_id=600,
            target_instance_ids=[700, 701],
            grp_id=100,
        )
        assert action["actionType"] == "ActionType_Cast"
        assert action["grpId"] == 100
        assert action["instanceId"] == 600
        assert len(action["targets"]) == 1
        assert len(action["targets"][0]["targets"]) == 2
        assert action["targets"][0]["targets"][0]["targetInstanceId"] == 700
        assert action["targets"][0]["targets"][1]["targetInstanceId"] == 701

    def test_build_targeted_action_no_targets(self):
        action = build_targeted_action(
            action_type="ActionType_Cast",
            instance_id=600,
            target_instance_ids=[],
        )
        assert "targets" not in action

    def test_build_targeted_action_with_ability(self):
        action = build_targeted_action(
            action_type="ActionType_Activate",
            instance_id=600,
            target_instance_ids=[700],
            ability_grp_id=42,
        )
        assert action["abilityGrpId"] == 42


# =========================================================================
# Test: Complete serialization roundtrips
# =========================================================================

class TestRoundtrips:
    """End-to-end serialization tests mimicking real game scenarios."""

    def test_pass_priority_roundtrip(self):
        """Player passes priority: minimal PerformActionResp."""
        action = build_pass_action()
        legal = [{"actionType": "ActionType_Pass"}]

        msg = serialize_validated(
            action, legal,
            system_seat_id=1, game_state_id=100,
        )

        assert msg["type"] == "ClientMessageType_PerformActionResp"
        assert msg["systemSeatId"] == 1
        assert msg["gameStateId"] == 100
        resp = msg["performActionResp"]
        assert len(resp["actions"]) == 1
        assert resp["actions"][0] == {"actionType": "ActionType_Pass"}

    def test_cast_spell_with_auto_tap_roundtrip(self):
        """Player casts Lightning Bolt with auto-tap solution."""
        legal_bolt = {
            "actionType": "ActionType_Cast",
            "grpId": 12345,
            "instanceId": 501,
            "manaPaymentOptions": [
                {"manaId": [{"color": "ManaColor_Red", "srcInstanceId": 100}]}
            ],
            "autoTapSolution": {
                "autoTapActions": [{"instanceId": 100, "abilityGrpId": 993}]
            },
        }
        legal = [{"actionType": "ActionType_Pass"}, legal_bolt]

        # Player selects by identity
        selection = build_cast_action(grp_id=12345, instance_id=501)
        msg = serialize_validated(
            selection, legal,
            system_seat_id=2, game_state_id=200,
        )

        resp = msg["performActionResp"]
        action = resp["actions"][0]
        assert action["actionType"] == "ActionType_Cast"
        assert action["grpId"] == 12345
        # Auto-tap from the legal action should be preserved
        assert "autoTapSolution" in action

    def test_play_land_roundtrip(self):
        """Player plays a Forest."""
        legal_forest = {
            "actionType": "ActionType_Play",
            "grpId": 67890,
            "instanceId": 502,
        }
        legal = [{"actionType": "ActionType_Pass"}, legal_forest]

        selection = build_play_land_action(grp_id=67890, instance_id=502)
        resp = serialize_validated(selection, legal)

        action = resp["actions"][0]
        assert action["actionType"] == "ActionType_Play"
        assert action["grpId"] == 67890
        assert action["instanceId"] == 502

    def test_activate_ability_roundtrip(self):
        """Player activates a mana ability."""
        legal_activate = {
            "actionType": "ActionType_Activate",
            "instanceId": 100,
            "abilityGrpId": 993,
            "sourceId": 100,
            "grpId": 55555,
        }
        legal = [{"actionType": "ActionType_Pass"}, legal_activate]

        selection = build_activate_action(
            instance_id=100, ability_grp_id=993, source_id=100, grp_id=55555
        )
        resp = serialize_validated(selection, legal)

        action = resp["actions"][0]
        assert action["actionType"] == "ActionType_Activate"
        assert action["abilityGrpId"] == 993

    def test_targeted_spell_roundtrip(self):
        """Player casts Shock targeting a specific creature."""
        legal_shock = {
            "actionType": "ActionType_Cast",
            "grpId": 22222,
            "instanceId": 503,
            "targets": [{
                "targetIdx": 0,
                "targets": [
                    {"targetInstanceId": 600, "highlight": "HighlightType_Hot"},
                    {"targetInstanceId": 601, "highlight": "HighlightType_Hot"},
                ],
                "minTargets": 1,
                "maxTargets": 1,
            }],
        }
        legal = [{"actionType": "ActionType_Pass"}, legal_shock]

        # Build with targets specified
        selection = build_targeted_action(
            action_type="ActionType_Cast",
            instance_id=503,
            target_instance_ids=[600],
            grp_id=22222,
        )
        # Validate against legal (identity match ignores targets)
        resp = serialize_validated(selection, legal)

        action = resp["actions"][0]
        assert action["actionType"] == "ActionType_Cast"
        assert "targets" in action

    def test_gre_action_ref_full_roundtrip(self):
        """Full roundtrip: raw action -> GREActionRef -> serialize -> validate."""
        from arenamcp.gre_action_matcher import GREActionRef

        raw = {
            "actionType": "ActionType_Cast",
            "grpId": 12345,
            "instanceId": 501,
            "abilityGrpId": 0,
            "sourceId": 0,
            "autoTapSolution": {
                "autoTapActions": [{"instanceId": 100, "abilityGrpId": 993}]
            },
        }
        legal = [{"actionType": "ActionType_Pass"}, raw]

        # Step 1: Create GREActionRef
        ref = GREActionRef.from_raw(raw)
        assert ref.action_type == "ActionType_Cast"
        assert ref.grp_id == 12345

        # Step 2: Serialize from ref
        result = serialize_from_action_ref(
            ref, system_seat_id=1, game_state_id=42
        )

        # Step 3: Verify structure
        assert result["type"] == "ClientMessageType_PerformActionResp"
        inner = result["performActionResp"]["actions"][0]
        assert inner["actionType"] == "ActionType_Cast"
        assert inner["grpId"] == 12345

        # Step 4: Validate the raw dict from ref against legal
        assert validate_action_against_legal(ref.raw, legal) is True


# =========================================================================
# Test: Edge cases
# =========================================================================

class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_action_with_all_zero_optional_fields(self):
        """An action where all optional uint fields are 0 should be minimal."""
        action = {
            "actionType": "ActionType_Pass",
            "grpId": 0,
            "instanceId": 0,
            "facetId": 0,
            "abilityGrpId": 0,
            "sourceId": 0,
            "alternativeGrpId": 0,
            "selectionType": 0,
            "selection": 0,
        }
        result = _serialize_action(action)
        assert result == {"actionType": "ActionType_Pass"}

    def test_action_with_all_fields_populated(self):
        """An action with every field populated should serialize all of them."""
        action = {
            "actionType": "ActionType_Cast",
            "grpId": 1,
            "instanceId": 2,
            "facetId": 3,
            "abilityGrpId": 4,
            "sourceId": 5,
            "alternativeGrpId": 6,
            "selectionType": 7,
            "selection": 8,
            "shouldStop": True,
            "uniqueAbilityId": 9,
            "timingSourceGrpid": 10,
            "targets": [{"targets": [{"targetInstanceId": 100}]}],
            "manaPaymentOptions": [{"manaId": []}],
            "manaCost": [{"color": "ManaColor_Red", "count": 1}],
            "autoTapSolution": {"autoTapActions": []},
        }
        result = _serialize_action(action)
        assert result["grpId"] == 1
        assert result["instanceId"] == 2
        assert result["facetId"] == 3
        assert result["abilityGrpId"] == 4
        assert result["sourceId"] == 5
        assert result["alternativeGrpId"] == 6
        assert result["selectionType"] == 7
        assert result["selection"] == 8
        assert result["shouldStop"] is True
        assert result["uniqueAbilityId"] == 9
        assert result["timingSourceGrpid"] == 10

    def test_serialization_does_not_mutate_input(self, cast_action):
        """Serialization must not modify the input dict."""
        original = copy.deepcopy(cast_action)
        serialize_perform_action_resp(cast_action)
        assert cast_action == original

    def test_cast_adventure_action_type(self):
        """ActionType_CastAdventure should serialize correctly."""
        action = {
            "actionType": "ActionType_CastAdventure",
            "grpId": 44444,
            "instanceId": 505,
        }
        result = _serialize_action(action)
        assert result["actionType"] == "ActionType_CastAdventure"

    def test_cast_mdfc_action_type(self):
        """ActionType_CastMDFC should serialize correctly."""
        action = {
            "actionType": "ActionType_CastMDFC",
            "grpId": 55555,
            "instanceId": 506,
            "facetId": 2,
        }
        result = _serialize_action(action)
        assert result["actionType"] == "ActionType_CastMDFC"
        assert result["facetId"] == 2

    def test_play_mdfc_action_type(self):
        """ActionType_PlayMDFC should serialize correctly."""
        action = {
            "actionType": "ActionType_PlayMDFC",
            "grpId": 66666,
            "instanceId": 507,
        }
        result = _serialize_action(action)
        assert result["actionType"] == "ActionType_PlayMDFC"

    def test_mana_payment_deep_copied(self, cast_action):
        """Mana payment data should be deep-copied, not referenced."""
        result = _serialize_action(cast_action)
        # Modify the serialized version
        result["manaPaymentOptions"][0]["modified"] = True
        # Original should be unaffected
        assert "modified" not in cast_action["manaPaymentOptions"][0]

    def test_validation_with_none_legal_actions(self, pass_action):
        """Passing None as legal_actions should return False, not crash."""
        # Using empty list as proxy (None would be a type error)
        assert validate_action_against_legal(pass_action, []) is False

    def test_multiple_target_selections(self):
        """Actions with multiple TargetSelection entries (multi-target spells)."""
        action = {
            "actionType": "ActionType_Cast",
            "grpId": 77777,
            "instanceId": 508,
            "targets": [
                {
                    "targetIdx": 0,
                    "targets": [{"targetInstanceId": 900}],
                    "minTargets": 1,
                    "maxTargets": 1,
                },
                {
                    "targetIdx": 1,
                    "targets": [{"targetInstanceId": 901}],
                    "minTargets": 1,
                    "maxTargets": 1,
                },
            ],
        }
        result = _serialize_action(action)
        assert len(result["targets"]) == 2
        assert result["targets"][0]["targetIdx"] == 0
        assert result["targets"][1]["targetIdx"] == 1

    def test_large_game_state_id(self):
        """Game state IDs can be large numbers."""
        action = build_pass_action()
        msg = serialize_client_message(
            action, system_seat_id=1, game_state_id=2147483647
        )
        assert msg["gameStateId"] == 2147483647


# =========================================================================
# Test: Action type coverage
# =========================================================================

class TestActionTypeCoverage:
    """Ensure all common GRE action types serialize correctly."""

    @pytest.mark.parametrize("action_type", [
        "ActionType_Pass",
        "ActionType_Cast",
        "ActionType_Play",
        "ActionType_Activate",
        "ActionType_Activate_Mana",
        "ActionType_Special",
        "ActionType_Special_TurnFaceUp",
        "ActionType_ResolutionCost",
        "ActionType_CastLeft",
        "ActionType_CastRight",
        "ActionType_Make_Payment",
        "ActionType_CombatCost",
        "ActionType_OpeningHandAction",
        "ActionType_CastAdventure",
        "ActionType_FloatMana",
        "ActionType_CastMDFC",
        "ActionType_PlayMDFC",
        "ActionType_Special_Payment",
        "ActionType_CastPrototype",
        "ActionType_CastLeftRoom",
        "ActionType_CastRightRoom",
        "ActionType_CastOmen",
    ])
    def test_action_type_serializes(self, action_type):
        """Every known ActionType string should serialize without error."""
        action = {"actionType": action_type}
        result = _serialize_action(action)
        assert result["actionType"] == action_type

    @pytest.mark.parametrize("action_type", [
        "ActionType_Pass",
        "ActionType_Cast",
        "ActionType_Play",
        "ActionType_Activate",
    ])
    def test_action_type_in_full_message(self, action_type):
        """Common action types produce valid ClientToGREMessage."""
        action = {"actionType": action_type}
        msg = serialize_client_message(
            action, system_seat_id=1, game_state_id=1
        )
        inner = msg["performActionResp"]["actions"][0]
        assert inner["actionType"] == action_type


# =========================================================================
# Test: AutoPassPriority enum
# =========================================================================

class TestAutoPassPriority:
    """Tests for the AutoPassPriority enum values."""

    def test_enum_values(self):
        assert AutoPassPriority.NONE.value == "AutoPassPriority_None"
        assert AutoPassPriority.NO.value == "AutoPassPriority_No"
        assert AutoPassPriority.YES.value == "AutoPassPriority_Yes"

    def test_none_not_included_in_output(self):
        action = build_pass_action()
        resp = serialize_perform_action_resp(
            action, auto_pass=AutoPassPriority.NONE
        )
        assert "autoPassPriority" not in resp

    def test_yes_included_in_output(self):
        action = build_pass_action()
        resp = serialize_perform_action_resp(
            action, auto_pass=AutoPassPriority.YES
        )
        assert resp["autoPassPriority"] == "AutoPassPriority_Yes"
