"""Validation tests for the 6 simulation fixes.

Tests SDF plugins, launch file logic, YAML config, and Dockerfile
without requiring ROS2 or Gazebo runtime.
"""

import ast
import os
import re
import xml.etree.ElementTree as ET

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SDF_PATH = "reach_avoid_ws/src/reach_avoid_sim/worlds/reach_avoid_arena.sdf"
SIM_LAUNCH_PATH = "reach_avoid_ws/src/reach_avoid_bringup/launch/simulation.launch.py"
FULL_LAUNCH_PATH = "reach_avoid_ws/src/reach_avoid_bringup/launch/full_game.launch.py"
SIM_PARAMS_PATH = "reach_avoid_ws/src/reach_avoid_bringup/config/simulation_params.yaml"
DOCKERFILE_PATH = "Dockerfile.sim"


# ===========================================================================
# 1. SDF World Plugin Validation
# ===========================================================================
class TestSDFPlugins:
    """Verify reach_avoid_arena.sdf has all required Gazebo system plugins."""

    @pytest.fixture(scope="class")
    def world(self):
        tree = ET.parse(SDF_PATH)
        root = tree.getroot()
        return root.find("world")

    REQUIRED_PLUGINS = [
        "gz::sim::systems::Physics",
        "gz::sim::systems::UserCommands",
        "gz::sim::systems::SceneBroadcaster",
        "gz::sim::systems::Contact",
        "gz::sim::systems::Imu",
        "gz::sim::systems::AirPressure",
        "gz::sim::systems::Magnetometer",
        "gz::sim::systems::Sensors",
    ]

    EXPECTED_FILENAMES = {
        "gz::sim::systems::Physics": "gz-sim-physics-system",
        "gz::sim::systems::UserCommands": "gz-sim-user-commands-system",
        "gz::sim::systems::SceneBroadcaster": "gz-sim-scene-broadcaster-system",
        "gz::sim::systems::Contact": "gz-sim-contact-system",
        "gz::sim::systems::Imu": "gz-sim-imu-system",
        "gz::sim::systems::AirPressure": "gz-sim-air-pressure-system",
        "gz::sim::systems::Magnetometer": "gz-sim-magnetometer-system",
        "gz::sim::systems::Sensors": "gz-sim-sensors-system",
    }

    def test_sdf_parses(self):
        """SDF file must be valid XML."""
        tree = ET.parse(SDF_PATH)
        assert tree.getroot().tag == "sdf"

    def test_world_exists(self, world):
        assert world is not None
        assert world.attrib.get("name") == "reach_avoid_arena"

    @pytest.mark.parametrize("plugin_name", REQUIRED_PLUGINS)
    def test_plugin_present(self, world, plugin_name):
        """Each required plugin must appear as <plugin name='...'>."""
        plugins = {p.attrib["name"] for p in world.findall("plugin")}
        assert plugin_name in plugins, f"Missing plugin: {plugin_name}"

    @pytest.mark.parametrize("plugin_name", REQUIRED_PLUGINS)
    def test_plugin_filename_correct(self, world, plugin_name):
        """Each plugin must use the correct shared-lib filename."""
        expected_fn = self.EXPECTED_FILENAMES[plugin_name]
        for p in world.findall("plugin"):
            if p.attrib.get("name") == plugin_name:
                assert p.attrib.get("filename") == expected_fn, (
                    f"{plugin_name}: expected filename={expected_fn}, "
                    f"got {p.attrib.get('filename')}"
                )
                return
        pytest.fail(f"Plugin {plugin_name} not found")

    def test_sensors_has_render_engine(self, world):
        """Sensors plugin should specify render_engine for headless."""
        for p in world.findall("plugin"):
            if p.attrib.get("name") == "gz::sim::systems::Sensors":
                re_elem = p.find("render_engine")
                assert re_elem is not None, "Sensors plugin missing <render_engine>"
                assert re_elem.text == "ogre2"
                return
        pytest.fail("Sensors plugin not found")

    def test_physics_tag_still_present(self, world):
        """The <physics> configuration tag should still exist."""
        physics = world.find("physics")
        assert physics is not None
        assert physics.attrib.get("type") == "dart"

    def test_plugin_count_at_least_8(self, world):
        """World must have at least 8 plugins."""
        plugins = world.findall("plugin")
        assert len(plugins) >= 8

    def test_plugins_before_lights(self, world):
        """Plugins should appear before <light> elements in the SDF."""
        children = list(world)
        first_plugin_idx = None
        first_light_idx = None
        for i, child in enumerate(children):
            if child.tag == "plugin" and first_plugin_idx is None:
                first_plugin_idx = i
            if child.tag == "light" and first_light_idx is None:
                first_light_idx = i
        assert first_plugin_idx is not None, "No plugins found"
        assert first_light_idx is not None, "No lights found"
        assert first_plugin_idx < first_light_idx, "Plugins should come before lights"


# ===========================================================================
# 2. simulation.launch.py - PX4 Path Discovery
# ===========================================================================
class TestSimulationLaunchPX4Path:
    """Verify PX4 path discovery includes Docker /opt/PX4-Autopilot."""

    @pytest.fixture(scope="class")
    def source(self):
        with open(SIM_LAUNCH_PATH) as f:
            return f.read()

    def test_python_syntax_valid(self, source):
        """Launch file must parse as valid Python."""
        ast.parse(source)

    def test_opt_px4_in_candidates(self, source):
        """'/opt/PX4-Autopilot' must be in the candidate list."""
        assert "/opt/PX4-Autopilot" in source

    def test_opt_px4_before_workspaces(self, source):
        """Docker path should be checked before devcontainer path."""
        idx_opt = source.index("/opt/PX4-Autopilot")
        idx_ws = source.index("/workspaces/ros2_ws/src/PX4-Autopilot")
        assert idx_opt < idx_ws, "/opt/PX4-Autopilot should be checked first"

    def test_env_override_still_first(self, source):
        """PX4_AUTOPILOT env var override should come before candidate list."""
        idx_env = source.index('PX4_AUTOPILOT')
        idx_candidates = source.index("for candidate in")
        assert idx_env < idx_candidates

    def test_home_fallback_still_present(self, source):
        """Home directory fallback should still be a candidate."""
        assert "PX4-Autopilot" in source
        assert 'expanduser("~")' in source or "expanduser('~')" in source


# ===========================================================================
# 3. simulation.launch.py - Indoor SITL Parameters
# ===========================================================================
class TestSimulationLaunchIndoorParams:
    """Verify indoor SITL PX4 parameters are in the env dict."""

    @pytest.fixture(scope="class")
    def source(self):
        with open(SIM_LAUNCH_PATH) as f:
            return f.read()

    REQUIRED_PARAMS = {
        "PX4_PARAM_COM_ARM_WO_GPS": "1",
        "PX4_PARAM_SYS_HAS_GPS": "0",
        "PX4_PARAM_EKF2_GPS_CTRL": "0",
        "PX4_PARAM_EKF2_HGT_REF": "2",
        "PX4_PARAM_EKF2_BARO_CTRL": "1",
        "PX4_PARAM_EKF2_MAG_TYPE": "1",
    }

    @pytest.mark.parametrize("param,value", REQUIRED_PARAMS.items())
    def test_indoor_param_present(self, source, param, value):
        """Each indoor SITL parameter must appear with correct value."""
        # Match 'PARAM': 'VALUE' pattern
        pattern = rf"'{param}':\s*'{value}'"
        assert re.search(pattern, source), (
            f"Missing or wrong: {param}={value}"
        )

    def test_existing_mag_params_preserved(self, source):
        """Pre-existing mag relaxation params should still be present."""
        assert "PX4_PARAM_COM_ARM_MAG_STR" in source
        assert "PX4_PARAM_EKF2_MAG_CHECK" in source
        assert "PX4_PARAM_EKF2_MAG_GATE" in source

    def test_standalone_mode_preserved(self, source):
        """PX4_GZ_STANDALONE=1 must still be set."""
        assert "'PX4_GZ_STANDALONE': '1'" in source


# ===========================================================================
# 4. full_game.launch.py - Path Search
# ===========================================================================
class TestFullGameLaunchPaths:
    """Verify full_game.launch.py uses path-search instead of hardcoded paths."""

    @pytest.fixture(scope="class")
    def source(self):
        with open(FULL_LAUNCH_PATH) as f:
            return f.read()

    @pytest.fixture(scope="class")
    def tree(self, source):
        return ast.parse(source)

    def test_python_syntax_valid(self, source):
        ast.parse(source)

    def test_find_path_helper_defined(self, source):
        """_find_path helper function should be defined."""
        assert "def _find_path(" in source

    def test_value_function_dir_uses_search(self, source):
        """value_function_dir default should come from _find_path, not hardcoded."""
        assert "vf_default = _find_path(" in source

    def test_game_params_uses_search(self, source):
        """game_params_file default should come from _find_path, not hardcoded."""
        assert "gp_default = _find_path(" in source

    def test_docker_vf_path_candidate(self, source):
        """Docker value_functions path ~/ws/data/value_functions should be a candidate."""
        assert "ws', 'data', 'value_functions'" in source or "ws/data/value_functions" in source

    def test_workspace_vf_path_candidate(self, source):
        """/workspace/data/value_functions should be a candidate."""
        assert "/workspace/data/value_functions" in source

    def test_docker_game_params_candidate(self, source):
        """Docker game_params path should be a candidate."""
        assert "ws', 'config', 'game_params.yaml'" in source or "ws/config/game_params.yaml" in source

    def test_venv_uses_candidate_search(self, source):
        """Venv path should search multiple candidates, not hardcoded."""
        assert "venv_candidates" in source

    def test_venv_has_docker_candidate(self, source):
        """Docker venv path should be in candidates."""
        assert "ws', '.venv'" in source or "ws/.venv" in source

    def test_no_sole_hardcoded_workspaces_path(self, source):
        """The only /workspaces/ references should be in candidate lists, not as sole defaults."""
        # Find lines with /workspaces/ that are NOT in a list/array context
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if "/workspaces/ros2_ws" in stripped and "default_value=" in stripped:
                pytest.fail(
                    f"Found hardcoded /workspaces/ as default_value: {stripped}"
                )


# ===========================================================================
# 5. simulation_params.yaml - Model Names
# ===========================================================================
class TestSimulationParamsYAML:
    """Verify simulation_params.yaml has correct PX4 model names."""

    @pytest.fixture(scope="class")
    def params(self):
        with open(SIM_PARAMS_PATH) as f:
            return yaml.safe_load(f)

    def test_yaml_parses(self):
        with open(SIM_PARAMS_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_defender_model_is_x500_1(self, params):
        assert params["gazebo"]["defender_model"] == "x500_1"

    def test_attacker_model_is_x500_2(self, params):
        assert params["gazebo"]["attacker_model"] == "x500_2"

    def test_world_name_unchanged(self, params):
        assert params["gazebo"]["world_name"] == "reach_avoid_arena"

    def test_defender_vehicle_id_matches(self, params):
        """defender_model x500_1 should match vehicle_id 1."""
        assert params["px4"]["defender"]["vehicle_id"] == 1
        assert params["gazebo"]["defender_model"] == "x500_1"

    def test_attacker_vehicle_id_matches(self, params):
        """attacker_model x500_2 should match vehicle_id 2."""
        assert params["px4"]["attacker"]["vehicle_id"] == 2
        assert params["gazebo"]["attacker_model"] == "x500_2"

    def test_no_custom_model_names(self, params):
        """Model names should NOT contain 'defender' or 'attacker' strings."""
        assert "defender" not in params["gazebo"]["defender_model"]
        assert "attacker" not in params["gazebo"]["attacker_model"]

    def test_topics_section_intact(self, params):
        """Topics section should be unchanged."""
        assert "defender_state" in params["topics"]
        assert "attacker_cmd_vel" in params["topics"]

    def test_model_names_consistent_with_launch(self):
        """Model names in YAML should match what simulation.launch.py uses."""
        with open(SIM_LAUNCH_PATH) as f:
            launch_src = f.read()
        with open(SIM_PARAMS_PATH) as f:
            params = yaml.safe_load(f)
        # simulation.launch.py ground_truth_relay uses x500_1, x500_2
        assert f"'defender_model_name': '{params['gazebo']['defender_model']}'" in launch_src
        assert f"'attacker_model_name': '{params['gazebo']['attacker_model']}'" in launch_src


# ===========================================================================
# 6. Dockerfile.sim Validation
# ===========================================================================
class TestDockerfile:
    """Verify Dockerfile.sim has PX4 env var and value_functions COPY."""

    @pytest.fixture(scope="class")
    def content(self):
        with open(DOCKERFILE_PATH) as f:
            return f.read()

    def test_px4_autopilot_env(self, content):
        """ENV PX4_AUTOPILOT=/opt/PX4-Autopilot should be set."""
        assert "ENV PX4_AUTOPILOT=/opt/PX4-Autopilot" in content

    def test_px4_env_after_build(self, content):
        """PX4_AUTOPILOT env should come after PX4 build (make px4_sitl)."""
        idx_build = content.index("make px4_sitl")
        idx_env = content.index("ENV PX4_AUTOPILOT")
        assert idx_env > idx_build

    def test_value_functions_copy(self, content):
        """Value functions should be copied into the container."""
        assert "data/value_functions" in content
        # Should be a COPY command
        lines = content.split("\n")
        vf_copy_lines = [l for l in lines if "value_functions" in l and "COPY" in l]
        assert len(vf_copy_lines) >= 1

    def test_value_functions_correct_dest(self, content):
        """Value functions should go to /home/simuser/ws/data/value_functions."""
        assert "/home/simuser/ws/data/value_functions" in content

    def test_config_copy_present(self, content):
        """Config directory should still be copied."""
        assert "COPY --chown=simuser:simuser config /home/simuser/ws/config" in content

    def test_workspace_copy_present(self, content):
        """Workspace src should still be copied."""
        assert "COPY --chown=simuser:simuser reach_avoid_ws/src /home/simuser/ws/src" in content

    def test_px4_installed_to_opt(self, content):
        """PX4 clone target should be /opt/PX4-Autopilot."""
        assert "/opt/PX4-Autopilot" in content

    def test_entrypoint_present(self, content):
        """Entrypoint script should still be present."""
        assert "ENTRYPOINT" in content


# ===========================================================================
# Cross-file Consistency
# ===========================================================================
class TestCrossFileConsistency:
    """Verify consistency between the modified files."""

    def test_sdf_world_name_matches_launch(self):
        """World name in SDF must match PX4_GZ_WORLD in launch file."""
        tree = ET.parse(SDF_PATH)
        sdf_world_name = tree.getroot().find("world").attrib["name"]

        with open(SIM_LAUNCH_PATH) as f:
            launch_src = f.read()
        assert f"'PX4_GZ_WORLD': '{sdf_world_name}'" in launch_src

    def test_sdf_world_name_matches_yaml(self):
        """World name in SDF must match YAML config."""
        tree = ET.parse(SDF_PATH)
        sdf_world_name = tree.getroot().find("world").attrib["name"]

        with open(SIM_PARAMS_PATH) as f:
            params = yaml.safe_load(f)
        assert params["gazebo"]["world_name"] == sdf_world_name

    def test_model_names_yaml_matches_launch(self):
        """YAML model names must match what launch ground_truth_relay uses."""
        with open(SIM_PARAMS_PATH) as f:
            params = yaml.safe_load(f)
        with open(SIM_LAUNCH_PATH) as f:
            src = f.read()
        assert params["gazebo"]["defender_model"] in src
        assert params["gazebo"]["attacker_model"] in src

    def test_docker_px4_path_in_launch_candidates(self):
        """Dockerfile PX4 install path must be in launch file candidates."""
        with open(DOCKERFILE_PATH) as f:
            dockerfile = f.read()
        # Find where PX4 is cloned to
        match = re.search(r"git clone .+ (/opt/PX4-Autopilot)", dockerfile)
        assert match, "PX4 clone path not found in Dockerfile"
        px4_path = match.group(1)

        with open(SIM_LAUNCH_PATH) as f:
            launch_src = f.read()
        assert px4_path in launch_src, (
            f"Dockerfile PX4 path {px4_path} not in launch candidates"
        )

    def test_docker_vf_path_in_full_game_candidates(self):
        """Dockerfile value_functions dest must match full_game.launch.py search."""
        # Docker copies to /home/simuser/ws/data/value_functions
        # full_game searches ~/ws/data/value_functions (which expands the same for simuser)
        with open(FULL_LAUNCH_PATH) as f:
            src = f.read()
        assert "'ws', 'data', 'value_functions'" in src or "ws/data/value_functions" in src
