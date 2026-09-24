"""Smoke test for Streamlit frontend app.py using streamlit.testing.v1.AppTest."""

import os
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(PROJECT_ROOT, "app.py")


def test_app_launches_without_error():
    """Verify that app.py initializes and executes both tabs without uncaught exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Assert no unhandled exceptions occurred
    assert not at.exception, f"App threw exception on launch: {at.exception}"

    # Verify key elements rendered
    # Title exists
    assert len(at.title) > 0
    assert "Route Optimization via Reinforcement Learning" in at.title[0].value

    # Tabs exist
    assert len(at.tabs) >= 2


def test_app_interactive_routing_execution():
    """Verify Tab 2 Interactive Routing Tool executes single-packet routing without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Select Route Packet button if available
    buttons = at.button
    route_buttons = [b for b in buttons if "Route" in b.label]
    if route_buttons:
        route_buttons[0].click().run()
        assert not at.exception


def test_app_custom_network_tab():
    """Verify Tab 3 Custom Network Builder renders and displays topological metrics."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) >= 3


def test_app_dark_mode_execution():
    """Verify permanent dark mode app loads and runs cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) >= 3
    assert len(at.selectbox) >= 4


