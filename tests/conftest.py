"""
conftest.py — Shared pytest fixtures.

Provides:
    - any_setup    : parametrized fixture, runs the test once per setup (matrix and UV)
    - matrix_setup : forces the matrix setup for a test
    - uv_setup     : forces the UV setup for a test
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import pytest
import src.spaces as spaces


@pytest.fixture(params=["matrix", "uv"])
def any_setup(request):
    """Parametrized fixture: the test runs once per setup (matrix, UV)."""
    if request.param == "matrix":
        spaces.SETUP = spaces.make_setup_matrix()
    else:
        spaces.SETUP = spaces.make_setup_uv()
    jax.clear_caches()
    yield request.param
    spaces.SETUP = spaces.make_setup_uv()
    jax.clear_caches()


@pytest.fixture
def matrix_setup():
    """Force the matrix setup for a test."""
    spaces.SETUP = spaces.make_setup_matrix()
    jax.clear_caches()
    yield
    spaces.SETUP = spaces.make_setup_uv()
    jax.clear_caches()


@pytest.fixture
def uv_setup():
    """Force the UV setup for a test."""
    spaces.SETUP = spaces.make_setup_uv()
    jax.clear_caches()
    yield
