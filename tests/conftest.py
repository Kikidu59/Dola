"""
conftest.py — Fixtures pytest partagées entre tous les fichiers de tests.

Fournit :
    - any_setup  : lance le test pour les deux setups (matrix et UV), parametrize
    - matrix_setup : force le setup matrix pour un test
    - uv_setup     : force le setup UV pour un test
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import pytest
import src.spaces as spaces


@pytest.fixture(params=["matrix", "uv"])
def any_setup(request):
    """Fixture parametrized : le test tourne une fois par setup (matrix, UV)."""
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
    """Force le setup matrix pour un test."""
    spaces.SETUP = spaces.make_setup_matrix()
    jax.clear_caches()
    yield
    spaces.SETUP = spaces.make_setup_uv()
    jax.clear_caches()


@pytest.fixture
def uv_setup():
    """Force le setup UV pour un test."""
    spaces.SETUP = spaces.make_setup_uv()
    jax.clear_caches()
    yield
