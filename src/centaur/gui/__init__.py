"""
CENTAUR GUI - Lightweight web dashboard for pipeline supervision.

This package provides a FastAPI-based web interface for monitoring
and controlling the CENTAUR research pipeline. The CLI remains the
primary interface; this GUI serves as a human supervision layer.

Usage
-----
    python src/pipeline.py centaur gui
    # or
    uvicorn centaur.gui.app:app --reload --port 8000
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app() -> "FastAPI":
    """
    Lazily import the FastAPI application factory.

    This keeps non-GUI CENTAUR commands usable even when optional web
    dependencies such as FastAPI are not installed.
    """
    from .app import create_app as _create_app

    return _create_app()

__all__ = ['create_app']
