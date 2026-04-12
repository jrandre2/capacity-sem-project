"""
AI Agents for CENTAUR

This package provides AI-powered tools for:
- Analyzing existing research project structures
- Mapping projects to standardized templates
- Generating migration plans
- Executing project transformations
"""

from .project_analyzer import ProjectAnalyzer, analyze_project
from .structure_mapper import StructureMapper, map_project
from .migration_planner import MigrationPlanner, generate_migration_plan
from .migration_executor import MigrationExecutor

__all__ = [
    'ProjectAnalyzer',
    'StructureMapper',
    'MigrationPlanner',
    'MigrationExecutor',
    'analyze_project',
    'map_project',
    'generate_migration_plan',
]
