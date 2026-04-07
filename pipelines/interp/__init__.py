"""Capture and analysis runtime for Xenon.

The live workflow surface in this package is:
- local capture fallback
- Modal vLLM capture orchestration and engine code
- Modal analysis orchestration and shared analysis code
- close capture/analysis helpers such as pooling and tool schemas

Research-specific workflows should not be added to this top-level package.
"""
