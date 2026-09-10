"""
Resource and Asset Path Resolution Helper.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Re-exports get_resource_path from contracts for standalone backward compatibility.
"""

from contracts import get_resource_path

__all__ = ["get_resource_path"]
