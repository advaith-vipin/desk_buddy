"""Desk Buddy — a live lip-sync desk avatar powered by pytoon + Ollama.

Package layout:
    desk_buddy.core           - data + logic (db, memory_handler, agents, crud_tools)
    desk_buddy.avatar         - avatar renderers (cube_face, front_face)
    desk_buddy.entrypoints    - runnable apps (pytoon_live, pytoon_cli)
    desk_buddy._paths         - runtime path anchors
    desk_buddy.prompts        - all LLM system prompts / instruction templates
"""

__version__ = "1.0.0"