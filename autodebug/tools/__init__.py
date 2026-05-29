from autodebug.tools.shared import make_read_file_tool, make_list_files_tool
from autodebug.memory import make_search_memory_tool
from autodebug.tools.repro import make_run_script_tool, make_submit_repro_tool
from autodebug.tools.bisect import (
    make_mark_bad_tool, make_mark_good_tool, make_mark_skip_tool,
    make_submit_result_tool, make_submit_culprit_tool,
)
from autodebug.tools.root_cause import (
    make_read_file_at_parent_tool, make_run_repro_with_traceback_tool, make_submit_root_cause_tool,
)
from autodebug.tools.fix import (
    make_apply_patch_tool, make_run_repro_tool, make_run_tests_tool, make_submit_fix_tool,
)
from autodebug.tools.skills import make_load_skill_tool

# Auto-discover factories from this module by naming convention make_{tool}_tool
_FACTORIES = {
    attr[len("make_"):-len("_tool")]: obj
    for attr, obj in globals().items()
    if attr.startswith("make_") and attr.endswith("_tool") and callable(obj)
}


def build_tools(names: list[str], **context) -> list:
    """Instantiate LangChain tools by name, injecting the provided context."""
    tools = []
    for name in names:
        if name not in _FACTORIES:
            raise KeyError(f"Unknown tool '{name}'. Available: {sorted(_FACTORIES)}")
        tools.append(_FACTORIES[name](**context))
    return tools
