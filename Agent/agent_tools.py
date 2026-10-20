# Agent/agent_tools.py

from .tools.code_handling import code_handling
from .tools.git_actions import GitActions
from .tools.plot_progress import plot_progress
from .tools.output_handeling import output_handeling


def get_tools(train_path, progress_path, repo_path):
    return {
        "code_handling": code_handling(train_path),
        "output_handeling": output_handeling(),
        "git_actions": GitActions(
            target_file=train_path,
            repo_path=repo_path,
        ),
        "plot_progress": lambda: plot_progress(progress_path),
    }