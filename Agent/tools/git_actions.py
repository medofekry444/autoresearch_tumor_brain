import subprocess
from pathlib import Path


class GitActions:
    """Manage version control for an agent-controlled file."""

    def __init__(self, target_file, repo_path=None):
        self.target_file = Path(target_file).resolve()

        if repo_path is None:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.target_file.parent),
                    "rev-parse",
                    "--show-toplevel",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            repo_path = result.stdout.strip()

        self.repo_path = Path(repo_path).resolve()

        if not self.target_file.is_relative_to(self.repo_path):
            raise ValueError(
                "Target file must be inside the Git repository."
            )

        self.relative_target = str(
            self.target_file.relative_to(self.repo_path)
        )

    def _run(self, *args):
        """Execute a Git command in the configured repository."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )

    def commit(self, message="Auto commit by agent"):
        """Commit changes to the agent-controlled file only."""
        try:
            self._run("add", "--", self.relative_target)

            result = subprocess.run(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--quiet",
                    "--",
                    self.relative_target,
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return "Nothing to commit."

            if result.returncode != 1:
                return (
                    "Error checking staged changes: "
                    + result.stderr.strip()
                )

            # Path-scoped commit avoids including unrelated staged files.
            self._run(
                "commit",
                "-m",
                message,
                "--only",
                "--",
                self.relative_target,
            )

            return "Changes committed successfully."

        except subprocess.CalledProcessError as error:
            return (
                "Error during git commit: "
                + (error.stderr or str(error)).strip()
            )

    def push(self):
        """Push the current branch to its configured remote."""
        try:
            result = self._run("push")

            return (
                "Changes pushed successfully. "
                + result.stdout.strip()
            )

        except subprocess.CalledProcessError as error:
            return (
                "Error during git push: "
                + (error.stderr or str(error)).strip()
            )

    def reset(self):
        """Restore only the target file to its last committed state."""
        try:
            self._run(
                "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                self.relative_target,
            )

            return "File restored to last commit."

        except subprocess.CalledProcessError as error:
            return (
                "Error during git reset: "
                + (error.stderr or str(error)).strip()
            )