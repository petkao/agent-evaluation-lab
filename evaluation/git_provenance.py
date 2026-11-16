
import subprocess


def git_command(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_git_provenance(require_clean=True):
    commit = git_command("rev-parse", "HEAD")
    branch = git_command(
        "rev-parse", "--abbrev-ref", "HEAD"
    )

    status = git_command(
        "status", "--porcelain", "--untracked-files=all"
    )

    is_clean = status == ""

    if require_clean and not is_clean:
        raise RuntimeError(
            "Git working tree is not clean. "
            "Commit your changes before evaluating."
        )

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_worktree_clean": is_clean,
    }
