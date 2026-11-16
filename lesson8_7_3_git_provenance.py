
from evaluation.git_provenance import get_git_provenance


def main():
    provenance = get_git_provenance(
        require_clean=False
    )

    print("Lesson 8.7.3 — Git Provenance")
    print("-----------------------------")
    print("Commit:", provenance["git_commit"])
    print("Branch:", provenance["git_branch"])
    print(
        "Working tree clean:",
        provenance["git_worktree_clean"]
    )


if __name__ == "__main__":
    main()
