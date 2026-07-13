"""Compatibility entry point for the reorganized environment test."""

from legged_mani.tests.test_environment import run_check


def main() -> None:
    run_check()


if __name__ == "__main__":
    main()
