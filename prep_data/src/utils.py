import sys
import subprocess


class Logger:
    """Handles conditional printing based on progress_bar and verbose."""
    def __init__(self, progress_bar=None, verbose=False):
        self.progress_bar = progress_bar
        self.verbose = verbose

    def log(self, message: str, file=sys.stdout, flush=True):
        """Main logging function."""
        if self.verbose:
            if self.progress_bar:
                self.progress_bar.write(message, file=file, flush=flush)
            else:
                print(message, file=file, flush=flush)

    def error(self, message: str, flush=True):
        """Prints errors (always visible)."""
        if self.progress_bar:
            self.progress_bar.write(message, file=sys.stderr, flush=flush)
        else:
            print(message, file=sys.stderr, flush=flush)


def run_cmd(command: str, logger: Logger):
    """
    Executes a shell command.

    Args:
        command (str): Command to execute.
        logger (Logger): Logger instance for printing.
    
    Raises:
        RuntimeError: If command fails.
    """
    logger.log(f"\n[EXEC] $ {command}", flush=True)
    
    stdout_target = None
    if not logger.verbose:
         stdout_target = subprocess.DEVNULL

    result = subprocess.run(command, shell=True, check=False, text=True, stdout=stdout_target)
    
    if result.returncode != 0:
        logger.error(f"\n--- ERROR ---")
        logger.error(f"Command failed with exit code: {result.returncode}")
        logger.error(f"Command: {command}")
        if result.stderr:
            logger.error(f"Stderr: {result.stderr}")
        raise RuntimeError(f"Command execution failed: {command}")
