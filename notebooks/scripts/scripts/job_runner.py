import datetime
import os
from pathlib import Path
import time
from typing import Any
import jbsub_tools

class JobRunnerBackEnd:
    """
    Abstract class for running jobs in a cluster or just in a background process on a machine.
    """

    def _submit_job(self, cmd: list[str], verbose=True):
        """Abstract method to submit a job in a cluster or a background process.

        :param cmd: List of strings that, when concatenated, form a linux command, e.g. ["echo", "'hello world'"]
        :param verbose: Print additional data about the job, defaults to True
        """
        pass

    def _wait_for_job(self, verbose=True):
        """Abstract method to block until the submitted running job completes.

        :param verbose: Print periodic information about the job status, defaults to True
        """
        pass


class TmuxJobRunner(JobRunnerBackEnd):
    """
    Runner for running jobs in tmux.
    """

    def __init__(self, tmux_params: dict, log_dir: Path, job_name: str):
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT-%H-%M-%SZ")
        self._tmux_session_id = f"{timestamp}_{job_name}"
        self._tmux_prefix = self.build_tmux_cmd(self._tmux_session_id)
        self._log_file = log_dir / f"{timestamp}_job_{job_name}.log"

    @staticmethod
    def build_tmux_cmd(session_id: str) -> list[str]:
        prefix = ["tmux", "new-session"]   # start a new session
        prefix.extend(["-s", session_id])  # name the session
        prefix.extend(["-d"])              # run in dettached mode (so it runs like a job in the background)
        return prefix
        
    def _submit_job(self, cmd: list[str], verbose=True):
        """Submit a job in a cluster or a background process.

        :param cmd: List of strings that, when concatenated, form a linux command, e.g. ["echo", "'hello world'"]
        :param verbose: Print additional data about the job, defaults to True
        """
        prefix = " ".join(str(c) for c in self._tmux_prefix)
        inner_cmd = " ".join(cmd)
        full_cmd = f"{prefix} '{inner_cmd} 2>&1 | tee {self._log_file}'"  
        if verbose:
            print(full_cmd)
        
        tmux_output = os.popen(full_cmd).read()
        
        assert tmux_output == "", "Could not submit job.  Here is the tmux output:\n" + tmux_output 

    def _wait_for_job(self, verbose=True):
        """Block until the submitted running job completes.

        :param verbose: Print periodic information about the job status, defaults to True
        """
        printed_session_info = False
        while True:
            tmux_sessions = os.popen("tmux list-sessions").read()
            this_job_sessions = [s for s in tmux_sessions.split() if s.startswith(self._tmux_session_id)]
            assert len(this_job_sessions) <= 1, this_job_sessions
            if this_job_sessions:
                if verbose:
                    print("\nTmux job running:")
                    print(this_job_sessions[0])
                    print("")
                    printed_session_info = True
            else:
                return  # session finished
            time.sleep(60)


class JbsubJobRunner(JobRunnerBackEnd):
    """
    Runner for running jobs on LSF using a particular interface called jbsub.
    """

    def __init__(self, jbsub_params: dict, log_dir: Path, job_name: str):
        self._jobid = ""
        self._jbsub_prefix = self.build_jbsub_cmd(jbsub_params, log_dir, job_name)

    @staticmethod
    def build_jbsub_cmd(cluster_params: dict, log_dir: Path, job_name="") -> list[str]:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT-%H-%M-%SZ")
        
        jbsub = ["jbsub"]
        jbsub.extend(["-q", cluster_params["queue"]])
        jbsub.extend(["-cores", f"{cluster_params['cpus']}+{cluster_params['gpus']}"])
        jbsub.extend(["-mem", f"{cluster_params['mem_gb']}G"])
        if cluster_params["require"]:
            jbsub.extend(["-require", cluster_params["require"]])
        jbsub.extend(["-out", str(log_dir / f"{timestamp}_{job_name}_job_%J_stdout.log")])  # %J is the jobid
        jbsub.extend(["-err", str(log_dir / f"{timestamp}_{job_name}_job_%J_stderr.log")])  # %J is the jobid
        return jbsub
        
    def _submit_job(self, cmd: list[str], verbose=True):
        """Submit a job in a cluster or a background process.

        :param cmd: List of strings that, when concatenated, form a linux command, e.g. ["echo", "'hello world'"]
        :param verbose: Print additional data about the job, defaults to True
        """
        self._jobid, _ = jbsub_tools.submit_job(
            jbsub_prefix=self._jbsub_prefix,
            cmd=cmd,
            verbose=verbose
        )

    def _wait_for_job(self, verbose=True):
        """Block until the submitted running job completes.

        :param verbose: Print periodic information about the job status, defaults to True
        """
        jbsub_tools.wait_for_job(self._jobid, verbose=True)


class JobRunner:
    """
    Run jobs in a variety of job runners
    """

    def __init__(self, job_params: dict, log_dir: Path, job_name: str):
        job_runner = job_params["job_runner"]
        if job_runner == "jbsub": 
            self.backend = JbsubJobRunner(job_params, log_dir, job_name)
        elif job_runner == "tmux":
            self.backend = TmuxJobRunner(job_params, log_dir, job_name)
        else:
            raise NotImplementedError(f"No support for job runner: {job_runner}")

    def run_cmd_in_background(self, cmd: list[Any], verbose=True):
        """Submit a job as a background process.

        :param cmd: List of string-like objects that, when concatenated, 
                    form a linux command, e.g. ["echo", "'hello world'"]
        :param verbose: Print additional data about the job, defaults to True
        """
        cmd = [str(c) for c in cmd]
        self.backend._submit_job(cmd, verbose=verbose)

    def run_cmd_and_block(self, cmd: list[Any], verbose=True):
        """Submit a job as a process and block until it is complete.

        (This is most useful if the job needs to run on a compute node in a cluster.)

        :param cmd: List of string-like objects that, when concatenated, 
                    form a linux command, e.g. ["echo", "'hello world'"]
        :param verbose: Print additional data about the job, defaults to True
        """
        cmd = [str(c) for c in cmd]
        self.backend._submit_job(cmd, verbose=verbose)
        self.backend._wait_for_job(verbose=verbose)