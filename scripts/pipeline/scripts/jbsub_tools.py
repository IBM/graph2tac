import os
import re
import time
from typing import Any

def submit_job(jbsub_prefix: list[Any], cmd: list[Any], verbose=True) -> tuple[str, str]:
    full_cmd = " ".join(str(c) for c in jbsub_prefix + cmd)
    if verbose:
        print(full_cmd)
    # TODO: This isn't capturing the stderr if the command has a mistake
    jbsub_output = os.popen(full_cmd).read()
    
    
    # extract jobid from jbsub output
    m = re.search(r"Job <(\d*)> is submitted", jbsub_output)
    assert m is not None, "Could not submit job.  Here is the jbsub output:\n" + jbsub_output 
    jobid = m[1]

    return jobid, jbsub_output

def job_info(jobid: str) -> tuple[str, str]:
    """Get job info.  (Work in progress)

    :return: full job info, job state (PEND, AVAIL, PSUSP, USUSP, RUN, SSUSP, DONE, EXIT)
    :rtype: tuple[str, str]
    """

    info = os.popen(f"jbinfo -long-long {jobid}").read()

    # get status
    m = re.search(r"Status <(\w*)>", info)
    assert m is not None, info
    return info, m[1]  # full_info, status

def wait_for_job(jobid: str, verbose=True, max_wait_sec=610) -> tuple[str, str]:
    prev_status = ""
    info_lines_printed = 0
    while True:
        # wait until the status has changed
        # use fibonacci numbers as amount of time to wait until reach max
        fib0 = 0
        fib1 = 1
        while True:
            time.sleep(min(fib0, max_wait_sec))
            fib1, fib0 = fib0 + fib1, fib1

            info, status = job_info(jobid)
            if status != prev_status:
                break
        
        if verbose:
            print()
            print("Job status:", status) 
            # find first line in info without `<`
            info_lines = info.split("\n")
            for line in info_lines[info_lines_printed:]:
                if "<" in line:
                    print(line)
                    info_lines_printed += 1
                else:
                    break  # stop at first line without `<...>` pattern

        if status in ["DONE", "EXIT"]:
            break
        else:
            prev_status = status
    
    if verbose:
        # wait one second to make sure I have all the info
        time.sleep(1)
        info, status = job_info(jobid)
        print("\n".join(info.split("\n")[info_lines_printed:]))
    
    return info, status

def wait_for_jobs(jobids: list[str], verbose=True, max_wait_sec=610) -> tuple[list[str], list[str]]:
    prev_statuses = ["" for _ in jobids]
    info_lines_printed = [0 for _ in jobids]
    while True:
        # wait until the status has changed
        # use fibonacci numbers as amount of time to wait until reach max
        fib0 = 0
        fib1 = 1
        while True:
            time.sleep(min(fib0, max_wait_sec))
            fib1, fib0 = fib0 + fib1, fib1

            infos = []
            statuses = []
            changed = []
            for jobid, prev_status in zip(jobids, prev_status):
                info, status = job_info(jobid)
                infos.append(info)
                statuses.append(status)
                changed.append(status != prev_status)
            
            if any(changed):
                break
        
        if verbose:
            new_info_lines_printed = []
            for jobid, info, status, info_line_printed, changed in zip(jobids, infos, statuses,info_lines_printed, changed):
                if not changed:
                    new_info_lines_printed.append(info_line_printed)
                    continue
                print()
                print(f"Job {jobid} status:", status)
                # find first line in info without `<`
                info_lines = info.split("\n")
                new_info_line_printed = info_line_printed 
                for line in info_lines[info_line_printed:]:
                    if "<" in line:
                        print(line)
                        new_info_line_printed += 1
                    else:
                        break  # stop at first line without `<...>` pattern
                new_info_lines_printed.append(new_info_line_printed)
            info_lines_printed = new_info_lines_printed
        if all(s in ["DONE", "EXIT"] for s in statuses):
            break
        else:
            prev_statuses = statuses
    
    
    # wait one second to make sure I have all the info
    time.sleep(1)
    infos = []
    statuses = []
    for jobid, info_line_printed in zip(jobids, info_lines_printed):
        info, status = job_info(jobid)
        infos.append(info)
        statuses.append(status)
        if verbose:
            print("\n".join(info.split("\n")[info_line_printed:]))
    
    return infos, statuses


def wait_while_job_is(jobid: str, statuses: list[str], max_wait_sec=610) -> tuple[str, str]:
    # use fibonacci numbers as amount of time to wait until reach max
    fib0 = 0
    fib1 = 1
    while True:
        time.sleep(min(fib0, max_wait_sec))
        fib1, fib0 = fib0 + fib1, fib1

        info, state = job_info(jobid)
        if state not in statuses:
            return info, state