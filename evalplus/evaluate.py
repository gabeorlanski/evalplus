import json
import multiprocessing
import os
import pickle
import threading
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn
import copy
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from datetime import timezone
import time

from evalplus.codegen import run_codegen
from evalplus.config import *
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    PASS,
    FAIL,
    NOT_RAN,
    TIMEOUT,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]
NOT_RAN_RES = (NOT_RAN, [], [])
DEFAULT_FAIL = (FAIL, [], [])


def get_groundtruth(
    problems, hashcode, tasks_only_output_not_none, disable_cache=False
):

    tbegin = datetime.now(timezone.utc)
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file) and not disable_cache:
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f), (datetime.now(timezone.utc) - tbegin).total_seconds()

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected output...")
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"] if problem["plus_input"] else problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    elapsed = (datetime.now(timezone.utc) - tbegin).total_seconds()

    print(f"Expected outputs computed in {elapsed:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output, elapsed


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    plus_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    max_test_cases: Optional[int] = None,
    max_time_limit: float = None,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    start_time = datetime.now(timezone.utc)
    base_inputs = copy.deepcopy(problem["base_input"])
    run_plus = not base_only
    run_base = not plus_only
    plus_inputs = copy.deepcopy(problem.get("plus_input", []))
    base_ref_time = expected_output["base_time"]
    plus_ref_time = expected_output["plus_time"]
    if not plus_inputs:
        plus_inputs = base_inputs
    if max_test_cases is not None:
        all_inputs = base_inputs + plus_inputs
        all_outputs = expected_output["base"] + expected_output["plus"]

        all_inputs = all_inputs[:max_test_cases]
        all_outputs = all_outputs[:max_test_cases]
        ret["base"] = untrusted_check(
            dataset=dataset,
            code=solution,
            inputs=all_inputs,
            entry_point=problem["entry_point"],
            expected=all_outputs,
            atol=problem["atol"],
            ref_time=base_ref_time,
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
            max_time_limit=max_time_limit,
        )
        if max_test_cases > len(base_inputs):
            ret["plus"] = ret["base"]
        else:
            ret["plus"] = NOT_RAN_RES

        ret["elapsed"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        return ret

    if run_base:
        ret["base"] = untrusted_check(
            dataset=dataset,
            code=solution,
            inputs=base_inputs,
            entry_point=problem["entry_point"],
            expected=expected_output["base"],
            atol=problem["atol"],
            ref_time=base_ref_time,
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
            max_time_limit=max_time_limit,
        )
    else:
        ret["base"] = NOT_RAN_RES

    if run_plus:
        ret["plus"] = untrusted_check(
            dataset=dataset,
            code=solution,
            inputs=plus_inputs,
            entry_point=problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=plus_ref_time,
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
            max_time_limit=max_time_limit,
        )
    else:
        ret["plus"] = NOT_RAN_RES
    ret["elapsed"] = (datetime.now(timezone.utc) - start_time).total_seconds()
    return ret


def evaluate(
    dataset: str,
    samples: Optional[str] = None,
    base_only: bool = False,
    plus_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    output_file: Optional[str] = None,
    gguf_file: Optional[str] = None,
    disable_cache: bool = False,
    max_test_cases: Optional[int] = None,
    max_time_limit: float = None,
    **model_kwargs,
):
    if model_kwargs:
        # To suppress the warning of tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )
        samples = run_codegen(
            dataset=dataset,
            gguf_file=gguf_file,
            **model_kwargs,
        )
    assert samples is not None, "No samples provided"

    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        # legacy compatibility
        if os.path.exists(samples.replace(".jsonl", "_eval_results.json")):
            result_path = samples.replace(".jsonl", "_eval_results.json")
        else:
            result_path = samples.replace(".jsonl", ".eval_results.json")
    print(result_path)
    if output_file is not None:
        result_path = output_file

    if os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    else:
        if dataset == "humaneval":
            problems = get_human_eval_plus(
                mini=mini, noextreme=noextreme, version=version
            )
            dataset_hash = get_human_eval_plus_hash(
                mini=mini, noextreme=noextreme, version=version
            )
            expected_output, get_gt_elapsed = get_groundtruth(
                problems,
                dataset_hash,
                [],
                disable_cache=disable_cache,
            )
        elif dataset == "mbpp":
            problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
            dataset_hash = get_mbpp_plus_hash(
                mini=mini, noextreme=noextreme, version=version
            )
            expected_output, get_gt_elapsed = get_groundtruth(
                problems,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
                disable_cache=disable_cache,
            )
        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }
        print(f"Using {max_test_cases} max tests")
        print(f"Using {max_time_limit} max timeout")
        print(f"Using {n_workers} workers")
        sample_meta_map = {}
        start_time = datetime.now(timezone.utc)
        with ProcessPoolExecutor(
            max_workers=n_workers,
        ) as executor:
            futures = {}
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(load_solutions(samples)):
                task_id = sample["task_id"]
                if task_id not in problems:
                    warn(
                        f"Task {task_id} is found in the samples but not found in the dataset"
                    )
                    continue
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                remainings.add(sample["_identifier"])
                args = (
                    dataset,
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    expected_output[task_id],
                    base_only,
                    plus_only,
                    not test_details,  # fast_check
                    sample["_identifier"],
                    min_time_limit,
                    gt_time_limit_factor,
                    max_test_cases,
                    max_time_limit,
                )
                future = executor.submit(check_correctness, *args)
                futures[future] = (
                    task_id,
                    completion_id[task_id],
                    sample["_identifier"],
                )
                completion_id[task_id] += 1
                n_samples += 1
                sample_meta_map[sample["_identifier"]] = sample["meta"]

            # assert n_samples == len(remainings), "Missing problems in unfinished"
            # assert len(completion_id) == len(problems), "Missing problems in samples"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potential stucking
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            threading.Thread(target=stucking_checker).start()

            for future in tqdm(as_completed(futures), total=n_samples):

                result = future.result()
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)

        results["execution_elapsed"] = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        results["get_gt_elapsed"] = get_gt_elapsed
        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:

                def get_failed_tests(stat, details, inputs) -> List[Any]:
                    if not details:
                        return []

                    if test_details:
                        return [
                            inputs[i] for i in range(len(details)) if not details[i]
                        ]

                    # else => simply return the only and the last fail test
                    try:
                        return [inputs[len(details) - 1]]
                    except Exception as e:
                        breakpoint()
                        return []

                base_stat = NOT_RAN
                base_details = []
                if not plus_only:
                    base_stat, base_details, base_timings = res["base"]
                    base_fail_tests = get_failed_tests(
                        base_stat, base_details, problems[task_id]["base_input"]
                    )

                # initialize plus tests
                plus_stat = NOT_RAN
                plus_fail_tests = []

                # with plus tests
                if not base_only and res.get("plus", NOT_RAN_RES) != NOT_RAN_RES:
                    plus_stat, plus_details, plus_timings = res.get("plus", NOT_RAN_RES)
                    plus_fail_tests = get_failed_tests(
                        plus_stat,
                        plus_details,
                        problems[task_id]["plus_input"]
                        or problems[task_id]["base_input"],
                    )
                else:
                    plus_timings = []
                    plus_details = []
                    plus_stat = NOT_RAN
                if dataset == "mbpp":
                    base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                    plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "elapsed": res["elapsed"],
                        "base_status": base_stat,
                        "plus_status": plus_stat,
                        "base_timings": base_timings,
                        "plus_timings": plus_timings,
                        "base_details": [bool(d) for d in base_details],
                        "plus_details": [bool(d) for d in plus_details],
                        **sample_meta_map[res["_identifier"]],
                    }
                )

    # Calculate pass@k.
    total = []
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = nc = rt = 0
        for r in res:
            rt += 1
            bc += r["base_status"] == PASS
            if not base_only:
                nc += r["plus_status"] == PASS
        total.append(rt)
        base_correct.append(bc)
        if not base_only:
            new_correct.append(nc)

    total = np.array(total)
    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")
    results["pass_at_k"] = {"base": pass_at_k}

    if new_correct:
        cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")
        results["pass_at_k"]["plus"] = pass_at_k

    # save results
    if os.path.isfile(result_path) and i_just_wanna_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)


def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()
