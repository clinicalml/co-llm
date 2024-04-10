import json
import os
import re
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import datasets
import evaluate
import fire
import numpy as np
import pandas as pd
import shortuuid
from tqdm import tqdm

exact_match = evaluate.load("exact_match")
rouge_metric = evaluate.load("rouge")


# The code is based on https://github.com/zhangir-azerbayev/MetaMath/blob/main/util.py
class MathEvaluator:
    @staticmethod
    def last_boxed_only(sample):
        q, a = sample
        a = MathEvaluator.last_boxed_only_string(a)
        if a == None:
            return None
        return (q, a)

    @staticmethod
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    @staticmethod
    def only_until_first_boxed_from_tokens(string, tokens):
        idx = string.find("\\boxed")
        if idx < 0:
            idx = string.find("\\fbox")
            if idx < 0:
                return None

        cum_length = 0
        for i, t in enumerate(tokens):
            cum_length += len(t)
            if cum_length >= idx:
                break

        return tokens[:i]

    @staticmethod
    def clean_numbers(sample):
        if not sample:
            return None
        new_sample = list()
        for s in sample:
            new_sample.append(MathEvaluator._clean_numbers(s))

        return tuple(new_sample)

    @staticmethod
    def _clean_numbers(string):
        """
        Clean Numbers in the given string

        >>> _clean_numbers(None, "Hello 123")
        'Hello 123'
        >>> _clean_numbers(None, "Hello 1234")
        'Hello 1,234'
        >>> _clean_numbers(None, "Hello 1234324asdasd")
        'Hello 1,234,324asdasd'
        """
        num_prev_digits = 0
        new_string = ""
        for i, c in enumerate(string):
            # isdigit() doesnt work here because of weird unicode chars.
            if c in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}:
                num_prev_digits += 1
            else:
                if num_prev_digits > 3:
                    # Some fixing
                    string_number = new_string[-num_prev_digits:]
                    new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
                num_prev_digits = 0
            new_string += c

        if num_prev_digits > 3:
            # Some fixing
            string_number = new_string[-num_prev_digits:]
            new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

        return new_string

    @staticmethod
    def fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    @staticmethod
    def fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except AssertionError:
            return string

    @staticmethod
    def remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    @staticmethod
    def fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    @staticmethod
    def strip_string(string):
        # linebreaks
        string = string.replace("\n", "")

        # remove inverse spaces
        string = string.replace("\\!", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = MathEvaluator.remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = MathEvaluator.fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = MathEvaluator.fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = MathEvaluator.fix_a_slash_b(string)

        return string

    @staticmethod
    def is_equiv(str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = MathEvaluator.strip_string(str1)
            ss2 = MathEvaluator.strip_string(str2)
            # pdb.set_trace()
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

    @staticmethod
    def remove_boxed(s):
        left = "\\boxed{"
        # try:
        #     assert s[: len(left)] == left
        #     assert s[-1] == "}"
        #     return s[len(left) : -1]
        # except:
        #     return None
        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
        else:
            return s

    @staticmethod
    def remove_text(s):
        left = "\\text{"

        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
        else:
            return s

    @staticmethod
    def remove_quotation(s):
        left = '"'

        if s[: len(left)] == left and s[-1] == '"':
            return s[len(left) : -1]
        else:
            return s

    def __init__(
        self,
        remove_dollar_sign=True,
        answer_start_str="the final answer is",
    ) -> None:
        self.invalid_outputs = []
        self.remove_dollar_sign = remove_dollar_sign
        self.answer_start_str = answer_start_str

    def process_results(
        self,
        doc,
        completion,
        answer,
        verbose,
    ) -> Tuple[bool, str, str]:
        answer = self.extract_gold_answer(answer)
        split_ans = completion.split(self.answer_start_str)
        if len(split_ans) > 1:
            ans = split_ans[-1]
            match = re.search(r"\$(.*?)\$", ans)
            if match:
                ans = match.group(1)
            extract_ans_temp = ans.split(".\n")[0]
            extract_ans_temp = extract_ans_temp.strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp

            if self.remove_dollar_sign:
                extract_ans = extract_ans.strip().replace("$", "")

            extract_ans = self.remove_text(self.remove_boxed(extract_ans))
            extract_ans = self.remove_quotation(
                extract_ans
            )  # there are cases like generations are in the format of `"even"``

            # print(extract_ans, answer)
            if self.is_equiv(extract_ans, answer, verbose=verbose):
                return True, extract_ans, answer
            else:
                return False, extract_ans, answer
        else:
            temp = {"question": doc, "output": completion, "answer": answer}
            self.invalid_outputs.append(temp)
            return False, None, answer

    def extract_gold_answer(self, gold_answer: str) -> str:
        return self.remove_text(self.remove_boxed(self.last_boxed_only_string(gold_answer)))


# This is based on the official documentation of the BioASQ dataset
class BioASQEval:

    @staticmethod
    def expand_acronym(s):
        pattern = re.compile(r"(\w+ \((\w+)\))")
        match = pattern.search(s)
        if not match:
            return [s]

        expanded = s[: match.start(2) - 1] + s[match.end(2) + 2 :]
        acronym = match.group(2) + s[match.end() :]
        return [s, expanded, acronym]

    @staticmethod
    def eval_yesno(gold: Dict[str, Any], pred: Dict[str, Any]):
        gold: str = gold["exact_answer"][0][0]
        pred: str = pred["exact_answer"]

        if pred is None:
            return {"accuracy": None}
        pred = pred.strip().split("\n")[0]

        return {
            "accuracy": 1 if pred == gold else 0,
        }

    @staticmethod
    def eval_list(gold: Dict[str, Any], pred: Dict[str, Any]):
        gold: List[List[str]] = gold["exact_answer"]
        pred: List[str] = pred["exact_answer"]

        if pred is None:
            return {"precision": None, "recall": None, "f1": None}

        # print(pred)
        transformed_pred = []
        for individual_pred in pred:
            transformed_pred.extend(BioASQEval.expand_acronym(individual_pred))

        pred = transformed_pred
        # print(pred)

        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        # Convert pred list to a set for O(1) lookup
        pred_set = set(pred)

        for gold_synonyms in gold:
            # Convert synonyms list to set
            synonym_set = set(gold_synonyms)

            # Count true positives (intersection between both sets)
            # tp += len(synonym_set & pred_set)

            if len(synonym_set & pred_set) > 0:
                tp += 1
            else:
                fn += 1

            # Count false negatives for this entity as synonyms not found in pred
            # fn += len(synonym_set & pred_set)

        # All elements in pred that are not in any gold synonym list are false positives
        fp = len(pred_set) - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def eval_factoid(
        gold: Dict[str, Any],
        pred: Dict[str, Any],
    ):
        gold: List[List[str]] = gold["exact_answer"]
        pred: List[str] = pred["exact_answer"]

        if pred is None or len(pred) == 0:
            return {"strict_accuracy": None, "lenient_accuracy": None, "mrr": None}

        # Strict Accuracy
        first_pred = pred[0]
        all_answers = set(sum(gold, []))

        # strict_accuracy = 1 if first_pred in all_answers else 0
        strict_accuracy = 1 if any([first_pred in syns for syns in all_answers]) else 0
        # if the prediction is just an exact substring of the target entry, we consider it correct.
        # For example, if the target is "X chromosome  (Xp21)" and the prediction is "X chromosome",
        # this is correct

        # Lenient Accuracy
        allowed_answers = pred[:5]  # same as the paper, we allow the top 5 predictions
        lenient_accuracy = 1 if any([any([pred in syns for syns in all_answers]) for pred in allowed_answers]) else 0

        # MRR
        mrr = 0
        for i, pred in enumerate(allowed_answers):
            if any([pred in syns for syns in all_answers]):
                mrr = 1 / (i + 1)

        return {
            "strict_accuracy": strict_accuracy,
            "lenient_accuracy": lenient_accuracy,
            "mrr": mrr,
        }

    @staticmethod
    def eval_summary(gold: Dict[str, Any], pred: Dict[str, Any]):
        assert pred["ideal_answer"] is not None, "Ideal answer cannot be None"

        references: List[List[str]] = [gold["ideal_answer"]]  # Multiple references to a target
        predictions: List[str] = [pred["ideal_answer"]]

        rouge_score = rouge_metric.compute(predictions=predictions, references=references, use_aggregator=True)
        return rouge_score

    @staticmethod
    def eval_ideal_answer(gold: Dict[str, Any], pred: Dict[str, Any]):
        references: List[List[str]] = gold["ideal_answer"]  # Multiple references to a target
        predictions: List[str] = [ele["ideal_answer"] for ele in pred]
        rouge_score = rouge_metric.compute(predictions=predictions, references=references, use_aggregator=True)
        rouge_score = {f"ideal_answer_{k}": v for k, v in rouge_score.items()}
        rouge_score["ideal_answer_aggregated"] = sum(rouge_score.values()) / len(rouge_score)
        return rouge_score

    def __init__(self):
        self.invalid_generations = []

    @staticmethod
    def parse_list_like(str):

        if str.startswith(":"):
            str = str[1:]

        if str and not str[0].isspace():
            # Presumably in this case, the answer is just a single string
            return [str.strip()]
        else:
            str = str.strip()
            if str.startswith("-"):
                pattern = re.compile(r"^-\s(.*)", re.MULTILINE)
                # Find all matches and create a list
                items = pattern.findall(str)
            elif str.startswith("1)"):
                pattern = re.compile(r"^\d+\)\s(.*)", re.MULTILINE)
                # Find all matches and create a list
                items = pattern.findall(str)
            else:
                items = [str.strip()]
        return items

    @staticmethod
    def parse_yesno(str):
        if "yes" in str.lower():
            return "yes"
        elif "no" in str.lower():
            return "no"

    def parse_generation(
        self,
        model_generation: str,
        qtype: str,
        answer_begin_str: str = "The final answer is",
        strip_answer: bool = False,
    ):
        if strip_answer:
            model_generation = model_generation.strip()
            model_generation = model_generation.split("Question: ")[0]

        split_ans = model_generation.split(answer_begin_str)

        if len(split_ans) <= 1:
            if qtype != "summary":
                self.invalid_generations.append(model_generation)  # TODO
            return {"exact_answer": None, "ideal_answer": model_generation.strip()}
        else:
            if qtype == "summary":
                self.invalid_generations.append(model_generation)  # TODO

            ideal_answer = split_ans[0].strip()
            exact_answer = split_ans[1]  # .strip()

            if qtype in ["factoid", "list"]:
                exact_answer = self.parse_list_like(exact_answer)
            elif qtype == "yesno":
                exact_answer = self.parse_yesno(exact_answer)
            return {"exact_answer": exact_answer, "ideal_answer": ideal_answer}

    def run_eval(
        self,
        orig_data,
        pred_data,
        separate_bad_format=False,
        answer_begin_str: str = "The final answer is:",
        strip_answer: bool = False,
    ):
        all_results = {}
        all_parsed_preds = []
        for qtype in ["factoid", "list", "yesno", "summary"]:
            all_results[qtype] = []
            eval_func = getattr(self, f"eval_{qtype}")

            for idx in range(len(orig_data)):
                if orig_data[idx]["type"] != qtype:
                    continue
                parsed_pred = self.parse_generation(
                    pred_data[idx]["generated_text"],
                    orig_data[idx]["type"],
                    answer_begin_str=answer_begin_str,
                    strip_answer=strip_answer,
                )
                # print("qtype:", qtype)
                # print("Orig data:", pred_data[idx]["generated_text"])
                # print("Parsed pred:", parsed_pred)
                # print("===" * 10)

                all_parsed_preds.append(parsed_pred)
                result = eval_func(orig_data[idx], parsed_pred)
                all_results[qtype].append(result)

        task_specific_metrics = self.calculate_metrics(all_results, separate_bad_format=separate_bad_format)
        overall_ideal_answer_metrics = self.eval_ideal_answer(orig_data, all_parsed_preds)
        return {**task_specific_metrics, **overall_ideal_answer_metrics}

    def calculate_metrics(self, result, separate_bad_format=False):
        if separate_bad_format:
            raise NotImplementedError
        else:
            all_metrics = pd.concat(
                [
                    pd.DataFrame(result["factoid"]).fillna(0).mean(),
                    pd.DataFrame(result["list"]).fillna(0).mean(),
                    pd.DataFrame(result["yesno"]).fillna(0).mean(),
                    pd.DataFrame(result["summary"]).fillna(0).mean(),
                ]
            )

            all_average = all_metrics.mean()
            selected_average = all_metrics[["strict_accuracy", "f1", "accuracy", "rouge2"]].mean()
            selected_average2 = all_metrics[["mrr", "f1", "accuracy", "rouge2"]].mean()

            all_metrics = all_metrics.to_dict()
            all_metrics["aggregated"] = all_average
            all_metrics["average"] = selected_average
            all_metrics["average2"] = selected_average2

        return all_metrics


def _add_deferral_probs(pred_data, result):

    # skip for empty predictions
    pred_data = pred_data.filter(lambda x: x["deferral_prob"] is not None)

    if "def_threshold" in pred_data.column_names:
        def_rate = np.array(
            [(np.array(ele["deferral_prob"]) > ele["def_threshold"]).mean() for ele in pred_data]
        ).mean()
        th = pred_data["def_threshold"][0]
        avg_deferral_prob = np.array([np.array(ele["deferral_prob"]).mean() for ele in pred_data]).mean()
    else:
        def_rate = np.array([np.array(ele["deferral_prob"]).mean() for ele in pred_data]).mean()
        avg_deferral_prob = None
        # For ablation studies, where we manually control the deferral
        th = None

    result["threshold"] = th
    result["def_rate"] = def_rate
    result["avg_deferral_prob"] = avg_deferral_prob


def eval_gsm8k(pred_data, orig_data=None, version="v1"):
    if "train" in pred_data.column_names:
        pred_data = pred_data["train"]

    predictions = []
    for idx, output in enumerate(pred_data["generated_text"]):
        # replace numbers like `x,xxx` with `xxxx`
        if output is None:
            print(f"empty prediction for {idx}")
            predictions.append("")
            continue

        if version == "v1":
            output = re.sub(r"(\d),(\d)", r"\1\2", output)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        elif version == "v2":
            # Not useful
            # numbers = re.findall(r'####\s(.*?)\n', output)
            # if numbers:
            #     numbers = re.sub(r"(\d),(\d)", r"\1\2", numbers[0])
            #     numbers = re.findall(r"[-+]?\d*\.\d+|\d+", numbers)
            numbers = [ele for ele in output.split("\n") if ele.startswith("####")]
            if numbers:
                numbers = re.sub(r"(\d),(\d)", r"\1\2", numbers[0])
                numbers = re.findall(r"####\s[-+]?\d*\.\d+|\d+", numbers)

        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)

    em_score = exact_match.compute(
        predictions=predictions,
        references=orig_data["target"],
        ignore_case=True,
        ignore_punctuation=True,
    )["exact_match"]

    result = {"exact_match": em_score}

    if "deferral_prob" in pred_data.column_names:
        _add_deferral_probs(pred_data, result)
    return result


def eval_bbh(pred_data, orig_data):
    if "train" in pred_data.column_names:
        pred_data = pred_data["train"]

    all_tasks = set(orig_data["task_name"])

    performance = {}
    for task_name in all_tasks:
        predictions = []
        targets = []
        for example, output in zip(orig_data, pred_data["generated_text"]):
            if example["task_name"] != task_name:
                continue
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", output)

            if extracted_answer:
                prediction = extracted_answer.group(1).strip()
            else:
                prediction = output.strip()
            predictions.append(prediction)
            targets.append(example["target"])

        performance[task_name] = exact_match.compute(
            predictions=predictions,
            references=targets,
            ignore_case=True,
            ignore_punctuation=True,
        )["exact_match"]

    em_score = sum(performance.values()) / len(performance)

    result = {"exact_match": em_score, "performance": performance}
    if "deferral_prob" in pred_data.column_names:
        def_rate = np.array(
            [(np.array(ele["deferral_prob"]) > ele["def_threshold"]).mean() for ele in pred_data]
        ).mean()

        th = pred_data["def_threshold"][0]

        result = {"threshold": th, "def_rate": def_rate, **result}

    return result


def eval_tydiqa(pred_data, orig_data):
    if "train" in pred_data.column_names:
        pred_data = pred_data["train"]

    metric = evaluate.load("squad")
    data_languages = set(orig_data["lang"])
    target = [ele.strip().split("\n\n")[0] if ele else "" for ele in pred_data["generated_text"]]

    eval_scores = {}
    for lang in data_languages:
        lang_predictions = [
            {"id": example["id"], "prediction_text": output}
            for example, output in zip(orig_data, target)
            if example["lang"] == lang
        ]
        lang_references = [
            {"id": example["id"], "answers": example["answers"]} for example in orig_data if example["lang"] == lang
        ]
        eval_scores[lang] = metric.compute(predictions=lang_predictions, references=lang_references)

    eval_scores["average"] = {
        metric: np.mean([scores[metric] for scores in eval_scores.values()]) for metric in ["exact_match", "f1"]
    }

    def_rate = np.array([(np.array(ele["deferral_prob"]) > ele["def_threshold"]).mean() for ele in pred_data]).mean()
    th = pred_data["def_threshold"][0]

    return {
        "threshold": th,
        "def_rate": def_rate,
        "exact_match": eval_scores["average"]["exact_match"],
        "f1": eval_scores["average"]["f1"],
        "performance": eval_scores,
    }


def eval_math(
    pred_data,
    orig_data,
    remove_dollar_sign=True,
    answer_start_str="the final answer is",
    verbose=False,
):
    if "train" in pred_data.column_names:
        pred_data = pred_data["train"]

    math_evaluator = MathEvaluator(
        remove_dollar_sign=remove_dollar_sign,
        answer_start_str=answer_start_str,
    )

    predictions = []

    for i in range(len(pred_data)):
        is_matched, prediction, gold_answer = math_evaluator.process_results(
            doc=orig_data[i]["question"],
            completion=pred_data[i]["generated_text"],
            answer=orig_data[i]["target"],
            verbose=verbose,
        )
        predictions.append(
            [
                is_matched,
                prediction,
                gold_answer,
                orig_data[i]["level"],
                orig_data[i]["type"],
            ]
        )

    pred_df = pd.DataFrame(
        predictions,
        columns=["is_matched", "prediction", "gold_answer", "level", "type"],
    )
    performances = pred_df.groupby(["level", "type"])["is_matched"].mean().to_dict()

    result = {
        "exact_match": np.mean([ele[0] for ele in predictions]),
        "performances": performances,
        "predictions": predictions,
    }
    if "deferral_prob" in pred_data.column_names:
        _add_deferral_probs(pred_data, result)
    return result


def eval_mmlu(pred_data, orig_data, verbose=False):
    if "train" in pred_data.column_names:
        pred_data = pred_data["train"]

    predictions = []
    for i in range(len(pred_data)):
        generated_answer = pred_data[i]["generated_text"]
        gold_answer = orig_data[i]["target"].lower()

        if "Question: " in generated_answer:
            generated_answer = generated_answer.split("Question: ")[0]
        if "So the answer is" in generated_answer:
            answer_span = generated_answer.split("So the answer is")[1]
            match = re.search(r"\(([A-Da-d])\)", answer_span)
            if match:
                predicted_answer = match.group(1)
            else:
                if verbose:
                    print(generated_answer)
        else:
            if verbose:
                print(generated_answer)

        predicted_answer = predicted_answer.lower()
        predictions.append([orig_data[i]["subset"], gold_answer, predicted_answer])

    pred_df = pd.DataFrame(predictions, columns=["subset", "label", "pred"])
    result = {
        "exact_match": (pred_df["label"] == pred_df["pred"]).mean(),
        "performance": pred_df.groupby("subset").apply(lambda gp: (gp["label"] == gp["pred"]).mean()).to_dict(),
    }

    if "deferral_prob" in pred_data.column_names:
        _add_deferral_probs(pred_data, result)
    return result


def eval_bioasq(
    pred_data,
    orig_data,
    answer_begin_str: str = "The final answer is:",
    strip_answer=False,
):
    bioasq_eval = BioASQEval()

    if "train" in pred_data.column_names:
        pred_data = pred_data["train"]

    result = bioasq_eval.run_eval(
        orig_data,
        pred_data,
        separate_bad_format=False,
        answer_begin_str=answer_begin_str,
        strip_answer=strip_answer,
    )

    if "deferral_prob" in pred_data.column_names:
        _add_deferral_probs(pred_data, result)
    return result


def process_mtbench(pred_data_path, model_name=""):

    ds = datasets.load_from_disk(os.path.join(pred_data_path, "dataset"))
    ds2 = datasets.load_dataset("json", data_files=os.path.join(pred_data_path, "generations.json"))["train"]

    # Cache the files
    ans_json = []
    for idx, ex in enumerate(ds2):
        ans_json.append(
            {
                "question_id": ds[idx]["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "choices": [{"index": 0, "turns": [ex["generated_text"]]}],
                "tstamp": time.time(),
            }
        )
    with open(os.path.join(pred_data_path, "mt_bench_format.jsonl"), "w") as f:
        for line in ans_json:
            f.write(json.dumps(line) + "\n")

    # Then after this, we need to switch to the mt_bench evalcode


def eval_mtbench(pred_data_path, orig_df):

    # Obtaining scores
    input_file = os.path.join(pred_data_path, "model_judgment/gpt-4_single.jsonl")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["question_id", "score", "judge"]].copy()
    df["judge"] = df["judge"].apply(lambda ele: ele[1])
    df = df[df["score"] != -1]

    result = {
        "mtbench_score": df["score"].mean(),
        "performance": df,
    }

    for _, gp in df.groupby("judge"):
        result["judge_" + gp["judge"].iloc[0]] = gp["score"].mean()

    df = df.merge(orig_df)
    for _, gp in df.groupby("category"):
        result["category_" + gp["category"].iloc[0]] = gp["score"].mean()

    # if "train" in pred_data.column_names:
    pred_data = datasets.load_dataset("json", data_files=os.path.join(pred_data_path, "generations.json"))["train"]
    if "deferral_prob" in pred_data.column_names:
        _add_deferral_probs(pred_data, result)

    return result


EVALUATORS = {
    "gsm8k": eval_gsm8k,
    "bbh": eval_bbh,
    "tydiqa": eval_tydiqa,
    "math": eval_math,
    "mmlu": eval_mmlu,
    "bioasq": eval_bioasq,
    "mtbench": eval_mtbench,
}

TARGET_METRIC = {
    "gsm8k": "exact_match",
    "math": "exact_match",
}


def eval(
    task_name,
    orig_data_path,
    pred_data_path,
    **kwargs,
):
    evaluator = EVALUATORS[task_name]
    orig_data = datasets.load_from_disk(orig_data_path)
    pred_data = datasets.load_dataset("json", data_files=pred_data_path)
    result = evaluator(pred_data, orig_data, **kwargs)

    pred_data_folder = Path(pred_data_path).parent

    with open(os.path.join(pred_data_folder, "eval.json"), "w") as f:
        json.dump(result, f, indent=4)

    return result


def eval_folder(
    task_name,
    orig_data_path,
    pred_data_folder,
    **kwargs,
):
    evaluator = EVALUATORS[task_name]
    orig_data = datasets.load_from_disk(orig_data_path)

    # print(kwargs)
    all_results = []
    for pred_data_path in tqdm(sorted(glob(pred_data_folder + "/defer-*/generations.json"))):

        pred_data = datasets.load_dataset("json", data_files=pred_data_path)
        result = evaluator(pred_data, orig_data, **kwargs)
        all_results.append(result)

    if task_name == "math":
        # fix the dict issue
        for result in all_results:
            result["performances"] = [[*key, val] for key, val in result["performances"].items()]

    print(all_results)
    with open(os.path.join(pred_data_folder, "eval.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    if os.path.exists(pred_data_folder + "/deferral_threshold.csv"):

        df_th = pd.read_csv(pred_data_folder + "/deferral_threshold.csv")
        df_acc = pd.DataFrame(all_results).rename(columns={"threshold": "deferral_threshold"})
        df_merged = pd.merge_asof(
            df_th.sort_values("deferral_threshold"),
            df_acc.sort_values("deferral_threshold"),
            on="deferral_threshold",
            tolerance=1e-5,
            direction="nearest",
        ).sort_values(TARGET_METRIC[task_name], ascending=False)

        df_merged.to_csv(pred_data_folder + "/eval_with_deferral_threshold.csv", index=False)

    return all_results


if __name__ == "__main__":
    fire.Fire(
        {
            "eval": eval,
            "eval_folder": eval_folder,
        }
    )
