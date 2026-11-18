"""Tests for hd_laya.bench: engine-agnostic benchmark runner (T7)."""
from hd_laya.bench import BenchmarkItem, run_benchmark


class FakeAgent:
    def __init__(self, mode="correct"):
        self.mode = mode

    def predict(self, state, questions):
        qid = list(questions.keys())[0]
        crit = questions[qid]["criteria"]
        labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        if self.mode == "correct":
            # predict the label embedded in the state
            pred = state.split(":")[1] if ":" in str(state) else labels[0]
            if pred not in labels:
                pred = labels[0]
        elif self.mode == "refuse":
            pred = None
        else:
            pred = labels[0]
        ans = {"choice": pred, "score": 1.0, "confidence": 0.9, "scores": {l: (1.0 if l == pred else 0.1) for l in labels}}
        if self.mode == "refuse":
            ans["refused"] = True
            ans["choice"] = "REFUSE"
        return {"answers": {qid: ans}, "usage": {}}


def test_bench_runs_and_computes_accuracy():
    items = [BenchmarkItem(state=f"s:{i % 5}", expected=str(i % 5)) for i in range(50)]
    def mq(item):
        labels = [str(j) for j in range(5)]
        return {"q": {"type": "choice", "criteria": {l: l for l in labels}}}
    res = run_benchmark(FakeAgent("correct"), items, mq)
    assert res["summary"]["n"] == 50
    assert res["summary"]["accuracy"] == 1.0
    assert res["summary"]["refusal_rate"] == 0.0
    assert res["summary"]["accuracy_on_answered"] == 1.0
    assert len(res["rows"]) == 50
    assert all(r["correct"] for r in res["rows"])


def test_bench_counts_refusals():
    items = [BenchmarkItem(state=f"s:{i}", expected=str(i % 3)) for i in range(10)]
    def mq(item):
        labels = [str(j) for j in range(3)]
        return {"q": {"type": "choice", "criteria": {l: l for l in labels}}}
    res = run_benchmark(FakeAgent("refuse"), items, mq)
    assert res["summary"]["refusal_rate"] == 1.0
    assert res["summary"]["answered"] == 0
    assert res["summary"]["accuracy"] == 0.0
    assert all(r["refused"] for r in res["rows"])


def test_bench_wrong_predictions_lower_accuracy():
    items = [BenchmarkItem(state=f"s:{i}", expected=str(i % 4)) for i in range(20)]
    def mq(item):
        labels = [str(j) for j in range(4)]
        return {"q": {"type": "choice", "criteria": {l: l for l in labels}}}
    # "wrong" mode predicts labels[0] always
    fa = FakeAgent("wrong")
    res = run_benchmark(fa, items, mq)
    # only items whose expected == "0" are correct
    expected_correct = sum(1 for i in range(20) if i % 4 == 0)
    assert res["summary"]["correct" if False else "n"] == 20
    assert abs(res["summary"]["accuracy"] - expected_correct / 20) < 1e-9
