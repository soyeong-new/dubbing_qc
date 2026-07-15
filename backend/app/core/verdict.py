from pathlib import Path
from typing import List, Optional
import yaml
from app.schemas import QCFinding, AxisScore, Verdict, AXES

_DEFAULT_CONFIG = Path(__file__).parent.parent / "qc_config.yaml"


def load_config(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _DEFAULT_CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def compute_axis_scores(findings: List[QCFinding], n_pairs: int, config: dict) -> List[AxisScore]:
    n_pairs = max(n_pairs, 1)
    deduction_w = config["deduction"]
    thresholds = config["mos_thresholds"]
    scores = []
    for axis in AXES:
        total = sum(
            deduction_w.get(f.severity, 0) for f in findings
            if f.axis == axis and f.finding_type == "quality"
        )
        rate = round(total / n_pairs * 100, 1)
        mos = 1
        for level in (5, 4, 3, 2):
            if rate <= thresholds[level]:
                mos = level
                break
        scores.append(AxisScore(axis=axis, mos=mos, deduction_rate=rate))
    return scores


def decide(axis_scores: List[AxisScore], findings: List[QCFinding], config: dict) -> Verdict:
    reasons = []
    high_findings = [f for f in findings if f.severity == "high"]
    min_mos = min(s.mos for s in axis_scores)
    pass_min = config["verdict"]["pass_min_mos"]
    cond_min = config["verdict"]["conditional_min_mos"]

    if high_findings:
        reasons.append(f"치명(high) 지적 {len(high_findings)}건 — 심각도 무관 즉시 반려 대상입니다.")
    for s in axis_scores:
        if s.mos < cond_min:
            reasons.append(f"{s.axis} MOS {s.mos} (감점률 {s.deduction_rate})")

    if high_findings or min_mos < cond_min:
        status = "fail"
    elif min_mos < pass_min:
        status = "conditional"
        reasons.append(f"최저 축 MOS {min_mos} — 수정 권고 후 통과 가능합니다.")
    else:
        status = "pass"
    return Verdict(status=status, axis_scores=axis_scores, reasons=reasons)
