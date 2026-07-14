from pathlib import Path
import yaml

_DEFAULT_DIR = Path(__file__).parent


def load_knowledge(dir_path: str | None = None) -> str:
    base = Path(dir_path) if dir_path else _DEFAULT_DIR
    lines = []
    for yml in sorted(base.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        for rule in data.get("rules", []):
            lines.append(
                f"- {rule.get('term', '')}: {rule.get('rule', '')}"
                f" (나쁜 예: {rule.get('bad', '-')} / 좋은 예: {rule.get('good', '-')})"
            )
    return "\n".join(lines)
