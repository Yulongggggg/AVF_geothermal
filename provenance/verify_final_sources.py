"""Compare final numerical functions to original source without executing a model."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "provenance/original_sources/final_geouno_20261015"


class Names(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        if node.name == "FinalGeoUNO":
            node.name = "HeadAblation"
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Public constructor defaults to the selected final variant; research default was joint64.
        if node.name == '__init__':
            for default in node.args.defaults:
                if isinstance(default, ast.Constant) and default.value == 'joint64':
                    default.value = 'resid64_p1'
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "FinalGeoUNO":
            node.id = "HeadAblation"
        return node


def definitions(path):
    tree = Names().visit(ast.parse(path.read_text()))
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }


def check():
    pairs = [
        ("model.py", "geouno/model/final.py", ["spd_solve", "HeadAblation", "load"]),
        (
            "train.py",
            "geouno/training/train.py",
            # run_stage has corrected restart handling; test_training_resume.py
            # exercises its checkpoint selection across interrupted epochs.
            ["build", "temperature_loss", "fine_lr", "admissible", "adopt_main"],
        ),
    ]
    results = []
    for original, current, functions in pairs:
        a, b = definitions(ORIGINAL / original), definitions(ROOT / current)
        for name in functions:
            results.append(dict(file=current, function=name, identical=a[name] == b[name]))
    return results


if __name__ == "__main__":
    checks = check()
    print(json.dumps(dict(checks=checks, model_execution=False), indent=2))
    raise SystemExit(int(not all(c["identical"] for c in checks)))
