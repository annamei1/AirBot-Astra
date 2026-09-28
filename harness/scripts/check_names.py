"""
Catch undefined module-level names before they reach the robot.

`ast.parse` and `import` both passed on a `skills.py` that raised NameError the moment the model
called follow_path, because the name was missing from a line that only runs inside a function. The
model hit it three times on the whiteboard, correctly diagnosed it as a broken tool rather than its
own mistake, gave up and put the eraser back — which is the right behaviour and still a wasted run.

    python -m harness.scripts.check_names

This is not a type checker. It answers one question: does every global name a function body reads
actually exist in that module, or come from a builtin, an import, or an enclosing scope? That is the
class of bug an edit that adds a call before its import produces, and nothing else here catches it.
"""
import ast
import builtins
import pathlib
import sys

ROOTS = ["harness"]


class Scope(ast.NodeVisitor):
    def __init__(self, module_names: set):
        self.module = module_names
        self.problems = []
        self.stack = []                      # a scope's local names, innermost last

    # --- things that bind a name ---
    def _bind(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                self.stack[-1].add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    self.stack[-1].add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                self.stack[-1].add(n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if n is not node:
                    self.stack[-1].add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                self.stack[-1].update(n.names)

    def visit_FunctionDef(self, node):
        local = {a.arg for a in
                 node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
        for extra in (node.args.vararg, node.args.kwarg):
            if extra:
                local.add(extra.arg)
        self.stack.append(local)
        self._bind(node)
        for child in node.body:
            self.generic_visit(child)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(set())
        self._bind(node)
        for child in node.body:
            self.generic_visit(child)
        self.stack.pop()

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load) or not self.stack:
            return
        name = node.id
        if (name in self.module or hasattr(builtins, name)
                or any(name in s for s in self.stack)):
            return
        self.problems.append((node.lineno, name))


def module_level_names(tree: ast.AST) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)                   # comprehension and lambda params, conservatively
    return out


def main() -> int:
    files = sorted(f for r in ROOTS for f in pathlib.Path(r).rglob("*.py"))
    bad = 0
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            print(f"{f}:{e.lineno}: SYNTAX {e.msg}")
            bad += 1
            continue
        v = Scope(module_level_names(tree))
        v.visit(tree)
        for line, name in v.problems:
            print(f"{f}:{line}: undefined name '{name}'")
            bad += 1
    print(f"\n{len(files)} files, {bad} problem{'' if bad == 1 else 's'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
