"""**Four classes held the same idea, and all four were missing the same method.**

`torch.topk(x, 2)` hands back a values-and-indices pair. This repository had four
classes for that idea — `borch._tensor._MinMax`, `borch._ops._named`'s stamped-out
type, `borch_webgpu._base._Pair` and `borch_webgpu._ops._MinMax` — and **none of the
three hand-written ones had a `__repr__`**, so `print(x.topk(3))` showed an object
address where torch prints the numbers. Only `_named`, the one built from a table, had
it.

Giving them one immediately surfaced a second defect the address had been hiding: six
of the eight index tensors were `float32` where torch's are `int64`.

That is the cost being measured here. It is not that duplication is untidy — it is
that **a fix reaches one copy**, and the copies that are written by hand are the ones
it does not reach. The repository already says so in prose, in a dozen comments and in
`borch/_tensor.py`'s note about forty-one in-place names ("Forty-one copies are not
written by hand"). Nothing checked it.

## What is asked

For each concept below: every class holding it must offer the same methods. A class
that grows one and leaves its siblings behind fails, naming both.

**Not that there should be one class.** There are honest reasons for several — a
tv_tensor cannot subclass the binding's slotted tensor the way it subclasses the core's,
and `aminmax` names its fields `min` and `max` where the rest say `values` and
`indices`. What is not honest is four classes with four different surfaces, which is
what an unwritten rule produces.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# A concept, and the classes that hold it. The path is where the class is written;
# a class stamped out by a factory is named by the factory.
CONCEPTS = {
    "a values-and-indices pair": [
        ("borch/_tensor.py", "_MinMax"),
        ("borch_webgpu/_base.py", "_Pair"),
        ("borch_webgpu/_ops.py", "_MinMax"),
    ],
    "a tv_tensor carrying a label": [
        ("borchvision.py", "TVTensor"),
    ],
}

# Methods a concept's classes must agree about. Listed rather than compared wholesale
# because the classes legitimately differ elsewhere: `_Pair` forwards unknown
# attributes to its values and the core's does not, which is about the binding's
# proxy and not about the idea.
MUST_AGREE = {
    "a values-and-indices pair": ("__repr__", "__iter__", "__getitem__", "__len__"),
    "a tv_tensor carrying a label": ("wrap", "_metadata"),
}


def _methods(path, class_name):
    """The methods a class defines itself. Bases are not followed — the point is
    what each hand-written copy actually carries."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {b.name for b in node.body
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return None


def test_every_class_of_one_concept_offers_the_same_methods():
    """**A method added to one copy has to reach the others.**

    This is the check that would have caught the missing `__repr__`: three classes for
    one idea, and the method on none of them. It fails the same way if somebody adds
    `__eq__` to one pair class and not the rest — which is the direction the defect
    usually arrives from, since the copy being edited is the one in front of you.
    """
    missing = []
    for concept, classes in CONCEPTS.items():
        wanted = MUST_AGREE[concept]
        for path, name in classes:
            have = _methods(path, name)
            assert have is not None, (
                f"{path} has no class `{name}` — this table names a class that moved. "
                "Point the row at where it went, or drop it if the concept is gone.")
            for method in wanted:
                if method not in have:
                    missing.append(f"{path}:{name} has no `{method}`  ({concept})")
    assert not missing, (
        "classes holding one concept do not offer the same methods:\n  "
        + "\n  ".join(missing)
        + "\n\n  A fix reaches one copy. Add it to the others, or — if this one truly "
          "should not have it —\n  take the method out of `MUST_AGREE` and say why "
          "there.")


def test_the_table_names_classes_that_exist():
    """A row pointing at a class that moved is a rule nobody is enforcing.

    It reads as covered and checks nothing, which is the failure this file's own
    subject is about — the same shape as `test_korean_ceiling.py`'s note on a ceiling
    over a directory that moved.
    """
    gone = [f"{path}:{name}" for classes in CONCEPTS.values()
            for path, name in classes if _methods(path, name) is None]
    assert not gone, (
        "these classes are not where the table says:\n  " + "\n  ".join(gone))
