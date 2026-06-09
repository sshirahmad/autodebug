"""Tests for autodebug.patch_utils.split_patch."""

from autodebug.patch_utils import split_patch, _is_test_path


def test_empty_diff_returns_empty_pair():
    assert split_patch("") == ("", "")
    assert split_patch(None) == ("", "")


def test_diff_with_only_test_file_routes_to_test_patch():
    diff = (
        "diff --git a/test/units/galaxy/test_collection.py b/test/units/galaxy/test_collection.py\n"
        "index abc..def 100644\n"
        "--- a/test/units/galaxy/test_collection.py\n"
        "+++ b/test/units/galaxy/test_collection.py\n"
        "@@ -1,2 +1,3 @@\n"
        " import pytest\n"
        "+def test_new(): assert True\n"
    )
    code, test = split_patch(diff)
    assert code == ""
    assert "test_collection.py" in test
    assert "def test_new()" in test


def test_diff_with_only_source_file_routes_to_code_patch():
    diff = (
        "diff --git a/lib/ansible/galaxy/collection.py b/lib/ansible/galaxy/collection.py\n"
        "@@ -1,2 +1,3 @@\n"
        " import os\n"
        "+import sys\n"
    )
    code, test = split_patch(diff)
    assert test == ""
    assert "collection.py" in code


def test_mixed_diff_splits_correctly():
    diff = (
        "diff --git a/lib/ansible/galaxy/collection.py b/lib/ansible/galaxy/collection.py\n"
        "@@ -1 +1 @@\n"
        "-old code\n"
        "+new code\n"
        "diff --git a/test/units/galaxy/test_collection.py b/test/units/galaxy/test_collection.py\n"
        "@@ -1 +1 @@\n"
        "-old test\n"
        "+new test\n"
    )
    code, test = split_patch(diff)
    assert "collection.py" in code and "test_collection.py" not in code
    assert "test_collection.py" in test and "lib/ansible/galaxy/collection.py" not in test
    assert "new code" in code
    assert "new test" in test


def test_is_test_path_recognizes_common_layouts():
    assert _is_test_path("test/units/galaxy/test_collection.py")
    assert _is_test_path("tests/test_foo.py")
    assert _is_test_path("pkg/foo/test_bar.py")
    assert _is_test_path("pkg/foo/bar_test.py")
    assert _is_test_path("src/testing/helpers.py")  # 'testing' dir
    assert not _is_test_path("lib/ansible/galaxy/collection.py")
    assert not _is_test_path("src/foo.py")
    assert not _is_test_path("docs/testing-guide.rst")  # word 'testing' not as path part


def test_real_ansible_manifest_patch_routes_test_to_test():
    """Smoke test against the actual ground_truth_patch shape from the dataset."""
    diff = (
        "diff --git a/lib/ansible/galaxy/collection.py b/lib/ansible/galaxy/collection.py\n"
        "index a055b08e71..856e54666f 100644\n"
        "--- a/lib/ansible/galaxy/collection.py\n"
        "+++ b/lib/ansible/galaxy/collection.py\n"
        "@@ -668,6 +668,11 @@ def verify_collections(...):\n"
        "+    raise AnsibleError(...)\n"
        "diff --git a/test/units/galaxy/test_collection.py b/test/units/galaxy/test_collection.py\n"
        "index aaa..bbb 100644\n"
        "--- a/test/units/galaxy/test_collection.py\n"
        "+++ b/test/units/galaxy/test_collection.py\n"
        "@@ -1100 +1100,5 @@\n"
        "+def test_verify_collections_no_version(...):\n"
        "+    pass\n"
    )
    code, test = split_patch(diff)
    assert "lib/ansible/galaxy/collection.py" in code
    assert "test/units/galaxy/test_collection.py" not in code
    assert "test_verify_collections_no_version" in test
    assert "raise AnsibleError" in code
