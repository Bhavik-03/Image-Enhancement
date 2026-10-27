import cv2
import numpy as np
import pytest

from data import PairedDataset, get_split, list_pairs


def make_root(tmp_path, n=10, size=(64, 80), same=True):
    """n synthetic pairs; ref is raw itself (same=True) or inverted, so alignment is checkable."""
    rng = np.random.default_rng(0)
    for d in ("raw", "ref"):
        (tmp_path / d).mkdir()
    for i in range(n):
        img = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "raw" / f"{i}.png"), img)
        cv2.imwrite(str(tmp_path / "ref" / f"{i}.png"), img if same else 255 - img)
    return tmp_path


def test_split_is_disjoint_complete_and_reproducible(tmp_path):
    root = make_root(tmp_path, n=20)
    split = get_split(root)
    all_stems = split["train"] + split["val"] + split["test"]
    assert sorted(all_stems) == sorted(list_pairs(root))
    assert len(set(all_stems)) == 20
    assert (len(split["val"]), len(split["test"])) == (2, 2)
    assert get_split(root) == split  # read back from split.json
    (root / "split.json").unlink()
    assert get_split(root) == split  # same seed -> same split


def test_train_item_is_cropped_aligned_and_in_range(tmp_path):
    ds = PairedDataset(make_root(tmp_path), "train", crop=32)
    raw, ref = ds[0]
    assert raw.shape == ref.shape == (3, 32, 32)
    assert 0 <= raw.min() and raw.max() <= 1
    assert (raw == ref).all()  # same crop and same flips applied to both


def test_train_crop_keeps_pair_alignment_for_different_images(tmp_path):
    ds = PairedDataset(make_root(tmp_path, same=False), "train", crop=32)
    for _ in range(5):
        raw, ref = ds[0]
        assert np.allclose(raw.numpy(), 1 - ref.numpy(), atol=1e-6)


def test_small_images_are_scaled_up_to_fit_the_crop(tmp_path):
    ds = PairedDataset(make_root(tmp_path, size=(20, 24)), "train", crop=32)
    assert ds[0][0].shape == (3, 32, 32)


def test_val_returns_full_resolution(tmp_path):
    ds = PairedDataset(make_root(tmp_path, size=(64, 80)), "val", crop=32)
    assert len(ds) == 1
    assert ds[0][0].shape == (3, 64, 80)


def test_unpaired_files_fail_early(tmp_path):
    root = make_root(tmp_path)
    (root / "ref" / "3.png").unlink()
    with pytest.raises(ValueError, match="not paired"):
        PairedDataset(root)


def test_missing_folder_fails_early(tmp_path):
    with pytest.raises(FileNotFoundError, match="raw"):
        PairedDataset(tmp_path)
