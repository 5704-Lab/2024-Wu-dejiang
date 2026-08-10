from step2_dataset.randomness import derive_seed, stable_image_id


def test_seed_is_stable_and_view_specific():
    first = derive_seed(1, "train", "0001_test", 0, "geometry")
    assert first == derive_seed(1, "train", "0001_test", 0, "geometry")
    assert first != derive_seed(1, "train", "0001_test", 0, "query-photo")
    assert first != derive_seed(1, "train", "0001_test", 1, "geometry")


def test_image_id_is_stable():
    assert stable_image_id("0001_test") == stable_image_id("0001_test")
    assert stable_image_id("0001_test") != stable_image_id("0002_test")
