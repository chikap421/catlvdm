import importlib


def test_evd_examples_import() -> None:
    importlib.import_module("examples.evd_catlvdm_train_step")
    importlib.import_module("examples.evd_catlvdm_inference_step")
    importlib.import_module("examples.evd_generic_dit_adapter")
