import importlib


def test_evd_examples_import() -> None:
    importlib.import_module("examples.evd_catlvdm_train_step")
    importlib.import_module("examples.evd_catlvdm_inference_step")
    importlib.import_module("examples.evd_stdit_adapter")
    importlib.import_module("examples.evd_train_2gpu")
