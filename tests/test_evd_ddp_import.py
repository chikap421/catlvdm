def test_ddp_helpers_and_example_import() -> None:
    from evd import distributed
    from examples import evd_train_2gpu

    assert distributed.get_rank() == 0
    assert distributed.get_world_size() == 1
    assert callable(evd_train_2gpu.main)
