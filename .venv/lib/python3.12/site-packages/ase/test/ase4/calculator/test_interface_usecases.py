def test_minimal(v4atoms, calculator):
    properties = ['energy', 'forces', 'stress']
    results = calculator.evaluate(v4atoms, properties=properties)
    v4atoms.store(results, prefix='test_')

    assert 'test_energy' in v4atoms.info
    assert 'test_forces' in v4atoms.arrays
    assert 'test_stress' in v4atoms.info
