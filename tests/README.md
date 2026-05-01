# Tests

Run the full test suite from the repo root:

```bash
PYTHONPATH=backend:. pytest
```

The `test_merchant_inference_service.py` tests that require large CSV and `.pkl` artifact files are guarded with `skipif` and will be skipped if those files are not present locally.
