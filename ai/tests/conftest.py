import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope='session')
def router(tmp_path_factory):
    """Latih model ke folder sementara, supaya uji tidak bergantung pada file .joblib lokal."""
    import joblib
    import pandas as pd
    from sika import config
    from sika.intent import IntentRouter
    from sika.preprocess import preprocess
    from sika.train import build_pipeline

    df = pd.read_csv(config.DATA_PATH)
    model = build_pipeline().fit(df['text'].apply(preprocess), df['intent'])
    path = tmp_path_factory.mktemp('model') / 'intent_classifier.joblib'
    joblib.dump(model, path)
    return IntentRouter(model_path=path)
