from rapid.schemas.models import ModelSelection


def test_pretrained_tokenizer_name_is_accepted():
    models = ModelSelection(
        teacher={"name": "teacher"},
        shallow={"name": "shallow"},
        pretrained={
            "name": "google/gemma-3-270m",
            "tokenizer_name": "google/gemma-3-270m-it",
        },
    )

    assert models.pretrained.tokenizer_name == "google/gemma-3-270m-it"
