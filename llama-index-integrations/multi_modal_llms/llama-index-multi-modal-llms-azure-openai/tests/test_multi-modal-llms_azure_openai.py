from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in AzureOpenAIMultiModal.__mro__]
    assert MultiModalLLM.__name__ in names_of_base_classes


def test_init():
    m = AzureOpenAIMultiModal(max_new_tokens=400, engine="fake", api_key="fake")
    assert m.max_new_tokens == 400