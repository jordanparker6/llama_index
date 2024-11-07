import json

from llama_index.core.schema import Document as LIDocument

from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import BaseNode

in_json_str = json.dumps(
    {
        "id_": "123",
        "embedding": None,
        "metadata": {},
        "excluded_embed_metadata_keys": [],
        "excluded_llm_metadata_keys": [],
        "relationships": {},
        "text": '{"schema_name": "DoclingDocument", "version": "1.0.0", "name": "sample", "origin": {"mimetype": "text/html", "binary_hash": 42, "filename": "sample.html"}, "furniture": {"self_ref": "#/furniture", "children": [], "name": "_root_", "label": "unspecified"}, "body": {"self_ref": "#/body", "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}], "name": "_root_", "label": "unspecified"}, "groups": [], "texts": [{"self_ref": "#/texts/0", "parent": {"$ref": "#/body"}, "children": [], "label": "paragraph", "prov": [], "orig": "Some text", "text": "Some text"}, {"self_ref": "#/texts/1", "parent": {"$ref": "#/body"}, "children": [], "label": "paragraph", "prov": [], "orig": "Another paragraph", "text": "Another paragraph"}], "pictures": [], "tables": [], "key_value_items": [], "pages": {}}',
        "mimetype": "text/plain",
        "start_char_idx": None,
        "end_char_idx": None,
        "text_template": "{metadata_str}\n\n{content}",
        "metadata_template": "{key}: {value}",
        "metadata_seperator": "\n",
        "class_name": "Document",
    }
)

out_get_nodes = {
    "root": [
        {
            "id_": "123_0",
            "embedding": None,
            "metadata": {
                "schema_name": "docling_core.transforms.chunker.DocMeta",
                "version": "1.0.0",
                "doc_items": [
                    {
                        "self_ref": "#/texts/0",
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "paragraph",
                        "prov": [],
                    }
                ],
                "origin": {
                    "mimetype": "text/html",
                    "binary_hash": 42,
                    "filename": "sample.html",
                },
            },
            "excluded_embed_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "excluded_llm_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "relationships": {
                "1": {
                    "node_id": "123",
                    "node_type": "4",
                    "metadata": {},
                    "hash": "95ee9366fededbecc38a68be8c58e87fb3cdf4e7fdfb83dbd00487489b941a4d",
                    "class_name": "RelatedNodeInfo",
                },
                "3": {
                    "node_id": "123_1",
                    "node_type": "1",
                    "metadata": {
                        "schema_name": "docling_core.transforms.chunker.DocMeta",
                        "version": "1.0.0",
                        "doc_items": [
                            {
                                "self_ref": "#/texts/1",
                                "parent": {"$ref": "#/body"},
                                "children": [],
                                "label": "paragraph",
                                "prov": [],
                            }
                        ],
                        "origin": {
                            "mimetype": "text/html",
                            "binary_hash": 42,
                            "filename": "sample.html",
                        },
                    },
                    "hash": "0a8df027ead9e42831f12f8aa680afe5138436ecd58c32a6289212bc4d0a644a",
                    "class_name": "RelatedNodeInfo",
                },
            },
            "text": "Some text",
            "mimetype": "text/plain",
            "start_char_idx": 529,
            "end_char_idx": 538,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
        {
            "id_": "123_1",
            "embedding": None,
            "metadata": {
                "schema_name": "docling_core.transforms.chunker.DocMeta",
                "version": "1.0.0",
                "doc_items": [
                    {
                        "self_ref": "#/texts/1",
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "paragraph",
                        "prov": [],
                    }
                ],
                "origin": {
                    "mimetype": "text/html",
                    "binary_hash": 42,
                    "filename": "sample.html",
                },
            },
            "excluded_embed_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "excluded_llm_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "relationships": {
                "1": {
                    "node_id": "123",
                    "node_type": "4",
                    "metadata": {},
                    "hash": "95ee9366fededbecc38a68be8c58e87fb3cdf4e7fdfb83dbd00487489b941a4d",
                    "class_name": "RelatedNodeInfo",
                },
                "2": {
                    "node_id": "123_0",
                    "node_type": "1",
                    "metadata": {
                        "schema_name": "docling_core.transforms.chunker.DocMeta",
                        "version": "1.0.0",
                        "doc_items": [
                            {
                                "self_ref": "#/texts/0",
                                "parent": {"$ref": "#/body"},
                                "children": [],
                                "label": "paragraph",
                                "prov": [],
                            }
                        ],
                        "origin": {
                            "mimetype": "text/html",
                            "binary_hash": 42,
                            "filename": "sample.html",
                        },
                    },
                    "hash": "fbfaa945f53349cff0ee00b81a8d3926ca76874fdaf3eac7888f41c5f6a74f0c",
                    "class_name": "RelatedNodeInfo",
                },
            },
            "text": "Another paragraph",
            "mimetype": "text/plain",
            "start_char_idx": 678,
            "end_char_idx": 695,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
    ]
}


out_parse_nodes = {
    "root": [
        {
            "id_": "123_0",
            "embedding": None,
            "metadata": {
                "schema_name": "docling_core.transforms.chunker.DocMeta",
                "version": "1.0.0",
                "doc_items": [
                    {
                        "self_ref": "#/texts/0",
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "paragraph",
                        "prov": [],
                    }
                ],
                "origin": {
                    "mimetype": "text/html",
                    "binary_hash": 42,
                    "filename": "sample.html",
                },
            },
            "excluded_embed_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "excluded_llm_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "relationships": {
                "1": {
                    "node_id": "123",
                    "node_type": "4",
                    "metadata": {},
                    "hash": "95ee9366fededbecc38a68be8c58e87fb3cdf4e7fdfb83dbd00487489b941a4d",
                    "class_name": "RelatedNodeInfo",
                }
            },
            "text": "Some text",
            "mimetype": "text/plain",
            "start_char_idx": None,
            "end_char_idx": None,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
        {
            "id_": "123_1",
            "embedding": None,
            "metadata": {
                "schema_name": "docling_core.transforms.chunker.DocMeta",
                "version": "1.0.0",
                "doc_items": [
                    {
                        "self_ref": "#/texts/1",
                        "parent": {"$ref": "#/body"},
                        "children": [],
                        "label": "paragraph",
                        "prov": [],
                    }
                ],
                "origin": {
                    "mimetype": "text/html",
                    "binary_hash": 42,
                    "filename": "sample.html",
                },
            },
            "excluded_embed_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "excluded_llm_metadata_keys": [
                "schema_name",
                "version",
                "doc_items",
                "origin",
            ],
            "relationships": {
                "1": {
                    "node_id": "123",
                    "node_type": "4",
                    "metadata": {},
                    "hash": "95ee9366fededbecc38a68be8c58e87fb3cdf4e7fdfb83dbd00487489b941a4d",
                    "class_name": "RelatedNodeInfo",
                }
            },
            "text": "Another paragraph",
            "mimetype": "text/plain",
            "start_char_idx": None,
            "end_char_idx": None,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
    ]
}


def _deterministic_id_func(i: int, node: BaseNode) -> str:
    return f"{node.node_id}_{i}"


def test_parse_nodes():
    li_doc = LIDocument.from_json(in_json_str)
    node_parser = DoclingNodeParser(
        id_func=_deterministic_id_func,
    )
    nodes = node_parser._parse_nodes(nodes=[li_doc])
    act_data = {"root": [n.model_dump() for n in nodes]}
    assert act_data == out_parse_nodes


def test_get_nodes_from_docs():
    li_doc = LIDocument.from_json(in_json_str)
    node_parser = DoclingNodeParser(
        id_func=_deterministic_id_func,
    )
    nodes = node_parser.get_nodes_from_documents(documents=[li_doc])
    act_data = {"root": [n.model_dump() for n in nodes]}
    assert act_data == out_get_nodes