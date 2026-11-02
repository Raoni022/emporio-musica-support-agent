"""JSON schemas of the tools for function calling.

Descriptions say *when* to use each tool (in Portuguese, like the prompt).
All schemas are strict (every property required, optional ones nullable,
no extra properties). validate_arguments() checks the same rules in code,
for providers that ignore or reject strict mode.
"""

from __future__ import annotations

from emporio_agent.data import StoreData
from emporio_agent.policies import TOPICS
from emporio_agent.tools import HANDOFF_REASONS, RETURN_REASONS, STRING_TYPES


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    properties = properties or {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _nullable(schema: dict) -> dict:
    schema = dict(schema)
    schema["type"] = [schema["type"], "null"]
    if "enum" in schema:
        schema["enum"] = [*schema["enum"], None]
    return schema


def build_tool_schemas(store: StoreData) -> list[dict]:
    categories = list(store.categories["name"])
    return [
        _tool(
            "search_products",
            "Busca produtos no catálogo com filtros. Use para perguntas abertas "
            "(\"violões até R$ 1000\", \"tem guitarra?\"). Por padrão só traz itens à venda, "
            "ordenados pelo preço final. Cada resultado já traz a promoção ativa (promotion), "
            "então não consulte os produtos um a um para saber se estão em promoção. Se "
            "truncated=true, diga quantas opções existem (total_count) e ofereça filtrar "
            "usando refine_by.",
            {
                "query": _nullable({"type": "string", "description": "Termos de busca livres."}),
                "category": _nullable({"type": "string", "enum": categories}),
                "min_price": _nullable({"type": "number"}),
                "max_price": _nullable({"type": "number"}),
                "string_type": _nullable({"type": "string", "enum": list(STRING_TYPES),
                                          "description": "nylon ou steel (aço)."}),
                "electronics": _nullable({"type": "boolean",
                                          "description": "Com captação/eletroacústico."}),
                "include_unavailable": {"type": "boolean",
                                        "description": "Incluir esgotados, descontinuados e "
                                                       "em breve. Normalmente false."},
            },
        ),
        _tool(
            "get_product_details",
            "Detalhes de UM produto pelo nome ou id: preço, promoção ativa, preço no PIX, "
            "estoque e especificações. Use sempre que o cliente citar um produto específico, "
            "antes de informar preço ou disponibilidade. Se vier ambiguous, pergunte qual; se "
            "approximate_match, confirme o nome. Se o produto estiver indisponível, diga isso e "
            "mostre já na mesma resposta as alternatives (nome e preço); não pergunte se pode "
            "mostrar.",
            {"product": {"type": "string", "description": "Nome (mesmo aproximado) ou id."}},
        ),
        _tool("list_categories",
              "Lista as categorias da loja com quantos produtos estão à venda em cada uma."),
        _tool("list_active_promotions",
              "Lista as promoções ativas agora. Use para qualquer pergunta geral sobre "
              "promoções (\"tem alguma promoção?\", \"o que está em oferta?\")."),
        _tool(
            "calculate_installments",
            "Opções de parcelamento sem juros no cartão, com as parcelas mínimas da loja, e "
            "se é possível combinar formas de pagamento. Para um produto, passe product "
            "(nome ou id): a ferramenta busca o preço final sozinha, então pode ser chamada "
            "junto com get_product_details. Use amount só para outros valores (ex.: soma de "
            "itens). Nunca calcule parcelas por conta própria.",
            {
                "product": _nullable({"type": "string"}),
                "amount": _nullable({"type": "number"}),
            },
        ),
        _tool(
            "get_order_status",
            "Status de um pedido. Exige o número do pedido e um contato do cadastro (o e-mail "
            "OU o telefone com DDD). Chame assim que tiver os dois, mesmo que vieram em "
            "mensagens diferentes: a verificação é feita pela ferramenta, não peça confirmação "
            "ao cliente. Informe o status, a previsão "
            "e, se houver, o código de rastreio (tracking_code). Se late=true, ofereça "
            "encaminhar à equipe.",
            {"order_id": {"type": "string"}, "contact": {"type": "string"}},
        ),
        _tool(
            "check_return_eligibility",
            "Verifica se um pedido pode ser devolvido ou trocado e até quando: regret "
            "(arrependimento), defect (defeito) ou preference (troca por cor/modelo). Exige "
            "número do pedido e e-mail ou telefone do cadastro. Pergunte o motivo se não "
            "estiver claro e confira com o cliente as exclusões retornadas. Sempre diga a data "
            "limite (deadline_display), não só o número de dias.",
            {
                "order_id": {"type": "string"},
                "contact": {"type": "string"},
                "reason": {"type": "string", "enum": list(RETURN_REASONS)},
            },
        ),
        _tool(
            "get_shipping_options",
            "Opções, prazos e custo de entrega para uma cidade e um valor de compra. Para "
            "cidades fora de Campo Grande o valor depende do CEP: não informe um preço.",
            {
                "city": {"type": "string"},
                "order_value": {"type": "number"},
                "product_ids": _nullable({"type": "array", "items": {"type": "integer"}}),
            },
        ),
        _tool("get_store_status",
              "Se a loja (e a equipe no WhatsApp) está aberta agora, o horário de hoje e "
              "quando reabre. Use para \"vocês estão abertos?\" e antes de dizer quando a "
              "equipe retorna."),
        _tool(
            "get_policy",
            "Texto de uma seção do manual de políticas da loja. Use antes de responder "
            "qualquer dúvida de regra: endereço e contato (about_and_contact), horários, "
            "pagamento, trocas, frete, promoções, atendimento, garantia, privacidade.",
            {"topic": {"type": "string", "enum": sorted(TOPICS)}},
        ),
        _tool(
            "register_handoff",
            "Encaminha o caso para um atendente humano: reclamação, intenção de compra, "
            "pedido atrasado, cancelamento, exclusão de dados ou dúvida sem resposta. Escreva "
            "um resumo que poupe o atendente de recomeçar a conversa.",
            {
                "reason": {"type": "string", "enum": list(HANDOFF_REASONS)},
                "summary": {"type": "string"},
                "contact": _nullable({"type": "string",
                                      "description": "E-mail ou telefone para retorno."}),
                "order_id": _nullable({"type": "string"}),
                "suggested_next_action": _nullable({"type": "string"}),
            },
        ),
    ]


_JSON_TYPES = {
    "string": str, "number": (int, float), "integer": int, "boolean": bool,
    "array": list, "object": dict, "null": type(None),
}


def _check(value, schema: dict, path: str) -> str | None:
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    ok = any(
        isinstance(value, _JSON_TYPES[t]) and not (t in ("number", "integer") and isinstance(value, bool))
        for t in types
    )
    if not ok:
        return f"{path} must be of type {'/'.join(types)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            error = _check(item, schema["items"], f"{path}[{i}]")
            if error:
                return error
    return None


def validate_arguments(schema: dict, arguments: dict) -> str | None:
    """Return an error message, or None if the arguments follow the schema."""
    parameters = schema["function"]["parameters"]
    properties = parameters["properties"]
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"
    extra = set(arguments) - set(properties)
    if extra:
        return f"unexpected arguments: {sorted(extra)}"
    for name, prop in properties.items():
        if name not in arguments:
            if "null" in (prop["type"] if isinstance(prop["type"], list) else [prop["type"]]):
                continue  # lenient for non-strict providers: missing optional = null
            return f"missing required argument: {name}"
        error = _check(arguments[name], prop, name)
        if error:
            return error
    return None
