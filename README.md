# Empório da Música — Support Agent ("Tom")

Prototype of a text-based customer support agent for Empório da Música, a fictional musical instrument store in Campo Grande/MS (Brazil). Built in Python for the Artefact AI Engineer technical case.

> Work in progress. Sections below are filled in as each step is implemented.

## 1. Overview

## 1b. Problem & scope

## 2. How to run

Requires Python 3.11+ (developed and tested on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in OPENAI_API_KEY
```

```bash
python cli.py                       # chat with Tom
python cli.py --debug               # also shows each tool call, its result, tokens, cost and latency
python cli.py --reference-date 2026-03-22T10:00   # e.g. a Sunday, to see "closed, reopens Monday"
pytest                              # tests, no LLM calls
python scripts/data_quality.py      # regenerates docs/data_quality.md
```

In the chat, `/reset` starts a new session and `/quit` exits. While a tool runs, a status line shows what Tom is doing (`🎵 Tom está consultando o catálogo...`).

**Why status lines instead of token streaming:** the final answer is not streamed, because it will be checked by the output guardrail (prices must come from the tools) before the customer reads it. The status line keeps the wait visible without showing an unchecked answer.

**History:** kept within a session (the customer often sends the order number in one message and the e-mail in the next), never across sessions, on purpose: the CLI has no stable customer identity, storing conversations means storing personal data (LGPD), and identity verification should not carry over from one session to another. In production: history per phone number with a TTL, and verification again in each session.

**Configuration (`.env`):**

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Key for OpenAI or any OpenAI-compatible provider. For a local server such as Ollama, any non-empty value |
| `OPENAI_BASE_URL` | empty (OpenAI) | Any OpenAI-compatible endpoint, e.g. Anthropic `https://api.anthropic.com/v1/`, Gemini `https://generativelanguage.googleapis.com/v1beta/openai/`, Groq `https://api.groq.com/openai/v1`, Ollama `http://localhost:11434/v1` |
| `MODEL` | `gpt-4.1-mini` | Model name as the provider expects it |
| `REFERENCE_DATE` | empty (real time) | Fixed "now" (`YYYY-MM-DD`, 10:00 assumed, or `YYYY-MM-DDTHH:MM`). The data ends in March 2026; with the real date every delivered order is outside any return window, so examples and tests use `2026-03-25T10:00` |
| `STRICT_TOOLS` | `true` | Set `false` for providers that reject strict tool schemas; arguments are validated in code either way |

The `.env` file wins over variables already set in the system (an old `OPENAI_API_KEY` in the environment would otherwise be used silently).

**Tested for real:** OpenAI `gpt-4.1-mini` with strict schemas on. Other providers are supported through `OPENAI_BASE_URL` but were not run end to end.

## 3. Architecture

```
customer message
      │
      ▼
 Agent.respond  ── length check (2,000 chars) ──► fixed reply, no LLM call
      │
      ▼  system prompt (persona + behavior + current store time) + session history
 LLM (function calling) ◄────────────────────────────┐
      │ tool calls (one or several in parallel)      │ results as JSON
      ▼                                              │
 validate arguments ─► tools.py (pure functions) ────┘
                         ├─ data.py      CSVs, normalized at load
                         ├─ policies.py  manual split into 10 sections (get_policy)
                         └─ handoffs.jsonl  queue for the human team
      │
      ▼  no more tool calls
 answer  (or a fixed fallback message on any failure)
```

- **`agent.py`**: the loop (decide → act → observe), at most 5 LLM calls and ~60 s per customer message. All tool calls in one model response run before the next LLM call.
- **`llm.py`**: the only module that imports `openai`; timeouts and retries come from the SDK. `FakeLLM` replays scripted responses so the loop is tested without an API.
- **`tool_schemas.py`**: strict JSON schemas; each description says *when* to use the tool. The category enum is built from the data.
- **`prompts.py`**: persona and behavior only, plus the fixed fallback messages.

## 4. Technical decisions and why

### Decisions vs. the options in the brief

### Alternatives considered

### Guardrails

### Evaluation

### Retrieval architecture

### Prompt strategy

### Data used (and not used)

## 5. Data issues and assumptions

The raw files in `data/` are never edited. `src/emporio_agent/data.py` normalizes formats at load time (types, lowercase e-mails, phone digits, parsed specs, translated order statuses) and adds derived fields (`availability`, active promotion and promotional price, accent-free search text). Business facts are never "fixed": when the right value is unknown, the field is suppressed, kept as recorded, or a choice is made and documented.

The full diagnosis, written as a report to hand back to the store, is in [docs/data_quality.md](docs/data_quality.md) (generated by `scripts/data_quality.py`). Main points:

| Issue | Decision |
|---|---|
| Descriptions in Baixos, Baterias and Teclados name other brands and contradict the specs (e.g. 4 strings vs. "5 cordas", 61 vs. "88 teclas") | Name, category, price and specs are the source of truth; descriptions in those 3 categories are never sent to the LLM |
| `strings` spec is a material (nylon/steel) for acoustic instruments and a count for electric ones | Split into `string_type` and `string_count`; the raw field is not exposed |
| Recorded order totals differ from items × price (orders 3 and 20) | Show the recorded total (what was charged); never recalculate |
| Product 130 is `coming_soon` but appears in orders 3 and 14 | Report the order as recorded; no speculation |
| Orders `pending`/`confirmed` for weeks, order 8 past its delivery estimate | Report the system status and offer a handoff when clearly late |
| `coming_soon` product with stock > 0 | Not sellable: status wins over stock |
| Several promotions for the same product (all inactive today) | Only `is_active == 1` counts; if more than one is active, the largest discount applies (not cumulative) |
| Customers 11 and 43 share an e-mail | Verification still correct (same e-mail owner); reported as a probable duplicate |
| Categories 6, 7 and 8 have no products | The agent says so, without inventing items |
| The manual lists two WhatsApp numbers: (67) 3341-4444 (cover, 1.2) and (67) 3321-4500 (section 7) | Use (67) 3341-4444. `get_policy` returns the section text unchanged plus a `notes` field saying which value to use |
| The store e-mail appears with an accent in table 1.2 (`empóriodamusica`) | Use `contato@emporiodamusica.com.br` (cover); same `notes` mechanism |
| The payment table says "minimum installment R$ 100,00" for credit; section 3.1 sets R$ 50 / 80 / 100 by number of installments | Section 3.1 applies; installments are computed by a tool. "No minimum (except below R$ 50,00)" is read as a R$ 50 minimum installment |
| Section 7.1 says to suggest partner stores, but none are listed | The agent does not name other stores |
| The brief asks for products "up to R$ X" but does not say whether that is the list or the promotional price | Search filters and sorts by the **final price** (promotional price when a promotion is active), i.e. what the customer pays. List price, discount and final price are always returned together |
| Manual 7.3 asks for "similar alternatives" to unavailable products without defining similar | Up to 3 available products from the same category (and same string type when applicable), with a final price within ±30% of the original; if none fall in that band, the closest prices in the category |
| Customers write product names loosely ("GD-20", "takamini") | Name lookup ignores separators and tolerates typos word by word; a typo match is flagged `approximate_match` so the agent confirms with the customer. Several matches return candidates instead of a guess |
| Installment minimums (3.1) say nothing about paying in a single installment | 1x on credit is always allowed; the R$ 50 / 80 / 100 minimums apply from 2x on |
| Splitting an amount into installments rarely gives whole cents (R$ 1.967,18 ÷ 10) | Each installment is rounded to the cent (half up, using `Decimal`); the total shown is always the exact amount. In practice the card operator absorbs the cents in one installment |
| Installments for a product | `calculate_installments` takes the **product** and looks up its final price itself (promotional when active). In the first real run the model called it in parallel with `get_product_details` and passed the previous product's price; now it never needs to know the price first. `amount` remains for other values (e.g. two items together) |
| How a customer proves an order is theirs (manual 9) | Order number + the e-mail (case-insensitive) or phone registered for **that order's** customer. Phone accepts any formatting and the +55 country code, but needs the area code (DDD). Unknown order, wrong contact and malformed input return the **same** message, so the response never reveals whether an order exists |
| What a verified order lookup returns | Status, date, items, recorded total, payment method, tracking and estimate, notes, and the customer's **first name only** (for the greeting). No e-mail, phone, full name or city: the agent does not need them (data minimization) |
| When an order counts as late | `shipped` past the estimated delivery, or `pending`/`confirmed` for more than 7 days since the order date. The tool returns `late` and `suggest_handoff`; the same threshold drives the data quality report |
| Return windows count from the receipt date, but the data has no actual delivery date | `estimated_delivery` is used as the receipt date, and the tool tells the agent to check with the customer. Regret and preference: 7 days after receipt. Defect: 30 days after purchase, then the 90-day legal warranty (from receipt) and the manufacturer's warranty |
| Section 4.3 gives 7 days for preference exchanges without saying from when | Counted from receipt, like the regret window |
| Section 4.1 is titled "Compras Online"; the data does not say where each order was placed | All orders in the data are treated as online orders |
| Returns before delivery (pending, confirmed, shipped) | Not covered by the manual (order 17 shows cancellations do happen): the tool returns `requires_human` and the agent offers a handoff |
| 4.4 exclusions (customized instruments, "final sale" items, mouthpieces) are not marked in the data | The tool always returns them and the agent asks the customer instead of assuming |
| "Região metropolitana de Campo Grande" is not defined in the manual | Only the city of Campo Grande gets the local rules |
| Free local shipping "acima de R$ 500,00" and a fee "abaixo de R$ 500,00": exactly R$ 500 is in neither | Strictly above R$ 500 is free; R$ 500 or less pays R$ 35 |
| Which products count as "large" (acoustic drums, digital pianos, double basses) | Acoustic drum kits in Baterias, pianos in Teclados, double basses in Cordas Orquestrais. Electric basses ship normally, even though some descriptions call them "contrabaixo" |
| Holidays | National and Mato Grosso do Sul holidays (`holidays` package). Campo Grande municipal holidays are not covered |
| Store timezone | Fixed UTC-4 (Campo Grande has had no daylight saving time since 2019); avoids depending on the IANA database, which Python does not bundle on Windows |
| "Prazo de retorno: 24 horas úteis" | Returned as-is ("até 24 horas úteis"), plus when the team is back if the store is closed now; not converted into a date |
| Handoff queue contents | `handoffs.jsonl` keeps the customer's contact **unmasked**, since the team needs it to call back; it is outside git. `identity_verified` is computed by the tool, never taken from the LLM |
| A customer message longer than 2,000 characters | Not sent to the LLM: a fixed reply asks the customer to summarize (cost and abuse protection) |
| Languages inside the system | The prompt and tool descriptions are in Portuguese (the agent talks to Brazilian customers); tool results are in English, as structured data the model reads; customer-facing fixed messages are in Portuguese |

## 5b. Where business rules live

Principle: a rule that can be **computed or verified lives in code**; an informational rule comes from **`get_policy`**; *when* to use each tool lives in the **tool descriptions**; only **behavior** lives in the prompt. The prompt does not repeat rules the code already enforces.

| Rule | Layer | Where | Tested by |
|---|---|---|---|
| Promotional price, discount % and original price | Code | `data.py` (derived at load), `search_products`, `get_product_details`, `list_active_promotions` | pytest |
| Only one active promotion per product (largest wins) | Code | `data.pick_active_promotions` | pytest |
| PIX 5% on the list price only, never on a promotional price | Code | `get_product_details` (`pix_price` is `null` with a `pix_note` when a promotion is active) | pytest |
| Minimum installments 50 / 80 / 100, at most 12x | Code | `calculate_installments` | pytest |
| Combining payment methods only above R$ 2.000 | Code | `calculate_installments` (`can_combine_payments`) | pytest |
| Availability (stock, discontinued, coming soon) | Code | `data.py` (`availability`) | pytest |
| Order identity verification (LGPD) | Code | `verify_order_owner`, used by `get_order_status` | pytest |
| Late-order detection | Code | `get_order_status` (`late`, `suggest_handoff`) | pytest |
| Regret, defect and preference windows; legal warranty | Code | `check_return_eligibility` | pytest |
| Campo Grande shipping (free above R$ 500, else R$ 35) and special freight for large items | Code | `get_shipping_options` | pytest |
| Opening hours, holidays and next opening | Code | `get_store_status` | pytest |
| Who gets handed to a human, and what the attendant receives | Code | `register_handoff` (reasons enum, verified identity, tools consulted) | pytest |
| When to consult data vs. policies | Tool descriptions + prompt | `tool_schemas.py`, `prompts.py` | eval |
| Persona, tone, flow, scope, handoff, security | Prompt | `prompts.py` | examples + eval |
| Policy text (returns, warranty, privacy...) and which contact to use | Manual | `get_policy` (`text` + `notes`) | pytest |

## 6. Example conversations

## 7. Observability

Every turn appends one JSON line to `logs/traces.jsonl` (outside git), written by `tracing.py` without any external service:

- the customer message and the final answer;
- each tool call with its arguments, latency and error (tool results are left out: they are large and can contain personal data);
- the manual version used by each `get_policy` call (to audit which policy an answer relied on);
- iterations, prompt and completion tokens, an **estimated** cost in USD (from a price table; `null` for unknown, free or local models), total latency;
- `error_type` when a fixed message replaced the LLM answer, and `resolved` (`false` on errors or when the agent handed off a question it could not answer).

**PII masking:** customer e-mails and phones are masked everywhere in the trace (`l***@jmail.com`, `(67) 9****-3456`), including inside tool arguments. Prices, dates, tracking codes, the CNPJ and the store's own public contact are left as they are. If the trace cannot be written, the conversation goes on.

Example line (formatted):

```json
{
  "timestamp": "2026-03-25T10:00:00-04:00",
  "session_id": "a1b2c3",
  "model": "gpt-4.1-mini",
  "user_message": "Pedido 3, meu e-mail é i***@jmail.com",
  "tool_calls": [
    {"name": "get_order_status", "arguments": {"order_id": "3", "contact": "i***@jmail.com"}, "latency_ms": 12, "error": null},
    {"name": "get_policy", "arguments": {"topic": "returns_and_exchanges"}, "latency_ms": 1, "error": null}
  ],
  "policy_versions": [{"section": 4, "version": "Versão 2.1 — Junho de 2025"}],
  "iterations": 2,
  "prompt_tokens": 1500,
  "completion_tokens": 200,
  "estimated_cost_usd": 0.00092,
  "latency_ms": 2300,
  "error_type": null,
  "resolved": true,
  "final_response": "Achei seu pedido, Isabela!"
}
```

Next step: send the same fields to Langfuse or OpenTelemetry instead of a local file.

## 7b. Failure handling

Principle: the customer always gets an answer and the agent never guesses. Every failure ends in a defined behavior (tested with `FakeLLM` in `tests/test_agent.py`).

| Situation | Behavior |
|---|---|
| LLM slow, timeout, rate limit (429), provider down (5xx) | The SDK retries twice with exponential backoff (30 s timeout); then a **fixed fallback message, no LLM**, with the store's WhatsApp, e-mail and hours |
| Empty answer from the model | Same fixed fallback |
| A tool raises (file, bug, integration) | Becomes `{"error": ..., "retryable": true}` for the model to explain; the turn continues |
| Unknown tool, invalid JSON or arguments outside the schema | Structured error back to the model, which can correct itself |
| Tool loop or slow turn | At most 5 LLM calls and ~60 s per message; then an apology and an offer to hand off |
| Message over 2,000 characters | Fixed reply asking to summarize |
| Missing or invalid configuration | Clear error at startup (`ConfigError`) |

After an LLM failure, the partial tool exchange of that turn is dropped from history (the customer's message stays), so the next request never carries an orphan tool call.

## 8. Known limitations and what I'd do with more time

- **Order verification can be guessed by brute force.** Order numbers are sequential and there is no limit on failed attempts per session. In production: lock or slow down after a few failures per phone number, and alert on many failures.

## 9. Path to production

## 10. Use of AI coding assistants
