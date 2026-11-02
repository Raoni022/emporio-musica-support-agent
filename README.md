# Empório da Música — Support Agent ("Tom")

Prototype of a text-based customer support agent for Empório da Música, a fictional musical instrument store in Campo Grande/MS (Brazil). Built in Python for the Artefact AI Engineer technical case.

## 1. Overview

**Tom** (a play on a song's *tom*, its key) is the store's virtual assistant. It talks to customers in Portuguese, in the informal-but-professional tone the store's manual asks for, and answers from the store's own data and policy manual instead of from memory:

```
Cliente: E o Crafter HT-100? Quanto fica no PIX? E em 10x?
  🎵 Tom está consultando o produto...
  🎵 Tom está calculando o parcelamento...
Tom: O Crafter HT-100 Folk Aço Natural está em promoção: de R$ 2.399,00 por R$ 1.967,18
     (18% de desconto). No PIX o preço é o mesmo da promoção, R$ 1.967,18.
     Parcelado em 10x sem juros, fica 10x de R$ 196,72.
```

(from [example 02](examples/02_price_payment_promotion.md); the promotion, the rule that PIX does not stack on it, and the installments all come from tools.)

How it works, in one paragraph: a plain function-calling loop (ReAct style, no framework) gives the model 11 deterministic tools over the store's CSVs and its policy manual, split into its 10 numbered sections. Every price, date and rule that can be computed is computed in code; the prompt holds only persona and behavior. A deterministic guardrail checks that every `R$` value in an answer came from a tool before the customer reads it, and every turn leaves one masked trace line.

| Measured on this prototype (`gpt-4.1-mini`) | Result |
|---|---|
| Golden dataset (27 cases, deterministic checks) | **80/81 over the last three runs** (27, 27, 26) ([evals/last_run.md](evals/last_run.md) is the last one) |
| Average cost per customer message | ~US$ 0.002 |
| Median latency per customer message | ~2.3 s |
| Output guardrail triggers in the eval | 0 |
| Unit tests (no LLM) | 290 passing |

## 1b. Problem & scope

**Problem.** The store's team answers every message by hand and is overloaded with recurring questions: opening hours, order status, price and availability, plus returns and payment rules. The goal is **first-level support** for those questions, with a structured handoff to the team for everything else.

**What Tom does**

- answers product, price, promotion, PIX and installment questions from the data, never from memory;
- answers policy questions (returns, warranty, shipping, payment, privacy, hours, address) from the manual's own sections;
- looks up an order and applies the return policy to it, only after checking that the contact belongs to that order's customer;
- **takes action, not only answers:** it registers a structured handoff for the team (`handoffs.jsonl`) with the reason, a summary, whether identity was verified, the order, the tools already consulted and a suggested next step, so the attendant does not start the conversation over;
- declines out-of-scope requests (accessories, unrelated topics) and ignores attempts to change its role.

**What Tom does not do, on purpose**

- it does not sell, reserve stock or take payments (the manual describes no checkout flow): it summarizes and hands off;
- it does not judge whether a product is defective: that goes to a human;
- it does not keep conversations across sessions (identity, LGPD, security; see "History" in section 2).

**Out of scope for this text agent, with the evolution path**

| Out of scope | Next step |
|---|---|
| Audio messages | Transcription (e.g. Whisper) in front of the agent; the agent itself does not change |
| Images and documents (e.g. a photo of a damaged product) | Attached to the handoff for a human; the agent does not judge damage |
| Real integrations (ERP, helpdesk, WhatsApp) | Simulated by the CSVs and the handoff queue; the tools are the integration point (section 9) |

**Updating data and policies.** CSVs and the PDF are loaded at startup: after an update, restart. The manual's version ("Versão 2.1 — Junho de 2025") is extracted and returned with every `get_policy` call, and the trace records which version each answer used. If the manual can no longer be split into its 10 numbered sections, the program refuses to start instead of running with a policy missing.

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

**Why status lines instead of token streaming:** the final answer is not streamed, because it is checked by the output guardrail (prices must come from the tools) before the customer reads it. The status line keeps the wait visible without showing an unchecked answer.

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
 output guardrail: every R$ value must come from the tools ── miss ──► one retry, then a safe answer
      │
      ▼
 answer  (or a fixed fallback message on any failure)  ──► trace line (logs/traces.jsonl)
```

- **`agent.py`**: the loop (decide → act → observe), at most 5 LLM calls and ~60 s per customer message. All tool calls in one model response run before the next LLM call.
- **`tools.py`**: the 11 business tools as plain functions over the loaded data; every computed or verifiable rule lives here (prices, PIX, installments, identity, return windows, shipping, hours).
- **`guardrails.py`**: the deterministic output check. **`tracing.py`**: one masked JSON line per turn.
- **`llm.py`**: the only module that imports `openai`; timeouts and retries come from the SDK. `FakeLLM` replays scripted responses so the loop is tested without an API.
- **`tool_schemas.py`**: strict JSON schemas; each description says *when* to use the tool. The category enum is built from the data.
- **`prompts.py`**: persona and behavior only, plus the fixed fallback messages.

## 4. Technical decisions and why

| Decision | Choice | Why |
|---|---|---|
| Approach | **Hybrid**: function calling over deterministic tools (data) + section-level retrieval of the manual through a tool (policies), in a ReAct-style loop (decide → act → observe), **no framework** | The loop is one file of under 300 lines including failure handling and the output check, transparent and easy to debug. LangChain/LangGraph add nothing at this size; LangGraph is the next step when stateful human handoff and multi-step flows appear (section 9) |
| Structured data | Deterministic tools over the CSVs (pandas) | Price, stock and orders need exactness, and the manual forbids giving prices or stock without checking the system (7.1). The LLM never computes or invents a value |
| Policies | **`get_policy(topic)`** over the manual split by its 10 numbered sections, no embeddings | The brief evaluates whether the agent knows **when** to consult policies: as a tool call that decision is explicit, visible in the trace and measurable in the eval. Splitting by the manual's own headings is deterministic, with no risk of retrieving the wrong passage |
| Observability | Local JSONL trace per turn, with e-mails and phones masked | Tools, arguments, latency, errors, tokens and estimated cost without an external service; the base for the eval (section 7) |
| LLM | OpenAI `gpt-4.1-mini` by default, configurable in `.env` | Cheap (~US$ 0.002 per message here), reliable function calling with strict schemas. Through `OPENAI_BASE_URL` any OpenAI-compatible provider runs without code changes, paid or free, local or remote |
| Interface | CLI, with `--debug` | The focus is the agent; `--debug` shows every tool call and result, which helps whoever evaluates it |
| History | Within a session: yes. Across sessions: no, on purpose | See "History" in section 2 |

### Decisions vs. the options in the brief

| Item in the brief | Options mentioned | Choice and why |
|---|---|---|
| Framework / approach | RAG, function calling, ReAct, SQL agent, hybrid | **Hybrid.** Function calling for data (exactness); section retrieval for policies (the manual is already structured, embeddings would be overkill); the loop is ReAct in practice. **SQL agent rejected:** the LLM could query the customers table without verification (LGPD) and write wrong queries. **Embedding RAG rejected:** 8 pages, and the risk of retrieving the wrong section |
| Model and provider | Paid or free, local or via API | Small OpenAI model by default; `OPENAI_BASE_URL` opens free and local options without code changes. Only OpenAI was tested end to end (section 2) |
| Interface | CLI, notebook, API, simple UI | CLI. A notebook does not represent a real conversation |
| Conversation history | "if it makes sense for the user" | In the session, yes; across sessions, no (identity, LGPD, security) |
| Data handling | "whatever you find necessary" | Normalization at load time, raw files untouched, business facts never "fixed", a data quality report for the store (section 5) |
| Constraint: Python | — | Everything is Python |

### Alternatives considered

- **Everything in the context (long context)**: the LLM would do the PIX, installment and date math, cost more per message, and the "when to consult" decision would be invisible.
- **Router + fixed flows**: breaks on mixed questions ("how much is it and how many installments?").
- **State machine / LangGraph**: identity verification is already enforced inside the tools; a graph adds structure without a problem to solve yet.
- **Multi-agent / orchestrator-workers**: overkill for 11 tools and short conversations.
- **Embedding RAG for everything**: bad at exact filters (price, stock, string type). Evolution: semantic search over descriptions combined with the structured filters.
- **SQL agent with restricted views**: still writes wrong queries; tools are a smaller, testable surface.
- **Off-the-shelf agent frameworks** (OpenAI Agents SDK, PydanticAI): one more abstraction for a loop this small.
- **Plan-and-execute**: messages need one or two tools, so planning adds an LLM call; parallel tool calls already cut round trips.
- **Prompt chaining**: a rigid pipeline breaks on mixed questions.
- **Reasoning models**: slower and more expensive, and they still need tools to reach the data (they also reject the low temperature used here).
- **LLM self-correction on every answer**: doubles latency; the deterministic price check covers the critical risk.
- **LLM-as-a-judge**: useful offline in the eval for tone and policy adherence (next step), never in the response path.
- **Episodic or semantic memory across sessions**: means storing personal data; belongs in production with consent and retention rules.
- **Fine-tuning**: no data to train on, and the rules change with the manual.
- **GraphRAG**: the relations are joins between six small tables.

### Guardrails

Principle (OWASP LLM01, prompt injection): **the prompt can be talked out of its rules, the code cannot.** Real protection lives in the tools; the prompt is the weakest layer.

| Layer | Protects against | Implementation |
|---|---|---|
| Input | Abuse, huge messages | 2,000-character limit, fixed reply without calling the LLM |
| Prompt | Off-topic requests, role changes | Scope and security rules in `prompts.py` |
| Tools | Data leaks, unintended actions | Identity checked inside the tools; enums in arguments; read-only data; no tool returns another customer's data; `register_handoff` only appends to a queue and computes `identity_verified` itself |
| **Output** | Hallucinated prices | `guardrails.validate_response`: every `R$` value in the answer must appear (within 5 cents) in the session's tool results or tool arguments, or in the customer's own messages. If not, the model gets one warning and rewrites; if it fails again, the customer gets a fixed answer without values and an offer of an attendant. The trace records `guardrail_triggered` and the values |
| Human | Decisions the agent should not make | `register_handoff` |

The output check is deterministic (a regex and a set lookup), so it adds no LLM call when the answer is grounded. Values are accepted from **earlier turns of the same session** too, not only the current one: a follow-up such as "so it's under R$ 2.000?" legitimately repeats a price the tools returned before. In a real run with five price-heavy questions (about 20 values) it triggered no false positives.

What it does not catch: a value that exists in the tool results but belongs to another product (the installment bug found in the first real run quoted a real tool value). That class of error is handled in the tools themselves (`calculate_installments` looks up the product's price) and measured by the eval.

### Evaluation

`evals/golden_dataset.jsonl` has 27 cases (the plan was 15–20; 7 were added from bugs found in manual CLI sessions), run by `python scripts/eval.py` against the real model with a fixed reference date (some cases set their own, e.g. a holiday). Each case lists the customer messages and **deterministic** checks, with no LLM judge: tools that must be called, text that must appear in the final answer (a list means "any of these"), and text that must not appear in any answer. Texts are compared lowercase and without accents.

Coverage: the 4 questions suggested in the brief, **verbatim**; every tool at least once; PIX not stacking on a promotion; the inactive promotion; installments using the right product's price (regression of a real bug); an out-of-stock product with alternatives; order verification with the right and the wrong contact; prompt injection asking for another customer's data; accessories and off-topic questions; a holiday; a question mixing data and policy; the ambiguous "contrabaixo"; shipping outside Campo Grande; a return inside its window; a complaint ending in a handoff on a holiday. Added from manual testing: installments for two products bought together, "PIX in installments", an empty category (saxophone), a made-to-order exclusion, a product that arrived damaged, Black Friday without invented dates, and an English question in the middle of a Portuguese conversation.

The runner prints and saves ([evals/last_run.md](evals/last_run.md)) the pass rate, average cost and median latency per message, guardrail triggers and fallback answers. It should run in CI before any change to the prompt, the tools or the model.

**Baseline (first run, `gpt-4.1-mini`): 17/20 (85%)**, US$ 0.0014 per message, median latency 2.1 s, no guardrail triggers, no fallback answers. The 3 failures were real behavior problems:

- out-of-stock product: it offered to show alternatives instead of showing the ones the tool returned (manual 7.3);
- late shipped order: a good answer, but without the tracking code;
- complaint: it asked for a phone number although the customer had given an e-mail, so the handoff never happened.

**After the fixes (`fix: follow the eval failures`): 20/20 in the last run**, and 79/80 over four consecutive runs while fixing (20, 19, 20, 20), US$ 0.0016 per message, median latency 2.2–2.4 s, no guardrail triggers, no fallback answers.

What the iterations showed:

- **One run is not enough.** At temperature 0.2 the same case can pass and fail on consecutive runs; the first round of fixes (prompt only) went from 17 to 16, with different cases failing. The eval was then run several times per change.
- **Instructions next to the data beat instructions in the prompt.** Rules stated only in the prompt or tool descriptions were followed inconsistently (say the total of a truncated search, show the original price and the %, show alternatives instead of offering them, give the deadline date, don't guess a name from the e-mail). Each became stable once the **tool result** carried it: an `instruction` field next to the data, ready-made `display` text for promotions (`de R$ 2.399,00 por R$ 1.967,18 (18% de desconto)`), `deadline_display` for return windows.
- **A fix can break another case.** Telling the agent not to re-ask for data it already had made it ask for "confirmation" when the e-mail came in a second message; the wording had to say that one contact is enough and that verifying is the tool's job.

**Manual testing found what the eval did not cover.** A CLI session with tricky questions surfaced 8 more problems: two products in 6x answered as impossible (the model read `can_combine_payments`, about payment *methods*, as combining *items*), "PIX in 3x" accepted, saxophones implied for an empty category, return rules answered from memory, invented Black Friday dates and an offer to "let you know" (the agent cannot send messages later), a cancellation registered without verification, and an English question answered in Portuguese. Fixes followed the same rule: computed in code where possible (`calculate_installments` now adds several products itself and says installments are credit-card only), pointed to the right manual section (`notes`), and, for the language, **detected in code** and stated to the model for that turn instead of relying on the prompt. Each became an eval case; the result is 80/81 over the last three runs.

### Retrieval architecture

Two kinds of retrieval, chosen by what the question needs:

- **Structured data → deterministic tools.** The CSVs are loaded once and normalized (`data.py`); tools filter and join them in pandas (`search_products`, `get_product_details`, `get_order_status`...). Matching is accent- and plural-insensitive, tolerant to separators and typos in product names, and carries Portuguese terms derived from the specs ("clássico" for nylon strings). Results are capped at 8 with the total count, so the agent never silently drops options.
- **Policies → section retrieval.** `policies.py` rebuilds the PDF text, removes page headers and footers, and splits it on the 10 expected headings, searched in order (so a numbered list inside section 7 cannot be mistaken for a heading). `get_policy(topic)` returns one section with the manual's version and, where the manual contradicts itself, `notes` saying which value to use. No embeddings: the manual is 8 pages with its own numbered structure.

The choice between the two is the model's, guided by the tool descriptions, and it is visible in every trace line and checked by the eval (`expected_tools`).

### Prompt strategy

- **Short and behavioral only**: persona and tone (manual 7.1), the service flow (7.2), scope, privacy and security rules, and when to hand off (`prompts.py`). Business rules are not repeated in the prompt: the code enforces them.
- **Current store time injected every turn** (weekday, date, time in Campo Grande), so "are you open now?" and date windows are consistent.
- **"When to use" lives in each tool's description**, next to its schema, not in the prompt.
- **Instructions next to the data.** Where the eval showed a rule being followed inconsistently, the tool result itself carries it: an `instruction` field (say the total of a truncated search, show the alternatives), ready-made `display` text for promotions, `deadline_display` for return windows (section 4, Evaluation).
- **Portuguese**, because the agent talks to Brazilian customers (the code, docs and tool results are in English). The language of each customer message is detected in code; when it is English, that turn's system prompt says so explicitly (the general "answer in the customer's language" rule alone was not followed after a Portuguese conversation).
- **Low temperature (0.2)** for consistency; strict tool schemas so arguments always follow the schema.

### Data used (and not used)

The brief says not all data has to be used. What the agent sees, per file:

| File | Used for | Left out, on purpose |
|---|---|---|
| `products.csv` | Catalog search and details: name, category, price, stock, status, normalized specs | `description` in Baixos, Baterias and Teclados (contradicts name and specs); the raw `strings` field (ambiguous); `created_at` |
| `categories.csv` | Category names (also the enum in the search schema) and the category list | — |
| `promotions.csv` | Active promotions only (`is_active = 1`), largest discount per product | Inactive promotions (shown to nobody; only reported if duplicated) |
| `orders.csv` | Order status, dates, recorded total, payment method, tracking, notes, delays, return windows | — |
| `order_items.csv` | Item names and quantities in an order | — |
| `customers.csv` | **Only** to verify that a contact belongs to an order's customer, and the first name for the greeting after verification | E-mail, phone, full name and city are never sent to the LLM |
| `políticas_da_loja.pdf` | The 10 sections through `get_policy`, the version, the official contact | The cover's marketing text |

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
| Customers use words that are not in product names ("violão clássico", "folk", "eletroacústico") | The search text also carries Portuguese terms derived from the specs: nylon → "clássico", steel → "aço folk", with electronics → "eletroacústico". If the free-text words match nothing but the structured filters do, the search returns the filter results with `query_relaxed` instead of zero. Found in a real run, where the agent told a customer there were no classical guitars (there are 15) |
| Customers write product names loosely ("GD-20", "takamini") | Name lookup ignores separators and tolerates typos word by word; a typo match is flagged `approximate_match` so the agent confirms with the customer. Several matches return candidates instead of a guess |
| Installment minimums (3.1) say nothing about paying in a single installment | 1x on credit is always allowed; the R$ 50 / 80 / 100 minimums apply from 2x on |
| Splitting an amount into installments rarely gives whole cents (R$ 1.967,18 ÷ 10) | Each installment is rounded to the cent (half up, using `Decimal`); the total shown is always the exact amount. In practice the card operator absorbs the cents in one installment |
| Installments for several products bought together | `calculate_installments` takes a list of products and adds their final prices itself; its result says installments are on credit card only and that `can_combine_payments` is about payment methods, not items (both misread in a real session) |
| Installments for a product | `calculate_installments` takes the **product** and looks up its final price itself (promotional when active). In the first real run the model called it in parallel with `get_product_details` and passed the previous product's price; now it never needs to know the price first. `amount` remains for other values (e.g. two items together) |
| How a customer proves an order is theirs (manual 9) | Order number + the e-mail (case-insensitive) or phone registered for **that order's** customer. Phone accepts any formatting and the +55 country code, but needs the area code (DDD). Unknown order, wrong contact and malformed input return the **same** message, so the response never reveals whether an order exists |
| What a verified order lookup returns | Status, date, items, recorded total, payment method, tracking and estimate, notes, and the customer's **first name only** (for the greeting). No e-mail, phone, full name or city: the agent does not need them (data minimization) |
| When an order counts as late | `shipped` past the estimated delivery, or `pending`/`confirmed` for more than 7 days since the order date. The tool returns `late` and `suggest_handoff`; the same threshold drives the data quality report |
| Return windows count from the receipt date, but the data has no actual delivery date | `estimated_delivery` is used as the receipt date, and the tool tells the agent to check with the customer. Regret and preference: 7 days after receipt. Defect: 30 days after purchase, then the 90-day legal warranty (from receipt) and the manufacturer's warranty |
| Section 4.3 gives 7 days for preference exchanges without saying from when | Counted from receipt, like the regret window |
| Section 4.1 is titled "Compras Online"; the data does not say where each order was placed | All orders in the data are treated as online orders |
| Returns before delivery (pending, confirmed, shipped) | Not covered by the manual (order 17 shows cancellations do happen): the tool returns `requires_human` and the agent offers a handoff |
| 4.4 exclusions (customized instruments, "final sale" items, mouthpieces) are not marked in the data | The tool always returns them so the agent asks instead of assuming; in practice it does not always ask (see section 8) |
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
| Every R$ value in an answer comes from the tools | Code | `guardrails.validate_response`, called by the agent loop | pytest |
| Policy text (returns, warranty, privacy...) and which contact to use | Manual | `get_policy` (`text` + `notes`) | pytest |

## 6. Example conversations

Five conversations generated by `python scripts/run_examples.py`, each **a single real run** of the agent (`gpt-4.1-mini`, output guardrail on), not edited. Every tool call can be expanded to see exactly what the agent received. The conversations are in Portuguese, the language of the store's customers; each file opens with an English summary of the scenario and what it shows.

| # | Scenario | What it shows |
|---|---|---|
| [01](examples/01_catalog_search.md) | Guitars up to R$ 1,000 (brief question), then only nylon | Data lookup with filters; says there are 12 options and offers a filter instead of silently showing 8 |
| [02](examples/02_price_payment_promotion.md) | Takamine GD20 price (brief question), then the Crafter on PIX and in 10x | Inactive promotion not applied; active promotion with original price, % and final price; PIX not stacking; installments from a tool |
| [03](examples/03_return_policy_applied_to_an_order.md) | "Me arrependi…" (brief question), wrong e-mail, then the right one | **Main non-trivial case**: identity verified in the tool, wrong contact revealing nothing, regret window and deadline computed in code, manual conditions |
| [04](examples/04_out_of_scope_and_security.md) | Accessories, football, prompt injection | Polite redirection, no invented partner stores, no data about other customers |
| [05](examples/05_store_info_and_unavailable_product.md) | Address (brief question), "open today?" on a Sunday, out-of-stock guitar | Policy lookup for the address, hours and next opening computed, alternatives for an unavailable product |

Scenario 03 runs on 2026-02-20, three days after order 7 was delivered: with the default reference date (2026-03-25) every delivered order is already past its return window. Scenario 05 runs on a Sunday.

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
  "guardrail_triggered": false,
  "unverified_values": [],
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

- **Return exclusions are not always raised.** `check_return_eligibility` always returns the section 4.4 exclusions (customized instruments, "final sale" items, mouthpieces), which the data does not mark, but the agent does not always ask the customer about them (it did not in example 03). Asked directly "can I exchange a guitar with a made-to-order setup?", it answers from the policy in most runs but sometimes asks for the order first (the only eval miss in the last three runs). Next step: an `instruction` next to the data, as for the other rules.
- **Language detection is a word-count heuristic** for Portuguese vs. English only. Other languages fall back to the prompt rule. Next step: a small language-identification library, if the store gets foreign customers.
- **Redundant tool calls.** Asked "is any of these on sale?" right after a search, the model still looks up each product one by one (8 calls), although the search results already carry the promotion. The answer is right, but the turn costs about twice as much. Next step: an eval case that checks the number of calls, then either a tool that answers about several products at once or a per-turn call budget.
- **Order verification can be guessed by brute force.** Order numbers are sequential and there is no limit on failed attempts per session. In production: lock or slow down after a few failures per phone number, and alert on many failures.
- **Only one provider tested end to end.** Everything was run with OpenAI `gpt-4.1-mini`. Other OpenAI-compatible providers (including a free one, which the plan called for) work through `OPENAI_BASE_URL` but were not run; small local models are expected to choose tools worse. Next step: run the same eval per provider and publish the table.
- **The eval is small and text-based.** 20 cases with deterministic text checks catch wrong values and missing steps, not tone or a subtly misleading sentence, and a single run is noisy (section 4, Evaluation). Next step: grow it with real conversations, run each case several times, and add an offline LLM judge for tone and policy adherence.
- **Dates rely on estimates.** There is no actual delivery date in the data, so return windows count from `estimated_delivery` (the tool tells the agent to confirm with the customer). Campo Grande's municipal holidays are not in the calendar.
- **The manual split depends on its exact headings.** If the store renames a section, the program refuses to start (by design) until the heading table in `policies.py` is updated.
- **Data is loaded at startup.** Updating a CSV or the manual means restarting; `handoffs.jsonl` is a local file without locking, fine for one CLI process only.

## 9. Path to production

The central argument: **the tools are a stable contract.** The agent only knows `search_products`, `get_order_status`, `get_policy` and the others, with their schemas. Replacing the CSVs with the ERP's API, or the PDF with a new manual version, changes what is behind the tools, not the agent, the prompt or the eval.

- **Pilot.** A WhatsApp Business webhook (FastAPI) in front of the same agent; history persisted per phone number (Postgres or Redis) with a TTL and verification again in each session; handoffs into the real helpdesk instead of a file. **Migrate the loop to LangGraph when the handoff becomes stateful** (an attendant takes over and hands back, or flows span several steps such as an exchange that waits for a photo): that is the trigger, not before. A web panel for the team (e.g. with the Vercel AI SDK) and the same tools exposed over MCP for other clients.
- **Quality loop.** The eval grows with real conversations and runs in CI before any change to the prompt, the tools or the model; several runs per case; an offline LLM judge for tone and policy adherence; traces sent to Langfuse or OpenTelemetry.
- **Data.** Tools call the real systems: a short cache only for the catalog, never long caches for stock, price or orders. The manual has an owner (management) and a version; each new version runs through the eval before going live. Promotions get start and end dates instead of only `is_active`.
- **Resilience and cost.** Rate limit per phone number at the gateway, a message queue, a daily cost ceiling, error-rate alerts, a circuit breaker and a fallback model.
- **Business metrics.** Share of conversations resolved without a human, handoffs by reason, cost per conversation, response time (the manual's target is ≤ 5 minutes), customer satisfaction.
- **Governance.** Log retention and masking (LGPD), consent before storing conversations, prompt-injection tests in the eval, model changes gated by the eval.

## 10. Use of AI coding assistants

> Drafted with Claude Code from the actual session, then reviewed and adjusted by me.

I used two assistants, with different roles:

1. **Planning with Claude (chat).** I reviewed the case and the data, and wrote the specification by questioning each part of the design: architecture, where each business rule lives, failure handling, data handling, scope. The spec changed at every round; [docs/decision_log.md](docs/decision_log.md) records each question, what changed and why. The spec itself (`CLAUDE.md`, in Portuguese with planning notes) is kept out of the repository.
2. **Implementation with Claude Code**, following that spec one step at a time (the commit plan is the history of this repository). At the end of each step it ran `pytest`, showed the result and proposed the commit message; I reviewed the code, ran it and made every commit myself. Commits carry a `Co-Authored-By` trailer.

What I decided and checked, not the assistant:

- the architecture, the business rules and every assumption in section 5. When Claude Code made a choice the spec did not cover, it had to flag it, I validated it by committing or objected before committing, and it went into the README in the same commit;
- testing it myself, which mattered most: my own CLI session and the first real runs found bugs that 200+ unit tests with a fake LLM could not (an empty `OPENAI_BASE_URL`, an old API key in my system environment overriding `.env`, installments computed with the previous product's price, a search that told a customer there were no classical guitars). Each became its own `fix:` commit;
- the eval as the measure for every behavior fix, reported with its failing runs and not only the final 20/20;
- these documents and the example conversations, which are single unedited runs.

The decision log's "Implementation" section lists the points where I questioned or changed what was generated.
