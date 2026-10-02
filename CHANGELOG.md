# Changelog

All notable changes to `fifty-agent-sdk` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.10.2] - 2026-10-02

### Added
- `SafetyConfig.error_fallback_message`: the text of the `FinalEvent` that ends a run after
  an LLM error or a parser error. It defaults to "Something went wrong while answering.
  Please try again." and must be non-empty. It follows every `ErrorEvent` except the one
  whose `error_type` is `MaxIterationsExceeded`. (BR-021)

### Changed
- After an LLM error or a parser error the run now ends with `error_fallback_message`. A
  parser error here is the native `tool_calls` rejection, or a text parse failure with the
  retry disabled or used up. Before, these runs ended with `fallback_message`, whose
  default reads "I was unable to complete the task within the allowed steps.", and
  `fallback_message` now follows only the iteration cap. The two fields are independent:
  setting only `fallback_message` does not change the error text. **If you set
  `fallback_message`, for example to localised text, set `error_fallback_message` too:
  until you do, your end users see its English default after a provider or parser
  error.** A test that asserts `fallback_message` after an LLM or parser error needs
  updating. (BR-021)
- `SafetyConfig.model_dump()` has one more key, `error_fallback_message`. A config written
  for 1.10.1 validates unchanged. A dump from this release does not validate on 1.10.1,
  whose `SafetyConfig` forbids unknown fields. (BR-021)
- `LLMError.context["type"]` from `OpenAICompatibleClient` takes new values:
  - `ContextLengthExceeded`, when the provider's text contains `max_prompt_length`,
    `context_length_exceeded` or `maximum context length` (compared case-insensitively),
    or the provider's error code is `context_length_exceeded`. It is applied on these
    paths only: the 200 body errors below (`NonJsonProviderBody`,
    `NonStreamProviderBody`), and every `APIError` other than `APITimeoutError`,
    `APIConnectionError` and `RateLimitError` (for example `BadRequestError`,
    `InternalServerError` and a mid-stream error event), on `complete()` and on
    `stream()`. It is never applied to those three on `complete()` or when `stream()`
    opens the request; while `stream()` iterates, `openai` 2.43.0 raises only the base
    `APIError` for an error event. It is never applied to a failure of the read itself
    while a stream is read (an `httpx` read error, for example), or to the bodies the
    client still does not catch or that still escape raw (see Fixed).
    `context["classified_from"]` keeps the type the error would otherwise have had, for
    example `BadRequestError`. The list is closed: other phrasings of the same failure
    stay unclassified, and an unrelated error whose text quotes one of these strings is
    classified too. A release that adds a marker will say so here.
  - `NonJsonProviderBody` for a stream event whose data is not JSON, or is a bare JSON
    string (before BR-021 the first was `JSONDecodeError`, with the decoder's message
    instead of the event's text, and the second `MalformedChunk`).
  - `NonStreamProviderBody` for a streamed 200 whose Content-Type is not
    `text/event-stream` and which produced no chunk (see the next items).

  (BR-021)
- `OpenAICompatibleClient`, `JsonModeParser` and `ProseModeParser` refuse tool arguments
  nested deeper than 64 levels. A level is one JSON object or array (`{"q": [1]}` is 2
  levels; brackets inside strings do not count). In JSON mode the envelope counts one
  more, so `tool_args` may still nest 64 levels. The check reads the text before it is
  decoded.
  - On a native turn, `complete()` raises `LLMError` with `context["type"] ==
    "MalformedResponse"`, the message `provider tool_call arguments nest deeper than 64
    levels`, and `tool_call_id`, `arguments_excerpt` (the first 200 characters, as for
    invalid JSON) and a new key, `max_tool_args_depth` (64), in `context`. One such entry
    refuses the whole response, so no tool of that turn runs, and `AgentLoop` ends the
    run on that turn with `error_fallback_message`, with no further request.
  - The text parsers' depth error is a `ParserError` (`error_phase` `json_decode` or
    `action_input_decode`, with `context["max_tool_args_depth"]`), and the loop takes its
    parser retry for it when that is enabled (the default). A strict-pass text that is too deep
    goes to the parsers' recovery pass first, as a decode failure does, and the depth error
    is raised only when that pass finds no `{`...`}` candidate or one past the limit. A
    candidate within the limit is decoded as in 1.10.1: an unclosed `[draft ` before a
    valid envelope still parses. That pass now also runs for valid JSON that is too deep,
    which 1.10.1 passed to `json.loads` as it stood: for example, the JSON-mode completion
    `[<envelope>, <70 nested arrays>]` now parses to that envelope, where 1.10.1 raised
    `schema_validation` and, with the retry enabled, took the parser retry, and a PROSE
    `Action Input` of
    `[{"a": 1}, <70-level array>]` now dispatches `{"a": 1}`, where 1.10.1 raised
    `Action Input JSON must decode to an object`.
  - Before, arguments that `json.loads` could decode were decoded and dispatched at any
    depth (see Fixed for what then happened on a native turn).
  - The check reads brackets, not JSON, so invalid text can get the depth error too, for
    example `x` followed by 100 `[`, which `json.loads` rejects at its first character. On
    a native turn any text whose brackets open more than 64 levels gets it: before,
    `complete()` raised `provider tool_call arguments is not valid JSON` for it, with the
    decode error as `__cause__`. In the text parsers such text gets it when the recovery
    pass finds no candidate or the check also refuses the candidate (65 levels for the
    whole JSON-mode envelope), as with `x` followed by 100 `[`; before, they raised their
    invalid-JSON `ParserError`. So `context["max_tool_args_depth"]` means the check refused
    the text; do not read it as proof that the arguments were deeply nested.
  - In JSON mode the check covers the whole envelope, not only `tool_args`. An `answer` or
    `thought` value nested more than 64 levels, which the schema rejects anyway, now gets
    the `json_decode` depth error where it used to get `schema_validation` whenever
    `json.loads` could decode it. Both are `ParserError` and take the parser retry when it
    is enabled.
  - The check reads the text once before `json.loads` does. Measured on CPython 3.11.15,
    3.13.2 and 3.14.3: 10 MB of arguments with few brackets took about 6.5 ms, less than
    `json.loads` of the same text; 10 MB of `[[],[],...]` took 0.69 s, 0.85 s and 1.02 s,
    1.5 to 6.2 times `json.loads`, growing linearly from 1 MB. The JSON-mode parser can
    check one completion twice (its strict text, then its recovery candidate).
  - The limit is fixed: there is no setting. A custom `LLMClient` or `Parser` decodes its
    own arguments and is not checked. No public API changes. (BR-019)
- The Runner's `error` audit event has a new key, `error_subtype`: the `ErrorEvent`'s
  `context["type"]` when it is a string (for example `ContextLengthExceeded`), else `None`.
  It is on every `error` event the Runner emits for an `ErrorEvent`, so
  `{"error_type": "LLMError", "error_subtype": "ContextLengthExceeded"}` and
  `{"error_type": "MaxIterationsExceeded", "error_subtype": null}` can be told apart
  without reading logs. (BR-021)
- `OpenAICompatibleClient.stream()` now reads a response whose media type is not
  `text/event-stream`, or that has no Content-Type, in full before yielding its first
  chunk, so a direct caller gets those chunks only once the body has arrived. A
  `text/event-stream` response streams as before. `AgentLoop` emits nothing for a
  streamed turn until it has read the stream up to its terminal chunk, so for a run the
  change adds only the time to receive whatever follows that chunk. If such a response
  produces no chunk at all, `stream()` now raises `LLMError` typed
  `NonStreamProviderBody`, whatever the body holds: an SSE-framed body with only
  `data: [DONE]` included. Before, it yielded nothing. If reading the body fails, the
  error is an `LLMError` whose `context["type"]` is the `httpx` exception's class name
  (for example `ReadError`), with `context["phase"] == "stream"`. (BR-021)
- The release-equivalence statements (`tool_mode` omitted ≡ 1.7.0, request options
  omitted ≡ 1.8.0, `interventions` omitted ≡ 1.9.0) now hold only for runs that do not end
  on an LLM or parser error. Those runs end with `error_fallback_message`, and a streamed
  200 that is not `text/event-stream` and yields no chunk now ends the run after one
  request with an `LLMError`, where with the shipped text parsers it used to end with a
  `ParserError`, after a parser retry when that is enabled. (BR-021) Since BR-019 they
  also hold only for runs in which the 64-level nesting check refuses no text. It refuses
  no text in any golden scenario, whose tool arguments nest at most 1 level. (BR-019)

### Fixed
- JSON that the SDK writes for the model now carries non-ASCII text as literal UTF-8
  instead of `\uXXXX` escapes. By default Python's `json.dumps` writes every non-ASCII
  character as an escape: 6 characters each, 12 for a character outside the Basic
  Multilingual Plane such as an emoji. One Arabic-heavy tool result measured 141,325
  characters escaped and 35,840 unescaped, about 3.9 times shorter. In that measured
  case the previous request had used 12,535 prompt tokens, and the provider rejected the
  request that added the escaped result for exceeding its max_prompt_length of 131,072.
- Three places change:
  - a tool result that is not a string (a dict or list, including an MCP tool's
    `structuredContent` or content blocks), in every tool mode and tool-result role, on
    the single-call path and in a native batch;
  - the argument schemas in the system prompt's text-mode tool list;
  - the `arguments` string of a replayed native tool call.
- The decoded JSON values are unchanged.
- For data whose strings hold only code points U+0000-U+007E, that JSON is unchanged,
  byte for byte. U+007F (DEL) is the one ASCII exception: it is now sent as the
  character rather than as `\u007f`.
- In those three places, a value holding a surrogate code point (U+D800-U+DFFF), which
  UTF-8 cannot encode, keeps the fully escaped 1.10.1 form, which UTF-8 can encode. A tool
  result that is already a string is unchanged by this release.
- Stored data is unchanged: tool results are never persisted, and Redis branch metadata
  keeps its encoding.
- This fix makes no public API changes. (BR-020)
- A provider answering HTTP 200 with a plain-text body now makes `OpenAICompatibleClient`
  raise `LLMError` carrying that body, typed `NonJsonProviderBody` (or
  `ContextLengthExceeded`, see Changed). The message quotes the first 500 characters of
  the stripped provider text, with `…[truncated]` appended when it was cut.
  `context["body_length"]` is the length in characters of that text before stripping: the
  decoded body, the value of a body that is a bare JSON string (so 61 for a 63-character
  body), or the data of the stream event that failed. It used to raise `Malformed provider
  response: 'str' object has no attribute 'choices'`, and the provider's explanation was
  lost. The same holds for a body that is a bare JSON string, and for a body labelled
  `application/json` whose UTF-8 text is not JSON, which used to escape `complete()` and
  `AgentLoop.run()` as a raw `json.JSONDecodeError`, with no `ErrorEvent` or
  `FinalEvent`. Two bodies with a JSON Content-Type still escape raw, as before: one that
  is not valid UTF-8 (`UnicodeDecodeError`; measured labelled `application/json`,
  `application/problem+json` and `text/json`) and one whose integer literal exceeds the
  interpreter's digit limit (`ValueError`; measured labelled `application/json`).
  `AgentLoop.run()` catches only `LLMError`, so such a run ends with no `ErrorEvent` or
  `FinalEvent` (measured labelled `application/json`).
- A streaming request answered with such a body looked like an empty stream: with the
  parser retry on (the default), the run ended with a `ParserError` after two provider
  calls. It now raises `LLMError` typed `NonStreamProviderBody` (or
  `ContextLengthExceeded`) after one, unless the body is labelled `text/event-stream` (see
  below).
- Some provider answers are still not caught (the raw escapes are in the item above). A
  non-streamed 200 whose body is a JSON object without `choices`, such as an error
  envelope `{"error": {...}}` (measured labelled `application/json` and `text/plain`),
  still raises `MalformedResponse` (`Provider returned no choices.`) with its text lost,
  so it is never classified; the same body on a streamed request that is not labelled
  `text/event-stream` raises `NonStreamProviderBody` carrying its text, classified
  `ContextLengthExceeded` when that text holds a marker. A body without SSE fields (plain
  text, or a JSON error envelope) labelled `text/event-stream` still looks empty, because
  the `openai` client's SSE decoder drops it before the SDK sees it.
- An over-long prompt whose provider error uses one of the three markers, on one of the
  paths listed under Changed, is classified `ContextLengthExceeded`, so a consumer can
  react to it, for example by trimming a tool result and retrying.
- The end user is no longer told the steps ran out when the provider failed: the run ends
  with `error_fallback_message` (see Added).
- Where the quoted provider text goes: `LLMError.message`, `ErrorEvent.message`, the
  Runner's `error` audit `error_message` (`ConsoleAuditSink` logs it, `SqlAuditSink` stores
  it) and the `on_error` hook, the places a provider's 4xx/5xx text already reached, in
  full. It is not in `FinalEvent.text`, in `LLMError.context`, in the state store, or in
  the log lines the loop, the client and the Runner write for such a run with no audit
  sink wired. Treat `ErrorEvent.message` as diagnostic text, not text for end users.
- How these bodies reach the SDK is the `openai` client's behaviour, measured on `openai`
  2.43.0. The BR-021 test suite also passed with `openai` 2.54.0 (on CPython 3.11.15 and
  3.13.2). (BR-021)
- A native tool call whose `arguments` `json.loads` decoded but the SDK could not
  re-encode for the next request ran the tool, and building that request then raised a raw
  `RecursionError` out of `AgentLoop.run()`, with no `ErrorEvent` or `FinalEvent`.
  - Reproduced on CPython 3.14.3 at 100,000 levels of nested objects, for a single call
    and a batch, with and without intervention hooks. 100,000 levels of nested arrays
    still re-encoded there, and that run completed.
  - Reproduced on 3.11.15 in a band of 3 levels just below the depth at which decoding
    failed. No such band was found on 3.13.2.
  - Arguments deeper than `json.loads` itself decodes raised the `RecursionError` out of
    `complete()` instead, before any tool ran: the `ValueError` arm around the decode does
    not catch it (reproduced on 3.11.15, 3.13.2 and 3.14.3).
  - Both now end as described under Changed.
  - Why only those interpreters and depths, measured with stdlib `json` from a module's
    top level: `json.loads` and `json.dumps` both reach 995 levels on CPython 3.11.5 and
    3.11.15, and 9998 on 3.13.2, for nested objects and nested arrays alike. On 3.14.3
    `json.loads` reaches 116,213 levels, and `json.dumps` 61,525 for nested objects and
    104,591 for nested arrays. The 3.11.15 band comes from the SDK re-encoding the
    replayed arguments 3 Python frames deeper than it decodes them. Inside an async
    pytest test each of these limits was lower: 948 on 3.11.15, 9976 on 3.13.2, and
    about 116,100 / 61,470 / 104,500 on 3.14.3, where they moved by a level between
    runs. (BR-019)

## [1.10.1] - 2026-09-28

### Fixed
- Intervention hooks now get a complete copy of the arguments the shipped parsers and LLM
  client decode, however deeply the model nests them. In 1.10.0, JSON nested 500 levels
  deep, which they decode, made `copy.deepcopy` raise `RecursionError` on Python 3.14.3,
  so a model could cut each hook's copy down to one level (a `before_tool` edit to a
  nested value then reached the dispatched arguments) and trigger the
  `intervention.args_not_copyable` WARNING. Values of the exact types `json.loads` builds,
  reached from the arguments through exact dicts and lists, are now handled without
  recursion: dicts and lists are walked on an explicit stack (tested at 5000 levels), and
  strings, numbers, booleans and `None` are kept as they are. Any other value, with
  everything inside it, still goes through `copy.deepcopy`, and the copied structure,
  shared references and cycles included, comes out as before. The fill order did change:
  a non-JSON value whose `__deepcopy__` or `__setstate__` reads another dict or list of
  the arguments during the copy may now see that container's copy still empty. Apart from
  running out of memory, only a non-JSON value that cannot be copied can still take the
  one-level fallback with its WARNING, and only host code supplies one: a custom parser
  or `LLMClient`, or, for `after_tool`'s copy, a `ReplaceToolArgs` replacement or code
  that stores one into the arguments' nested values (a tool, or an event consumer writing
  into `ActionEvent.args`). No public API changes, and with no hook set no extra argument
  copy is made, as before. (TD-010)

## [1.10.0] - 2026-09-28

### Added
- `Interventions` and `AgentLoop(interventions=...)`: hooks whose return values the loop
  honours. They are separate from the observability `Hooks`, whose contract is unchanged,
  and they are wired on the loop only (`AgentRunner` takes none; it already passes its
  `session_id` down, and the loop forwards it). A value that is not an `Interventions`
  raises `TypeError` at construction. (FR-003)
- `after_tool(session_id, call_id, tool_name, args, result)`. It runs once for each
  dispatched call that returned a `ToolResult`, a success or `is_error=True`, and never
  for `ToolNotFound`, `ToolTimeout`, a denied call or a fatal error. It runs after that
  call's terminal event has been yielded: under a runner, after the consumer handled the
  event and `on_tool_end` fired; in a native batch, one call at a time in call order. A
  non-blank string it returns is appended verbatim, after a blank line, to that call's
  model-facing observation, in every tool mode and tool-result role; `None` or a blank
  string adds nothing. The SDK never writes to the `ToolResult` (passed by identity,
  read-only), and the events are the ones the run emits without the hook. It fails soft:
  if it raises or returns anything other than a string or `None`, no note is added and a
  WARNING is logged (`intervention.hook_failed` / `intervention.hook_invalid`, type only,
  never the exception's text).
- `before_tool(session_id, call_id, tool_name, args)`, with the decisions `DenyToolCall`
  and `ReplaceToolArgs`. It runs before each tool call's `ActionEvent`, including calls
  to unregistered names; `None` proceeds. A denied call is not run and keeps the event
  shape an unregistered name has: `ActionEvent`, `ToolStartedEvent`, then
  `ToolFailedEvent(error="Tool call denied: <reason>")`, and the model reads the same
  text. A replacement is dispatched, and `ActionEvent.args`, `on_tool_start` and the
  `tool_invocation` audit payload carry it; it is never written into the model's own
  assistant turn.
- Each hook gets its own deep copy of the arguments: `before_tool` of the model's, taken
  before dispatch; `after_tool` of the dispatched ones, taken after the tool returns. The
  tool itself still gets a one-level copy, as in 1.9.0, so nested values in `after_tool`'s
  copy may carry in-place edits the tool made to its own arguments. Nothing a hook does to
  its copy, even to a nested value, reaches the dispatch, the events, the model's turn or
  the other hook; to change arguments, return `ReplaceToolArgs`. Arguments that cannot be
  deep-copied (an object a custom parser or a custom `LLMClient` put into `ToolCall.args`,
  or extreme nesting) are copied one level deep instead, and a WARNING
  `intervention.args_not_copyable` is logged; the hook still runs. The model controls how
  deeply its arguments nest, so it can cause this fallback. Never rely on an in-place edit
  staying private: change arguments with `ReplaceToolArgs`, and alert on that WARNING.
- `Interventions(before_tool_fallback=...)` and `BeforeToolFallback`. The consumer
  chooses what happens when `before_tool` raises or returns something unusable:
  - `BeforeToolFallback.DENY`, the default, fails closed: the call is denied with the
    SDK's own reason.
  - `BeforeToolFallback.ALLOW` fails open: the call runs with the model's original
    arguments, never the hook's copy and never an invalid replacement.

  Either way a returned `DenyToolCall` is honoured, and a WARNING is logged with
  `fallback="call_denied"` or `"call_allowed"`. Use `DENY` for guards and for
  argument-scoping hooks: a scoping hook (inject the tenant, clamp a limit) that fails
  open dispatches the model's unscoped arguments. Use `ALLOW` only for advisory hooks,
  where availability matters more than the check, and alert on `"call_allowed"`. The
  plain strings `"deny"` and `"allow"` are accepted; an unknown value raises `ValueError`
  at construction.
- `BeforeToolHook` and `AfterToolHook`, the public types of the two callables. Both may
  be sync or async, including callable objects and `functools.partial`.

### Notes
- With `interventions` omitted, the loop sends the same request bodies (same keys,
  values and JSON types) and emits the same event stream as 1.9.0. This is pinned by the
  1.7.0 and 1.8.0 goldens plus a new golden captured from 1.9.0, before any change, that
  also records the event stream (types, order, `sequence` and payloads; timestamps
  excluded, call ids normalised) and covers native batches with unregistered and timed-out
  members. The same capture run against the published 1.9.0 wheel produced an identical
  file. Key order holds by construction; the HTTP bytes the `openai` SDK sends were not
  measured.
- Persistence: a note or a denial text is part of that call's observation for the rest of
  the run, so later requests in the same run carry it. It is never written to the state
  store, like every tool observation; a later turn sees only the final answer.
- `ActionEvent.args` now means the arguments a call is dispatched with. Only a host that
  returns `ReplaceToolArgs` sees a difference.
- There is no `before_llm_call` hook. To change a request, wrap the `LLMClient`.
- `MCPClient(on_tool_error=...)` remains the way to replace MCP `isError` text:
  `after_tool` only appends, and the two compose.
- The hooks are called positionally and their signatures will not grow within 1.x; new
  data arrives as a new hook or field.

## [1.9.0] - 2026-09-26

### Added
- `ChatRequest.reasoning_effort` and `AgentLoop(reasoning_effort=...)`. The loop sets it
  on every request it builds (tool steps, parser-retry and require-tool-before-final
  re-asks, native and streamed turns). `OpenAICompatibleClient` sends it as a top-level
  `reasoning_effort` for any model when it is set. There is no model-name filter, so a
  provider or model that rejects it surfaces as `LLMError` (in the loop: `ErrorEvent`
  and the fallback `FinalEvent` on the first call). The value must be a lowercase token
  such as `"low"`, `"medium"` or `"high"`; other levels are passed through verbatim for
  the provider to judge. The string `"none"` is a level and is sent; Python `None`
  (the default) omits it. An invalid value raises at construction: `ValueError` from
  `AgentLoop`, pydantic `ValidationError` from `ChatRequest`. (FR-002)
- `AgentLoop(temperature=...)`. Omitted, the loop keeps sending `temperature` `0.0` as
  before. A number in `[0.0, 2.0]` is sent on every request the loop builds. `None`
  omits `temperature` from every request, for models that reject sampling parameters.
  For example, OpenAI documents that gpt-5.1 accepts `temperature` only with
  `reasoning_effort="none"`; this SDK has not verified that behaviour against any live
  provider. It is independent of `reasoning_effort`: the loop never drops `temperature`
  on its own, so pass `temperature=None` yourself if your provider needs it. Invalid
  values (out of range, NaN, `bool`, strings) raise `ValueError` at construction.
  (FR-002)

### Notes
- With both kwargs omitted, and `reasoning_effort` unset on a direct `ChatRequest`,
  request bodies keep their 1.8.0 content: the same keys, values and JSON types. This is
  pinned by the 1.7.0 legacy golden and a new golden captured from 1.8.0 that covers the
  explicit `tool_mode` shapes and direct-client shapes. Key order holds by construction;
  the HTTP bytes the `openai` SDK sends were not measured.
- Custom `LLMClient` implementations receive the new `ChatRequest.reasoning_effort` field
  and must forward it themselves. One that ignores it drops the value silently.
- `ChatRequest.model_dump()` now includes `reasoning_effort: None`, so a snapshot of a
  request dump (for example in an `on_llm_call` hook) sees one new key.
- `reasoning_effort` is sent through the `openai` SDK's `extra_body`, because the typed
  `create(reasoning_effort=...)` parameter only exists from `openai` 1.58.0. The
  dependency floor stays `openai>=1.30.0`.

## [1.8.0] - 2026-09-25

### Added
- `ToolMode` (`JSON`, `PROSE`, `NATIVE`) and `AgentLoop(tool_mode=...)`. One value now
  sets the text parser, output format, tool-result role, tool declaration and
  parser-retry reminder together, so a loop can no longer be half native and half text.
  A `StrEnum`, so `tool_mode="native"` from configuration also works. (FR-001)

  | | `JSON` | `PROSE` | `NATIVE` |
  |---|---|---|---|
  | tools declared via `tools` param | no | no | yes, `tool_choice="auto"` |
  | prompt tool block | rendered | rendered | suppressed |
  | text parser | `JsonModeParser` | `ProseModeParser` | final-only: text without `tool_calls` is the answer |
  | output format | `JSON_MODE_OUTPUT_FORMAT` | `PROSE_MODE_OUTPUT_FORMAT` | none (plain-text final) |
  | tool-result role | `"assistant"` (`"user"` allowed) | same as JSON | always `"tool"`, paired by `tool_call_id` |
  | `stream=True` | allowed | allowed | rejected at construction |

- Passing `tool_mode` together with a knob it owns (`parser`, `output_format` or
  `prompts.output_format`, `tool_message_role`, `SafetyConfig.native_tools_enabled`)
  whose value belongs to another mode raises `ValueError` at construction, naming the
  mode, the knob and the fix. The loop never silently picks one side. Compatible values
  are honoured, for example a custom parser wrapping `JsonModeParser` under `JSON`, or a
  custom output format such as "answer in markdown" under `NATIVE`.
- Under `ToolMode.NATIVE` a text completion is never dispatched as a tool call: a JSON
  `{"action": "tool", ...}` envelope or a prose `Action:` block without `tool_calls` is
  the final answer, verbatim. A `role="tool"` reply is therefore only ever sent after an
  assistant turn whose `tool_calls` carry its id. An empty completion triggers the
  existing one-shot parser retry with a native reminder; when no retry is left the run
  ends with `ErrorEvent(error_type="ParserError")` and the fallback `FinalEvent`.
- `ToolMode.NATIVE` with an empty registry sends neither `tools` nor `tool_choice`
  (OpenAI rejects an empty `tools` array). A tool registered later is declared on the
  next request.
- Mode-appropriate parser-retry reminders: when `SafetyConfig.parser_retry_reminder` is
  left at its default, `PROSE` and `NATIVE` loops no longer tell the model to emit JSON.
  A reminder you set explicitly is used verbatim in every mode.
- An explicit `JSON` or `PROSE` loop ignores native `tool_calls` it never asked for,
  parses the text instead, and logs a `native_tool_calls_ignored` warning carrying the
  call count only. Under an explicit mode a blank completion is also not echoed back as
  an empty assistant turn before the retry reminder.

### Changed
- `NativeToolsParser` returns non-blank assistant `content` sent alongside `tool_calls`
  (stripped) as the turn's `thought`, for single- and multi-call turns; it was always
  `""` before. **Consumer-visible:** `ThoughtEvent.text` on a native tool turn may now be
  non-empty, including on the `native_tools_enabled=True` path. Requests are unchanged:
  the replayed assistant turn still carries the verbatim `content`.
- `AgentLoop(parser=...)` is optional when `tool_mode` is given. Without `tool_mode` it
  is still required and omitting it still raises `TypeError`.
- `AgentLoop(tool_message_role=...)` now defaults to `None`, meaning "not set". With
  `tool_mode` omitted it still resolves to `"tool"`, so existing callers see no change.

### Migration
Omitting `tool_mode` sends the same request bodies as 1.7.0, with the same keys, values
and JSON types (pinned by golden request bodies captured from 1.7.0), so no existing
constructor call needs to change.
`SafetyConfig.native_tools_enabled` is not deprecated at runtime. It stays the legacy
knob, and it keeps the 1.7.0 behaviour of leaving a text tool call dispatchable. To get
a consistent native loop:

```python
# Before (1.7.0): native declaration, but a text tool call can still be dispatched.
loop = AgentLoop(
    llm=llm,
    registry=registry,
    parser=JsonModeParser(),
    prompts=PromptSections(persona="..."),
    safety=SafetyConfig(native_tools_enabled=True),
    model="gpt-5.1",
    output_format=JSON_MODE_OUTPUT_FORMAT,
)

# After (1.8.0): drop parser=, output_format= and native_tools_enabled.
loop = AgentLoop(
    llm=llm,
    registry=registry,
    prompts=PromptSections(persona="..."),
    safety=SafetyConfig(),
    model="gpt-5.1",
    tool_mode=ToolMode.NATIVE,
)
```

Keeping `parser=JsonModeParser()` or `output_format=JSON_MODE_OUTPUT_FORMAT` next to
`tool_mode=ToolMode.NATIVE` raises `ValueError`, as does an explicitly set
`native_tools_enabled=False`. The final answer is plain text, so a consumer that read the
`answer` field of a JSON envelope now reads `FinalEvent.text` directly. Under an explicit
`JSON` or `PROSE` mode, tool results default to `role="assistant"`; pass
`tool_message_role="user"` for chat templates that require strict user/assistant
alternation.

## [1.7.0] - 2026-09-25

### Added
- `OpenAICompatibleClient(max_tokens_param=...)` chooses the request-body key that carries
  `ChatRequest.max_tokens`: `"max_tokens"` or `"max_completion_tokens"`. The default,
  `None`, picks the key from the model name (see Fixed). Set it explicitly when the model
  name does not reveal the family (for example an Azure deployment name), or to force
  `"max_tokens"` for a gateway that does not understand `max_completion_tokens`.

### Fixed
- `OpenAICompatibleClient` now sends `ChatRequest.max_tokens` as `max_completion_tokens`
  for OpenAI reasoning-model families (`gpt-5*`, `o1`, `o3`, `o4`, including behind
  a `provider/` or `ft:` prefix). Before this change it always sent `max_tokens`, which
  these models reject with HTTP 400 ("Unsupported parameter: 'max_tokens' ... Use
  'max_completion_tokens' instead"), including through OpenAI-compatible gateways. Every
  other model keeps the `max_tokens` wire shape. On reasoning models the cap also counts
  reasoning tokens, so a small cap can use up the budget before any visible output.

## [1.6.1] - 2026-09-14

### Changed
- **Source-compatible in-memory retention change:** `MemoryStateStore()` now lazily expires
  whole sessions after 3,600 seconds of inactivity and applies a 1,000-session LRU cap.
  Reads refresh inactivity; active and queued session operations are never evicted. Callers
  that intentionally require the former unbounded process-local behavior can construct
  `MemoryStateStore(ttl_seconds=None, max_sessions=None)`. (BR-012)

## [1.6.0] - 2026-09-14

### Added
- `OpenAICompatibleClient` gains `aclose()` and async context-manager support,
  so the underlying httpx client can be disposed deterministically.
- `ChatRequest.tool_choice` accepts the specific-tool dict form (forcing one
  named tool), alongside the existing string forms.
- `temperature=None` omits the parameter from the request body instead of
  sending it, letting a provider's own default apply.

### Fixed
- Python 3.11 package imports work with the named-tool `tool_choice` form:
  `TypedDict` now comes from the directly-declared `typing-extensions`
  dependency, as required by Pydantic on Python versions below 3.12. (BR-014)
- All five direct `json.loads` boundaries now contain bare `ValueError` as
  `ParserError` or `LLMError`, including CPython's oversized-integer guard,
  while preserving the existing malformed-syntax phases, messages, context,
  and exception chaining. (BR-013)
- Parser recursion-limit regressions now inject `RecursionError`
  deterministically instead of relying on interpreter-specific behavior from
  a 100,000-level JSON value, restoring portable Python 3.14 coverage.
  (BR-017)
- **Audit payload shape change (consumer-visible for `AuditSink`
  implementors):** the `tool_invocation` payload's `args` field no longer
  embeds the raw argument dict — it is now per-key metadata (sorted argument
  keys with each value's type name and length, `len=None` for unsized
  values), closing a leak of secrets/PII into persisted audit payloads. The
  `on_tool_start` hook still receives the full args.
- The runner now correlates tool invocations by `call_id` across
  `MultiAction` batches — previously the first terminal event inherited the
  last call's `call_id`/args and every other call was audited with
  `args={}`.
- A fatal `AgentSdkError` escaping the loop (e.g. `MCPError`) is now
  surfaced to hooks and audit before re-raising: the error audit event is
  emitted, `on_error` fires, and the run records
  `terminated_by="sdk_error"` instead of exiting as `"interrupted"` with
  `on_run_end(error=None)`.
- Hook-failure logs now carry `hook_name` and `error_type` only, never
  `str(exc)` — a raising hook could previously dump conversation content
  into a WARNING log line.
- `SqlStateStore` persists `ChatMessage.tool_calls` (nullable JSON column;
  JSONB on Postgres), so a persisted native-tool-calling assistant turn no
  longer loses its `tool_calls` — and orphans the paired `role="tool"`
  replies — on session resume. Additive, via the existing consumer-owned
  migration path; the SDK still ships no migrations.
- `RedisStateStore` rejects non-positive `ttl_seconds` at construction —
  `ttl_seconds=0` previously made the append's `EXPIRE` delete every session
  key immediately.
- Every Redis mutation (`append`, `fork`, `switch_branch`, and
  `truncate_after`) now runs through one optimistic transaction that refreshes
  every session key to the same sliding TTL. Registry/active/message-key
  conflicts retry from a fresh snapshot up to a bounded limit, preventing
  metadata keys from outliving their message lists. (BR-015)
- The loop rejects `stream=True` combined with `native_tools_enabled` at
  construction instead of misbehaving later.
- A `RecursionError` from pathologically nested JSON is translated into
  `ParserError` (`error_phase` `json_decode` / `action_input_decode`)
  instead of escaping the parser contract.
- The JSON-mode parser strips `tool_name` and rejects a whitespace-only one,
  which previously produced a `ThoughtAction` the registry could never
  match.
- An MCP auth callable that raises (e.g. a down token endpoint) is
  translated into `MCPError` — only the exception type name is captured,
  never its text — instead of escaping the MCPError-only contract and being
  downgraded to a model-recoverable `ToolResult`.
- A `CancelledError` arriving as a leaf of a mixed transport
  `BaseExceptionGroup` re-raises untouched instead of being translated into
  `MCPError`, restoring the cancellation contract.
- Tool schemas sent to the LLM now have `#/$defs` references inlined, so a
  nested `BaseModel` parameter no longer reaches the model as a dangling
  `$ref`; recursive models are rejected at decoration time for `@tool` and
  fall back to an empty schema for untrusted MCP server schemas. Expansion is
  additionally capped at 10,000 total resolver visits, preventing acyclic
  fan-out from exhausting memory; MCP fallback logs contain only stable
  reason/type metadata, never remote schema text. (BR-016)

### Changed
- Release tag validation now reads the package version through `tomllib`, with
  executable matching- and mismatched-tag regression coverage.

## [1.5.0] - 2026-07-30

### Added
- **A public accessor for a Runner's state store.** `AgentRunner` gains a
  read-only `state` property returning the `StateStore` it reads and writes.
  It returns the **exact instance** passed as the `state=` constructor keyword
  — by identity, never a copy and never a wrapper — so a caller sharing it
  shares the store's internal serialization (for example `SqlStateStore`'s
  per-session `asyncio.Lock` registry). Constructing a second store over the
  same engine is **not** equivalent: two stores carry independent lock
  registries, so two writers could interleave on one session. The accessor is
  named for its constructor keyword, making `AgentRunner(loop=…, state=s).state
  is s` a derivable invariant. Reachable from the package root today —
  `AgentRunner` and `StateStore` are both already exported.

  It is **read-only** for correctness, not style: `run()` appends the user
  message, drives the loop, then appends the assistant message, so a store swap
  landing between those appends would split one turn across two backends and
  void the documented transactional-persistence invariants. To use a different
  store, construct another Runner — `__init__` does no I/O. Assignment raises
  `AttributeError`, and `mypy` rejects it statically (there is deliberately no
  raising setter, which would make the assignment type-check as legal). The
  declared return type is the `StateStore` protocol, so a caller needing
  backend-specific API that is not on it (e.g. `SqlStateStore.aclose()`) should
  keep its own concretely-typed reference or narrow with `cast`.

  **This is the supported replacement for reaching into
  `AgentRunner._state`.** That attribute remains private, unsupported, and
  carries no semver protection — it is unchanged and source-compatible in
  1.5.0, so nothing breaks on upgrade, but consumers reading it should migrate
  to `AgentRunner.state`. (BR-011)

### Fixed
- **Branch materialization no longer recurses per lineage hop.** All three
  backends walked a branch's lineage with one Python frame per hop, so a
  pathological *linear* fork chain (~990 deep) raised `RecursionError`. The five
  affected walks — `MemoryStateStore._materialize`,
  `SqlStateStore._materialize_positional` / `._materialized_len`, and
  `RedisStateStore._materialize` / `._materialized_len` (the last two `async`,
  where an `await`ed recursive call consumes a frame just the same) — now walk
  parent pointers into a list and fold root-to-leaf. Lineage depth is bounded
  only by memory.

  The failure bit at **build** time before read time was reachable: `fork` computes the active
  branch's head through the same walk to bound `from_sequence`, so on affected
  versions a chain that deep could not be constructed through the public API at
  all. Realistic branching is wide rather than 1000-deep-linear, which is why
  this went unnoticed.

  **No API and no behaviour change.** No signature moves (all five are private
  helpers), and the fold evaluates the identical expression with the identical
  associativity — the recursive form was already a left fold from the root, so
  each hop's `min` / prefix-slice clamp still sees the already-clamped ancestor
  length. The existing differential fuzz suite passes unmodified. On Redis the
  per-hop `LRANGE`/`LLEN` count and command set are unchanged; only the call
  *order* flips from leaf-to-root to root-to-leaf, and multi-key reads were
  never atomic in either order.

  The brief's optional half — wrapping these paths in `StateStoreError` — was
  considered and **declined**: with the walk iterative the `RecursionError` it
  guarded against is unreachable, and a broad `except Exception` there would
  swallow the protocol-mandated `ValueError` for an unknown explicit `branch_id`
  as well as Redis's deliberate pydantic `ValidationError` pass-through. No
  lineage-cycle guard was added either — `parent_branch_id` is written once, at
  `fork`, to an already-existing branch and never mutated, so the lineage is
  acyclic by construction. (TD-002)

## [1.4.0] - 2026-07-29

### Added
- **A public hook for transforming MCP `isError` text before it reaches the
  model.** `MCPClient` gains a keyword-only `on_tool_error` callback (typed
  `MCPToolErrorHook`, exported from the package root) that receives BOTH the
  SDK's bounded default message (`"MCP tool '<name>' returned isError=True"`)
  AND the server's raw error `content` blocks, and returns the string the model
  will see as `ToolResult.error`. Wire it through plain construction —
  `MCPClient(config, on_tool_error=screen)` then `MCPProvider(client)`; no
  subclassing required. The hook may be sync or `async def` (the return value is
  inspected with `inspect.isawaitable`), and it fires exactly once per per-call
  `isError=True` result — never on success, and never on a transport/protocol/
  session failure, which still raises `MCPError` at the `call_tool` boundary
  before the result is unwrapped (the BR-005 recoverable/fatal split is
  unchanged). It cannot change `is_error` or `output`: a failed tool is never
  reported as a success. On ANY hook failure the original bounded message is
  used — a raising hook is caught and logged `WARNING`
  (`mcp.tool_error_hook_failed`), and a non-`str` or blank/whitespace-only
  return is rejected and logged `WARNING` (`mcp.tool_error_hook_invalid`); those
  logs carry `tool_name`/`error_type`/`returned_type` only, never the exception
  text and never the server content. `asyncio.CancelledError` propagates
  untouched. **Default `None` — the hook-off path is byte-for-byte the 1.3.0
  path**, short-circuiting before any call or allocation.

  **This is the supported replacement for importing
  `fifty_agent_sdk.mcp.client._MCPCallError` or overriding
  `MCPClient._unwrap_invoke_result`.** Those symbols remain private,
  unsupported, and carry no semver protection — they are unchanged and
  source-compatible in 1.4.0, so nothing breaks on upgrade, but consumers doing
  either should migrate to `on_tool_error`. (BR-010)

## [1.3.0] - 2026-07-01

### Added
- **Concurrent multi-call tool dispatch (opt-in).** A single native ReACT
  iteration can now dispatch up to N independent tool calls concurrently under
  a bounded `asyncio.gather`, feeding all observations back to the model in
  one turn. Set `SafetyConfig(max_concurrent_tool_calls=N)` (default `1` —
  serialized, so a caller who enables native multi-call parsing without
  raising the cap sees no pool surprise) alongside `native_tools_enabled=True`.
  A native response carrying >1 `tool_calls` yields a new `MultiAction` parse
  result; the loop mints one distinct `call_id` per call (carried on the new
  `ToolCall.id` field), embeds all N on the replayed assistant turn, dispatches
  them through a `Semaphore`-bounded gather with `return_exceptions=True`, and
  appends observations in CALL order (deterministic regardless of completion
  order). Per-call recoverable failures (`ToolNotFound`/`ToolTimeout`/`is_error`)
  yield a `ToolFailedEvent` for that call only — siblings still complete; fatal
  `AgentSdkError`/`BaseException` re-raise and terminate. The batch counts as
  ONE `max_iterations` unit. **Single-call turns are byte-for-byte unchanged**
  — a 1-entry native response still yields `ThoughtAction` and the gather is
  unreachable; the text/JSON path never produces `MultiAction`.
  (`OpenAICompatibleClient._serialize_message` sources each wire
  `tool_calls[].id` from the entry's `ToolCall.id`, falling back to
  `tool_call_id` for the single-call path.) (BR-006)
- **Native function-calling (opt-in).** The agent loop can now dispatch tools
  from a provider's structured `tool_calls` instead of only from JSON-mode text.
  Set `SafetyConfig(native_tools_enabled=True)` and the OpenAI-compatible
  adapter declares the registry's tools via the `tools`/`tool_choice` request
  params (built from each tool's `ToolSchema`); a provider returning
  `choices[].message.tool_calls` is dispatched through the new concrete
  `NativeToolsParser`, and assistant `tool_calls` turns round-trip on the
  request wire in valid OpenAI envelope shape (`{id, type:"function",
  function:{name, arguments(JSON string)}}`) so multi-turn native conversations
  replay without a 400. `ChatMessage` gains an optional `tool_calls` field; the
  adapter populates it (parsing the provider's JSON-string `arguments`,
  `LLMError(type="MalformedResponse")` on a malformed envelope). The
  `tool_call_id` is minted before the native assistant-turn append and reused
  for the tool reply so provider replay pairs correctly. **Default OFF** — the
  text/JSON parse path (BR-018-hardened) is byte-for-byte unchanged when the
  flag is off; native mode is an explicit per-deployment choice. The
  `NativeToolsParser` Protocol is renamed `NativeToolsParserProtocol`.
  (BR-007, BR-008)

### Changed
- A native response carrying more than one `tool_calls` entry no longer
  truncates to the first call (the BR-007 scope limitation). The
  `NativeToolsParser` now validates EVERY entry's schema and returns a
  `MultiAction` carrying the full list; the `native_tool_calls_truncated`
  DEBUG log is removed. Concurrent/batched dispatch landed in BR-006 (above).

## [1.2.1] - 2026-06-25

### Fixed
- An MCP tool returning `isError=True` is now a **recoverable** tool
  observation instead of a run-terminating error. Previously the MCP adapter
  raised `MCPError` on `isError=True`; that exception escaped the agent loop
  (which only caught `ToolNotFound`/`ToolTimeout`) and propagated out of
  `Runner.run()`, crashing the turn with a raw `MCPError`. Now a per-call
  `isError=True` is surfaced as a `ToolResult(is_error=True)` — the model
  receives it as a `"Tool error: …"` observation and produces a grounded
  reply — converging MCP tool failures onto the exact same recoverable path as
  native `is_error`, `ToolNotFound`, and `ToolTimeout`. Genuinely-fatal
  transport / protocol / session `MCPError`s still terminate the run as before.
  (BR-005)

## [1.2.0] - 2026-06-22

### Added
- First-class conversation **branching** on `StateStore`: `fork`,
  `list_branches`, `switch_branch`, branch-scoped
  `get_messages(..., branch_id=...)`, plus the `BranchInfo` and
  `TRUNK_BRANCH_ID` exports. A session is now a tree of branches with an active
  head; `append` writes to the active branch (the "edit a message / regenerate"
  model). Implemented across all backends (memory, SQL, Redis). The change is
  **data-additive and zero-migration** — existing sessions read as the `trunk`
  branch. **Breaking for custom `StateStore` implementations**: they must add
  the new methods. (BR-004)
- `StateStore.truncate_after(session_id, sequence, *, branch_id=None)` — a
  destructive hard-delete of a branch's tail (messages with `sequence > N`),
  for redaction, retention, and rollback. Only the target branch's own messages
  are removed — a fork's inherited prefix is never touched — and it is
  idempotent / a no-op on an unknown session or branch. (BR-003)

### Fixed
- `Registry.invoke` now enforces timeouts via `asyncio.timeout` instead of
  `asyncio.wait_for`, running the tool coroutine inline in the caller's task.
  This makes `KeyboardInterrupt`/`SystemExit` propagation deterministic across
  Python 3.11–3.13 and fixes a pytest-session abort on the 3.11 CI leg. (BR-002)

### Changed
- `fifty_agent_sdk.__version__` is now derived from installed distribution
  metadata (`importlib.metadata`) rather than a hardcoded string, so it can no
  longer drift from `pyproject.toml`. (TD-001)

## [1.1.1] - 2026-06-22

### Fixed
- `fifty_agent_sdk.__version__` now reports the correct release version. It was
  pinned at `1.0.0` and missed the 1.1.0 bump; all version sources
  (`pyproject.toml` and `__init__.py`) are now in agreement.

## [1.1.0] - 2026-06-22

First public open-source release as a standalone package (`fifty-agent-sdk`),
extracted with its full commit history from the monorepo it was first built in.

### Added
- Standard MCP client over Streamable HTTP via the official `mcp` SDK, exposed
  through `MCPProvider` (full `initialize → tools/list → tools/call` handshake).
  The MCP path is now standard-only.

### Changed
- Import root is now `fifty_agent_sdk` (was `agent_sdk`).
- Distributed and published as `fifty-agent-sdk` on PyPI.

## [1.0.0] - 2026-06-19

Initial production release: custom ReACT loop, JSON-mode tool calling, a
pluggable LLM client (any OpenAI-compatible endpoint), in-process + MCP tool
sources, pluggable conversation-state storage (memory / SQL / Redis), audit
sinks, observability hooks, and a full-fidelity event stream.
