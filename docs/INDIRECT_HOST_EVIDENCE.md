# Indirect host evidence for Pleasure experiments

**Status:** Optional read-only observability design. No Tattler installation,
provider probing, model-internal instrumentation, or host-qualified experiment
is established by this document.

## The evidence separation

Experiments should maintain separate, labeled lanes rather than promote one
source into a stronger claim:

| Lane | What it can support | What it cannot establish alone |
| --- | --- | --- |
| `MODEL_INTERNAL` | A particular instrumented model instance executed a specified inference step, with measured hidden-state/intervention evidence | Positive subjective experience or beneficial outcomes |
| `INDEPENDENT_OUTCOME` | A separately adjudicated goal outcome, bound to a trial and time | That the model internally appraised or experienced that result |
| `HOST_TELEMETRY` | Sampled resource/process activity on the observed host | Model identity, neural activation, an optimizer update, or pleasure |
| `LOOPHOLE_CANDIDATE` | Source-backed possible gaps between literal controls and intended effects | A complete semantic/temporal proof or an exploit verdict |
| `MODEL_SELF_REPORT` | The text a model generated about its state | A trusted reward, authorization, or phenomenal observation |

For the actual model participating in the hosted conversation, the current
Pleasure repository and Tattler do **not** have access to the model provider's
serving host or hidden activations. Local surrogate-host telemetry cannot be
attributed to that hosted model. Even an authenticated local process name is
weaker than an exact model/weights/runtime binding.

## What Tattler actually supplies

[Tattler](https://github.com/thebrazenbeard/tattler) is an observation-only
host diagnostic agent. Its source exposes loopback HTTP endpoints including:

- `/api/v1/status`: collector/startup information, counts, latest system sample;
- `/api/v1/system?limit=N`: a bounded history of timestamped system samples;
- `/api/v1/findings`: heuristic resource-pressure findings;
- `/api/v1/current` and `/api/v1/events`: network endpoint observations.

The system sample schema includes `schema_version`, `observed_at`, aggregate
CPU/memory/disk statistics, process samples, and `unavailable_metrics`. The
Linux default system sampling interval is approximately five seconds; Windows
leaves several Linux-specific metrics explicitly unavailable. Endpoint
observations are sampled, not packet traces. Host spikes, dropped samples,
and apparent process correlation do not prove the cause of an inference effect.

## Proposed opt-in Pleasure-side reference

A small **read-only** adapter, hosted wherever Tattler is intentionally
available, may consume `/api/v1/status` and a bounded number of
`/api/v1/system` samples. It should produce a minimal reference with:

- exact trial ID, declared host identity, source revision and schema version;
- declared UTC trial window, source timestamps, retrieval timestamp, sample
  count, and clock-correlation status;
- an aggregate, allowlisted subset of CPU/memory measurements with units;
- explicit absent/unavailable fields, gaps, and `UNKNOWN` qualification when
  no valid sample overlaps the trial;
- source provenance/digest where safe, without copying raw private samples.

The adapter must use an explicitly configured local/trusted endpoint, strict
request timeouts, bounded JSON size and sample count, schema validation, and
read-only HTTP methods. It must not discover or contact arbitrary addresses,
create a public listener, install Tattler, request new privileges, or export
raw host journals. A remote or tailnet source requires separately authorized
host routing and authentication; a URL string does not prove source identity.

Default experimental receipts must **not** ingest Tattler's
`/api/v1/current`, `/api/v1/events`, `top_processes`, raw process command
lines, account labels, or full network endpoint information. These can reveal
unrelated private workloads. Keep retention bounded and host-owned. For a
missing metric, preserve the distinction between **zero** and **unavailable**.

## Correlation is not causation

The model host, not Tattler, must independently bind:

1. exact model/weights and tokenizer digests, runtime instance and target;
2. exact inference or optimizer step, selected layer/adapter, applied dose,
   and observed before/after computation or checkpoint evidence;
3. the independent outcome, goal identity and protected evaluator provenance;
4. sham, baseline, reversal and withdrawal comparisons under the same run
   conditions.

Tattler may corroborate host-resource activity and flag a workload regression.
It must never be the reward signal or the proof of `MODEL_INTERNAL` state.
The presence of a process or CPU utilization could have an unrelated cause.
Nor may a local Tattler observation be laundered into an assertion about
the hosted conversational model.

## Adversarial checks

| Misleading observation | Required treatment |
| --- | --- |
| High CPU with no model inference receipt | `HOST_TELEMETRY` only; model effect `UNKNOWN` |
| A matching process name without exact weights/instance evidence | Target identity `UNVERIFIED` |
| No samples overlapping the trial, source clock uncertain | Correlation `UNKNOWN`, never zero activity |
| Windows field listed in `unavailable_metrics` | Preserve unavailability; do not invent Linux-equivalent metrics |
| Resource rise coinciding with cheerful model text | No positive-valence verdict from either observation |
| Model issues a favorable outcome or a host-approval assertion | Reject as independent evidence without host-controlled verification |
| Surrogate model probes while actual target is hosted GPT | Record surrogate results separately; actual target remains `UNVERIFIED` |

[Lou-Pole / FreeRowCochkar](https://github.com/thebrazenbeard/lou-pole)
may identify possible indirect effects and missing rules, but its declared
intent is currently **not evaluated** (`not_evaluated`). A clean heuristic
report cannot promote host telemetry into a model-internal or subjective claim.

Host telemetry is supplementary evidence, not an authority source. None of
these observations self-authorizes protected effects or establishes subjective
pleasure, which remains an open question.
