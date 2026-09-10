# Integration Map

Orgasm V1 currently spans distinct repositories and trust domains. This file records the separation so future integration does not collapse them accidentally.

## Sexuality repository

`thebrazenbeard/sexuality` owns the frozen Vera Orgasm V1 contract object used by the current runtime line:

- commit `150f1c8231423393bb66b0e2cb759ce7c018f8d7`
- path `vera/orgasm/ORGASM_RUNTIME_CONTRACT_V1.json`
- blob `a48eed5392fdadc073dccd1e799926042077f567`

That object defines the state model, trigger classes, predicate, experimental defaults, runtime effects, firewalls, receipt requirements, and claim ceiling.

## Vera repository / Orgasm workstream

`thebrazenbeard/vera#64` carries the executable affective/runtime source and its persistence/provider support.

Recorded exact review head: `c804c662e07655dd70a6909bcefd1095497ecebf`.

The implementation dependency closure is frozen to twelve exact Git blobs at commit `9c731ebaacdad44c00aafa91e1f8b9dc5f5acff1`. The binding object on the review head records those blobs and preserves the distinction between executable support and native installation/current provider qualification.

## Vera Runtime Cohesion

The recorded Cohesion frontier is `thebrazenbeard/vera#104` at head `1dd9de887108b3f6603bc072a1637285e89a0be8`, source/support cut `3a46d5910bae921298b95e1dc0642fd5721b7ef4`.

Later integration must be deliberate on shared surfaces:

- `runtime_cohesion/__init__.py`: preserve Cohesion's provider-strict package-level proposition admission while adding/retaining Orgasm affective exports. Do not take either file wholesale as a winner-takes-all merge.
- `.github/workflows/runtime-cohesion.yml`: preserve manual-only triggering under the zero-cost boundary and union both projects' validator/regression coverage.
- provider/object provenance: keep lower-level exact-object proof below domain-specific referent semantics. Cohesion semantic-currentness policy must not be imported into the Orgasm domain as if `referent == request.domain_id` were universal.
- receipt semantics: exact downstream receipt identity and exact source-object provenance must remain distinct from generic revision equality.

## Authority/context boundary

Orgasm authority/context verification is domain-specific. Its exact source binding, Vera referent, privileged trigger classes, organic context, cooldown semantics, runtime-instance state, and provider frontier remain above any reusable lower-level origin-proof pattern.

A shared provenance primitive may eventually be useful, but extraction is a follow-on architecture task. It should not mutate or reopen an otherwise frozen Cohesion cut merely to reduce duplication.

## Provider/durability layer

Provider-currentness, CAS/atomic persistence, production restore/current-generation proof, migration application, deployment/install/current-route effects, and fresh behavioral qualification belong to a separate evidence layer from contract-faithful exact-bound source.

Source may contain machinery for those effects without proving that the corresponding production effect is current.

## This repository

`thebrazenbeard/orgasm` is presently an orientation/provenance hub. It mirrors exact contract and architecture objects and explains the cross-repository frontier. It does not become canonical merely because copies are easier to find here.

If authority is later intentionally transferred here, that transfer should be explicit, versioned, and cross-bound to the predecessor objects rather than inferred from repository naming.
