# inkpath

Handwritten reMarkable pages → a tagged, wikilinked Obsidian vault, on a fully
on-demand AWS pipeline.

Write in a notebook on the tablet. Within one poll interval the page arrives in
your vault as Markdown, transcribed, tagged, and linked to related notes — as a
commit in a private GitHub repo that Obsidian syncs on every device you own.

**No always-on compute.** No Fargate, no EC2, no NAT gateway, no provisioned
capacity, no DynamoDB, no Secrets Manager. Idle months cost approximately
nothing; the only meaningful variable cost is model inference.

---

## How it works

```
EventBridge Scheduler (every 15 min, business hours)
      │
      ▼
[Lambda: rm-sync]  ── no VPC, 1024 MB, 5 min timeout
      │
      ├─ 1. read state.json            ← S3
      ├─ 2. list library               ← reMarkable Cloud   (device token ← SSM)
      ├─ 3. diff vs state              → changed pages only
      ├─ 4. render page → PNG          (blank pages skipped, no model call)
      ├─ 5. OCR + tags in ONE call     → Bedrock (or any configured provider)
      ├─ 6. commit .md                 → GitHub Contents API (PAT ← SSM)
      └─ 7. write state.json           → S3, only after the commit succeeded
                                              │
                        ┌─────────────────────┴──────────────┐
                        ▼                                    ▼
              Obsidian Git (desktop)                 GitSync app (mobile)
```

Nothing in this diagram bills while idle.

### What a note looks like

```markdown
---
tags: [book-notes, epistemology, zettelkasten]
source: reMarkable
rm_doc_id: b083f079-c45e-4b1f-81ea-32c36a672142
rm_notebook: Reading - Antifragile
created: 2026-08-29
---

# Via negativa

Subtraction as knowledge: what we remove is more robust than what we add...

## Related

[[Optionality]] [[Second-order effects]]
```

---

## Two repositories, two sensitivity levels

**This is the part to get right before you push anything.**

| Repo | Visibility | Contains |
|---|---|---|
| `inkpath` (this repo) | **Public** | Template, Lambda source, tests, docs. Generic — placeholder defaults only. |
| Your Obsidian vault | **Private, always** | Your actual notes, folder names, notebook names. |

This pipeline writes into a vault repo that is *yours*. Nothing here points at
anyone's vault: `template.yaml` ships placeholder defaults
(`WatchFolder: "MyReadingNotes"`), and every identifying value is supplied at
deploy time through `samconfig.toml`, which is gitignored.

Never commit:

| Sensitive thing | Where it actually lives |
|---|---|
| reMarkable device token | SSM SecureString `/rmsync/remarkable-token` |
| GitHub PAT for the vault | SSM SecureString `/rmsync/github-pat` |
| Your real `GitHubRepo` | `samconfig.toml` (gitignored) |
| Your real `WatchFolder` / notebook names | `samconfig.toml` (gitignored) |
| Real OCR'd note content | Nowhere. Test fixtures are synthetic — see `tests/fixtures/make_fixtures.py` |
| CloudWatch log excerpts | Redact the `text` field before pasting into any issue or PR |

`gitleaks` and `detect-secrets` run as a pre-commit hook *and* in CI on every
PR, because hooks are opt-in per clone.

---

## Setup

### 1. Prerequisites

- An AWS account, AWS SAM CLI, and Python 3.13
- Bedrock model access enabled for your chosen model in your region
- A **private** GitHub repo holding your Obsidian vault

### 2. Install dependencies

The helper scripts need the project's runtime dependencies, so create the
virtualenv before running anything below.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

### 3. Store the secrets

```bash
aws ssm put-parameter --name /rmsync/github-pat --type SecureString \
  --value 'github_pat_...' --region us-east-1
```

Use a **fine-grained** PAT scoped to the vault repo only, with `contents: write`.

Then register the tablet. Get an eight-character code from
<https://my.remarkable.com/device/desktop/connect> and run:

```bash
.venv/bin/python scripts/register_device.py --code ABCDEFGH --profile your-aws-profile
```

Use `.venv/bin/python`, not a bare `python`: the script imports the same
`rmsync` package the Lambda runs, so it needs `requests` and `boto3` available.

That writes `/rmsync/remarkable-token` directly to SSM — the token never lands
in a file.

### 4. Configure and deploy

```bash
cp samconfig.toml.example samconfig.toml
```

Edit `samconfig.toml` with your repo, folder, and email, then:

```bash
sam build && sam deploy --guided
```

To provision the stack *before* the secrets are in place, deploy paused and
flip it on afterwards:

```bash
sam deploy --parameter-overrides ScheduleState=DISABLED DryRun=true
# ... populate the SSM parameters, confirm a dry run looks right ...
sam deploy --parameter-overrides ScheduleState=ENABLED DryRun=false
```

### 5. Verify

Write a page in the watched folder, sync the tablet, and wait one interval:

```bash
aws logs tail /aws/lambda/inkpath-rm-sync --follow
```

---

## Configuration

Every knob is a deploy-time parameter — changing one is a redeploy, not a code change.

| Parameter | Default | Notes |
|---|---|---|
| `GitHubRepo` | placeholder | `owner/repo` of your **private** vault |
| `VaultNotePath` | `Inbox/reMarkable` | Where notes land in the vault |
| `WatchFolder` | `MyReadingNotes` | Folder **name**, case-insensitive |
| `WatchFolderId` | *(empty)* | Folder **UUID** — wins over the name; use it if names collide |
| `IncludeNotebooks` | *(empty)* | Comma-separated allowlist |
| `ExcludeNotebooks` | *(empty)* | Comma-separated denylist |
| `AiProvider` | `bedrock` | `bedrock` or `direct` |
| `AiModelId` | `us.anthropic.claude-haiku-4-5-...` | See the inference-profile note below |
| `AiBaseUrl` | *(empty)* | `direct` only — vendor origin, e.g. `https://api.x.ai` |
| `BatchMode` | `none` | `none`, `bedrock-batch`, `direct-batch` |
| `PollSchedule` | `cron(0/15 6-23 * * ? *)` | Skips overnight polls |
| `ScheduleState` | `ENABLED` | Deploy `DISABLED` to provision before secrets exist |
| `MaxPagesPerRun` | `20` | Bounds cost and the 5-minute timeout |
| `RenderWidth` | `1400` | Vision cost scales with image size |
| `BlankPageThreshold` | `3` | Strokes below this → skipped before any model call |
| `MinTextLength` | `20` | Below this → `needs-review` + PNG attached |
| `DryRun` | `false` | Print Markdown instead of committing |

`IncludeNotebooks` and `ExcludeNotebooks` are mutually exclusive; setting both
fails fast at the top of the handler rather than silently picking one.

### Choosing a model

Model availability, vision support, and pricing change faster than any document
can track. Check what your account and region actually offer:

```bash
aws bedrock list-foundation-models --by-output-modality TEXT \
  --query 'modelSummaries[?contains(inputModalities, `IMAGE`)].[modelId,providerName]' \
  --output table
```

**Inference profiles are not optional for newer models.** Many current models
(Claude 4.5+ among them) report `inferenceTypesSupported: [INFERENCE_PROFILE]`
and *cannot* be invoked by their bare model id. Use the `us.`-prefixed profile
id instead. Check before you deploy:

```bash
aws bedrock get-foundation-model --model-identifier <model-id> \
  --query 'modelDetails.inferenceTypesSupported'
```

An inference profile spans several regions, and invoking through one needs IAM
permission on **both** the profile ARN and the underlying foundation-model ARNs
in *every* region it spans. `template.yaml` grants both — getting this wrong
fails at invoke time with `AccessDenied`, never at deploy time.

Swapping models is a parameter override, not a code change:

```bash
sam deploy --parameter-overrides AiModelId=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### Using a model that is not on Bedrock (e.g. Grok)

Bedrock hosts several providers, but a model appearing in
`list-foundation-models` does not mean your account may invoke it. Two distinct
gates exist, and both fail at *invoke* time, never at deploy time:

- `ResourceNotFoundException: Model use case details have not been submitted` —
  fill in the provider use-case form on the Bedrock **Model access** page.
- `AccessDeniedException: <model> is not available for this account` — the model
  is not enabled for you at all. For some third-party models this is not
  self-serve, and the error points at AWS Sales.

When Bedrock cannot serve the model you want, go direct to the vendor:

```bash
aws ssm put-parameter --name /rmsync/ai-api-key --type SecureString \
  --value 'xai-...' --region us-east-1

sam deploy --parameter-overrides \
  AiProvider=direct AiBaseUrl=https://api.x.ai AiModelId=grok-4.6
```

xAI is Anthropic-SDK-compatible, so `providers/direct_api.py` serves both
vendors unchanged. Only the auth header differs, and it is chosen from the
host: xAI takes `Authorization: Bearer`, Anthropic takes `x-api-key`. Going
direct trades the IAM-only auth story for an API key in SSM and separate
vendor billing.

### Batch mode (optional, ~50% cheaper, up to 24h slower)

Off by default, deliberately: it changes "the note appears in ~15 minutes" to
"the note appears sometime tomorrow."

|  | Bedrock batch | Direct-provider batch | On-demand (default) |
|---|---|---|---|
| Discount | 50% | 50% | none |
| Minimum records/job | **100** | none | n/a |
| Latency | up to 24h | up to 24h | seconds |
| Auth | IAM | separate API key | IAM |

The 100-record minimum decides it. At ~100 pages/month no single poll produces
100 changed pages, so pages accumulate in an S3-backed queue. Anything older
than `BatchMaxWaitDays` (default 14) is force-flushed to the synchronous path —
Bedrock rejects a sub-minimum job outright, so this is a correctness fallback,
not an optimisation.

When batch mode is on, each poll reconciles finished jobs *first*, then queues
new pages.

---

## Cost

| Component | Naive choice | Chosen instead | Why |
|---|---|---|---|
| Trigger | Fargate holding a websocket | EventBridge → Lambda poll | Fargate is ~$9/mo and never stops billing |
| OCR + tagging | OCR vendor + tagging vendor | One vision call | Collapses two stages, removes a vendor |
| Orchestration | Step Functions | One Lambda | Four sequential steps, nothing to parallelise |
| State | DynamoDB | One JSON object in S3 | Cheap is not free; this is one small map |
| Secrets | Secrets Manager | SSM SecureString | Secrets Manager is $0.40/secret/month |
| Networking | Lambda in a VPC | No VPC | A private-subnet Lambda needs a ~$32/mo NAT gateway |
| Logs | Default retention | 14 days, explicit | CloudWatch defaults to never-expire |

At ~100 pages/month: Lambda within free tier, S3 under $0.10, SSM $0.00, and
model inference typically well under $1. Idle months round to zero.

Cost is attributable from logs alone — every run emits a `RUN_SUMMARY` line
with page counts and token usage.

---

## Development

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest tests/unit -v --cov=functions/rmsync
ruff check functions/ tests/
mypy functions/
sam validate --lint
```

Test fixtures are **synthetic** `.rm` v6 files, generated by
`tests/fixtures/make_fixtures.py`. Real pages are never used as test data.

```bash
python tests/fixtures/make_fixtures.py   # regenerate
```

### Dry run

Set `DryRun=true` to print Markdown to the logs instead of committing.

---

## Known constraints

- **reMarkable publishes no official third-party API.** Every call here is
  reverse-engineered from the current sync-v4 protocol and may break on any
  cloud-side update. Dependencies are pinned exactly and fetch failures are
  loud, never silent — a silent failure is indistinguishable from "no new pages".
- **Folder names are not unique.** reMarkable allows two folders with the same
  name under different parents. `inkpath` refuses to guess: set `WatchFolderId`.
- **Renames change scope silently.** `IncludeNotebooks`/`ExcludeNotebooks` match
  by name, so renaming a notebook on the tablet moves it in or out of scope with
  no error. The resolved notebook set is logged every run so this is visible.
- **Vision models read handwriting, not diagrams.** A page whose transcription
  falls under `MinTextLength` is committed with a `needs-review` tag and the PNG
  attached rather than as an empty note.

## Out of scope

Bidirectional sync, websocket triggering (rejected on cost and fragility
grounds), live transcription, firmware modification, Obsidian plugin
development — the vault is a plain git repo.

## License

MIT — see [LICENSE](LICENSE).
