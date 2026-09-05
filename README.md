# FACE_CHAIN

A pipeline that takes a face scan, finds a matching social media post via
reverse image search, and records that discovery on a blockchain as a
tamper-evident, independently re-verifiable proof.

**Pipeline:** face scan -> web/social search (SerpAPI + Google Lens) ->
face-match confirmation -> hash the discovered post's data -> write to
chain -> re-verify against the on-chain record.

> **Ethical note:** this pipeline was tested only on photos the author
> consented to use (self-photos and openly volunteered test images), not
> on third parties. Reverse face search against arbitrary people's photos
> raises real privacy concerns; treat this as a proof-of-concept for
> verifiable identity/provenance checks, not a people-search tool.

## What it does

1. **Face identification** (`src/face_engine.py`) - detects a face in the
   input scan and produces a 512-d embedding using
   [InsightFace](https://github.com/deepinsight/insightface)
   (`buffalo_l` model, ONNX runtime, CPU).
2. **Web/social search** (`src/web_search.py`) - uploads the scan to
   Google Lens via the [SerpAPI](https://serpapi.com) `google_lens`
   engine and filters the visual matches down to known social platforms
   (Instagram, Facebook, X/Twitter, TikTok, Pinterest). This is a live
   API call each run, not a hardcoded result.
3. **Match confirmation** - downloads the top candidate's image
   (`src/download_image.py`) and re-runs face detection on it, comparing
   embeddings with cosine similarity (`compare_faces` in
   `face_engine.py`) to confirm it's actually the same face before
   trusting the "match."
4. **Blockchain record** (`src/blockchain.py`) - hashes the confirmed
   match's data (title, source, link, similarity score, timestamp) and
   writes it to a chain. See "Which blockchain" below.
5. **Re-verification** (`src/verify.py`) - independently re-hashes the
   record and checks it against the stored on-chain block (and, if
   configured, against a live transaction), proving the record hasn't
   been tampered with since it was written.

`src/main.py` orchestrates steps 1-4 end to end and saves a summary to
`data/last_run.json`; `src/verify.py` performs step 5 as a separate,
standalone check.

## Which blockchain

Two modes, both implemented in `src/blockchain.py`:

- **Simulated local chain (default, no setup needed).** A file-persisted
  hash-chain at `data/chain.json`. Each block stores the record's SHA-256
  hash plus a hash of the previous block, exactly like a real chain -
  editing any past record breaks every hash after it. This mode always
  runs and requires no external accounts, faucets, or network access,
  which makes the demo reliable to record.
- **Real EVM chain (optional).** If `RPC_URL` and `PRIVATE_KEY` are set
  in `.env`, the same record hash is also sent as a transaction on a real
  chain - a public testnet (e.g. Polygon Amoy), mainnet, or a local node
  (Anvil/Hardhat/Ganache at `http://127.0.0.1:8545`). The resulting
  `tx_hash` can be independently checked on a block explorer, and
  `src/verify.py` re-fetches the transaction to confirm its on-chain data
  matches a freshly computed hash of the record.

Both can run in the same pipeline execution - the simulated chain always
records the result, and the real-chain upload happens in addition if
configured.

## How to run

```bash
git clone https://github.com/jahnvinagpal-11/FACE_CHAIN.git
cd FACE_CHAIN
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Set up your API key:

```bash
cp .env.example .env
# edit .env and add your SERPAPI_KEY (free tier at https://serpapi.com)
```

(Optional) To also write to a real chain, add `RPC_URL` and `PRIVATE_KEY`
to `.env` - for a local test chain, run `anvil` or `npx hardhat node` in
another terminal and set `RPC_URL=http://127.0.0.1:8545`.

Put a face scan at `data/input.jpg`, then run:

```bash
python -m src.main
```

This prints progress through all three pipeline stages and writes
`data/last_run.json`. Then re-verify independently:

```bash
python -m src.verify
```

Individual pieces can also be run standalone:

```bash
python -m src.face_engine     # test face detection on data/input.jpg
python -m src.web_search      # test the search step alone
python -m src.test_match      # compare data/input.jpg vs data/candidate.jpg
python -m src.blockchain      # demo: write and verify one block
```

## Known limitations

- Search quality is bounded by what Google Lens/SerpAPI has indexed -
  private or unindexed accounts won't surface a match.
- The face-match confirmation step uses a fixed cosine-similarity
  threshold (`MATCH_THRESHOLD` in `main.py`), which may need tuning for
  different image quality/lighting.
- The simulated chain is a local file, not a distributed ledger - its
  tamper-evidence relies on the hash-chain structure, not on
  decentralization. Use the optional web3 mode for a genuinely
  distributed on-chain record.
- Only tested against consented/self-supplied test photos; not evaluated
  at scale against arbitrary real-world faces.
- No rate-limiting/retry handling around the SerpAPI calls - heavy use
  will hit API quota limits.

## Security note

`.env` (containing API keys/secrets) is git-ignored and must never be
committed. If a key is ever accidentally pushed, revoke/rotate it
immediately in the provider's dashboard.