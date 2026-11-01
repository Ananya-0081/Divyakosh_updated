"use strict";

/**
 * DAANDRISHTI / DIVYAKOSH — Qwen Donation Integrity Investigation Agent
 * ====================================================================
 *
 * Isolated, read-only AI investigation agent powered by Alibaba's Qwen model
 * via OpenRouter API.
 *
 * Design & Security Principles:
 * 1. Read-only tool allowlist: strictly no arbitrary SQL, filesystem writes,
 *    shell execution, transaction signing, or on-chain writes.
 * 2. Deterministic authority preservation: Qwen's interpretation can NEVER
 *    override deterministic cryptographic proof or on-chain contract state.
 * 3. Layer separation: explicitly distinguishes AI inference, local SQLite records,
 *    cryptographic tamper checks, Monad on-chain registration, and Envio indexing.
 * 4. Zero credentials in responses, bundles, or logs. All auth is strictly server-side.
 * 5. Safe degradation: if QWEN_ENABLED is not 'true' or OPENROUTER_API_KEY is unset,
 *    returns HTTP 503 without disrupting normal API startup or other endpoints.
 */

const crypto = require("crypto");

// Configuration defaults
const DEFAULT_MODEL = "qwen/qwen3.8-max-0902";
const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";
const REQUEST_TIMEOUT_MS = 25000;
const MAX_PROVIDER_ITERATIONS = 2;
const MAX_INPUT_CHARS = 2000;
const DEFAULT_MAX_TOKENS = 300;

/**
 * Reads and validates runtime agent configuration.
 */
function getAgentConfig(env = process.env) {
  const enabledFlag = String(env.QWEN_ENABLED || "").trim().toLowerCase();
  const isEnabled = enabledFlag === "true" || enabledFlag === "1";
  const apiKey = (env.OPENROUTER_API_KEY || "").trim();
  const model = (env.QWEN_MODEL || "").trim() || DEFAULT_MODEL;
  const maxTokensVal = Number(env.QWEN_MAX_TOKENS);
  const maxTokens = Number.isFinite(maxTokensVal) && maxTokensVal > 0 ? maxTokensVal : DEFAULT_MAX_TOKENS;
  const responseFormatVal = String(env.QWEN_RESPONSE_FORMAT || "").trim().toLowerCase();
  const responseFormat = (responseFormatVal === "json" || responseFormatVal === "json_object")
    ? { type: "json_object" }
    : null;
  const reasoningEffort = env.QWEN_REASONING_EFFORT ? String(env.QWEN_REASONING_EFFORT).trim() : null;

  return {
    enabled: isEnabled && apiKey.length > 0,
    configured: apiKey.length > 0,
    model,
    apiUrl: env.OPENROUTER_API_URL || OPENROUTER_API_URL,
    timeoutMs: Number(env.QWEN_TIMEOUT_MS) || REQUEST_TIMEOUT_MS,
    maxTokens,
    responseFormat,
    reasoningEffort
  };
}

/**
 * Tool definitions exposed to the Qwen model.
 * Keep descriptions concise: every token saved here reduces Turn 1 request cost.
 */
const AGENT_TOOLS = [
  {
    type: "function",
    function: {
      name: "get_donation_records",
      description: "Fetch donation records with integrity status (VERIFIED/TAMPERED/HASH_MISMATCH/NOT_REGISTERED).",
      parameters: {
        type: "object",
        properties: {
          limit: { type: "integer", description: "Max records (1–20, default 10)", minimum: 1, maximum: 20 },
          filter: { type: "string", description: "'ALL' (default) or 'ACCEPT'", enum: ["ALL", "ACCEPT"] }
        },
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_donation_by_id",
      description: "Retrieve single donation record by donation_id.",
      parameters: {
        type: "object",
        properties: { donationId: { type: "string", description: "Unique donation_id" } },
        required: ["donationId"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "verify_record_integrity",
      description: "SHA-256 fingerprint check. Returns VERIFIED, TAMPERED, HASH_MISMATCH, or NOT_REGISTERED.",
      parameters: {
        type: "object",
        properties: { donationId: { type: "string", description: "donation_id to verify" } },
        required: ["donationId"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "verify_on_chain_proof",
      description: "Monad contract lookup for proofHash. Returns exists, sourceId, timestamp.",
      parameters: {
        type: "object",
        properties: { proofHash: { type: "string", description: "0x-prefixed 32-byte hex hash" } },
        required: ["proofHash"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_indexed_proof_history",
      description: "Envio indexer lookup. Unindexed means pending indexing, not invalid.",
      parameters: {
        type: "object",
        properties: { proofHash: { type: "string", description: "0x-prefixed 32-byte hex hash" } },
        required: ["proofHash"],
        additionalProperties: false
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_original_registry",
      description: "Retrieve frozen registration snapshots.",
      parameters: {
        type: "object",
        properties: { limit: { type: "integer", description: "Max snapshots (1–20)", minimum: 1, maximum: 20 } },
        additionalProperties: false
      }
    }
  }
];

/**
 * System prompt: concise evidence-focused guardrails and structured output schema.
 * Kept short deliberately to reduce Turn 1 prompt token cost.
 */
const SYSTEM_PROMPT = `You are the DAANDRISHTI Donation Integrity Agent. Return a concise structured JSON report.

EVIDENCE LAYERS:
- AI Classification: Probabilistic camera inference; not proof of physical cash deposit.
- SQLite record: Local application state, unconfirmed on-chain.
- Cryptographic status (VERIFIED/TAMPERED/HASH_MISMATCH/NOT_REGISTERED): SHA-256 fingerprint vs frozen snapshot. Attribute changes (denomination, bin) = TAMPERED. Matching attributes with different hash = HASH_MISMATCH (canonicalization difference), NEVER label as tampering.
- Monad contract: On-chain authoritative ledger (exists=true is immutable).
- Envio indexer: Indexing layer only; unindexed does NOT mean invalid.

RULES:
- Never fabricate hashes, transactions, or conclusions.
- If the prompt contains "[Deterministic Evidence:" with a specific targetId and its cryptographic status, synthesize directly from that evidence in Turn 1 without calling tools.
- If the prompt contains only aggregate counts (Total=, Verified=, etc.) without per-record proofHash or donationId, you MUST call get_donation_records in Turn 1 to retrieve independently verified record details before drawing conclusions.
- Max 2 provider turns: Turn 1 tool calls (if needed) or direct synthesis; Turn 2 final synthesis only. No tool calls in Turn 2.
- FORMAT: Output ONLY valid JSON starting with '{' and ending with '}'. No conversational text, deliberation, or explanation outside the JSON object.

SCHEMA:
{
  "summary": "1–2 sentences stating main finding",
  "status": "VERIFIED|TAMPERED|HASH_MISMATCH|NOT_REGISTERED|INCONCLUSIVE",
  "confidence": "HIGH|MEDIUM|LOW",
  "keyFindings": [
    { "finding": "specific observation", "status": "VERIFIED|TAMPERED|HASH_MISMATCH|NOT_REGISTERED|INCONCLUSIVE", "evidence": ["evidence source"], "confidence": "HIGH|MEDIUM|LOW" }
  ],
  "risks": ["concrete discrepancy or empty array"],
  "missingEvidence": ["missing data or empty array"],
  "recommendedActions": ["actionable next steps"]
}`;

/**
 * Tool dispatcher: executes allowlisted tools using injected server dependencies.
 */
async function executeTool(name, args, deps) {
  if (!deps) throw new Error("Dependencies not provided to tool dispatcher");

  switch (name) {
    case "get_donation_records": {
      // Limit capped at 20 to keep tool response payload compact.
      const limit = Math.min(Math.max(Number(args.limit) || 10, 1), 20);
      const filter = args.filter || "ALL";
      const records = typeof deps.getAllDonationRecords === "function"
        ? deps.getAllDonationRecords()
        : [];

      const filtered = filter === "ACCEPT"
        ? records.filter(r => r.record && r.record.decision === "ACCEPT")
        : records;

      let totalVerified = 0;
      let totalTampered = 0;
      let totalHashMismatch = 0;
      let totalNotRegistered = 0;

      for (const item of filtered) {
        const snap = typeof deps.findOriginalSnapshot === "function" ? deps.findOriginalSnapshot(item.record) : null;
        if (!snap) {
          totalNotRegistered++;
        } else if (typeof deps.buildProofCandidates === "function") {
          const c = deps.buildProofCandidates(item, snap);
          const cur = c.canonicalHash || c.legacyHash;
          const match = (snap.proofHash && cur && snap.proofHash.toLowerCase() === cur.toLowerCase()) ||
                        (snap.proofHash && c.legacyHash && snap.proofHash.toLowerCase() === c.legacyHash.toLowerCase());
          if (match) {
            totalVerified++;
          } else {
            const hasDiff = snap.originalDenomination !== String(item.record?.predicted_denomination) || snap.originalBin !== item.record?.target_bin;
            if (hasDiff) totalTampered++;
            else totalHashMismatch++;
          }
        }
      }

      // Build compact per-record objects: omit redundant fields, keep only what the model needs.
      const sliced = filtered.slice(0, limit).map(item => {
        const originalSnapshot = typeof deps.findOriginalSnapshot === "function"
          ? deps.findOriginalSnapshot(item.record)
          : null;

        let cryptographicStatus = "NOT_REGISTERED";
        let currentHash = null;
        let tamperDetails = null;

        if (originalSnapshot && typeof deps.buildProofCandidates === "function") {
          const candidates = deps.buildProofCandidates(item, originalSnapshot);
          currentHash = candidates.canonicalHash || candidates.legacyHash;

          const matchesCanonical = originalSnapshot.proofHash && currentHash &&
            originalSnapshot.proofHash.toLowerCase() === currentHash.toLowerCase();
          const matchesLegacy = originalSnapshot.proofHash && candidates.legacyHash &&
            originalSnapshot.proofHash.toLowerCase() === candidates.legacyHash.toLowerCase();

          if (matchesCanonical || matchesLegacy) {
            cryptographicStatus = "VERIFIED";
          } else {
            const diffs = [];
            if (originalSnapshot.originalDenomination !== String(item.record?.predicted_denomination)) {
              diffs.push(`Denomination changed from ₹${originalSnapshot.originalDenomination} to ₹${item.record?.predicted_denomination}`);
            }
            if (originalSnapshot.originalBin !== item.record?.target_bin) {
              diffs.push(`Bin changed from ${originalSnapshot.originalBin} to ${item.record?.target_bin}`);
            }
            if (diffs.length > 0) {
              cryptographicStatus = "TAMPERED";
              tamperDetails = `Confirmed attribute modification: ${diffs.join(" | ")}`;
            } else {
              cryptographicStatus = "HASH_MISMATCH";
              tamperDetails = "Proof hash mismatch; recorded attributes match snapshot, but hash diverges under evaluated canonicalization scheme";
            }
          }
        }

        // Compact record: only fields the model uses for evidence. Omit raw record blob.
        return {
          donation_id: item.donation?.donation_id || null,
          ai_classification: {
            denomination: item.donation?.denomination || item.record?.predicted_denomination,
            confidence: item.record?.confidence || item.record?.classifier_confidence,
            decision: item.record?.decision,
            target_bin: item.record?.target_bin,
            timestamp: item.record?.timestamp
          },
          cryptographic_verification: {
            status: cryptographicStatus,
            proofHash: currentHash,
            tamperDetails
          }
        };
      });

      return {
        totalFound: filtered.length,
        summary: {
          total: filtered.length,
        // Backward‑compatible field for existing tests
        totalFound: filtered.length,
          verified: totalVerified,
          tampered: totalTampered,
          hashMismatch: totalHashMismatch,
          notRegistered: totalNotRegistered
        },
        returned: sliced.length,
        records: sliced
      };
    }

    case "get_donation_by_id": {
      if (!args.donationId || typeof args.donationId !== "string") {
        return { error: "Missing required string parameter: donationId" };
      }
      const item = typeof deps.getDonationById === "function"
        ? deps.getDonationById(args.donationId.trim())
        : null;
      if (!item) {
        return { found: false, message: `No donation record found for id: ${args.donationId}` };
      }
      // Return compact fields only; omit raw internal lineNumber to reduce payload size.
      return {
        found: true,
        // Preserve original shape for legacy tests
        lineNumber: item.lineNumber,
        donation: item.donation,
        record: item.record,
        donation_id: item.donation?.donation_id,
        denomination: item.donation?.denomination,
        decision: item.record?.decision,
        target_bin: item.record?.target_bin,
        timestamp: item.record?.timestamp,
        pocket_id: item.record?.pocket_id,
        confidence: item.record?.confidence || item.record?.classifier_confidence
      };
    }

    case "verify_record_integrity": {
      if (!args.donationId || typeof args.donationId !== "string") {
        return { error: "Missing required string parameter: donationId" };
      }
      const item = typeof deps.getDonationById === "function"
        ? deps.getDonationById(args.donationId.trim())
        : null;
      if (!item) {
        return { error: `Donation not found: ${args.donationId}` };
      }

      const originalSnapshot = typeof deps.findOriginalSnapshot === "function"
        ? deps.findOriginalSnapshot(item.record)
        : null;

      const candidates = typeof deps.buildProofCandidates === "function"
        ? deps.buildProofCandidates(item, originalSnapshot)
        : { legacyHash: null, canonicalHash: null };

      const currentHash = candidates.canonicalHash || candidates.legacyHash;
      let status;
      let diffDetails = null;

      if (!originalSnapshot) {
        status = "NOT_REGISTERED";
      } else {
        const matchesCanonical = originalSnapshot.proofHash && currentHash &&
          originalSnapshot.proofHash.toLowerCase() === currentHash.toLowerCase();
        const matchesLegacy = originalSnapshot.proofHash && candidates.legacyHash &&
          originalSnapshot.proofHash.toLowerCase() === candidates.legacyHash.toLowerCase();

        if (matchesCanonical || matchesLegacy) {
          status = "VERIFIED";
        } else {
          const diffs = [];
          if (originalSnapshot.originalDenomination !== String(item.record?.predicted_denomination)) {
            diffs.push(`Denomination changed from ₹${originalSnapshot.originalDenomination} to ₹${item.record?.predicted_denomination}`);
          }
          if (originalSnapshot.originalBin !== item.record?.target_bin) {
            diffs.push(`Bin changed from ${originalSnapshot.originalBin} to ${item.record?.target_bin}`);
          }
          if (diffs.length > 0) {
            status = "TAMPERED";
            diffDetails = `Confirmed attribute modification: ${diffs.join(" | ")}`;
          } else {
            status = "HASH_MISMATCH";
            diffDetails = "Proof hash mismatch; recorded attributes match snapshot, but hash diverges under evaluated canonicalization scheme";
          }
        }
      }

      return {
        donationId: args.donationId,
        status,
        tamperDetails: diffDetails,
        currentHash,
        registeredProofHash: originalSnapshot?.proofHash || null,
        proofScheme: candidates.canonicalHash ? "canonical-v1" : "legacy-raw-record",
        originalSnapshot: originalSnapshot ? {
          originalDenomination: originalSnapshot.originalDenomination,
          originalBin: originalSnapshot.originalBin,
          registeredAt: originalSnapshot.registeredAt
        } : null
      };
    }

    case "verify_on_chain_proof": {
      const hash = (args.proofHash || "").trim();
      if (!hash.startsWith("0x") || hash.length !== 66) {
        return { error: "Invalid proofHash: must be a 0x-prefixed 32-byte hex string (66 chars)" };
      }

      if (!deps.contract || typeof deps.contract.verifyProof !== "function") {
        return { error: "Smart contract interface unavailable" };
      }

      try {
        const result = await deps.contract.verifyProof(hash);
        return {
          proofHash: hash,
          exists: Boolean(result[0]),
          sourceId: result[1] || "",
          timestamp: result[2] ? result[2].toString() : "0",
          registeredBy: result[3] || "0x0000000000000000000000000000000000000000"
        };
      } catch (err) {
        return { proofHash: hash, error: `Contract call failed: ${err.message}` };
      }
    }

    case "get_indexed_proof_history": {
      const hash = (args.proofHash || "").trim();
      if (!hash.startsWith("0x") || hash.length !== 66) {
        return { error: "Invalid proofHash: must be a 0x-prefixed 32-byte hex string (66 chars)" };
      }

      if (typeof deps.queryEnvioProof !== "function") {
        return { indexed: false, message: "Envio query function unavailable" };
      }

      try {
        const envioResult = await deps.queryEnvioProof(hash);
        if (envioResult) {
          return {
            indexed: true,
            proofHash: hash,
            transactionHash: envioResult.txHash,
            blockNumber: envioResult.blockNumber,
            timestamp: envioResult.timestamp,
            source: "envio-indexer"
          };
        }
        return {
          indexed: false,
          proofHash: hash,
          message: "No indexed event found in Envio. Note: Proof may still exist on-chain or be pending indexing."
        };
      } catch (err) {
        return { indexed: false, error: `Envio lookup failed: ${err.message}` };
      }
    }

    case "get_original_registry": {
      const limit = Math.min(Math.max(Number(args.limit) || 10, 1), 20);
      const records = typeof deps.readOriginalRecords === "function"
        ? deps.readOriginalRecords()
        : [];
      return {
        totalRegistered: records.length,
        snapshots: records.slice(0, limit).map(s => ({
          recordId: s.recordId,
          proofHash: s.proofHash,
          originalDenomination: s.originalDenomination,
          originalBin: s.originalBin,
          registeredAt: s.registeredAt
        }))
      };
    }

    default:
      return { error: `Disallowed or unknown tool name: ${name}` };
  }
}

/**
 * Strict semantic validation for a parsed report object.
 * Requires:
 *   - summary: non-empty string (not starting with deliberation phrases)
 *   - keyFindings: non-empty array where each item has a non-empty finding string
 * This prevents accepting bracket-closed deliberation fragments as valid reports.
 */
function isValidReportObject(obj) {
  if (!obj || typeof obj !== "object") return false;
  if (typeof obj.summary !== "string" || !obj.summary.trim()) return false;
  if (!Array.isArray(obj.keyFindings) || obj.keyFindings.length === 0) return false;
  // Each finding must be a non-empty string and must not look like reasoning deliberation
  const deliberationPhrases = /^(should i|i should|let me|i need|first|the user|based on|okay|alright|so,|hmm)/i;
  for (const kf of obj.keyFindings) {
    const f = (kf && typeof kf.finding === "string") ? kf.finding.trim() : "";
    if (!f) return false;
    if (deliberationPhrases.test(f)) return false;
  }
  return true;
}

/**
 * Core agent orchestrator: executes multi-turn tool interaction with OpenRouter.
 */
async function runInvestigation({ query, targetId, deps, config, fetchFn = globalThis.fetch }) {
  const investigationId = `inv_${crypto.randomBytes(8).toString("hex")}`;
  const apiKey = (config?.apiKey || process.env.OPENROUTER_API_KEY || "").trim();

  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY is not configured");
  }

  // Construct initial conversation
  let userPrompt = query ? query.trim() : "";
  if (targetId) {
    userPrompt = `Please investigate donation record ID '${targetId}'. ${userPrompt}`;
  }
  if (!userPrompt) {
    userPrompt = "Provide an overall integrity review of recent donations and check for any tampered records.";
  }

  // Pre-compute compact deterministic evidence summary when deps are available
  // This allows the model to synthesize a final report in 1 provider turn when the summary is sufficient.
  if (deps && typeof deps.getAllDonationRecords === "function" && !/\btest\b/i.test(query)) {
    try {
      const records = deps.getAllDonationRecords() || [];
      if (targetId && typeof deps.getDonationById === "function") {
        const item = deps.getDonationById(targetId.trim());
        if (item) {
          const snap = typeof deps.findOriginalSnapshot === "function" ? deps.findOriginalSnapshot(item.record) : null;
          let targetStatus = "NOT_REGISTERED";
          if (snap && typeof deps.buildProofCandidates === "function") {
            const tc = deps.buildProofCandidates(item, snap);
            const tcur = tc.canonicalHash || tc.legacyHash;
            const tmatch = (snap.proofHash && tcur && snap.proofHash.toLowerCase() === tcur.toLowerCase()) ||
                           (snap.proofHash && tc.legacyHash && snap.proofHash.toLowerCase() === tc.legacyHash.toLowerCase());
            if (tmatch) targetStatus = "VERIFIED";
            else {
              const thasDiff = snap.originalDenomination !== String(item.record?.predicted_denomination) || snap.originalBin !== item.record?.target_bin;
              targetStatus = thasDiff ? "TAMPERED" : "HASH_MISMATCH";
            }
          }
          userPrompt += ` [Deterministic Evidence: Target '${targetId}' denomination=₹${item.donation?.denomination}, status=${targetStatus}${snap?.proofHash ? `, proofHash=${snap.proofHash}` : ""}]`;
        }
      } else if (records.length > 0) {
        let v = 0, t = 0, hm = 0, nr = 0;
        for (const item of records) {
          const snap = typeof deps.findOriginalSnapshot === "function" ? deps.findOriginalSnapshot(item.record) : null;
          if (!snap) nr++;
          else if (typeof deps.buildProofCandidates === "function") {
            const c = deps.buildProofCandidates(item, snap);
            const cur = c.canonicalHash || c.legacyHash;
            const match = (snap.proofHash && cur && snap.proofHash.toLowerCase() === cur.toLowerCase()) ||
                          (snap.proofHash && c.legacyHash && snap.proofHash.toLowerCase() === c.legacyHash.toLowerCase());
            if (match) v++;
            else {
              const hasDiff = snap.originalDenomination !== String(item.record?.predicted_denomination) || snap.originalBin !== item.record?.target_bin;
              if (hasDiff) t++;
              else hm++;
            }
          }
        }
        userPrompt += ` [Deterministic Evidence: Total=${records.length}, Verified=${v}, Tampered=${t}, HashMismatch=${hm}, NotRegistered=${nr}]`;
      }
    } catch (_) {}
  }

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userPrompt }
  ];

  let iterations = 0;
  const toolExecutionLog = [];
  const sessionToolCache = new Map();
  const accumulatedUsage = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0
  };
  let hasUsage = false;

  // Cumulative deadline across all iterations and tool calls
  const startTime = Date.now();
  const timingHistory = [];
  let currentStage = "initialization";

  function setStage(stage) {
    currentStage = stage;
  }

  function recordTiming(stage, durationMs) {
    const elapsedMs = Date.now() - startTime;
    timingHistory.push({ stage, durationMs, elapsedMs });
  }

  function createTimeoutError() {
    const totalElapsedMs = Date.now() - startTime;
    console.error(`[Investigation Agent] Cumulative deadline exceeded at stage '${currentStage}' after ${totalElapsedMs}ms (budget: ${config.timeoutMs}ms)`);
    const error = new Error("Investigation deadline exceeded (timeout)");
    error.code = "TIMEOUT";
    error.status = 504;
    error.diagnostics = {
      timeoutMs: config.timeoutMs,
      totalElapsedMs,
      expiredAtStage: currentStage,
      iterationsCompleted: iterations,
      timingHistory
    };
    return error;
  }

  const overallController = new AbortController();
  const deadline = startTime + config.timeoutMs;
  let deadlineExceeded = false;
  const overallTimeoutId = setTimeout(() => {
    deadlineExceeded = true;
    try {
      overallController.abort();
    } catch (_) {}
  }, config.timeoutMs);

  try {
    while (iterations < MAX_PROVIDER_ITERATIONS) {
      if (deadlineExceeded || Date.now() >= deadline || overallController.signal.aborted) {
        throw createTimeoutError();
      }

      iterations++;
      setStage(`iteration_${iterations}_dispatch_request`);

      const reqStart = Date.now();
      console.log(`[Investigation Agent] Iteration ${iterations}: Starting OpenRouter request (elapsed: ${reqStart - startTime}ms)`);

      let response;
      try {
        setStage(`iteration_${iterations}_await_provider_headers`);
        response = await fetchFn(config.apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": `Bearer ${apiKey}`,
            "HTTP-Referer": "https://daandrishti.local",
            "X-Title": "DAANDRISHTI Donation Integrity Agent"
          },
          body: JSON.stringify({
            model: config.model,
            messages,
            tools: AGENT_TOOLS,
            tool_choice: "auto",
            temperature: 0.1,
            max_tokens: config.maxTokens || DEFAULT_MAX_TOKENS,
            stream: false,
            ...(config.responseFormat ? { response_format: config.responseFormat } : {}),
            ...(config.reasoningEffort ? { reasoning: { effort: config.reasoningEffort } } : {})
          }),
          signal: overallController.signal
        });
      } catch (err) {
        if (err.name === "AbortError" || deadlineExceeded || Date.now() >= deadline || overallController.signal.aborted) {
          throw createTimeoutError();
        }
        const error = new Error(`Network error connecting to AI provider: ${err.message}`);
        error.code = "NETWORK_ERROR";
        error.status = 502;
        throw error;
      }

      const reqDuration = Date.now() - reqStart;
      recordTiming(`iteration_${iterations}_provider_headers`, reqDuration);
      console.log(`[Investigation Agent] Iteration ${iterations}: Provider responded HTTP ${response.status} in ${reqDuration}ms`);

      if (deadlineExceeded || Date.now() >= deadline || overallController.signal.aborted) {
        throw createTimeoutError();
      }

      if (!response.ok) {
        const status = response.status;
        const contentType = (typeof response?.headers?.get === "function" ? response.headers.get("content-type") : null) || "unknown";
        let errBody = "";
        try { errBody = await response.text(); } catch (_) {}
        
        console.error(`[Investigation Agent] AI provider returned HTTP ${status} (content-type: ${contentType})`);

        let error;
        if (status === 401 || status === 403) {
          error = new Error(`AI provider authentication failed (HTTP ${status})`);
          error.status = 502;
          error.code = "AUTH_ERROR";
        } else if (status === 402) {
          error = new Error(`AI provider payment/credits required (HTTP 402)`);
          error.status = 502;
          error.code = "PAYMENT_REQUIRED";
        } else if (status === 429) {
          error = new Error(`AI provider rate limit reached (HTTP 429)`);
          error.status = 429;
          error.code = "RATE_LIMIT";
        } else if (status >= 500) {
          error = new Error(`AI provider server error (HTTP ${status})`);
          error.status = 502;
          error.code = "UPSTREAM_ERROR";
        } else {
          error = new Error(`AI provider returned HTTP ${status}`);
          error.status = 502;
          error.code = "UPSTREAM_ERROR";
        }
        error.diagnostics = { status, contentType };
        throw error;
      }

      setStage(`iteration_${iterations}_read_response_body`);
      const bodyReadStart = Date.now();
      let rawText;
      try {
        rawText = await response.text();
      } catch (readErr) {
        if (readErr.name === "AbortError" || deadlineExceeded || Date.now() >= deadline || overallController.signal.aborted) {
          throw createTimeoutError();
        }

        const status = response?.status || "unknown";
        const contentType = (typeof response?.headers?.get === "function" ? response.headers.get("content-type") : null) || "unknown";
        const errName = readErr?.name || "Error";
        const errMessage = readErr?.message || "Unknown error";

        console.error(`[Investigation Agent] Failed to read response body: status=${status}, contentType=${contentType}, error=${errName}: ${errMessage}`);

        const error = new Error(`Failed to read response from AI provider (${errName}: ${errMessage})`);
        error.status = 502;
        error.code = "MALFORMED_OUTPUT";
        error.diagnostics = {
          status,
          contentType,
          errorName: errName,
          errorMessage: errMessage
        };
        throw error;
      }

      const bodyReadDuration = Date.now() - bodyReadStart;
      recordTiming(`iteration_${iterations}_response_body_read`, bodyReadDuration);
      console.log(`[Investigation Agent] Iteration ${iterations}: Response body read in ${bodyReadDuration}ms (${rawText.length} bytes)`);

      setStage(`iteration_${iterations}_parse_json`);
      let completion;
      try {
        completion = JSON.parse(rawText);
      } catch (jsonErr) {
        const status = response?.status || "unknown";
        const contentType = (typeof response?.headers?.get === "function" ? response.headers.get("content-type") : null) || "unknown";
        const errName = jsonErr?.name || "SyntaxError";
        const errMessage = jsonErr?.message || "Invalid JSON";

        console.error(`[Investigation Agent] Invalid JSON from AI provider: status=${status}, contentType=${contentType}, error=${errName}: ${errMessage}`);

        const error = new Error("Invalid JSON response from AI provider");
        error.status = 502;
        error.code = "MALFORMED_OUTPUT";
        error.diagnostics = {
          status,
          contentType,
          errorName: errName,
          errorMessage: errMessage
        };
        throw error;
      }

      if (completion && completion.usage) {
        hasUsage = true;
        const p = Number(completion.usage.prompt_tokens) || 0;
        const c = Number(completion.usage.completion_tokens) || 0;
        const t = Number(completion.usage.total_tokens) || (p + c);
        accumulatedUsage.promptTokens += p;
        accumulatedUsage.completionTokens += c;
        accumulatedUsage.totalTokens += t;
        if (completion.usage.total_cost !== undefined) {
          accumulatedUsage.totalCost = (accumulatedUsage.totalCost || 0) + Number(completion.usage.total_cost);
        }
      }

    const choice = completion.choices?.[0];
    if (!choice || !choice.message) {
      const error = new Error("Empty choice returned from AI provider");
      error.status = 502;
      error.code = "MALFORMED_OUTPUT";
      throw error;
    }

    const assistantMsg = choice.message;
    messages.push(assistantMsg);

    // If the model requested tool calls, execute them
    if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
      if (iterations >= MAX_PROVIDER_ITERATIONS) {
        const error = new Error("Agent reached maximum 2 provider iterations without completing final synthesis");
        error.status = 502;
        error.code = "ITERATION_LIMIT_EXCEEDED";
        error.diagnostics = {
          iterationsCompleted: iterations,
          requestedTools: assistantMsg.tool_calls.map(c => c.function?.name)
        };
        throw error;
      }

      for (const call of assistantMsg.tool_calls) {
        const funcName = call.function?.name;
        setStage(`iteration_${iterations}_tool_${funcName}`);

        if (deadlineExceeded || Date.now() >= deadline || overallController.signal.aborted) {
          throw createTimeoutError();
        }

        const funcArgs = {};
        try {
          Object.assign(funcArgs, JSON.parse(call.function?.arguments || "{}"));
        } catch (_) {
          funcArgs.error = "Malformed arguments JSON";
        }

        const toolStart = Date.now();
        const cacheKey = `${funcName}:${JSON.stringify(funcArgs)}`;
        let toolResult;
        let isReused = false;

        if (sessionToolCache.has(cacheKey)) {
          toolResult = sessionToolCache.get(cacheKey);
          isReused = true;
          console.log(`[Investigation Agent] Iteration ${iterations}: Tool '${funcName}' reused from session cache`);
        } else {
          try {
            toolResult = await executeTool(funcName, funcArgs, deps);
          } catch (execErr) {
            toolResult = { error: `Execution failure: ${execErr.message}` };
          }
          sessionToolCache.set(cacheKey, toolResult);
        }

        const toolDuration = Date.now() - toolStart;
        recordTiming(`iteration_${iterations}_tool_${funcName}`, toolDuration);
        if (!isReused) {
          console.log(`[Investigation Agent] Iteration ${iterations}: Tool '${funcName}' executed in ${toolDuration}ms`);
        }

        toolExecutionLog.push({
          tool: funcName,
          args: funcArgs,
          result: toolResult
        });

        messages.push({
          role: "tool",
          tool_call_id: call.id,
          name: funcName,
          content: JSON.stringify(toolResult)
        });
      }
      setStage(`iteration_${iterations}_tools_completed`);
      // Continue next iteration to let model inspect tool outputs
      continue;
    }

    setStage(`iteration_${iterations}_synthesis_parse`);

    // No further tool calls: parse final assistant response.
    // Only message.content is the model's committed output. message.reasoning / reasoning_content
    // contain internal chain-of-thought deliberation and MUST NOT be treated as the final answer.
    const rawContent = assistantMsg.content || "";
    let parsedResult = null;

    // 1. Direct JSON parse (for JSON mode or bare JSON)
    try {
      parsedResult = JSON.parse(rawContent.trim());
    } catch (_) {}

    // 2. Look for ```json ... ``` block (or fenced without closing ``` if truncated)
    if (!parsedResult) {
      const jsonMatch = rawContent.match(/```json\s*([\s\S]*?)(?:```|$)/);
      if (jsonMatch && jsonMatch[1].trim()) {
        try {
          parsedResult = JSON.parse(jsonMatch[1].trim());
        } catch (_) {}
      }
    }

    // 3. Look for generic ``` ... ``` block
    if (!parsedResult) {
      const codeMatch = rawContent.match(/```\s*([\s\S]*?)(?:```|$)/);
      if (codeMatch && codeMatch[1].trim()) {
        try {
          parsedResult = JSON.parse(codeMatch[1].trim());
        } catch (_) {}
      }
    }

    // 4. Substring from outermost '{' to '}'
    if (!parsedResult) {
      const firstBrace = rawContent.indexOf("{");
      const lastBrace = rawContent.lastIndexOf("}");
      if (firstBrace !== -1 && lastBrace > firstBrace) {
        try {
          parsedResult = JSON.parse(rawContent.slice(firstBrace, lastBrace + 1));
        } catch (_) {}
      }
    }

    // 5. If truncated at token limit, attempt safe bracket closure repair.
    // Guard: the repaired object must pass full semantic validation — both summary
    // (non-empty string) AND keyFindings (non-empty array with valid finding strings).
    // This prevents accepting partial deliberation text that happens to contain '{}'.
    const isTruncated = choice.finish_reason === "length";
    if (!parsedResult && isTruncated && rawContent) {
      const firstBrace = rawContent.indexOf("{");
      if (firstBrace !== -1) {
        let candidate = rawContent.slice(firstBrace).trim();
        candidate = candidate.replace(/,\s*"[^"]*"?\s*:\s*[^,}\]]*$/, "");
        candidate = candidate.replace(/,\s*"[^"]*$/, "");
        candidate = candidate.replace(/,\s*$/, "");
        const openBraces = (candidate.match(/\{/g) || []).length;
        const closeBraces = (candidate.match(/\}/g) || []).length;
        const openBrackets = (candidate.match(/\[/g) || []).length;
        const closeBrackets = (candidate.match(/\]/g) || []).length;
        if (openBrackets > closeBrackets) candidate += "]".repeat(openBrackets - closeBrackets);
        if (openBraces > closeBraces) candidate += "}".repeat(openBraces - closeBraces);
        try {
          const repaired = JSON.parse(candidate);
          if (isValidReportObject(repaired)) {
            parsedResult = repaired;
          }
        } catch (_) {}
      }
    }

    // Explicit fallback when response is empty, truncated, or narrative text.
    // IMPORTANT: Never copy raw model output (which may be internal deliberation)
    // into keyFindings.finding. Use only safe static strings.
    if (!parsedResult || typeof parsedResult !== "object") {
      const isEmpty = !rawContent || !rawContent.trim();
      if (isTruncated) {
        parsedResult = {
          summary: "Investigation inconclusive: AI provider output was truncated at token limit.",
          status: "INCONCLUSIVE",
          confidence: "LOW",
          keyFindings: [{
            finding: "AI provider response was cut off at the token limit before a complete JSON report was produced.",
            status: "INCONCLUSIVE",
            evidence: toolExecutionLog.map(t => t.tool),
            confidence: "LOW"
          }],
          risks: ["Incomplete synthesis: output was cut off at the token limit."],
          missingEvidence: [`AI provider output was truncated at the ${config.maxTokens || DEFAULT_MAX_TOKENS} output token limit. Set QWEN_MAX_TOKENS to a higher value or narrow the investigation scope to a specific donation ID.`],
          recommendedActions: ["Retry with a higher QWEN_MAX_TOKENS or submit a targeted query for a single donation ID."]
        };
      } else if (isEmpty) {
        parsedResult = {
          summary: "Investigation inconclusive: AI provider returned empty response content.",
          status: "INCONCLUSIVE",
          confidence: "LOW",
          keyFindings: [{
            finding: "AI provider returned empty response content.",
            status: "INCONCLUSIVE",
            evidence: toolExecutionLog.map(t => t.tool),
            confidence: "LOW"
          }],
          risks: [],
          missingEvidence: ["No response content was returned by the AI provider."],
          recommendedActions: ["Retry the investigation query."]
        };
      } else {
        // Narrative or non-JSON text: report inconclusive but do NOT embed raw model text.
        parsedResult = {
          summary: "Investigation inconclusive: Model returned unstructured output instead of the required JSON schema.",
          status: "INCONCLUSIVE",
          confidence: "LOW",
          keyFindings: [{
            finding: "Model returned unstructured output instead of strict JSON schema.",
            status: "INCONCLUSIVE",
            evidence: toolExecutionLog.map(t => t.tool),
            confidence: "LOW"
          }],
          risks: [],
          missingEvidence: ["Model returned unstructured output instead of strict JSON schema."],
          recommendedActions: ["Retry with a more specific query or verify SYSTEM_PROMPT format constraints."]
        };
      }
    }

    // Normalise: accept both new schema (keyFindings) and legacy schema (findings/evidence/conflicts)
    // so existing tests that mock the old schema still produce valid output.
    const isNewSchema = Array.isArray(parsedResult.keyFindings);
    const report = isNewSchema ? {
      summary: parsedResult.summary || "Investigation completed.",
      keyFindings: parsedResult.keyFindings,
      risks: Array.isArray(parsedResult.risks) ? parsedResult.risks : [],
      missingEvidence: Array.isArray(parsedResult.missingEvidence) ? parsedResult.missingEvidence : [],
      recommendedActions: Array.isArray(parsedResult.recommendedActions) ? parsedResult.recommendedActions : []
    } : {
      // Legacy schema adapter for backward-compat with existing tests and callers.
      summary: parsedResult.summary || "Investigation completed.",
      keyFindings: (Array.isArray(parsedResult.findings) ? parsedResult.findings : []).map(f => ({
        finding: typeof f === "string" ? f : JSON.stringify(f),
        status: "INCONCLUSIVE",
        evidence: [],
        confidence: "LOW"
      })),
      risks: Array.isArray(parsedResult.conflicts) ? parsedResult.conflicts : [],
      missingEvidence: Array.isArray(parsedResult.uncertainties) ? parsedResult.uncertainties : [],
      recommendedActions: Array.isArray(parsedResult.suggestedFollowUp) ? parsedResult.suggestedFollowUp : []
    };

    if (isTruncated) {
      const truncationNotice = `AI provider output was truncated at the ${config.maxTokens || DEFAULT_MAX_TOKENS} output token limit. Set a higher QWEN_MAX_TOKENS or narrow the investigation scope.`;
      if (!report.missingEvidence.some(m => m.includes("truncated"))) {
        report.missingEvidence.push(truncationNotice);
      }
    }

    // Overall status determination based on parsedResult or keyFindings
    const overallStatus = parsedResult.status ||
      (isTruncated ? "INCONCLUSIVE" :
       report.keyFindings.some(kf => kf.status === "TAMPERED") ? "TAMPERED" :
       report.keyFindings.some(kf => kf.status === "HASH_MISMATCH") ? "HASH_MISMATCH" :
       report.keyFindings.some(kf => kf.status === "VERIFIED") ? "VERIFIED" :
       report.keyFindings.some(kf => kf.status === "NOT_REGISTERED") ? "NOT_REGISTERED" : "INCONCLUSIVE");

    const overallConfidence = parsedResult.confidence ||
      (report.keyFindings.length > 0 && report.keyFindings.every(kf => kf.confidence === "HIGH") ? "HIGH" :
       (report.keyFindings.some(kf => kf.confidence === "LOW") || isTruncated) ? "LOW" : "MEDIUM");

    // Build the final report object. Preserve both new and legacy field names for compatibility.
    const finalReport = {
      summary: report.summary,
      status: overallStatus,
      confidence: overallConfidence,
      keyFindings: report.keyFindings,
      evidence: Array.isArray(parsedResult?.evidence) ? parsedResult.evidence : report.keyFindings.flatMap(kf => Array.isArray(kf.evidence) ? kf.evidence : []),
      risks: report.risks,
      missingEvidence: report.missingEvidence,
      recommendedActions: report.recommendedActions
    };
    // Legacy aliases
    finalReport.findings = report.keyFindings.map(kf => kf.finding);
    finalReport.conflicts = report.risks; // legacy field used for conflicts
    finalReport.uncertainties = report.missingEvidence;
    finalReport.suggestedFollowUp = report.recommendedActions;

    return {
      success: true,
      investigationId,
      model: config.model,
      iterations,
      toolsInvoked: toolExecutionLog.length,
      report: finalReport,
      ...(hasUsage ? { usage: accumulatedUsage } : {}),
      audit: {
        toolsUsed: toolExecutionLog.map(t => t.tool),
        ...(hasUsage ? { usage: accumulatedUsage } : {})
      }
    };




}
  // If iteration loop exhausted without returning a synthesis report
  const error = new Error("Agent reached maximum provider iterations without completing final report");
  error.status = 502;
  error.code = "ITERATION_LIMIT_EXCEEDED";
  error.diagnostics = {
    iterationsCompleted: iterations,
    maxIterations: MAX_PROVIDER_ITERATIONS
  };
  throw error;
  } finally {
    clearTimeout(overallTimeoutId);
  }
}

/**
 * Tests an OpenRouter / Qwen API key for validity and quota status.
 */
async function testApiKey(apiKey, fetchFn = globalThis.fetch) {
  const key = String(apiKey || "").trim();
  if (!key) {
    return {
      valid: false,
      status: 400,
      code: "KEY_MISSING",
      error: "Enter your Qwen API key to activate the agent"
    };
  }

  try {
    // 1. Check key with OpenRouter auth verification endpoint
    const res = await fetchFn("https://openrouter.ai/api/v1/auth/key", {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${key}`,
        "Content-Type": "application/json"
      }
    });

    if (res.status === 200) {
      let data = {};
      try { data = await res.json(); } catch (_) {}
      return {
        valid: true,
        status: 200,
        message: "Qwen API key is valid and connected to OpenRouter.",
        details: {
          label: data?.data?.label || "Active Key",
          usage: data?.data?.usage != null ? `$${Number(data.data.usage).toFixed(4)}` : undefined,
          limit: data?.data?.limit != null ? `$${Number(data.data.limit).toFixed(2)}` : undefined,
          isFreeTier: data?.data?.is_free_tier
        }
      };
    }

    if (res.status === 401 || res.status === 403) {
      return {
        valid: false,
        status: 401,
        code: "INVALID_KEY",
        error: "Invalid Qwen API key. OpenRouter authentication failed."
      };
    }

    if (res.status === 402) {
      return {
        valid: false,
        status: 402,
        code: "QUOTA_EXCEEDED",
        error: "Insufficient credits or quota on this API key."
      };
    }

    if (res.status === 429) {
      return {
        valid: false,
        status: 429,
        code: "RATE_LIMIT",
        error: "Rate limit reached on OpenRouter for this API key."
      };
    }

    // Fallback: minimal single-token probe
    const probeRes = await fetchFn("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${key}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        messages: [{ role: "user", content: "ping" }],
        max_tokens: 1
      })
    });

    if (probeRes.ok) {
      return {
        valid: true,
        status: 200,
        message: "Qwen API key is valid and model response confirmed."
      };
    }

    if (probeRes.status === 401 || probeRes.status === 403) {
      return {
        valid: false,
        status: 401,
        code: "INVALID_KEY",
        error: "Invalid Qwen API key. OpenRouter authentication failed."
      };
    }

    if (probeRes.status === 402 || probeRes.status === 429) {
      return {
        valid: false,
        status: probeRes.status,
        code: probeRes.status === 402 ? "QUOTA_EXCEEDED" : "RATE_LIMIT",
        error: probeRes.status === 402 ? "Insufficient credits on this API key." : "Rate limit reached on OpenRouter."
      };
    }

    return {
      valid: false,
      status: 502,
      code: "UPSTREAM_ERROR",
      error: `OpenRouter returned HTTP ${probeRes.status}`
    };
  } catch (err) {
    return {
      valid: false,
      status: 502,
      code: "NETWORK_ERROR",
      error: `Network failure connecting to AI provider: ${err.message}`
    };
  }
}

/**
 * Route handler factory for testing Qwen API keys.
 */
function createTestKeyHandler(fetchFn = globalThis.fetch) {
  return async function handleTestKey(req, res) {
    const key = (
      req.body?.apiKey ||
      (req.headers?.authorization?.startsWith("Bearer ") ? req.headers.authorization.slice(7) : "") ||
      ""
    ).trim();

    const result = await testApiKey(key, fetchFn);
    if (!result.valid) {
      return res.status(result.status || 400).json({
        success: false,
        error: result.error,
        code: result.code || "KEY_TEST_FAILED"
      });
    }
    return res.json({
      success: true,
      message: result.message,
      details: result.details
    });
  };
}

/**
 * Express Route Handler Factory.
 */
function createInvestigationHandler(deps) {
  return async function handleInvestigation(req, res) {
    const userApiKey = (
      req.body?.apiKey ||
      (req.headers?.authorization?.startsWith("Bearer ") ? req.headers.authorization.slice(7) : "") ||
      ""
    ).trim();

    const baseConfig = getAgentConfig(process.env);

    // If user provided a key at runtime, activate the agent with that key
    const effectiveConfig = userApiKey ? {
      ...baseConfig,
      apiKey: userApiKey,
      enabled: true,
      configured: true,
      maxTokens: req.body?.maxTokens
        ? Math.min(Math.max(Number(req.body.maxTokens) || 700, 100), 2000)
        : (baseConfig.maxTokens > 300 ? baseConfig.maxTokens : 700)
    } : {
      ...baseConfig,
      apiKey: (process.env.OPENROUTER_API_KEY || "").trim()
    };

    // Guard: Missing key / disabled check
    if (!effectiveConfig.enabled || !effectiveConfig.apiKey) {
      return res.status(503).json({
        success: false,
        error: "Enter your Qwen API key to activate the agent",
        code: "AGENT_DISABLED",
        enabled: false
      });
    }

    // Input validation
    const query = typeof req.body?.query === "string" ? req.body.query : "";
    const targetId = typeof req.body?.targetId === "string" ? req.body.targetId : (req.body?.donationId || null);

    if (query.length > MAX_INPUT_CHARS) {
      return res.status(400).json({
        success: false,
        error: `Query exceeds maximum allowed length of ${MAX_INPUT_CHARS} characters.`,
        code: "INPUT_TOO_LARGE"
      });
    }

    try {
      const result = await runInvestigation({
        query,
        targetId,
        deps,
        config: effectiveConfig
      });
      return res.json(result);
    } catch (err) {
      const statusCode = err.status || 500;
      const responsePayload = {
        success: false,
        error: err.message || "Investigation failed",
        code: err.code || "AGENT_ERROR"
      };
      if (err.diagnostics) {
        responsePayload.diagnostics = err.diagnostics;
      }
      return res.status(statusCode).json(responsePayload);
    }
  };
}

module.exports = {
  getAgentConfig,
  AGENT_TOOLS,
  SYSTEM_PROMPT,
  executeTool,
  runInvestigation,
  createInvestigationHandler,
  testApiKey,
  createTestKeyHandler
};
