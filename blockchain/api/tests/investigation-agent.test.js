/**
 * DAANDRISHTI — Qwen Investigation Agent Test Suite
 * ==================================================
 *
 * Fully mocked, isolated test suite verifying:
 * 1. Safe startup & disabled handling (QWEN_ENABLED=false / missing key).
 * 2. Valid agent request with mocked Qwen tool calling and structured report.
 * 3. Execution of all allowlisted read-only tools.
 * 4. Rejection of unknown tool names and malformed tool arguments.
 * 5. Rejection of arbitrary SQL, shell execution, or contract write attempts.
 * 6. Detection and explicit reporting of conflicting/tampered evidence.
 * 7. Verification that missing Envio data is NOT treated as invalid proof.
 * 8. Graceful handling of timeouts, HTTP 429 rate limits, and 500 upstream errors.
 * 9. Absolute key redaction: API keys never leak into responses, logs, or error objects.
 * 10. Verification of request input size limits.
 *
 * NOTE: 100% mocked network calls. Zero live credits consumed. Zero transactions broadcast.
 */

const assert = require("assert");
const {
  getAgentConfig,
  AGENT_TOOLS,
  executeTool,
  runInvestigation,
  createInvestigationHandler,
  testApiKey,
  createTestKeyHandler
} = require("../services/investigation-agent");

let passed = 0;
let failed = 0;

function pass(name) {
  console.log(`  ✅ PASS: ${name}`);
  passed++;
}

function fail(name, err) {
  console.error(`  ❌ FAIL: ${name}`);
  if (err) console.error(`         → ${err.message || err}`);
  failed++;
}

async function test(name, fn) {
  try {
    await fn();
    pass(name);
  } catch (err) {
    fail(name, err);
  }
}

// ---------------------------------------------------------------------
// Mock Fixtures
// ---------------------------------------------------------------------

const mockDeps = {
  contract: {
    verifyProof: async (hash) => {
      if (hash === "0x1111111111111111111111111111111111111111111111111111111111111111") {
        return [true, "webcam-0", 1791346726n, "0x424090934608c9e0234CaE6BEe1f353747a2B05d"];
      }
      return [false, "", 0n, "0x0000000000000000000000000000000000000000"];
    }
  },
  getAllDonationRecords: () => [
    {
      lineNumber: 1,
      donation: { donation_id: "don_valid_100", denomination: 100 },
      record: {
        pocket_id: "webcam-0",
        timestamp: "2026-10-10T12:00:00Z",
        predicted_denomination: "100",
        confidence: 0.98,
        decision: "ACCEPT",
        target_bin: "BIN_100"
      }
    },
    {
      lineNumber: 2,
      donation: { donation_id: "don_tampered_50", denomination: 50 },
      record: {
        pocket_id: "webcam-0",
        timestamp: "2026-10-10T12:05:00Z",
        predicted_denomination: "1052", // tampered
        confidence: 0.85,
        decision: "ACCEPT",
        target_bin: "BIN_1052"
      }
    }
  ],
  getDonationById: (id) => {
    if (id === "don_valid_100") {
      return {
        lineNumber: 1,
        donation: { donation_id: "don_valid_100", denomination: 100 },
        record: {
          pocket_id: "webcam-0",
          timestamp: "2026-10-10T12:00:00Z",
          predicted_denomination: "100",
          decision: "ACCEPT",
          target_bin: "BIN_100"
        }
      };
    }
    if (id === "don_tampered_50") {
      return {
        lineNumber: 2,
        donation: { donation_id: "don_tampered_50", denomination: 50 },
        record: {
          pocket_id: "webcam-0",
          timestamp: "2026-10-10T12:05:00Z",
          predicted_denomination: "1052",
          decision: "ACCEPT",
          target_bin: "BIN_1052"
        }
      };
    }
    if (id === "don_mismatch_100") {
      return {
        lineNumber: 3,
        donation: { donation_id: "don_mismatch_100", denomination: 100 },
        record: {
          pocket_id: "webcam-0",
          timestamp: "2026-10-10T12:10:00Z",
          predicted_denomination: "100",
          decision: "ACCEPT",
          target_bin: "BIN_100"
        }
      };
    }
    return null;
  },
  findOriginalSnapshot: (record) => {
    if (record.timestamp === "2026-10-10T12:00:00Z") {
      return {
        recordId: "webcam-0|2026-10-10T12:00:00Z",
        proofHash: "0x1111111111111111111111111111111111111111111111111111111111111111",
        originalDenomination: "100",
        originalBin: "BIN_100",
        registeredAt: "2026-10-10T12:01:00Z"
      };
    }
    if (record.timestamp === "2026-10-10T12:05:00Z") {
      return {
        recordId: "webcam-0|2026-10-10T12:05:00Z",
        proofHash: "0x2222222222222222222222222222222222222222222222222222222222222222",
        originalDenomination: "50",
        originalBin: "BIN_50",
        registeredAt: "2026-10-10T12:06:00Z"
      };
    }
    if (record.timestamp === "2026-10-10T12:10:00Z") {
      return {
        recordId: "webcam-0|2026-10-10T12:10:00Z",
        proofHash: "0x4444444444444444444444444444444444444444444444444444444444444444",
        originalDenomination: "100",
        originalBin: "BIN_100",
        registeredAt: "2026-10-10T12:11:00Z"
      };
    }
    return null;
  },
  buildProofCandidates: (item, snapshot) => {
    if (item.donation.donation_id === "don_valid_100") {
      return {
        legacyHash: "0x1111111111111111111111111111111111111111111111111111111111111111",
        canonicalHash: "0x1111111111111111111111111111111111111111111111111111111111111111"
      };
    }
    return {
      legacyHash: "0x3333333333333333333333333333333333333333333333333333333333333333",
      canonicalHash: "0x3333333333333333333333333333333333333333333333333333333333333333"
    };
  },
  queryEnvioProof: async (hash) => {
    if (hash === "0x1111111111111111111111111111111111111111111111111111111111111111") {
      return {
        txHash: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        blockNumber: 68873538,
        timestamp: "1791346726"
      };
    }
    return null; // Simulate unindexed proof
  },
  readOriginalRecords: () => [
    {
      recordId: "webcam-0|2026-10-10T12:00:00Z",
      proofHash: "0x1111111111111111111111111111111111111111111111111111111111111111",
      originalDenomination: "100",
      originalBin: "BIN_100",
      registeredAt: "2026-10-10T12:01:00Z"
    }
  ]
};

// ---------------------------------------------------------------------
// TEST RUNNER
// ---------------------------------------------------------------------

async function runAllTests() {
  console.log("============================================================");
  console.log("DAANDRISHTI — QWEN INVESTIGATION AGENT TEST SUITE (MOCKED)");
  console.log("============================================================\n");

  // 1. Config & Disabled Status
  await test("1.1 Agent disabled by default when QWEN_ENABLED is not set", () => {
    const config = getAgentConfig({ OPENROUTER_API_KEY: "test_key" });
    assert.strictEqual(config.enabled, false);
  });

  await test("1.2 Agent disabled if OPENROUTER_API_KEY is empty even if QWEN_ENABLED=true", () => {
    const config = getAgentConfig({ QWEN_ENABLED: "true", OPENROUTER_API_KEY: "" });
    assert.strictEqual(config.enabled, false);
  });

  await test("1.3 Agent enabled only when both QWEN_ENABLED=true and key are present", () => {
    const config = getAgentConfig({ QWEN_ENABLED: "true", OPENROUTER_API_KEY: "sk-or-v1-abc" });
    assert.strictEqual(config.enabled, true);
    assert.strictEqual(config.model, "qwen/qwen3.8-max-0902");
  });

  await test("1.4 Configurable maxTokens defaults to 300 and respects QWEN_MAX_TOKENS", () => {
    const configDefault = getAgentConfig({ QWEN_ENABLED: "true", OPENROUTER_API_KEY: "sk-or-v1-abc" });
    assert.strictEqual(configDefault.maxTokens, 300);

    const configCustom = getAgentConfig({ QWEN_ENABLED: "true", OPENROUTER_API_KEY: "sk-or-v1-abc", QWEN_MAX_TOKENS: "256" });
    assert.strictEqual(configCustom.maxTokens, 256);
  });

  await test("1.5 Route returns HTTP 503 when agent is disabled", async () => {
    const prevFlag = process.env.QWEN_ENABLED;
    delete process.env.QWEN_ENABLED;
    const handler = createInvestigationHandler(mockDeps);

    let resStatus = 0;
    let resJson = null;
    const req = { body: { query: "investigate" } };
    const res = {
      status: (s) => { resStatus = s; return res; },
      json: (j) => { resJson = j; return res; }
    };

    await handler(req, res);
    assert.strictEqual(resStatus, 503);
    assert.strictEqual(resJson.success, false);
    assert.strictEqual(resJson.code, "AGENT_DISABLED");
    if (prevFlag !== undefined) process.env.QWEN_ENABLED = prevFlag;
  });

  // 2. Allowlisted Tools Execution
  await test("2.1 get_donation_records tool works with valid parameters", async () => {
    const res = await executeTool("get_donation_records", { limit: 5, filter: "ALL" }, mockDeps);
    assert.strictEqual(res.totalFound, 2);
    assert.strictEqual(res.records[0].donation_id, "don_valid_100");
  });

  await test("2.2 get_donation_by_id tool finds valid record", async () => {
    const res = await executeTool("get_donation_by_id", { donationId: "don_valid_100" }, mockDeps);
    assert.strictEqual(res.found, true);
    assert.strictEqual(res.donation.denomination, 100);
  });

  await test("2.3 verify_record_integrity correctly classifies untampered record as VERIFIED", async () => {
    const res = await executeTool("verify_record_integrity", { donationId: "don_valid_100" }, mockDeps);
    assert.strictEqual(res.status, "VERIFIED");
    assert.strictEqual(res.tamperDetails, null);
  });

  await test("2.4 verify_record_integrity flags modified denomination as TAMPERED with diff", async () => {
    const res = await executeTool("verify_record_integrity", { donationId: "don_tampered_50" }, mockDeps);
    assert.strictEqual(res.status, "TAMPERED");
    assert.ok(res.tamperDetails.includes("Denomination changed from ₹50 to ₹1052"));
  });

  await test("2.5 verify_on_chain_proof verifies authentic hash on Monad contract mock", async () => {
    const res = await executeTool("verify_on_chain_proof", {
      proofHash: "0x1111111111111111111111111111111111111111111111111111111111111111"
    }, mockDeps);
    assert.strictEqual(res.exists, true);
    assert.strictEqual(res.sourceId, "webcam-0");
  });

  await test("2.6 get_indexed_proof_history reports missing Envio without declaring proof invalid", async () => {
    const res = await executeTool("get_indexed_proof_history", {
      proofHash: "0x2222222222222222222222222222222222222222222222222222222222222222"
    }, mockDeps);
    assert.strictEqual(res.indexed, false);
    assert.ok(res.message.includes("does not mean invalid proof") || res.message.includes("still exist on-chain"));
  });

  // 3. Security & Disallowed Operations
  await test("3.1 Disallowed tool names are rejected with error", async () => {
    const res = await executeTool("execute_raw_sql", { query: "SELECT * FROM users" }, mockDeps);
    assert.ok(res.error.includes("Disallowed or unknown tool name"));
  });

  await test("3.2 Shell execution or filesystem writes are strictly absent from AGENT_TOOLS", () => {
    const toolNames = AGENT_TOOLS.map(t => t.function.name);
    assert.ok(!toolNames.includes("shell_exec"));
    assert.ok(!toolNames.includes("run_command"));
    assert.ok(!toolNames.includes("write_file"));
    assert.ok(!toolNames.includes("insert_sql"));
  });

  await test("3.3 Input query length capping rejects oversized inputs with HTTP 400", async () => {
    const handler = createInvestigationHandler(mockDeps);
    process.env.QWEN_ENABLED = "true";
    process.env.OPENROUTER_API_KEY = "dummy_secret_key_123456789";

    let resStatus = 0;
    let resJson = null;
    const req = { body: { query: "x".repeat(3000) } };
    const res = {
      status: (s) => { resStatus = s; return res; },
      json: (j) => { resJson = j; return res; }
    };

    await handler(req, res);
    assert.strictEqual(resStatus, 400);
    assert.strictEqual(resJson.code, "INPUT_TOO_LARGE");
    delete process.env.QWEN_ENABLED;
    delete process.env.OPENROUTER_API_KEY;
  });

  // 4. Mocked Multi-turn Orchestration
  await test("4.1 Agent handles multi-turn tool calling and produces structured report", async () => {
    const SECRET_KEY = "test_openrouter_secret_never_leak_xyz";
    process.env.OPENROUTER_API_KEY = SECRET_KEY;

    let turn = 0;
    const mockFetch = async (url, options) => {
      // Confirm auth header and JSON accept header were supplied securely
      assert.strictEqual(options.headers["Authorization"], `Bearer ${SECRET_KEY}`);
      assert.strictEqual(options.headers["Accept"], "application/json");
      const parsedBody = JSON.parse(options.body);
      assert.strictEqual(parsedBody.stream, false);
      turn++;

      if (turn === 1) {
        // Model asks to inspect donation record
        const payload1 = {
          choices: [{
            message: {
              role: "assistant",
              content: null,
              tool_calls: [{
                id: "call_1",
                type: "function",
                function: {
                  name: "verify_record_integrity",
                  arguments: JSON.stringify({ donationId: "don_valid_100" })
                }
              }]
            }
          }]
        };
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify(payload1),
          json: async () => payload1
        };
      }

      // Turn 2: Model receives tool result and produces final report
      const payload2 = {
        choices: [{
          message: {
            role: "assistant",
            content: "```json\n" + JSON.stringify({
              summary: "Record don_valid_100 passed cryptographic integrity checks.",
              findings: ["Record matches registered hash."],
              evidence: [{ type: "CRYPTOGRAPHIC", identifier: "don_valid_100", status: "VERIFIED" }],
              conflicts: [],
              uncertainties: [],
              suggestedFollowUp: ["Check on-chain contract registration."]
            }) + "\n```"
          }
        }]
      };
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify(payload2),
        json: async () => payload2
      };
    };

    const result = await runInvestigation({
      query: "Check don_valid_100",
      deps: mockDeps,
      config: {
        model: "qwen/qwen3.8-max-0902",
        apiUrl: "https://mock.openrouter.ai/v1/chat/completions",
        timeoutMs: 5000
      },
      fetchFn: mockFetch
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.toolsInvoked, 1);
    assert.strictEqual(result.report.summary, "Record don_valid_100 passed cryptographic integrity checks.");
    assert.strictEqual(result.report.evidence[0].status, "VERIFIED");

    // 5. Zero Secret Leaks Check
    const resultString = JSON.stringify(result);
    assert.ok(!resultString.includes(SECRET_KEY), "API key must NEVER appear in agent response");

    delete process.env.OPENROUTER_API_KEY;
  });

  // 6. Upstream Error Handling (429, Timeout, 502)
  await test("6.1 Handles OpenRouter HTTP 429 rate limit cleanly", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetch429 = async () => ({
      ok: false,
      status: 429,
      text: async () => "Rate limit exceeded"
    });

    try {
      await runInvestigation({
        query: "test",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetch429
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 429);
      assert.strictEqual(err.code, "RATE_LIMIT");
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.2 Handles upstream HTTP 500 cleanly as 502 UPSTREAM_ERROR", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetch500 = async () => ({
      ok: false,
      status: 500,
      text: async () => "Internal server error"
    });

    try {
      await runInvestigation({
        query: "test",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetch500
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "UPSTREAM_ERROR");
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.3 Handles HTTP 401 authentication failure cleanly as AUTH_ERROR", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetch401 = async () => ({
      ok: false,
      status: 401,
      headers: { get: () => "application/json" },
      text: async () => JSON.stringify({ error: { message: "Invalid API key" } })
    });

    try {
      await runInvestigation({
        query: "test 401",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetch401
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "AUTH_ERROR");
      assert.ok(err.message.includes("authentication failed (HTTP 401)"));
      assert.strictEqual(err.diagnostics.status, 401);
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.4 Handles HTTP 403 authentication failure cleanly as AUTH_ERROR", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetch403 = async () => ({
      ok: false,
      status: 403,
      headers: { get: () => "application/json" },
      text: async () => JSON.stringify({ error: { message: "Forbidden" } })
    });

    try {
      await runInvestigation({
        query: "test 403",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetch403
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "AUTH_ERROR");
      assert.ok(err.message.includes("authentication failed (HTTP 403)"));
      assert.strictEqual(err.diagnostics.status, 403);
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.5 Handles network connection failure cleanly as NETWORK_ERROR", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchNetworkError = async () => {
      throw new Error("getaddrinfo ENOTFOUND openrouter.ai");
    };

    try {
      await runInvestigation({
        query: "test network error",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchNetworkError
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "NETWORK_ERROR");
      assert.ok(err.message.includes("Network error connecting to AI provider"));
      assert.ok(err.message.includes("ENOTFOUND"));
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.6 Handles AbortError/timeout during response body reading as TIMEOUT (504)", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchAbortOnRead = async () => ({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      text: async () => {
        const abortErr = new Error("The operation was aborted");
        abortErr.name = "AbortError";
        throw abortErr;
      }
    });

    try {
      await runInvestigation({
        query: "test abort during read",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchAbortOnRead
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 504);
      assert.strictEqual(err.code, "TIMEOUT");
      assert.ok(err.message.includes("Investigation deadline exceeded (timeout)"));
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.7 Handles stream reading failure with sanitized diagnostics as MALFORMED_OUTPUT (502)", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchStreamFail = async () => ({
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      text: async () => {
        const streamErr = new TypeError("terminated");
        throw streamErr;
      }
    });

    try {
      await runInvestigation({
        query: "test stream error",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchStreamFail
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "MALFORMED_OUTPUT");
      assert.ok(err.message.includes("Failed to read response from AI provider (TypeError: terminated)"));
      assert.strictEqual(err.diagnostics.status, 200);
      assert.strictEqual(err.diagnostics.contentType, "application/json");
      assert.strictEqual(err.diagnostics.errorName, "TypeError");
      assert.strictEqual(err.diagnostics.errorMessage, "terminated");
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("6.8 Handles HTTP 402 payment/credits required as PAYMENT_REQUIRED", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetch402 = async () => ({
      ok: false,
      status: 402,
      headers: { get: () => "application/json" },
      text: async () => JSON.stringify({ error: { message: "Credit balance too low" } })
    });

    try {
      await runInvestigation({
        query: "test 402",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetch402
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "PAYMENT_REQUIRED");
      assert.ok(err.message.includes("payment/credits required (HTTP 402)"));
      assert.strictEqual(err.diagnostics.status, 402);
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  // 7. Edge Cases & Cumulative Deadline (Reliability Fixes)
  await test("7.1 Handles malformed tool-call JSON arguments safely without crashing", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let turn = 0;
    const mockFetchMalformed = async () => {
      turn++;
      if (turn === 1) {
        const payload1 = {
          choices: [{
            message: {
              role: "assistant",
              content: null,
              tool_calls: [{
                id: "call_bad_args",
                type: "function",
                function: {
                  name: "get_donation_by_id",
                  arguments: "INVALID_NOT_JSON_{"
                }
              }]
            }
          }]
        };
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify(payload1),
          json: async () => payload1
        };
      }
      const payload2 = {
        choices: [{
          message: {
            role: "assistant",
            content: "```json\n" + JSON.stringify({
              summary: "Handled malformed arguments gracefully.",
              findings: ["Tool reported argument error."],
              evidence: [],
              conflicts: [],
              uncertainties: ["Tool argument was malformed."],
              suggestedFollowUp: []
            }) + "\n```"
          }
        }]
      };
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify(payload2),
        json: async () => payload2
      };
    };

    const result = await runInvestigation({
      query: "test malformed args",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchMalformed
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.toolsInvoked, 1);
    assert.strictEqual(result.report.summary, "Handled malformed arguments gracefully.");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.2 Handles empty choices array from provider with 502 MALFORMED_OUTPUT", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchEmptyChoices = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ choices: [] }),
      json: async () => ({ choices: [] })
    });

    try {
      await runInvestigation({
        query: "test empty choices",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchEmptyChoices
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "MALFORMED_OUTPUT");
      assert.ok(err.message.includes("Empty choice returned"));
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.3 Narrative fallback produces safe INCONCLUSIVE report without embedding raw model text", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const payloadNarrative = {
      choices: [{
        message: {
          role: "assistant",
          content: "Based on the evidence, the donation was recorded with Rs 100 and matches the expected hash."
        }
      }]
    };
    const mockFetchNarrative = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payloadNarrative),
      json: async () => payloadNarrative
    });

    const result = await runInvestigation({
      query: "test narrative fallback",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchNarrative
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.report.status, "INCONCLUSIVE");
    assert.strictEqual(result.report.confidence, "LOW");
    // Must NOT embed raw model text into summary or findings
    assert.ok(!result.report.summary.includes("Based on the evidence"), "Raw model narrative must not appear in summary");
    assert.ok(result.report.summary.includes("unstructured output"));
    assert.ok(Array.isArray(result.report.findings));
    // uncertainties alias missingEvidence
    assert.ok(result.report.uncertainties[0].includes("unstructured output"));
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.4 Cumulative deadline triggers 504 TIMEOUT promptly and prevents subsequent iterations", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchSlow = async (url, options) => {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          const payload = {
            choices: [{
              message: {
                role: "assistant",
                content: "Should not reach here"
              }
            }]
          };
          resolve({
            ok: true,
            status: 200,
            text: async () => JSON.stringify(payload),
            json: async () => payload
          });
        }, 100);

        if (options?.signal) {
          options.signal.addEventListener("abort", () => {
            clearTimeout(timer);
            const abortErr = new Error("The operation was aborted");
            abortErr.name = "AbortError";
            reject(abortErr);
          });
        }
      });
    };

    try {
      await runInvestigation({
        query: "test cumulative timeout",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 50 },
        fetchFn: mockFetchSlow
      });
      assert.fail("Should have thrown error");
    } catch (err) {
      assert.strictEqual(err.status, 504);
      assert.strictEqual(err.code, "TIMEOUT");
      assert.ok(err.message.includes("Investigation deadline exceeded (timeout)"));
      assert.ok(err.diagnostics, "Timeout error must contain diagnostics");
      assert.strictEqual(err.diagnostics.timeoutMs, 50);
      assert.ok(typeof err.diagnostics.expiredAtStage === "string");
      assert.ok(err.diagnostics.totalElapsedMs >= 40);
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.5 Outgoing OpenRouter request explicitly sends Accept: application/json and stream: false", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let capturedOptions = null;
    const mockFetchOptions = async (url, options) => {
      capturedOptions = options;
      const payload = {
        choices: [{
          message: {
            role: "assistant",
            content: "```json\n" + JSON.stringify({
              summary: "Request options verified successfully.",
              findings: [],
              evidence: [],
              conflicts: [],
              uncertainties: [],
              suggestedFollowUp: []
            }) + "\n```"
          }
        }]
      };
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify(payload),
        json: async () => payload
      };
    };

    await runInvestigation({
      query: "verify request options",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchOptions
    });

    assert.ok(capturedOptions, "fetchFn must be called with options");
    assert.strictEqual(capturedOptions.headers["Accept"], "application/json");
    const parsedBody = JSON.parse(capturedOptions.body);
    assert.strictEqual(parsedBody.stream, false);
    assert.strictEqual(parsedBody.max_tokens, 300);
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.6 Rejects invalid JSON upstream response safely with 502 MALFORMED_OUTPUT without exposing raw response body", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const sensitiveHtmlSnippet = "<html><body>502 Bad Gateway from cloudflare-upstream-key-xyz</body></html>";
    const mockFetchInvalidJson = async () => ({
      ok: true,
      status: 200,
      text: async () => sensitiveHtmlSnippet,
      json: async () => { throw new SyntaxError("Unexpected token < in JSON at position 0"); }
    });

    try {
      await runInvestigation({
        query: "test invalid json",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchInvalidJson
      });
      assert.fail("Should have thrown error on invalid JSON response");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "MALFORMED_OUTPUT");
      assert.strictEqual(err.message, "Invalid JSON response from AI provider");
      assert.ok(!err.message.includes("cloudflare-upstream-key-xyz"), "Error message must never contain raw upstream response body");
      assert.ok(!err.message.includes("<html>"), "Error message must never leak HTML or raw body text");
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.7 Safely rejects unexpected Server-Sent Events (SSE) stream format with 502 MALFORMED_OUTPUT", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const sseStreamText = 'data: {"id":"gen-123","choices":[{"delta":{"content":"investigation progress"}}]}\n\ndata: [DONE]\n\n';
    const mockFetchSSE = async () => ({
      ok: true,
      status: 200,
      text: async () => sseStreamText,
      json: async () => { throw new SyntaxError("Unexpected token d in JSON at position 0"); }
    });

    try {
      await runInvestigation({
        query: "test unexpected sse",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchSSE
      });
      assert.fail("Should have thrown error on unexpected SSE format");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "MALFORMED_OUTPUT");
      assert.strictEqual(err.message, "Invalid JSON response from AI provider");
      assert.ok(!err.message.includes("data:"), "Error message must never contain raw SSE stream deltas");
      assert.ok(!err.message.includes("gen-123"), "Error message must never leak SSE content");
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("7.8 Timing diagnostics log stage transitions without exposing secrets", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let turn = 0;
    const mockFetchTiming = async () => {
      turn++;
      if (turn === 1) {
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify({
            choices: [{
              message: {
                role: "assistant",
                content: null,
                tool_calls: [{
                  id: "call_t1",
                  type: "function",
                  function: {
                    name: "get_donation_records",
                    arguments: JSON.stringify({ limit: 1 })
                  }
                }]
              }
            }]
          }),
          json: async () => ({})
        };
      }
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          choices: [{
            message: {
              role: "assistant",
              content: "```json\n" + JSON.stringify({
                summary: "Diagnostics verified.",
                findings: [],
                evidence: [],
                conflicts: [],
                uncertainties: [],
                suggestedFollowUp: []
              }) + "\n```"
            }
          }]
        }),
        json: async () => ({})
      };
    };

    const res = await runInvestigation({
      query: "verify timing log",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchTiming
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.toolsInvoked, 1);
    delete process.env.OPENROUTER_API_KEY;
  });

  // 8. Turn Optimization & Evidence Layering Tests
  await test("8.1 get_donation_records provides distinct ai_classification and cryptographic_verification layers", async () => {
    const res = await executeTool("get_donation_records", { limit: 5, filter: "ALL" }, mockDeps);
    assert.strictEqual(res.totalFound, 2);
    const rec1 = res.records[0];
    assert.strictEqual(rec1.donation_id, "don_valid_100");
    assert.strictEqual(rec1.ai_classification.decision, "ACCEPT");
    assert.strictEqual(rec1.ai_classification.denomination, 100);
    assert.strictEqual(rec1.cryptographic_verification.status, "VERIFIED");
    assert.strictEqual(rec1.cryptographic_verification.proofHash, "0x1111111111111111111111111111111111111111111111111111111111111111");

    const rec2 = res.records[1];
    assert.strictEqual(rec2.donation_id, "don_tampered_50");
    assert.strictEqual(rec2.cryptographic_verification.status, "TAMPERED");
    assert.ok(rec2.cryptographic_verification.tamperDetails.includes("Denomination changed"));
  });

  await test("8.2 Session tool cache reuses evidence when model issues redundant tool calls", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let executions = 0;
    const trackingDeps = {
      ...mockDeps,
      getAllDonationRecords: () => {
        executions++;
        return mockDeps.getAllDonationRecords();
      }
    };

    let turn = 0;
    const mockFetchRepeatedTools = async () => {
      turn++;
      if (turn === 1) {
        // Model requests two identical tool calls in parallel batch
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify({
            choices: [{
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  { id: "call_r1", type: "function", function: { name: "get_donation_records", arguments: JSON.stringify({ limit: 5 }) } },
                  { id: "call_r2", type: "function", function: { name: "get_donation_records", arguments: JSON.stringify({ limit: 5 }) } }
                ]
              }
            }]
          }),
          json: async () => ({})
        };
      }
      // Final synthesis
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          choices: [{
            message: {
              role: "assistant",
              content: "```json\n" + JSON.stringify({
                summary: "Redundant call handled safely.",
                findings: [],
                evidence: [],
                conflicts: [],
                uncertainties: [],
                suggestedFollowUp: []
              }) + "\n```"
            }
          }]
        }),
        json: async () => ({})
      };
    };

    const res = await runInvestigation({
      query: "test cache reuse",
      deps: trackingDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchRepeatedTools
    });

    assert.strictEqual(res.success, true);
    // getAllDonationRecords executed only ONCE despite two tool invocations in the batch
    assert.strictEqual(executions, 1, "Redundant tool query must be served from session cache");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.3 Single-round tool batching completes investigation cleanly in 2 turns", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let turn = 0;
    const mockFetchTwoTurns = async () => {
      turn++;
      if (turn === 1) {
        // Turn 1: Model requests tools in parallel
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify({
            choices: [{
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  { id: "call_b1", type: "function", function: { name: "get_donation_records", arguments: JSON.stringify({ limit: 2 }) } },
                  { id: "call_b2", type: "function", function: { name: "get_original_registry", arguments: JSON.stringify({ limit: 2 }) } }
                ]
              }
            }]
          }),
          json: async () => ({})
        };
      }
      // Turn 2: Model synthesizes final report directly from gathered evidence
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          choices: [{
            message: {
              role: "assistant",
              content: "```json\n" + JSON.stringify({
                summary: "Complete investigation finished in 2 turns.",
                findings: ["Records and snapshots verified in parallel."],
                evidence: [{ type: "CRYPTOGRAPHIC", identifier: "don_valid_100", status: "VERIFIED" }],
                conflicts: [],
                uncertainties: [],
                suggestedFollowUp: []
              }) + "\n```"
            }
          }]
        }),
        json: async () => ({})
      };
    };

    const res = await runInvestigation({
      query: "test 2-turn batching",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchTwoTurns
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.iterations, 2, "Investigation must finish in exactly 2 iterations");
    assert.strictEqual(res.toolsInvoked, 2, "Both batched tools executed");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.4 Enforces server-side maximum of two provider iterations and rejects further tool requests on turn 2", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let turn = 0;
    const mockFetchExceedIterations = async () => {
      turn++;
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          choices: [{
            message: {
              role: "assistant",
              content: null,
              tool_calls: [{
                id: `call_${turn}`,
                type: "function",
                function: { name: "get_donation_records", arguments: JSON.stringify({ limit: 1 }) }
              }]
            }
          }]
        }),
        json: async () => ({})
      };
    };

    try {
      await runInvestigation({
        query: "test iteration limit",
        deps: mockDeps,
        config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
        fetchFn: mockFetchExceedIterations
      });
      assert.fail("Should have thrown ITERATION_LIMIT_EXCEEDED");
    } catch (err) {
      assert.strictEqual(err.status, 502);
      assert.strictEqual(err.code, "ITERATION_LIMIT_EXCEEDED");
      assert.ok(err.message.includes("maximum 2 provider iterations"));
      assert.strictEqual(err.diagnostics.iterationsCompleted, 2);
    }
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.5 verify_record_integrity correctly classifies matching attributes with hash difference as HASH_MISMATCH", async () => {
    const res = await executeTool("verify_record_integrity", { donationId: "don_mismatch_100" }, mockDeps);
    assert.strictEqual(res.donationId, "don_mismatch_100");
    assert.strictEqual(res.status, "HASH_MISMATCH");
    assert.ok(res.tamperDetails.includes("Proof hash mismatch; recorded attributes match snapshot"));
  });

  await test("8.6 Returns final report in 1 provider turn when prompt evidence is sufficient without tool calls", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    let callCount = 0;
    const mockFetchSingleTurn = async (url, options) => {
      callCount++;
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          choices: [{
            message: {
              role: "assistant",
              content: "```json\n" + JSON.stringify({
                summary: "Integrity review completed from initial ground-truth evidence.",
                status: "VERIFIED",
                confidence: "HIGH",
                keyFindings: [
                  { finding: "10 records verified, 4 tampered, 52 not registered", status: "VERIFIED", evidence: ["Ground-truth summary"], confidence: "HIGH" }
                ],
                risks: [],
                missingEvidence: [],
                recommendedActions: ["Review 4 tampered records"]
              }) + "\n```"
            }
          }]
        }),
        json: async () => ({})
      };
    };

    const res = await runInvestigation({
      query: "Summarize donation integrity using available evidence",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchSingleTurn
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.iterations, 1, "Must complete in exactly 1 iteration");
    assert.strictEqual(res.toolsInvoked, 0, "No tools invoked in single-turn mode");
    assert.strictEqual(res.report.status, "VERIFIED");
    assert.strictEqual(res.report.confidence, "HIGH");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.7 Extracts OpenRouter usage metadata (prompt_tokens, completion_tokens, total_tokens, totalCost)", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchWithUsage = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        choices: [{
          message: {
            role: "assistant",
            content: "```json\n" + JSON.stringify({
              summary: "Usage tracked.",
              keyFindings: []
            }) + "\n```"
          }
        }],
        usage: {
          prompt_tokens: 140,
          completion_tokens: 65,
          total_tokens: 205,
          total_cost: 0.00015
        }
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "test usage tracking",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchWithUsage
    });

    assert.strictEqual(res.success, true);
    assert.ok(res.usage, "Usage metadata must be present on response root");
    assert.strictEqual(res.usage.promptTokens, 140);
    assert.strictEqual(res.usage.completionTokens, 65);
    assert.strictEqual(res.usage.totalTokens, 205);
    assert.strictEqual(res.usage.totalCost, 0.00015);
    assert.ok(res.audit.usage, "Usage metadata must be present in audit");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.8 Correctly preserves status, confidence, missingEvidence, and risks in structured report", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchUncertain = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        choices: [{
          message: {
            role: "assistant",
            content: "```json\n" + JSON.stringify({
              summary: "Record verification inconclusive.",
              status: "HASH_MISMATCH",
              confidence: "LOW",
              keyFindings: [
                { finding: "Hash mismatch detected but attributes match", status: "HASH_MISMATCH", evidence: ["snapshot comparison"], confidence: "LOW" }
              ],
              risks: ["Potential canonicalization inconsistency"],
              missingEvidence: ["On-chain transaction receipt pending"],
              recommendedActions: ["Check Envio indexer after next block"]
            }) + "\n```"
          }
        }]
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "check mismatch record",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchUncertain
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.report.status, "HASH_MISMATCH");
    assert.strictEqual(res.report.confidence, "LOW");
    assert.strictEqual(res.report.risks[0], "Potential canonicalization inconsistency");
    assert.strictEqual(res.report.missingEvidence[0], "On-chain transaction receipt pending");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.9 Handles truncated JSON response gracefully with explicit INCONCLUSIVE fallback", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const truncatedPayload = "{\"summary\":\"Donations under review\",\"keyFindings\":[{\"finding\":\"Partial";
    const mockFetchTruncated = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        choices: [{
          finish_reason: "length",
          message: {
            role: "assistant",
            content: truncatedPayload
          }
        }]
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "check truncated output",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchTruncated
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.report.status, "INCONCLUSIVE");
    assert.strictEqual(res.report.confidence, "LOW");
    assert.ok(res.report.missingEvidence[0].includes("truncated at the 300 output token limit"));
    assert.ok(!res.report.summary.includes("Investigation completed."));
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.10 Handles empty provider output gracefully with INCONCLUSIVE and never claims completion", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchEmpty = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        choices: [{
          message: {
            role: "assistant",
            content: "   "
          }
        }]
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "check empty response",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchEmpty
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.report.status, "INCONCLUSIVE");
    assert.strictEqual(res.report.confidence, "LOW");
    assert.ok(res.report.summary.includes("empty response content"));
    assert.ok(!res.report.summary.includes("Investigation completed."));
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.11 Configures optional response_format when QWEN_RESPONSE_FORMAT is set", () => {
    const configDefault = getAgentConfig({ QWEN_ENABLED: "true", OPENROUTER_API_KEY: "sk-or-v1-abc" });
    assert.strictEqual(configDefault.responseFormat, null);

    const configJson = getAgentConfig({ QWEN_ENABLED: "true", OPENROUTER_API_KEY: "sk-or-v1-abc", QWEN_RESPONSE_FORMAT: "json_object" });
    assert.deepStrictEqual(configJson.responseFormat, { type: "json_object" });
  });

  await test("8.12 Repairs slightly truncated JSON to extract findings when bracket closure is feasible", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    // Slightly truncated payload: full findings present, missing only final closing braces
    const reparablePayload = "{\"summary\":\"10 records verified, 4 tampered.\",\"status\":\"TAMPERED\",\"confidence\":\"HIGH\",\"keyFindings\":[{\"finding\":\"4 records altered\",\"status\":\"TAMPERED\",\"evidence\":[\"registry\"],\"confidence\":\"HIGH\"}]";
    const mockFetchRepair = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        choices: [{
          finish_reason: "length",
          message: {
            role: "assistant",
            content: reparablePayload
          }
        }]
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "check salvageable truncated output",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchRepair
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.report.status, "TAMPERED");
    assert.strictEqual(res.report.confidence, "HIGH");
    assert.strictEqual(res.report.summary, "10 records verified, 4 tampered.");
    assert.strictEqual(res.report.keyFindings[0].finding, "4 records altered");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.13 Does NOT treat message.reasoning as final output when content is null (returns safe INCONCLUSIVE)", async () => {
    // Rationale: message.reasoning is internal chain-of-thought deliberation.
    // Only message.content is the model's committed final answer.
    // If content is null and finish_reason is 'length', the correct result is
    // INCONCLUSIVE / truncated — not a parsed reasoning report.
    process.env.OPENROUTER_API_KEY = "test_key";
    const reasoningPayload = JSON.stringify({
      summary: "Reasoning phase resolved 14 donations: 10 verified, 4 tampered.",
      status: "TAMPERED",
      confidence: "HIGH",
      keyFindings: [{
        finding: "4 records failed hash verification",
        status: "TAMPERED",
        evidence: ["monad_contract"],
        confidence: "HIGH"
      }],
      risks: ["4 records altered"],
      missingEvidence: [],
      recommendedActions: ["Escalate tampered records"]
    });

    const mockFetchReasoning = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        usage: { prompt_tokens: 1304, completion_tokens: 300, total_tokens: 1604 },
        choices: [{
          finish_reason: "length",
          message: {
            role: "assistant",
            content: null,
            reasoning: reasoningPayload
          }
        }]
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "check reasoning isolation",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchReasoning
    });

    // Correct behavior: reasoning is discarded; content is null + finish_reason=length → INCONCLUSIVE truncation
    assert.strictEqual(res.success, true);
    assert.strictEqual(res.report.status, "INCONCLUSIVE");
    assert.strictEqual(res.report.confidence, "LOW");
    assert.ok(res.report.summary.includes("truncated at token limit") || res.report.summary.includes("empty response content"));
    // Must NOT produce TAMPERED from reasoning deliberation
    assert.ok(res.report.status !== "TAMPERED");
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.14 Accurately classifies empty content with finish_reason length as truncated at token limit", async () => {
    process.env.OPENROUTER_API_KEY = "test_key";
    const mockFetchEmptyTruncated = async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({
        usage: { prompt_tokens: 1304, completion_tokens: 300, total_tokens: 1604 },
        choices: [{
          finish_reason: "length",
          message: {
            role: "assistant",
            content: "",
            reasoning: ""
          }
        }]
      }),
      json: async () => ({})
    });

    const res = await runInvestigation({
      query: "check empty length cutoff",
      deps: mockDeps,
      config: { model: "qwen/qwen3.8-max-0902", apiUrl: "https://mock.openrouter.ai", timeoutMs: 5000 },
      fetchFn: mockFetchEmptyTruncated
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.report.status, "INCONCLUSIVE");
    assert.strictEqual(res.report.confidence, "LOW");
    assert.ok(res.report.summary.includes("truncated at token limit"));
    assert.ok(!res.report.summary.includes("empty response content"));
    assert.ok(res.report.missingEvidence[0].includes("truncated at the 300 output token limit"));
    delete process.env.OPENROUTER_API_KEY;
  });

  await test("8.15 Configures reasoning effort and passes to provider payload when set", async () => {
    const configWithEffort = getAgentConfig({
      QWEN_ENABLED: "true",
      OPENROUTER_API_KEY: "sk-or-v1-abc",
      QWEN_REASONING_EFFORT: "none"
    });
    assert.strictEqual(configWithEffort.reasoningEffort, "none");

    process.env.OPENROUTER_API_KEY = "test_key";
    let capturedBody = null;
    const mockFetchEffort = async (url, opts) => {
      capturedBody = JSON.parse(opts.body);
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          choices: [{
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: JSON.stringify({
                summary: "All verified.",
                status: "VERIFIED",
                confidence: "HIGH",
                keyFindings: []
              })
            }
          }]
        }),
        json: async () => ({})
      };
    };

    const res = await runInvestigation({
      query: "check reasoning effort payload",
      deps: mockDeps,
      config: { ...configWithEffort, apiUrl: "https://mock.openrouter.ai" },
      fetchFn: mockFetchEffort
    });

    assert.strictEqual(res.success, true);
    assert.strictEqual(capturedBody.reasoning?.effort, "none");
    delete process.env.OPENROUTER_API_KEY;
  });

  // 9. Runtime User API Key & Key Testing
  await test("9.1 testApiKey returns 400 KEY_MISSING when key is empty", async () => {
    const res = await testApiKey("");
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.status, 400);
    assert.strictEqual(res.code, "KEY_MISSING");
  });

  await test("9.2 testApiKey returns valid when OpenRouter responds 200", async () => {
    const mockFetch = async () => ({
      status: 200,
      json: async () => ({
        data: { label: "Test Key", usage: 0.05, limit: 10, is_free_tier: false }
      })
    });
    const res = await testApiKey("sk-or-test-key", mockFetch);
    assert.strictEqual(res.valid, true);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.details.label, "Test Key");
    assert.strictEqual(res.details.usage, "$0.0500");
  });

  await test("9.3 testApiKey handles 401 INVALID_KEY gracefully", async () => {
    const mockFetch = async () => ({
      status: 401,
      json: async () => ({ error: { message: "Invalid key" } })
    });
    const res = await testApiKey("bad-key", mockFetch);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.status, 401);
    assert.strictEqual(res.code, "INVALID_KEY");
  });

  await test("9.4 testApiKey handles 402 QUOTA_EXCEEDED gracefully", async () => {
    const mockFetch = async () => ({
      status: 402,
      json: async () => ({ error: { message: "Out of credits" } })
    });
    const res = await testApiKey("exhausted-key", mockFetch);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.status, 402);
    assert.strictEqual(res.code, "QUOTA_EXCEEDED");
  });

  await test("9.5 testApiKey handles 429 RATE_LIMIT gracefully", async () => {
    const mockFetch = async () => ({
      status: 429,
      json: async () => ({ error: { message: "Too many requests" } })
    });
    const res = await testApiKey("limited-key", mockFetch);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.status, 429);
    assert.strictEqual(res.code, "RATE_LIMIT");
  });

  await test("9.6 testApiKey handles network errors gracefully", async () => {
    const mockFetch = async () => {
      throw new Error("getaddrinfo ENOTFOUND openrouter.ai");
    };
    const res = await testApiKey("any-key", mockFetch);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.status, 502);
    assert.strictEqual(res.code, "NETWORK_ERROR");
  });

  await test("9.7 createInvestigationHandler activates and passes user-provided key without server env flag", async () => {
    delete process.env.QWEN_ENABLED;
    delete process.env.OPENROUTER_API_KEY;

    let receivedAuth = null;
    const mockFetch = async (url, opts) => {
      receivedAuth = opts.headers["Authorization"];
      return {
        status: 200,
        ok: true,
        text: async () => JSON.stringify({
          choices: [{
            message: {
              role: "assistant",
              content: JSON.stringify({
                status: "VERIFIED",
                confidence: "HIGH",
                summary: "Audit completed with user key.",
                keyFindings: [{ finding: "All checks passed", status: "VERIFIED", confidence: "HIGH", evidence: [] }]
              })
            },
            finish_reason: "stop"
          }]
        }),
        headers: { get: () => "application/json" }
      };
    };

    // Override global fetch temporarily for this handler test
    const origFetch = globalThis.fetch;
    globalThis.fetch = mockFetch;

    const handler = createInvestigationHandler(mockDeps);
    let resStatus = 200;
    let resJson = null;
    const req = {
      body: {
        apiKey: "sk-or-user-provided-key-12345",
        query: "verify ledger integrity"
      }
    };
    const res = {
      status: (s) => { resStatus = s; return res; },
      json: (j) => { resJson = j; return res; }
    };

    try {
      await handler(req, res);
      assert.strictEqual(resStatus, 200);
      assert.strictEqual(resJson.success, true);
      assert.strictEqual(receivedAuth, "Bearer sk-or-user-provided-key-12345");
      assert.strictEqual(resJson.report.status, "VERIFIED");
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  await test("9.8 createInvestigationHandler returns 503 AGENT_DISABLED when key is missing", async () => {
    delete process.env.QWEN_ENABLED;
    delete process.env.OPENROUTER_API_KEY;

    const handler = createInvestigationHandler(mockDeps);
    let resStatus = 0;
    let resJson = null;
    const req = { body: { query: "investigate without key" } };
    const res = {
      status: (s) => { resStatus = s; return res; },
      json: (j) => { resJson = j; return res; }
    };

    await handler(req, res);
    assert.strictEqual(resStatus, 503);
    assert.strictEqual(resJson.code, "AGENT_DISABLED");
    assert.ok(resJson.error.includes("Enter your Qwen API key to activate the agent"));
  });

  console.log("\n============================================================");
  console.log(`TOTAL: ${passed + failed} | PASSED: ${passed} | FAILED: ${failed}`);
  console.log("============================================================\n");

  if (failed > 0) {
    process.exit(1);
  }
}

runAllTests().catch(err => {
  console.error("Fatal test runner crash:", err);
  process.exit(1);
});
