/**
 * One-shot patch: replaces AGENT_TOOLS block in investigation-agent.js
 * with compact tool descriptions that reduce token usage.
 * Run once, then delete.
 */
"use strict";
const fs = require("fs");
const path = require("path");

const filePath = path.join(__dirname, "../services/investigation-agent.js");
let src = fs.readFileSync(filePath, "utf8");

// ─── 1. Replace AGENT_TOOLS block ────────────────────────────────────────────
const toolsStart = "/**\n * Tool definitions exposed to the Qwen model.\n */\nconst AGENT_TOOLS = [";
const toolsEnd   = "];\n\n/**\n * System prompt";

const si = src.indexOf(toolsStart);
const ei = src.indexOf(toolsEnd);
if (si === -1 || ei === -1) {
  console.error("Could not locate AGENT_TOOLS block. Aborting.");
  process.exit(1);
}

const compactTools = `/**
 * Tool definitions exposed to the Qwen model.
 * Keep descriptions concise: every token saved here reduces Turn 1 request cost.
 */
const AGENT_TOOLS = [
  {
    type: "function",
    function: {
      name: "get_donation_records",
      description: "Fetch recent donation records with cryptographic integrity status (VERIFIED/TAMPERED/HASH_MISMATCH/NOT_REGISTERED) and a summary count. Each record has ai_classification and cryptographic_verification. Use limit\u226410 unless broader coverage is required.",
      parameters: {
        type: "object",
        properties: {
          limit: { type: "integer", description: "Max records (1\u201320, default 10)", minimum: 1, maximum: 20 },
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
      description: "Retrieve one donation record by donation_id. Use for targeted single-record investigations.",
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
      description: "SHA-256 fingerprint check for a single donation. Returns VERIFIED, TAMPERED, HASH_MISMATCH, or NOT_REGISTERED with tamper details. Preferred for single-record verification.",
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
      description: "Read-only Monad contract lookup for a proof hash. Returns exists, sourceId, timestamp, registeredBy.",
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
      description: "Envio HyperIndex lookup for a proof's tx hash and block. Absence means unindexed, not invalid.",
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
      description: "Frozen registration snapshots. Use only when get_donation_records or verify_record_integrity do not supply sufficient snapshot data.",
      parameters: {
        type: "object",
        properties: { limit: { type: "integer", description: "Max snapshots (1\u201320)", minimum: 1, maximum: 20 } },
        additionalProperties: false
      }
    }
  }`;

src = src.slice(0, si) + compactTools + src.slice(ei);

fs.writeFileSync(filePath, src, "utf8");
console.log("✅ AGENT_TOOLS block replaced successfully.");
console.log("New file size:", Buffer.byteLength(src, "utf8"), "bytes");
