
require("dotenv").config();

async function testQwen() {
    const apiKey = process.env.OPENROUTER_API_KEY;
    const model =
        process.env.QWEN_MODEL || "qwen/qwen3.8-max-0902";

    if (!apiKey) {
        console.error("OPENROUTER_API_KEY is missing from .env");
        return;
    }

    console.log("Selected model:", model);

    const response = await fetch(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            method: "POST",
            headers: {
                Authorization: `Bearer ${apiKey}`,
                "Content-Type": "application/json",
                Accept: "application/json"
            },
            body: JSON.stringify({
                model,
                messages: [
                    {
                        role: "user",
                        content:
                            "In two sentences, explain how donation integrity can be checked."
                    }
                ],
                max_tokens: 256,
                stream: false
            })
        }
    );

    const data = await response.json();

    if (!response.ok) {
        console.error("OpenRouter error:", response.status, data);
        return;
    }

    const answer = data.choices?.[0]?.message?.content;

    if (!answer) {
        console.error("No answer returned by the model:", data);
        return;
    }

    console.log("\nQwen response:\n", answer);
}

testQwen().catch((error) => {
    console.error("Request failed:", error.message);
});
