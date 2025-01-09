import { Buffer } from "node:buffer";

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "genai-js/0.21.0";
const DEFAULT_MODEL = "gemini-1.5-pro-latest";
const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
const harmCategory = [
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
    category,
    threshold: "BLOCK_NONE",
}));
const reasonsMap = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
};
const delimiter = "\n\n";
const fieldsMap = {
    stop: "stopSequences",
    n: "candidateCount",
    max_tokens: "maxOutputTokens",
    max_completion_tokens: "maxOutputTokens",
    temperature: "temperature",
    top_p: "topP",
    top_k: "topK",
    frequency_penalty: "frequencyPenalty",
    presence_penalty: "presencePenalty",
};

const makeHeaders = (apiKey, more) => ({
    "x-goog-api-client": API_CLIENT,
    ...(apiKey && { "x-goog-api-key": apiKey }),
    ...more
});

const fixCors = ({ headers, status, statusText }) => {
    headers = new Headers(headers);
    headers.set("Access-Control-Allow-Origin", "*");
    return { headers, status, statusText };
};

const handleOPTIONS = async () => {
    return new Response(null, {
        headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    });
};

class HttpError extends Error {
    constructor(message, status) {
        super(message);
        this.name = this.constructor.name;
        this.status = status;
    }
}

const assertMethod = (request, method) => {
    if (request.method !== method) {
        throw new HttpError(`Method Not Allowed. Expected ${method}, but got ${request.method}`, 405);
    }
};

const handleErrors = (err) => {
    console.error(err);
    return new Response(JSON.stringify({ error: { message: err.message, type: err.name, status: err.status || 500 } }), fixCors({ status: err.status ?? 500, headers: { 'Content-Type': 'application/json' } }));
};

const generateChatcmplId = () => {
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
    return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const transformConfig = (req) => {
    let cfg = {};
    for (let key in req) {
        const matchedKey = fieldsMap[key];
        if (matchedKey) {
            cfg[matchedKey] = req[key];
        }
    }
    if (req.response_format) {
        switch (req.response_format.type) {
            case "json_schema":
                cfg.responseSchema = req.response_format.json_schema?.schema;
                if (cfg.responseSchema && "enum" in cfg.responseSchema) {
                    cfg.responseMimeType = "text/x.enum";
                    break;
                }
            case "json_object":
                cfg.responseMimeType = "application/json";
                break;
            case "text":
                cfg.responseMimeType = "text/plain";
                break;
            default:
                throw new HttpError("Unsupported response_format.type", 400);
        }
    }
    return cfg;
};

const parseImg = async (url) => {
    let mimeType, data;
    if (url.startsWith("http://") || url.startsWith("https://")) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`${response.status} ${response.statusText} (${url})`);
            }
            mimeType = response.headers.get("content-type");
            data = Buffer.from(await response.arrayBuffer()).toString("base64");
        } catch (err) {
            throw new Error("Error fetching image: " + err.toString());
        }
    } else {
        const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
        if (!match) {
            throw new Error("Invalid image data: " + url);
        }
        ({ mimeType, data } = match.groups);
    }
    return {
        inlineData: {
            mimeType,
            data,
        },
    };
};

const transformMsg = async ({ role, content }) => {
    const parts = [];
    if (!Array.isArray(content)) {
        parts.push({ text: content });
        return { role, parts };
    }
    for (const item of content) {
        switch (item.type) {
            case "text":
                parts.push({ text: item.text });
                break;
            case "image_url":
                parts.push(await parseImg(item.image_url.url));
                break;
            case "input_audio":
                parts.push({
                    inlineData: {
                        mimeType: "audio/" + item.input_audio.format,
                        data: item.input_audio.data,
                    }
                });
                break;
            default:
                throw new TypeError(`Unknown "content" item type: "${item.type}"`);
        }
    }
    if (content.every(item => item.type === "image_url")) {
        parts.push({ text: "" });
    }
    return { role, parts };
};

const transformMessages = async (messages) => {
    if (!messages) { return; }
    const contents = [];
    let system_instruction;
    for (const item of messages) {
        if (item.role === "system") {
            delete item.role;
            system_instruction = await transformMsg(item);
        } else {
            item.role = item.role === "assistant" ? "model" : "user";
            contents.push(await transformMsg(item));
        }
    }
    if (system_instruction && contents.length === 0) {
        contents.push({ role: "model", parts: { text: " " } });
    }
    return { system_instruction, contents };
};

const transformRequest = async (req) => {
    const transformed = {
        ...await transformMessages(req.messages),
        safetySettings,
        generationConfig: transformConfig(req),
    };
    if (req.tools) {
        transformed.tools = req.tools;
    }
    if (req.tool_choice) {
        transformed.tool_choice = req.tool_choice;
    }
    return transformed;
};

const transformCandidates = (key, cand) => ({
    index: cand.index || 0,
    [key]: {
        role: "assistant",
        content: cand.content?.parts.map(p => p.text).join(delimiter),
        tool_calls: cand.toolCalls?.map(toolCall => ({
            id: toolCall.id,
            type: "function",
            function: {
                name: toolCall.function.name,
                arguments: toolCall.function.args,
            }
        }))
    },
    logprobs: null,
    finish_reason: reasonsMap[cand.finishReason] || cand.finishReason,
});
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data) => ({
    completion_tokens: data.candidatesTokenCount,
    prompt_tokens: data.promptTokenCount,
    total_tokens: data.totalTokenCount
});

const processCompletionsResponse = (data, model, id) => {
    return JSON.stringify({
        id,
        choices: data.candidates.map(transformCandidatesMessage),
        created: Math.floor(Date.now() / 1000),
        model,
        object: "chat.completion",
        usage: transformUsage(data.usageMetadata),
    });
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream(chunk, controller) {
    chunk = await chunk;
    if (!chunk) { return; }
    this.buffer += chunk;
    do {
        const match = this.buffer.match(responseLineRE);
        if (!match) { break; }
        controller.enqueue(match[1]);
        this.buffer = this.buffer.substring(match[0].length);
    } while (true);
}
async function parseStreamFlush(controller) {
    if (this.buffer) {
        console.error("Invalid data:", this.buffer);
        controller.enqueue(this.buffer);
    }
}

function transformResponseStream(data, stop, first) {
    const item = transformCandidatesDelta(data.candidates[0]);
    if (stop) { item.delta = {}; } else { item.finish_reason = null; }
    if (first) { item.delta.content = ""; } else { delete item.delta.role; }
    const output = {
        id: this.id,
        choices: [item],
        created: Math.floor(Date.now() / 1000),
        model: this.model,
        object: "chat.completion.chunk",
    };
    if (data.usageMetadata && this.streamIncludeUsage) {
        output.usage = stop ? transformUsage(data.usageMetadata) : null;
    }
    return "data: " + JSON.stringify(output) + delimiter;
}

async function toOpenAiStream(chunk, controller) {
    const transform = transformResponseStream.bind(this);
    const line = await chunk;
    if (!line) { return; }
    let data;
    try {
        data = JSON.parse(line);
    } catch (err) {
        console.error(line);
        console.error(err);
        const length = this.last.length || 1;
        const candidates = Array.from({ length }, (_, index) => ({
            finishReason: "error",
            content: { parts: [{ text: err.message }] },
            index,
        }));
        data = { candidates };
    }
    const cand = data.candidates[0];
    console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
    cand.index = cand.index || 0;
    if (!this.last[cand.index]) {
        controller.enqueue(transform(data, false, "first"));
    }
    this.last[cand.index] = data;
    if (cand.content || cand.toolCalls) {
        controller.enqueue(transform(data));
    }
}
async function toOpenAiStreamFlush(controller) {
    const transform = transformResponseStream.bind(this);
    if (this.last.length > 0) {
        for (const data of this.last) {
            controller.enqueue(transform(data, "stop"));
        }
        controller.enqueue("data: [DONE]" + delimiter);
    }
}

async function handleModels(apiKey) {
    const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
        headers: makeHeaders(apiKey),
    });
    if (!response.ok) {
        throw new HttpError(`Failed to fetch models: ${response.status} ${response.statusText}`, response.status);
    }
    const { models } = await response.json();
    const body = JSON.stringify({
        object: "list",
        data: models.map(({ name }) => ({
            id: name.replace("models/", ""),
            object: "model",
            created: 0,
            owned_by: "",
        })),
    }, null, "  ");
    return new Response(body, fixCors(response));
}

async function handleEmbeddings(req, apiKey) {
    if (typeof req.model !== "string") {
        throw new HttpError("model is not specified", 400);
    }
    if (!Array.isArray(req.input)) {
        req.input = [req.input];
    }
    let model;
    if (req.model.startsWith("models/")) {
        model = req.model;
    } else {
        req.model = DEFAULT_EMBEDDINGS_MODEL;
        model = "models/" + req.model;
    }
    const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
        method: "POST",
        headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
        body: JSON.stringify({
            "requests": req.input.map(text => ({
                model,
                content: { parts: { text } },
                outputDimensionality: req.dimensions,
            }))
        })
    });
    if (!response.ok) {
        throw new HttpError(`Failed to fetch embeddings: ${response.status} ${response.statusText}`, response.status);
    }
    const { embeddings } = await response.json();
    const body = JSON.stringify({
        object: "list",
        data: embeddings.map(({ values }, index) => ({
            object: "embedding",
            index,
            embedding: values,
        })),
        model: req.model,
    }, null, "  ");
    return new Response(body, fixCors(response));
}

async function handleCompletions(req, apiKey) {
    let model = DEFAULT_MODEL;
    switch (true) {
        case typeof req.model !== "string":
            break;
        case req.model.startsWith("models/"):
            model = req.model.substring(7);
            break;
        case req.model.startsWith("gemini-"):
        case req.model.startsWith("learnlm-"):
            model = req.model;
    }
    const TASK = req.stream ? "streamGenerateContent" : "generateContent";
    let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
    if (req.stream) { url += "?alt=sse"; }
    
    const transformedRequest = await transformRequest(req);
    console.log("Transformed Request:", JSON.stringify(transformedRequest, null, 2)); // Log the transformed request
    
    const response = await fetch(url, {
        method: "POST",
        headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
        body: JSON.stringify(transformedRequest),
    });
    
    if (!response.ok) {
        const errorText = await response.text();
        console.error("Gemini API Error:", response.status, response.statusText, errorText); // Log the error response
        throw new HttpError(`Failed to fetch completions: ${response.status} ${response.statusText} ${errorText}`, response.status);
    }
    
    let body = response.body;
    if (req.stream) {
        body = response.body
            .pipeThrough(new TextDecoderStream())
            .pipeThrough(new TransformStream({
                transform: parseStream,
                flush: parseStreamFlush,
                buffer: "",
            }))
            .pipeThrough(new TransformStream({
                transform: toOpenAiStream,
                flush: toOpenAiStreamFlush,
                streamIncludeUsage: req.stream_options?.include_usage,
                model, id: generateChatcmplId(), last: [],
            }))
            .pipeThrough(new TextEncoderStream());
    } else {
        body = await response.text();
        body = processCompletionsResponse(JSON.parse(body), model, generateChatcmplId());
    }
    return new Response(body, fixCors(response));
}

export default {
    async fetch(request) {
        if (request.method === "OPTIONS") {
            return handleOPTIONS();
        }
        try {
            const auth = request.headers.get("Authorization");
            const apiKey = auth?.split(" ")[1];
            const { pathname } = new URL(request.url);
            const openaiBase = "/v1beta/openai";
            switch (true) {
                case pathname.startsWith(openaiBase + "/chat/completions"):
                    assertMethod(request, "POST");
                    return handleCompletions(await request.json(), apiKey)
                        .catch(handleErrors);
                case pathname.startsWith(openaiBase + "/embeddings"):
                    assertMethod(request, "POST");
                    return handleEmbeddings(await request.json(), apiKey)
                        .catch(handleErrors);
                case pathname.startsWith(openaiBase + "/models"):
                    assertMethod(request, "GET");
                    return handleModels(apiKey)
                        .catch(handleErrors);
                default:
                    throw new HttpError("404 Not Found", 404);
            }
        } catch (err) {
            return handleErrors(err);
        }
    }
};
