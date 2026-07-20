// ================================================================================
// 生成プロンプトをクリップボードへコピーする UI 補助スクリプト
// ================================================================================
(function () {
    var PROMPT_IDS = [
        "llm_prompt_ollama_idea_generated_prompt",
        "llm_prompt_ollama_vlm_generated_prompt",
        // legacy id (pre 3-tab layout)
        "llm_prompt_ollama_generated_prompt",
    ];
    var BTN_IDS = [
        "llm_prompt_ollama_idea_copy_prompt_btn",
        "llm_prompt_ollama_vlm_copy_prompt_btn",
        "llm_prompt_ollama_copy_prompt_btn",
    ];

    // ================================================================================
    // 指定ルート配下のテキストエリア値を取得する
    // ================================================================================
    function getPromptTextNear(btn) {
        var root = btn.closest(".tabitem, .gradio-container, body") || document;
        for (var i = 0; i < PROMPT_IDS.length; i++) {
            var el = root.querySelector("#" + PROMPT_IDS[i]);
            if (!el) continue;
            var ta = el.querySelector("textarea");
            if (ta && ta.value) return ta.value;
        }
        for (var j = 0; j < PROMPT_IDS.length; j++) {
            var global = document.getElementById(PROMPT_IDS[j]);
            if (!global) continue;
            var gta = global.querySelector("textarea");
            if (gta) return gta.value || "";
        }
        return "";
    }

    // ================================================================================
    // Copy ボタン要素を解決する
    // ================================================================================
    function resolveButton(id) {
        var el = document.getElementById(id);
        if (!el) return null;
        if (el.tagName === "BUTTON") return el;
        return el.querySelector("button") || el;
    }

    // ================================================================================
    // クリップボードへ書き込む
    // ================================================================================
    async function copyText(text) {
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
        } catch (err) {
            var ta = document.createElement("textarea");
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
        }
    }

    // ================================================================================
    // Copy ボタンにクリップボード書き込みを紐付ける
    // ================================================================================
    function bind() {
        BTN_IDS.forEach(function (id) {
            var btn = resolveButton(id);
            if (!btn || btn.dataset.llmPromptOllamaCopyBound === "1") return;
            btn.dataset.llmPromptOllamaCopyBound = "1";
            btn.addEventListener("click", function () {
                copyText(getPromptTextNear(btn));
            });
        });
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(bind);
    } else {
        document.addEventListener("DOMContentLoaded", bind);
        setTimeout(bind, 1500);
    }
})();
